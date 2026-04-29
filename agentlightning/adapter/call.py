# Copyright (c) Microsoft. All rights reserved.

"""Handles chat completion / response calls. Extracts them from spans and annotates them with annotations."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, cast

from openai.types.chat import (
    ChatCompletion,
    CompletionCreateParams,
)
from pydantic import TypeAdapter

from agentlightning.types.adapter import (
    AdaptingSpan,
    AnnotatedChatCompletionCall,
    Annotation,
    BaseAdaptingSequence,
    ChatCompletionCall,
    Tree,
)
from agentlightning.types.tracer import Span
from agentlightning.utils.otel import filter_and_unflatten_attributes, query_linked_spans
from agentlightning.utils.pydantic import to_plain_object

from .base import SequenceAdapter

CompletionCreateParamsType: TypeAdapter[CompletionCreateParams] = TypeAdapter(CompletionCreateParams)


class IdentifyChatCompletionCalls(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
    """Curate the chat completion calls from the spans."""

    def _parse_request(self, span: AdaptingSpan) -> Dict[str, Any]:
        request = filter_and_unflatten_attributes(span.attributes, "gen_ai.request")
        if not isinstance(request, dict):
            raise ValueError(f"Invalid request format in span attributes: {request}")
        if "functions" in request:
            if not isinstance(request["functions"], list):
                raise ValueError(f"Invalid functions format in request. Must be a list: {request['functions']}")
            for function in cast(List[Any], request["functions"]):
                if not isinstance(function, dict):
                    raise ValueError(f"Invalid function format in request. Must be a dict: {function}")
                if "parameters" in function and isinstance(function["parameters"], str):
                    function["parameters"] = json.loads(function["parameters"])
        return request

    def _parse_response(self, span: AdaptingSpan) -> Dict[str, Any]:
        response = filter_and_unflatten_attributes(span.attributes, "gen_ai.response")
        if not isinstance(response, dict):
            raise ValueError(f"Invalid response format in span attributes: {response}")
        if "created" not in response:
            response["created"] = int(span.ensure_end_time())
        if "object" not in response:
            response["object"] = "chat.completion"
        return response

    def _parse_completion_choices(self, span: AdaptingSpan) -> List[Dict[str, Any]]:
        completion_choices = filter_and_unflatten_attributes(span.attributes, "gen_ai.completion")
        if not isinstance(completion_choices, list):
            raise ValueError(f"Invalid completion choices format in span attributes: {completion_choices}")
        for index, choice in enumerate(completion_choices):
            if not isinstance(choice, dict):
                raise ValueError(
                    f"Invalid completion choice format in span attributes. Choice must be a dict: {choice}"
                )

            choice["index"] = index

            # Uncover the message from the choice
            message: Dict[str, Any] = {
                "role": "assistant",
            }
            if "content" in choice:
                message["content"] = cast(Dict[str, Any], choice).pop("content")
            if isinstance(span.container, Tree):
                # Get additional fields from child spans if any
                for child in span.children():
                    tool_call = self._parse_agentops_tool_calls(child)
                    if tool_call is not None:
                        message.setdefault("tool_calls", []).append(tool_call)
            choice["message"] = message

        return completion_choices

    def _parse_prompt_messages(self, span: AdaptingSpan) -> List[Dict[str, Any]]:
        prompt_messages = filter_and_unflatten_attributes(span.attributes, "gen_ai.prompt")
        if not isinstance(prompt_messages, list) or not all(isinstance(msg, dict) for msg in prompt_messages):
            raise ValueError(f"Invalid prompt messages format in span attributes: {prompt_messages}")

        # Fix discrepency between OpenAI API and AGL span attributes.
        for message in prompt_messages:
            if not isinstance(message, dict):
                raise ValueError(f"Invalid message format in prompt messages. Must be a dict: {message}")
            if "tool_calls" in message:
                if not isinstance(message["tool_calls"], list):
                    raise ValueError(f"Invalid tool calls format in message. Must be a list: {message['tool_calls']}")
                for tool_call in cast(List[Any], message["tool_calls"]):
                    if not isinstance(tool_call, dict):
                        raise ValueError(f"Invalid tool call format in message. Must be a dict: {tool_call}")
                    if "type" not in tool_call:
                        tool_call["type"] = "function"
                    if "function" not in tool_call and "name" in tool_call and "arguments" in tool_call:
                        tool_call["function"] = {
                            "name": cast(Dict[str, Any], tool_call).pop("name"),
                            "arguments": cast(Dict[str, Any], tool_call).pop("arguments"),
                        }
        return prompt_messages

    def _parse_usages(self, span: AdaptingSpan) -> Dict[str, Any]:
        usages = filter_and_unflatten_attributes(span.attributes, "gen_ai.usage")
        if not isinstance(usages, dict):
            raise ValueError(f"Invalid usages format in span attributes: {usages}")
        if "prompt_tokens" not in usages:
            usages["prompt_tokens"] = 0
        if "completion_tokens" not in usages:
            usages["completion_tokens"] = 0
        if "total_tokens" not in usages:
            usages["total_tokens"] = usages["prompt_tokens"] + usages["completion_tokens"]
        return usages

    def _parse_agentops_tool_calls(self, span: Span) -> Optional[Dict[str, Any]]:
        if span.name.startswith("tool_call."):
            tool_call_data = filter_and_unflatten_attributes(span.attributes, "tool")
            if isinstance(tool_call_data, dict):
                if "call" in tool_call_data:
                    # Example tool_call_data:
                    # {'tool.name': 'get_rooms', 'tool.parameters': '{"date": ...}',
                    #  'tool.call.id': 'call_owd6', 'tool.call.type': 'function'}
                    tool_call_data = {
                        **tool_call_data["call"],
                        "function": {k: v for k, v in tool_call_data.items() if k != "call"},
                    }
                if (
                    "function" in tool_call_data
                    and isinstance(tool_call_data["function"], dict)
                    and "parameters" in tool_call_data["function"]
                ):
                    tool_call_data["function"]["arguments"] = tool_call_data["function"].pop("parameters")  # type: ignore

            return cast(Dict[str, Any], tool_call_data)
        return None

    def _normalize_request(self, request_body: Dict[str, Any]) -> CompletionCreateParams:
        validated_request = CompletionCreateParamsType.validate_python(request_body)
        return to_plain_object(validated_request, [])

    def _parse_openai_chat_completion_create(self, span: AdaptingSpan) -> ChatCompletionCall:
        prompt_messages = self._parse_prompt_messages(span)
        request_metadata = self._parse_request(span)
        completion_choices = self._parse_completion_choices(span)
        usages = self._parse_usages(span)
        response_metadata = self._parse_response(span)

        return self._construct_chat_completion_call(
            {"messages": prompt_messages, **request_metadata},
            {
                **response_metadata,
                "choices": completion_choices,
                "usage": usages,
            },
        )

    def _augment_litellm_raw_gen_ai_request(
        self, span: AdaptingSpan, request: Dict[str, Any], response: Dict[str, Any]
    ) -> None:
        """Augment the request/response with more rich info from the sibling raw_gen_ai_request span.

        The request and response are modified in place.
        """
        hosted_vllm = filter_and_unflatten_attributes(span.attributes, "llm.hosted_vllm")
        if not hosted_vllm:
            return

        if not isinstance(hosted_vllm, dict):
            raise ValueError(f"Invalid hosted_vllm format in span attributes: {hosted_vllm}")

        if "choices" in hosted_vllm:
            choices = ast.literal_eval(hosted_vllm["choices"])
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError(f"Invalid choices format in hosted_vllm: {choices}")
            if "token_ids" in choices[0]:
                response["choices"][0]["token_ids"] = choices[0]["token_ids"]

        if "prompt_token_ids" in hosted_vllm:
            request["prompt_token_ids"] = ast.literal_eval(hosted_vllm["prompt_token_ids"])

    def _parse_litellm_request(self, span: AdaptingSpan) -> ChatCompletionCall:
        prompt_messages = self._parse_prompt_messages(span)
        completion_choices = self._parse_completion_choices(span)
        usages = self._parse_usages(span)
        request_metadata = self._parse_request(span)
        response_metadata = self._parse_response(span)

        request_body = {"messages": prompt_messages, **request_metadata}
        response_body = {
            **response_metadata,
            "choices": completion_choices,
            "usage": usages,
        }

        # If the underlying backend is vllm, we have more rich info in sibling span.
        for sibling in span.siblings():
            if sibling.name == "raw_gen_ai_request":
                self._augment_litellm_raw_gen_ai_request(sibling, request_body, response_body)

        return self._construct_chat_completion_call(request_body, response_body)

    def _construct_chat_completion_call(
        self, request_body: Dict[str, Any], response_body: Dict[str, Any]
    ) -> ChatCompletionCall:
        request = self._normalize_request(request_body)
        response = ChatCompletion.model_validate(response_body, extra="allow")
        return ChatCompletionCall.model_construct(
            request=request,
            response=response,
            malformed_fields={},  # TODO: malformed fields
        )

    def adapt_one(self, source: AdaptingSpan) -> AdaptingSpan:
        if source.name == "openai.chat.completion.create" or source.name == "openai.chat.completion":
            chat_completion_call = self._parse_openai_chat_completion_create(source)
            return source.with_data(chat_completion_call)
        elif source.name == "litellm_request":
            # Litellm request span
            chat_completion_call = self._parse_litellm_request(source)
            return source.with_data(chat_completion_call)
        else:
            # Not a chat completion call span. Do nothing
            return source


class AnnotateChatCompletionCalls(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
    """Annotate chat completion calls with the given annotations.

    The intersection of "effective radius" of annotations and chat completion calls is used to determine
    which annotations apply to which chat completion calls.

    If an annotation is not linked to any span, try to use `RepairMissingLinks` first to link it to spans.
    """

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> BaseAdaptingSequence[AdaptingSpan]:
        annotation_spans = [span for span in source if isinstance(span.data, Annotation)]
        span_id_to_updated_annotation: Dict[str, AnnotatedChatCompletionCall] = {}
        for annotation_span in annotation_spans:
            annotation = cast(Annotation, annotation_span.data)
            for linked_span in query_linked_spans(source, annotation.links):
                if isinstance(linked_span.container, Tree):
                    linked_spans = list(linked_span.container.traverse())
                else:
                    linked_spans = [linked_span]

                for linked_span in linked_spans:
                    if isinstance(linked_span.data, ChatCompletionCall):
                        existing_annotations: List[Annotation] = (
                            list(linked_span.data.annotations)
                            if isinstance(linked_span.data, AnnotatedChatCompletionCall)
                            else []
                        )
                        # Annotate the chat completion call
                        annotated_call = AnnotatedChatCompletionCall(
                            request=linked_span.data.request,
                            response=linked_span.data.response,
                            malformed_fields=linked_span.data.malformed_fields,
                            annotations=existing_annotations + [annotation],
                        )
                        # Update the linked span with the annotated call
                        span_id_to_updated_annotation[linked_span.span_id] = annotated_call

        def _update_span(span: AdaptingSpan) -> AdaptingSpan:
            if span.span_id in span_id_to_updated_annotation:
                annotated_call = span_id_to_updated_annotation[span.span_id]
                return span.with_data(annotated_call, override="silent")  # override is expected here
            else:
                return span

        return source.map(_update_span)

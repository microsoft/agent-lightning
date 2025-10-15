# Copyright (c) Microsoft. All rights reserved.

import json
from collections import defaultdict
from typing import Any, Dict, Generator, Iterable, List, Optional, TypedDict, Union, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)
from pydantic import TypeAdapter

from agentlightning.types import Span

from .base import TraceAdapter


class OpenAIMessages(TypedDict):
    messages: List[ChatCompletionMessageParam]
    tools: Optional[List[ChatCompletionFunctionToolParam]]


class _RawSpanInfo(TypedDict):
    prompt: List[Dict[str, Any]]
    completion: List[Dict[str, Any]]
    request: Dict[str, Any]
    response: Dict[str, Any]
    tools: List[Dict[str, Any]]


def group_genai_dict(data: Dict[str, Any], prefix: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Convert a flat dict with keys like 'gen_ai.prompt.0.role'
    into structured nested dicts or lists under the given prefix.

    Args:
        data: Flat dictionary (keys are dotted paths).
        prefix: Top-level key to extract (e.g., 'gen_ai.prompt').

    Returns:
        A nested dict (if no index detected) or list (if indexed).
    """
    result: Union[Dict[str, Any], List[Any]] = {}

    # Collect keys that match the prefix
    relevant = {k[len(prefix) + 1 :]: v for k, v in data.items() if k.startswith(prefix + ".")}

    # Detect if we have numeric indices (-> list) or not (-> dict)
    indexed = any(part.split(".")[0].isdigit() for part in relevant.keys())

    if indexed:
        # Group by index
        grouped: Dict[int, Dict[str, Any]] = defaultdict(dict)
        for k, v in relevant.items():
            parts = k.split(".")
            if not parts[0].isdigit():
                continue
            idx, rest = int(parts[0]), ".".join(parts[1:])
            grouped[idx][rest] = v
        # Recursively build
        result = []
        for i in sorted(grouped.keys()):
            result.append(group_genai_dict({f"{prefix}.{rest}": val for rest, val in grouped[i].items()}, prefix))
    else:
        # No indices: build dict
        nested: Dict[str, Any] = defaultdict(dict)
        for k, v in relevant.items():
            if "." in k:
                head, _tail = k.split(".", 1)
                nested[head][f"{prefix}.{k}"] = v
            else:
                result[k] = v
        # Recurse into nested dicts
        for head, subdict in nested.items():
            result[head] = group_genai_dict(subdict, prefix + "." + head)

    return result


def convert_to_openai_messages(prompt_completion_list: List[_RawSpanInfo]) -> Generator[OpenAIMessages, None, None]:
    """
    Convert raw tool call traces + prompt/completion list
    into OpenAI fine-tuning JSONL format (tool calling style).

    https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-functions
    """
    for pc_entry in prompt_completion_list:
        messages: List[ChatCompletionMessageParam] = []

        # Extract messages
        for msg in pc_entry["prompt"]:
            role = msg["role"]

            if role == "assistant" and "tool_calls" in msg:
                # Use the tool_calls directly
                # This branch is usually not used in the wild.
                tool_calls: List[ChatCompletionMessageFunctionToolCallParam] = [
                    ChatCompletionMessageFunctionToolCallParam(
                        id=call["id"],
                        type="function",
                        function={"name": call["name"], "arguments": call["arguments"]},
                    )
                    for call in msg["tool_calls"]
                ]
                messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=None, tool_calls=tool_calls)
                )
            else:
                # Normal user/system/tool content
                message = cast(
                    ChatCompletionMessageParam,
                    TypeAdapter(ChatCompletionMessageParam).validate_python(
                        dict(role=role, content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", None))
                    ),
                )
                messages.append(message)

        # Extract completions (assistant outputs after tool responses)
        for comp in pc_entry["completion"]:
            if comp.get("role") == "assistant":
                content = comp.get("content")
                if pc_entry["tools"]:
                    tool_calls = [
                        ChatCompletionMessageFunctionToolCallParam(
                            id=tool["call"]["id"],
                            type=tool["call"]["type"],
                            function={"name": tool["name"], "arguments": tool["parameters"]},
                        )
                        for tool in pc_entry["tools"]
                    ]
                    messages.append(
                        ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=tool_calls)
                    )
                else:
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=content))

        # Build tools definitions (if available)
        if "functions" in pc_entry["request"]:
            tools = [
                ChatCompletionFunctionToolParam(
                    type="function",
                    function={
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": (
                            json.loads(fn["parameters"]) if isinstance(fn["parameters"], str) else fn["parameters"]
                        ),
                    },
                )
                for fn in pc_entry["request"]["functions"]
            ]
            yield OpenAIMessages(messages=messages, tools=tools)
        else:
            yield OpenAIMessages(messages=messages, tools=None)


class TraceToMessages(TraceAdapter[List[OpenAIMessages]]):
    """
    Adapter that converts OpenTelemetry trace spans into OpenAI-compatible message format.

    This adapter processes trace spans containing LLM conversation data and transforms them
    into structured OpenAI message format suitable for fine-tuning or analysis. It extracts
    prompts, completions, tool calls, and function definitions from trace attributes and
    reconstructs the conversation flow.

    The adapter handles:
    - Converting flat trace attributes into structured message objects
    - Extracting and matching tool calls with their corresponding requests
    - Building proper OpenAI ChatCompletionMessage objects with roles, content, and tool calls
    - Generating function definitions for tools used in conversations
    """

    def get_tool_calls(self, completion: Span, all_spans: List[Span], /) -> Iterable[Dict[str, Any]]:
        """Find tool calls in the trace. Returns a dict with the tool call id, name, and arguments.

        The spans that are direct children of the completion span are the tool calls.
        """
        # Get all the spans that are children of the completion span
        children = [span for span in all_spans if span.parent_id == completion.span_id]
        # Get the tool calls from the children
        for maybe_tool_call in children:
            tool_call = group_genai_dict(maybe_tool_call.attributes, "tool")
            if not isinstance(tool_call, dict):
                raise ValueError(f"Extracted tool call from trace is not a dict: {tool_call}")
            if tool_call:
                yield tool_call

    def adapt(self, source: List[Span], /) -> List[OpenAIMessages]:
        raw_prompt_completions: List[_RawSpanInfo] = []

        for span in source:
            attributes = {k: v for k, v in span.attributes.items()}

            # Get all related information from the trace span
            prompt = group_genai_dict(attributes, "gen_ai.prompt") or []
            completion = group_genai_dict(attributes, "gen_ai.completion") or []
            request = group_genai_dict(attributes, "gen_ai.request") or {}
            response = group_genai_dict(attributes, "gen_ai.response") or {}
            if not isinstance(prompt, list):
                raise ValueError(f"Extracted prompt from trace is not a list: {prompt}")
            if not isinstance(completion, list):
                raise ValueError(f"Extracted completion from trace is not a list: {completion}")
            if not isinstance(request, dict):
                raise ValueError(f"Extracted request from trace is not a dict: {request}")
            if not isinstance(response, dict):
                raise ValueError(f"Extracted response from trace is not a dict: {response}")
            if prompt or completion or request or response:
                tools = list(self.get_tool_calls(span, source)) or []
                raw_prompt_completions.append(
                    _RawSpanInfo(
                        prompt=prompt or [], completion=completion, request=request, response=response, tools=tools
                    )
                )

        return list(convert_to_openai_messages(raw_prompt_completions))

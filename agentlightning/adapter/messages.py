# Copyright (c) Microsoft. All rights reserved.

import json
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Sequence, TypedDict, Union, cast

from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, TypeAdapter

from agentlightning.types import Span

from .base import TraceAdapter


class OpenAIMessages(BaseModel):
    messages: List[Union[ChatCompletionMessage, ChatCompletionMessageParam]]
    tools: Optional[List[ChatCompletionFunctionToolParam]] = None


class _RawSpanInfo(TypedDict):
    prompt: List[Dict[str, Any]]
    completion: List[Dict[str, Any]]
    request: Dict[str, Any]
    response: Dict[str, Any]


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


def convert_to_openai_messages(
    prompt_completion_list: List[_RawSpanInfo], tool_requests: List[Dict[str, Any]]
) -> Generator[OpenAIMessages, None, None]:
    """
    Convert raw tool call traces + prompt/completion list
    into OpenAI fine-tuning JSONL format (tool calling style).

    Since promopt-completions sometimes do not contain the generated tool calls,
    the tool call requests need to be provided separately.
    The tool calls are then matched in a first-come-first-served basis to the tool call requests.

    https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-functions
    """
    for pc_entry in prompt_completion_list:
        messages: List[Union[ChatCompletionMessage, ChatCompletionMessageParam]] = []
        tools: List[ChatCompletionFunctionToolParam] = []

        # Extract messages
        for msg in pc_entry["prompt"]:
            role = msg["role"]

            if role == "assistant" and "tool_calls" in msg:
                # Use the tool_calls directly
                tool_calls: Sequence[ChatCompletionMessageFunctionToolCall] = []
                for call in msg["tool_calls"]:
                    function = Function(name=call["name"], arguments=call["arguments"])
                    tool_calls.append(
                        ChatCompletionMessageFunctionToolCall(
                            id=call["id"],
                            type="function",
                            function=function,
                        )
                    )
                messages.append(ChatCompletionMessage(role="assistant", tool_calls=list(tool_calls)))
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
                if comp.get("content"):
                    message = ChatCompletionMessage(role="assistant", content=comp["content"])
                    messages.append(message)
                elif comp.get("finish_reason") == "tool_calls":
                    if len(tool_requests) == 0:
                        raise ValueError("No tool requests available for tool_calls completion")
                    tool_req = tool_requests.pop(0)
                    # TODO: this is a hack because tracing frameworks did not report the tool call properly (?)
                    message = ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageFunctionToolCall(
                                id=tool_req["call"]["id"],
                                type=tool_req["call"]["type"],
                                function=Function(name=tool_req["name"], arguments=tool_req["parameters"]),
                            )
                        ],
                    )
                    messages.append(message)
                else:
                    raise ValueError(f"Unsupported assistant completion: {comp}")

        # Build tools definitions (if available)
        if "functions" in pc_entry["request"]:
            for fn in pc_entry["request"]["functions"]:
                tools.append(
                    ChatCompletionFunctionToolParam(
                        type="function",
                        function=FunctionDefinition(
                            name=fn["name"],
                            description=fn.get("description", ""),
                            parameters=(
                                json.loads(fn["parameters"]) if isinstance(fn["parameters"], str) else fn["parameters"]
                            ),
                        ),
                    )
                )
            yield OpenAIMessages(messages=messages, tools=tools)
        else:
            yield OpenAIMessages(messages=messages, tools=tools)


class TraceMessagesAdapter(TraceAdapter[List[OpenAIMessages]]):
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

    Returns:
        List[OpenAIMessages]: A list of structured message conversations with associated tools
    """

    def adapt(self, source: List[Span], /) -> List[OpenAIMessages]:
        raw_tool_calls: List[Dict[str, Any]] = []
        raw_prompt_completions: List[_RawSpanInfo] = []

        for span in source:
            attributes = {k: v for k, v in span.attributes.items()}

            # Otherwise we strip all the tool calls and prompts and responses
            tool_call = group_genai_dict(dict(attributes), "tool")
            if not isinstance(tool_call, dict):
                raise ValueError(f"Extracted tool call from trace is not a dict: {tool_call}")
            if tool_call:
                raw_tool_calls.append(tool_call)

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
                raw_prompt_completions.append(
                    _RawSpanInfo(prompt=prompt or [], completion=completion, request=request, response=response)
                )

        return list(convert_to_openai_messages(raw_prompt_completions, raw_tool_calls))

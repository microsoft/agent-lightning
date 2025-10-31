# Copyright (c) Microsoft. All rights reserved.

import json
from importlib.metadata import version
from typing import Any, Dict, Optional

import pytest
from packaging.version import Version

from agentlightning.adapter.messages import TraceToMessages
from agentlightning.types import OtelResource, Span, TraceStatus


def make_span(
    name: str,
    attributes: Dict[str, Any],
    sequence_id: int,
    *,
    parent_id: Optional[str] = None,
) -> Span:
    return Span(
        rollout_id="rollout-id",
        attempt_id="attempt-id",
        sequence_id=sequence_id,
        trace_id=f"trace-{sequence_id}",
        span_id=f"span-{sequence_id}",
        parent_id=parent_id,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes=attributes,
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=None,
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


_openai_version = Version(version("openai"))
_skip_for_openai_lt_1_100_0 = _openai_version < Version("1.100.0")


@pytest.mark.skipif(
    _skip_for_openai_lt_1_100_0,
    reason="Requires openai>=1.100.0",
)
def test_trace_messages_adapter_builds_expected_conversations():
    system_prompt = "You are a scheduling assistant."
    user_prompt = "Find a room."
    tool_name = "get_rooms_and_availability"
    tool_call_id = "call_sZkwxqiOmCx4n1iIQw5KhoQ0"
    tool_parameters = json.dumps({"date": "2025-10-13", "duration_min": 30, "time": "16:30"})
    tool_definition = json.dumps(
        {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM 24h local"},
                "duration_min": {
                    "type": "integer",
                    "description": "Meeting duration minutes",
                },
            },
            "required": ["date", "time", "duration_min"],
        }
    )
    tool_response = '{"rooms": [{"id": "Lyra", "capacity": 10, "free": true}]}'
    assistant_decision = "final_choice: No Room"
    grader_system_prompt = "Be a strict grader of exact room choice."
    grader_user_prompt = "Task output:\n    final_choice: No Room\n    ..."
    grader_result = '{"score": 1, "reason": "Matches expected."}'

    spans = [
        make_span(
            "tool_call.get_rooms_and_availability",
            {
                "tool.name": tool_name,
                "tool.parameters": tool_parameters,
                "tool.call.id": tool_call_id,
                "tool.call.type": "function",
            },
            0,
            parent_id="span-1",
        ),
        make_span(
            "openai.chat.completion",
            {
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-5-mini",
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": system_prompt,
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": user_prompt,
                "gen_ai.request.functions.0.name": tool_name,
                "gen_ai.request.functions.0.description": "Return meeting rooms with...",
                "gen_ai.request.functions.0.parameters": tool_definition,
                "gen_ai.response.id": "chatcmpl-CQFrAgBDvyZbWXSBBEQ2bm8qOAjeu",
                "gen_ai.response.model": "gpt-5-mini-2025-08-07",
                "gen_ai.usage.total_tokens": 391,
                "gen_ai.usage.prompt_tokens": 332,
                "gen_ai.usage.completion_tokens": 59,
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.finish_reason": "tool_calls",
            },
            1,
        ),
        make_span(
            "openai.chat.completion",
            {
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": system_prompt,
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": user_prompt,
                "gen_ai.prompt.2.role": "tool",
                "gen_ai.prompt.2.content": tool_response,
                "gen_ai.prompt.2.tool_call_id": tool_call_id,
                "gen_ai.response.id": "chatcmpl-CQFrE6lkDgdOzyrJdvS4FF27KcQj9",
                "gen_ai.response.model": "gpt-5-mini-2025-08-07",
                "gen_ai.usage.total_tokens": 924,
                "gen_ai.usage.prompt_tokens": 691,
                "gen_ai.usage.completion_tokens": 233,
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": assistant_decision,
                "gen_ai.completion.0.finish_reason": "stop",
            },
            2,
        ),
        make_span(
            "openai.chat.completion",
            {
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": grader_system_prompt,
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": grader_user_prompt,
                "gen_ai.response.id": "chatcmpl-CQFrJaQqYCxnO9K70q2D1xlESJeix",
                "gen_ai.response.model": "gpt-4.1-mini-2025-04-14",
                "gen_ai.usage.total_tokens": 120,
                "gen_ai.usage.prompt_tokens": 98,
                "gen_ai.usage.completion_tokens": 22,
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": grader_result,
                "gen_ai.completion.0.finish_reason": "stop",
            },
            3,
        ),
    ]

    adapter = TraceToMessages()

    expected = [
        {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_parameters},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Return meeting rooms with...",
                        "parameters": json.loads(tool_definition),
                    },
                }
            ],
        },
        {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
                {
                    "content": tool_response,
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                },
                {"content": assistant_decision, "role": "assistant"},
            ],
            "tools": None,
        },
        {
            "messages": [
                {"content": grader_system_prompt, "role": "system"},
                {"content": grader_user_prompt, "role": "user"},
                {"content": grader_result, "role": "assistant"},
            ],
            "tools": None,
        },
    ]

    assert adapter.adapt(spans) == expected


@pytest.mark.skipif(
    _skip_for_openai_lt_1_100_0,
    reason="Requires openai>=1.100.0",
)
def test_trace_messages_adapter_handles_multiple_tool_calls():
    system_prompt = "You are a scheduling assistant."
    user_prompt = "Find a room at 16:30 for 30 minutes. Needs projector + confphone. Accessible."
    tool_name = "get_rooms_and_availability"
    tool_parameters = json.dumps({"date": "2025-10-13", "time": "16:30", "duration_min": 30})
    tool_definition = json.dumps(
        {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM 24h local"},
                "duration_min": {
                    "type": "integer",
                    "description": "Meeting duration minutes",
                },
            },
            "required": ["date", "time", "duration_min"],
        }
    )
    tool_payload = json.dumps({"rooms": [{"id": "Orion", "free": True}]})
    assistant_response = (
        "Based on availability...\n\n"
        "1. **Nova**  \n   - Capacity: 12  \n   - Equipment: confphone  \n   - Accessibility: Yes  \n   - Distance: 45m  \n\n"
        "Please let me know if you'd like to book this room."
    )

    spans = [
        make_span(
            "tool_call.get_rooms_and_availability",
            {
                "tool.name": tool_name,
                "tool.parameters": tool_parameters,
                "tool.call.id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                "tool.call.type": "function",
            },
            0,
            parent_id="span-2",
        ),
        make_span(
            "tool_call.get_rooms_and_availability",
            {
                "tool.name": tool_name,
                "tool.parameters": tool_parameters,
                "tool.call.id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                "tool.call.type": "function",
            },
            1,
            parent_id="span-2",
        ),
        make_span(
            "openai.chat.completion",
            {
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": system_prompt,
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": user_prompt,
                "gen_ai.request.functions.0.name": tool_name,
                "gen_ai.request.functions.0.description": "Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.",
                "gen_ai.request.functions.0.parameters": tool_definition,
                "gen_ai.response.id": "chatcmpl-CQPL1FxUhG2xeOfin1hPTsYQvkRlL",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.usage.total_tokens": 211,
                "gen_ai.usage.prompt_tokens": 128,
                "gen_ai.usage.completion_tokens": 83,
                "gen_ai.completion.0.finish_reason": "tool_calls",
                "gen_ai.completion.0.role": "assistant",
            },
            2,
        ),
        make_span(
            "openai.chat.completion",
            {
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": system_prompt,
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": user_prompt,
                "gen_ai.prompt.2.role": "assistant",
                "gen_ai.prompt.2.tool_calls.0.id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                "gen_ai.prompt.2.tool_calls.0.name": tool_name,
                "gen_ai.prompt.2.tool_calls.0.arguments": tool_parameters,
                "gen_ai.prompt.2.tool_calls.1.id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                "gen_ai.prompt.2.tool_calls.1.name": tool_name,
                "gen_ai.prompt.2.tool_calls.1.arguments": tool_parameters,
                "gen_ai.prompt.3.role": "tool",
                "gen_ai.prompt.3.content": tool_payload,
                "gen_ai.prompt.3.tool_call_id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                "gen_ai.prompt.4.role": "tool",
                "gen_ai.prompt.4.content": tool_payload,
                "gen_ai.prompt.4.tool_call_id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                "gen_ai.response.id": "chatcmpl-CQPL2AOaq21yYW3ihE53x1xKf8lYk",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.usage.total_tokens": 1176,
                "gen_ai.usage.prompt_tokens": 1082,
                "gen_ai.usage.completion_tokens": 94,
                "gen_ai.completion.0.finish_reason": "stop",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": assistant_response,
            },
            3,
        ),
    ]

    adapter = TraceToMessages()

    expected = [
        {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_parameters},
                        },
                        {
                            "id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_parameters},
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.",
                        "parameters": json.loads(tool_definition),
                    },
                }
            ],
        },
        {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_parameters},
                        },
                        {
                            "id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_parameters},
                        },
                    ],
                },
                {
                    "content": tool_payload,
                    "role": "tool",
                    "tool_call_id": "call_CyJ3ooO7a7K1s9VXhWjBZCjo",
                },
                {
                    "content": tool_payload,
                    "role": "tool",
                    "tool_call_id": "call_EvoOTBfXMoIuMDHD1X9xVZPe",
                },
                {"content": assistant_response, "role": "assistant"},
            ],
            "tools": None,
        },
    ]

    assert adapter.adapt(spans) == expected

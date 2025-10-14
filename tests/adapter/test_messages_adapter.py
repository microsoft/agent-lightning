# Copyright (c) Microsoft. All rights reserved.

import json
from typing import Any, Dict

from agentlightning.adapter.messages import TraceMessagesAdapter
from agentlightning.types import Resource, Span, TraceStatus


def make_span(name: str, attributes: Dict[str, Any], sequence_id: int) -> Span:
    return Span(
        rollout_id="rollout-id",
        attempt_id="attempt-id",
        sequence_id=sequence_id,
        trace_id=f"trace-{sequence_id}",
        span_id=f"span-{sequence_id}",
        parent_id=None,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes=attributes,
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=None,
        parent=None,
        resource=Resource(attributes={}, schema_url=""),
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
    grader_user_prompt = "Task output:\n" "    final_choice: No Room\n......" "    "
    grader_result = '{"score": 1, "reason": "The final choice matches exactly with the expected answer."}'

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

    adapter = TraceMessagesAdapter()
    result = adapter.adapt(spans)

    actual = [conversation.model_dump() for conversation in result]

    expected = [
        {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
                {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "function": {"arguments": tool_parameters, "name": tool_name},
                            "type": "function",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "function": {
                        "name": tool_name,
                        "description": "Return meeting rooms with...",
                        "parameters": json.loads(tool_definition),
                    },
                    "type": "function",
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
                {
                    "content": assistant_decision,
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            ],
            "tools": [],
        },
        {
            "messages": [
                {"content": grader_system_prompt, "role": "system"},
                {"content": grader_user_prompt, "role": "user"},
                {
                    "content": grader_result,
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            ],
            "tools": [],
        },
    ]

    assert actual == expected

# Copyright (c) Microsoft. All rights reserved.

from typing import List

from agentlightning.adapter.base import Chain
from agentlightning.adapter.call import IdentifyChatCompletionCalls
from agentlightning.adapter.preprocess import RepairMalformedSpans, ToTree
from agentlightning.types.tracer import Span


def test_openai_calls():
    spans: List[Span] = [
        Span.from_attributes(
            attributes={
                "tool.name": "get_rooms_and_availability",
                "tool.parameters": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "tool.call.id": "call_owd6...l9Gv",
                "tool.call.type": "function",
            },
            name="tool_call.get_rooms_and_availability",
            span_id="18edfd18ea659820",
            parent_id="e858708413368c22",
            sequence_id=0,
        ),
        Span.from_attributes(
            attributes={
                "tool.name": "get_rooms_and_availability",
                "tool.parameters": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "tool.call.id": "call_VKsy...5ULX",
                "tool.call.type": "function",
            },
            name="tool_call.get_rooms_and_availability",
            span_id="6c9ba649e512a7f8",
            parent_id="e858708413368c22",
            sequence_id=1,
        ),
        Span.from_attributes(
            attributes={
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.temperature": 0.0,
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are a scheduling assistant.",
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": "Find a room 2025-10-13 16:30 for 60m, 12 attendees; needs confphone+whiteboard; accessible.",
                "gen_ai.request.functions.0.name": "get_rooms_and_availability",
                "gen_ai.request.functions.0.description": "Return rooms with capacity/equipment/accessibility/bookings.",
                "gen_ai.request.functions.0.parameters": '{"type":"object","properties":{"date":{"type":"string"},"time":{"type":"string"},"duration_min":{"type":"integer"}},"required":["date","time","duration_min"]}',
                "gen_ai.response.id": "chatcmpl-...1jC4",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.openai.system_fingerprint": "fp_03e44fcc34",
                "gen_ai.usage.total_tokens": 211,
                "gen_ai.usage.prompt_tokens": 128,
                "gen_ai.usage.completion_tokens": 83,
                "gen_ai.completion.0.finish_reason": "tool_calls",
                "gen_ai.completion.0.role": "assistant",
            },
            name="openai.chat.completion",
            span_id="e858708413368c22",
            parent_id="3db86425087d211f",
            sequence_id=2,
        ),
        Span.from_attributes(
            attributes={
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.temperature": 0.0,
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are a scheduling assistant.",
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": "Find a room 2025-10-13 16:30 for 60m, 12 attendees; needs confphone+whiteboard; accessible.",
                "gen_ai.prompt.2.role": "assistant",
                "gen_ai.prompt.2.tool_calls.0.id": "call_owd6...l9Gv",
                "gen_ai.prompt.2.tool_calls.0.name": "get_rooms_and_availability",
                "gen_ai.prompt.2.tool_calls.0.arguments": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "gen_ai.prompt.2.tool_calls.1.id": "call_VKsy...5ULX",
                "gen_ai.prompt.2.tool_calls.1.name": "get_rooms_and_availability",
                "gen_ai.prompt.2.tool_calls.1.arguments": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "gen_ai.prompt.3.role": "tool",
                "gen_ai.prompt.3.content": '{"rooms":[{"id":"Nova","capacity":12,"equipment":["whiteboard","confphone"],"accessible":true,"distance_m":45,"booked":[],"free":true},{"id":"Pulse","capacity":8,"equipment":["whiteboard","confphone"],"accessible":true,"booked":[["2025-10-13","16:30",30]],"free":false}]}',
                "gen_ai.prompt.3.tool_call_id": "call_owd6...l9Gv",
                "gen_ai.prompt.4.role": "tool",
                "gen_ai.prompt.4.content": '{"rooms":[{"id":"Nova","capacity":12,"equipment":["whiteboard","confphone"],"accessible":true,"free":true},{"id":"Pulse","capacity":8,"equipment":["whiteboard","confphone"],"accessible":true,"free":false}]}',
                "gen_ai.prompt.4.tool_call_id": "call_VKsy...5ULX",
                "gen_ai.response.id": "chatcmpl-...Syso",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.openai.system_fingerprint": "fp_03e44fcc34",
                "gen_ai.usage.total_tokens": 1189,
                "gen_ai.usage.prompt_tokens": 1082,
                "gen_ai.usage.completion_tokens": 107,
                "gen_ai.completion.0.finish_reason": "stop",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Available rooms: Nova (cap 12, whiteboard+confphone, accessible); Lyra (cap 10...).",
            },
            name="openai.chat.completion",
            span_id="9a44818e0901d0a1",
            parent_id="3db86425087d211f",
            sequence_id=3,
        ),
        Span.from_attributes(
            attributes={
                "agentops.span.kind": "session",
                "operation.name": "ro-90201d0a24cb",
            },
            name="ro-90201d0a24cb.session",
            span_id="3db86425087d211f",
            parent_id=None,
            sequence_id=4,
        ),
        Span.from_attributes(
            attributes={
                "agentlightning.reward.0.name": "primary",
                "agentlightning.reward.0.value": 1.0,
            },
            name="agentlightning.annotation",
            span_id="dc5e3c27f4378b6e",
            parent_id=None,
            sequence_id=5,
        ),
    ]

    # r1 = RepairMalformedSpans()(spans)
    # r2 = ToTree()(r1)
    # r2.visualize(filename="test_openai_calls", item_to_str=lambda span: span.name)

    adapter = Chain(
        RepairMalformedSpans(),
        ToTree(),
        IdentifyChatCompletionCalls(),
    )

    adapted_spans = adapter(spans)

    for span in adapted_spans:
        print(span)


def test_litellm_call():
    """Test LiteLLM proxy spans with chat completion calls."""
    spans: List[Span] = [
        # proxy_pre_call span
        Span.from_attributes(
            attributes={
                "call_type": "add_litellm_data_to_request",
                "service": "proxy_pre_call",
            },
            name="proxy_pre_call",
            span_id="a82547704417abf4",
            parent_id="9e1058bdd104a886",
            sequence_id=0,
        ),
        # router span (async_get_available_deployment)
        Span.from_attributes(
            attributes={
                "call_type": "async_get_available_deployment",
                "service": "router",
            },
            name="router",
            span_id="48086befab5d70cd",
            parent_id="9e1058bdd104a886",
            sequence_id=1,
        ),
        # self span (make_openai_chat_completion_request)
        Span.from_attributes(
            attributes={
                "call_type": "make_openai_chat_completion_request <- track_llm_api_timing",
                "service": "self",
            },
            name="self",
            span_id="7b2c9b9d544c2107",
            parent_id="9e1058bdd104a886",
            sequence_id=2,
        ),
        # router span (acompletion)
        Span.from_attributes(
            attributes={
                "call_type": "acompletion",
                "service": "router",
            },
            name="router",
            span_id="44f9efc7cd957922",
            parent_id="9e1058bdd104a886",
            sequence_id=3,
        ),
        # litellm_request span - main LLM call span with gen_ai attributes
        Span.from_attributes(
            attributes={
                "metadata.user_api_key_hash": "",
                "metadata.user_api_key_alias": "",
                "metadata.user_api_key_spend": 0.0,
                "metadata.user_api_key_max_budget": "",
                "metadata.user_api_key_budget_reset_at": "",
                "metadata.user_api_key_team_id": "",
                "metadata.user_api_key_org_id": "",
                "metadata.user_api_key_user_id": "",
                "metadata.user_api_key_team_alias": "",
                "metadata.user_api_key_user_email": "",
                "metadata.user_api_key_end_user_id": "",
                "metadata.user_api_key_request_route": "/v1/chat/completions",
                "metadata.spend_logs_metadata": "",
                "metadata.requester_ip_address": "",
                "metadata.requester_metadata": "{}",
                "metadata.prompt_management_metadata": "",
                "metadata.applied_guardrails": "[]",
                "metadata.mcp_tool_call_metadata": "",
                "metadata.vector_store_request_metadata": "",
                "metadata.usage_object": "{'completion_tokens': 48, 'prompt_tokens': 36, 'total_tokens': 84}",
                "metadata.requester_custom_headers": "{'x-rollout-id': 'ro-0b4d59a7d478'}",
                "metadata.cold_storage_object_key": "",
                "metadata.user_api_key_auth_metadata": "{}",
                "hidden_params": '{"model_id": "f6746e78...b010", "api_base": "http://localhost:45177/v1"}',
                "gen_ai.cost.input_cost": 0,
                "gen_ai.cost.output_cost": 0,
                "gen_ai.cost.total_cost": 0.0,
                "gen_ai.cost.tool_usage_cost": 0.0,
                "gen_ai.cost.original_cost": 0.0,
                "gen_ai.cost.discount_percent": 0.0,
                "gen_ai.cost.discount_amount": 0.0,
                "gen_ai.request.model": "Qwen/Qwen2.5-0.5B-Instruct",
                "llm.request.type": "acompletion",
                "gen_ai.system": "hosted_vllm",
                "llm.is_streaming": "False",
                "gen_ai.response.id": "chatcmpl-8b25...3e6e",
                "gen_ai.response.model": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
                "llm.usage.total_tokens": 84,
                "gen_ai.usage.completion_tokens": 48,
                "gen_ai.usage.prompt_tokens": 36,
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "Hello, what's your name?",
                "gen_ai.completion.0.finish_reason": "stop",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Hello! I am Qwen, an AI assistant created by Alibaba Cloud.",
            },
            name="litellm_request",
            span_id="c43d325d68344786",
            parent_id="9e1058bdd104a886",
            sequence_id=4,
        ),
        # raw_gen_ai_request span - sibling of litellm_request (same parent)
        Span.from_attributes(
            attributes={
                "llm.hosted_vllm.messages": '[{"role": "user", "content": "Hello, what\'s your name?"}]',
                "llm.hosted_vllm.extra_body": "{'return_token_ids': True}",
                "llm.hosted_vllm.id": "chatcmpl-8b25...3e6e",
                "llm.hosted_vllm.choices": '[{"finish_reason": "stop", "index": 0, "message": {"content": "Hello! I am Qwen...", "role": "assistant"}}]',
                "llm.hosted_vllm.created": 1767842789,
                "llm.hosted_vllm.model": "Qwen/Qwen2.5-0.5B-Instruct",
                "llm.hosted_vllm.object": "chat.completion",
                "llm.hosted_vllm.service_tier": "",
                "llm.hosted_vllm.system_fingerprint": "",
                "llm.hosted_vllm.usage": "{'completion_tokens': 48, 'prompt_tokens': 36, 'total_tokens': 84}",
                "llm.hosted_vllm.prompt_logprobs": "",
                "llm.hosted_vllm.prompt_token_ids": "[151644, 8948, 198, ...]",
                "llm.hosted_vllm.kv_transfer_params": "",
            },
            name="raw_gen_ai_request",
            span_id="bb5a15dd04c9e74b",
            parent_id="9e1058bdd104a886",  # Same parent as litellm_request - they are siblings
            sequence_id=5,
        ),
        # Root span - Received Proxy Server Request
        Span.from_attributes(
            attributes={},
            name="Received Proxy Server Request",
            span_id="9e1058bdd104a886",
            parent_id=None,
            sequence_id=6,
        ),
    ]

    adapter = Chain(
        RepairMalformedSpans(),
        ToTree(),
        IdentifyChatCompletionCalls(),
    )

    adapted_spans = adapter(spans)

    for span in adapted_spans:
        print(span)

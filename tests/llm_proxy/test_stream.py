# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, cast

import pytest

from agentlightning.llm_proxy import StreamConversionMiddleware


def merge_openai_streaming(chunks: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge chunks from OpenAI chat completion streaming into a single message dict.

    Returns a dict with keys:
      - role: "assistant" (or whatever)
      - content: full concatenated content string
      - function_call: optional dict with keys name, arguments (string or JSON parsed)
    """
    role: Optional[str] = None
    content_parts: List[str] = []
    function_name: Optional[str] = None
    function_args_str: Optional[str] = None

    for chunk in chunks:
        choice = chunk.get("choices", [])[0]
        delta = choice.get("delta", {})

        if "role" in delta and delta["role"] is not None:
            role = delta["role"]

        if "content" in delta and delta["content"] is not None:
            content_parts.append(delta["content"])

        # existing format: function_call
        if "function_call" in delta and delta["function_call"] is not None:
            fn = delta["function_call"]
            if function_name is None:
                function_name = fn.get("name")
                function_args_str = fn.get("arguments", "")
            else:
                function_args_str += fn.get("arguments", "")

        # new format: tool_calls array
        if "tool_calls" in delta and delta["tool_calls"]:
            for tc in delta["tool_calls"]:
                func = tc.get("function", {})
                # set name if first time
                if function_name is None and func.get("name"):
                    function_name = func["name"]
                # accumulate arguments
                if func.get("arguments") is not None:
                    if function_args_str is None:
                        function_args_str = func["arguments"]
                    else:
                        function_args_str += func["arguments"]

    full_content = "".join(content_parts)
    result: Dict[str, Any] = {"role": role or "assistant", "content": full_content}
    if function_name is not None:
        try:
            function_args = json.loads(function_args_str or "")  # type: ignore
        except Exception:
            function_args = function_args_str
        result["function_call"] = {"name": function_name, "arguments": function_args}
    return result


def merge_anthropic_streaming(chunks: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge chunks from Anthropic streaming into a single message dict.

    Returns a dict with keys:
      - role: "assistant"
      - content_text: full content text (concatenated)
      - tool_calls: list of dicts { name, input } if any
    """
    role: Optional[str] = None
    content_text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    current_tool: Optional[Dict[str, Any]] = None
    current_tool_input_str: Optional[str] = None

    for chunk in chunks:
        # role
        if role is None and "role" in chunk:
            role = chunk["role"]

        # handle content_block style (fine-grained)
        typ = chunk.get("type")
        if typ == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                # finish previous tool if exists
                if current_tool is not None:
                    try:
                        input_obj = json.loads(current_tool_input_str or "")
                    except Exception:
                        input_obj = current_tool_input_str
                    current_tool["input"] = input_obj
                    tool_calls.append(current_tool)
                current_tool = {"name": block.get("name"), "id": block.get("id"), "input": None}
                current_tool_input_str = ""
            continue

        if typ == "content_block_delta":
            delta = chunk.get("delta", {})
            dtyp = delta.get("type")
            if dtyp == "input_json_delta":
                current_tool_input_str = (current_tool_input_str or "") + delta.get("partial_json", "")
            elif dtyp == "text_delta":
                content_text_parts.append(delta.get("text", ""))
            continue

        if typ == "content_block_stop":
            if current_tool is not None:
                try:
                    input_obj = json.loads(current_tool_input_str or "")
                except Exception:
                    input_obj = current_tool_input_str
                current_tool["input"] = input_obj
                tool_calls.append(current_tool)
                current_tool = None
                current_tool_input_str = None
            continue

        # handle normal content items
        content_items = chunk.get("content", [])
        for item in content_items:
            t = item.get("type")
            if t == "text":
                content_text_parts.append(item.get("text", ""))
            elif t == "tool_use":
                tool_id = item.get("id")
                name = item.get("name")
                inp = item.get("input", {})
                if current_tool and current_tool.get("id") == tool_id:
                    # merge into same tool
                    try:
                        existing = json.loads(current_tool_input_str or "{}")
                    except Exception:
                        existing: Dict[str, Any] = {}
                    if isinstance(existing, dict):  # type: ignore
                        existing.update(inp)
                        current_tool_input_str = json.dumps(existing)
                    else:
                        # fallback: treat as string concatenation
                        current_tool_input_str += json.dumps(inp)
                else:
                    # finish previous tool
                    if current_tool is not None:
                        try:
                            input_obj = json.loads(current_tool_input_str or "")
                        except Exception:
                            input_obj = current_tool_input_str
                        current_tool["input"] = input_obj
                        tool_calls.append(current_tool)
                    current_tool = {"name": name, "id": tool_id, "input": None}
                    current_tool_input_str = json.dumps(inp)
            # else: ignore
    # end loop

    # finish any open tool
    if current_tool is not None:
        try:
            input_obj = json.loads(current_tool_input_str or "")
        except Exception:
            input_obj = current_tool_input_str
        current_tool["input"] = input_obj
        tool_calls.append(current_tool)

    full_text = "".join(content_text_parts)
    result: Dict[str, Any] = {"role": role or "assistant", "content_text": full_text}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def test_openai_text_only_short():
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ],
        )
    )
    merged = merge_openai_streaming(chunks)
    assert merged["role"] == "assistant"
    assert merged["content"] == "Hello world!"
    assert "function_call" not in merged


def test_openai_text_and_function_call_arguments_split():
    # Mixed content + function_call arguments spread over multiple deltas
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": "Starting… "}}]},
                {
                    "choices": [
                        {"index": 0, "delta": {"function_call": {"name": "get_weather", "arguments": '{"city": "'}}}
                    ]
                },
                {"choices": [{"index": 0, "delta": {"function_call": {"arguments": 'Singapore", "unit": "'}}}]},
                {"choices": [{"index": 0, "delta": {"function_call": {"arguments": 'celsius"}'}}}]},
                {"choices": [{"index": 0, "delta": {"content": "done."}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ],
        )
    )
    merged = merge_openai_streaming(chunks)
    assert merged["content"] == "Starting… done."
    assert merged["function_call"]["name"] == "get_weather"
    assert merged["function_call"]["arguments"] == {"city": "Singapore", "unit": "celsius"}


def test_openai_tool_calls_via_tool_calls_field():
    # Newer shape: delta.tool_calls with function.name/arguments segments
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "search"}}
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": "'}}]}}
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'python streaming"}'}}]},
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ],
        )
    )
    merged = merge_openai_streaming(chunks)
    assert merged["function_call"]["name"] == "search"
    assert merged["function_call"]["arguments"] == {"q": "python streaming"}


def test_openai_invalid_json_arguments_falls_back_to_string():
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {
                    "choices": [{"index": 0, "delta": {"function_call": {"name": "do", "arguments": '{"bad": '}}}]
                },  # truncated
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ],
        )
    )
    merged = merge_openai_streaming(chunks)
    assert merged["function_call"]["name"] == "do"
    # Should be raw string because JSON parsing fails
    assert isinstance(merged["function_call"]["arguments"], str)
    assert merged["function_call"]["arguments"].startswith('{"bad": ')


def test_anthropic_text_only_multiple_blocks():
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"role": "assistant", "content": [{"type": "text", "text": "Hello "}]},
                {"content": [{"type": "text", "text": "world!"}]},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            ],
        )
    )
    merged = merge_anthropic_streaming(chunks)
    assert merged["role"] == "assistant"
    assert merged["content_text"] == "Hello world!"
    assert "tool_calls" not in merged


def test_anthropic_tool_use_split_inputs_merge():
    # Tool input is delivered as multiple content fragments that should be merged
    chunks = iter(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "Working… "}]},
            {"content": [{"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"a": 1}}]},
            {"content": [{"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"b": 2}}]},
            {"content": [{"type": "text", "text": "done."}]},
            {"type": "message_stop"},
        ]
    )
    merged = merge_anthropic_streaming(chunks)
    assert merged["content_text"] == "Working… done."
    assert merged["tool_calls"][0]["name"] == "calculate"
    assert merged["tool_calls"][0]["input"] == {"a": 1, "b": 2}


def test_anthropic_fine_grained_input_json_delta():
    # Simulate SSE-style events: content_block_start(tool_use) + multiple input_json_delta pieces
    chunks = iter(
        [
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_x", "name": "fetch"},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"url": "'},
                "active_tool_id": "toolu_x",
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'https://example.com"}'},
                "active_tool_id": "toolu_x",
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_stop"},
        ]
    )
    merged = merge_anthropic_streaming(chunks)
    [tool] = merged["tool_calls"]
    assert tool["id"] == "toolu_x"
    assert tool["name"] == "fetch"
    assert tool["input"] == {"url": "https://example.com"}


def test_anthropic_text_and_tool_interleaved_with_text_deltas():
    # Mix text via text_delta and plain text content items
    chunks = iter(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "Start "}]},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "middle "}},
            {"content": [{"type": "text", "text": "end."}]},
            {"type": "message_stop"},
        ]
    )
    merged = merge_anthropic_streaming(chunks)
    assert merged["content_text"] == "Start middle end."


def test_anthropic_partial_json_left_as_string_when_invalid():
    # Provide malformed JSON parts; merger should keep raw string for tool input
    chunks = iter(
        [
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "toolu_bad", "name": "ingest"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"alpha": 1, '},
                "active_tool_id": "toolu_bad",
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '"beta": 2'},
                "active_tool_id": "toolu_bad",
            },
            # missing closing brace
            {"type": "content_block_stop", "index": 2},
            {"type": "message_stop"},
        ]
    )
    merged = merge_anthropic_streaming(chunks)
    [tool] = merged["tool_calls"]
    assert tool["id"] == "toolu_bad"
    assert isinstance(tool["input"], str)
    assert tool["input"].startswith('{"alpha": 1, ')


@pytest.mark.parametrize("text_len", [1, 50, 500])
def test_openai_long_text_stream_rounds_up(text_len: int):
    # Create a synthetic long content split into ~20-40 char pieces as the merger would see
    text = "x" * text_len
    # Simulate content arriving in three chunks
    part1, part2, part3 = text[: text_len // 3], text[text_len // 3 : 2 * text_len // 3], text[2 * text_len // 3 :]
    chunks = iter(
        cast(
            List[Dict[str, Any]],
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": part1}}]},
                {"choices": [{"index": 0, "delta": {"content": part2}}]},
                {"choices": [{"index": 0, "delta": {"content": part3}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ],
        )
    )
    merged = merge_openai_streaming(chunks)
    assert merged["content"] == text


async def collect_sse(gen: AsyncGenerator[str, Any]) -> List[str]:
    """Drain an async generator of SSE strings into a list."""
    out: List[str] = []
    async for s in gen:
        assert isinstance(s, str)
        out.append(s)
    return out


def parse_openai_sse_to_json_events(sse_chunks: List[str]) -> List[Dict[str, Any]]:
    """From the OpenAI stream (which uses only 'data:' lines), return JSON events.
    Filters out the literal DONE sentinel.
    """
    events: List[Dict[str, Any]] = []
    for chunk in sse_chunks:
        # each chunk looks like 'data: {...}\n\n' OR 'data: [DONE]\n\n'
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return events


def parse_anthropic_sse_to_json_payloads(sse_chunks: List[str]) -> List[Dict[str, Any]]:
    """Extract the JSON payload from each Anthropic SSE event (ignore pings)."""
    out: List[Dict[str, Any]] = []
    for chunk in sse_chunks:
        # chunks look like 'event: <name>\ndata: {json}\n\n'
        if "data:" not in chunk:
            continue
        data_line = [ln for ln in chunk.splitlines() if ln.startswith("data:")]
        if not data_line:
            continue
        payload = data_line[0][len("data:") :].strip()
        obj = json.loads(payload)
        if obj.get("type") == "ping":
            continue
        out.append(obj)
    return out


@pytest.fixture
def mw() -> StreamConversionMiddleware:
    # BaseHTTPMiddleware requires an ASGI app; we only need the instance for bound methods.
    class _DummyApp:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            pass

    return StreamConversionMiddleware(_DummyApp())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, finish_reason",
    [
        ("Hello world.", "stop"),
        ("This answer was cut off on purpose.", "length"),
    ],
)
async def test_openai_content_only_stream_roundtrip(mw: StreamConversionMiddleware, text: str, finish_reason: str):
    response_json = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
                # include logprobs to ensure it doesn't interfere with streaming
                "logprobs": None,
            }
        ],
    }

    sse_chunks = await collect_sse(mw.openai_stream_generator(response_json))

    # basic shape checks
    assert any('"delta": {"role": ' in s for s in sse_chunks)
    assert any("[DONE]" in s for s in sse_chunks)

    events = parse_openai_sse_to_json_events(sse_chunks)
    assert events, "Expected JSON events from stream"

    # the last JSON event before [DONE] should contain the finish_reason
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == finish_reason

    merged = merge_openai_streaming(iter(events))
    assert merged["role"] == "assistant"
    assert merged["content"] == text
    assert "function_call" not in merged


@pytest.mark.asyncio
async def test_openai_long_text_chunking_and_reassembly(mw: StreamConversionMiddleware):
    long_text = """
        This is a deliberately long sentence that should be broken into multiple streaming deltas by the
        chunking logic so that we can verify reassembly yields the exact same content without loss. """.strip()

    response_json = {
        "id": "chatcmpl-long",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": long_text}, "finish_reason": "stop"}],
    }

    sse_chunks = await collect_sse(mw.openai_stream_generator(response_json))
    events = parse_openai_sse_to_json_events(sse_chunks)

    # ensure multiple content delta chunks were emitted
    content_deltas = [ev for ev in events if ev["choices"][0]["delta"].get("content")]
    assert len(content_deltas) > 1

    merged = merge_openai_streaming(iter(events))
    assert merged["content"] == long_text


@pytest.mark.asyncio
async def test_openai_tool_call_only_stream_roundtrip(mw: StreamConversionMiddleware):
    response_json = {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Boston"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    sse_chunks = await collect_sse(mw.openai_stream_generator(response_json))
    events = parse_openai_sse_to_json_events(sse_chunks)

    # expect at least one tool_calls delta with name, followed by deltas with arguments
    assert any(
        (tc := ev["choices"][0]["delta"].get("tool_calls")) and tc[0].get("function", {}).get("name") == "get_weather"
        for ev in events
    )
    assert any(
        (tc := ev["choices"][0]["delta"].get("tool_calls")) and "arguments" in tc[0].get("function", {})
        for ev in events
    )

    merged = merge_openai_streaming(iter(events))
    assert merged["function_call"]["name"] == "get_weather"
    assert merged["function_call"]["arguments"] == {"location": "Boston"}


@pytest.mark.asyncio
async def test_openai_content_and_tool_call_stream_roundtrip(mw: StreamConversionMiddleware):
    response_json = {
        "id": "chatcmpl-mixed",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll call the weather tool now...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Singapore", "units": "metric"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    sse_chunks = await collect_sse(mw.openai_stream_generator(response_json))
    events = parse_openai_sse_to_json_events(sse_chunks)

    merged = merge_openai_streaming(iter(events))
    assert merged["content"].startswith("I'll call the weather tool")
    assert merged["function_call"]["name"] == "get_weather"
    assert merged["function_call"]["arguments"] == {"location": "Singapore", "units": "metric"}


@pytest.mark.asyncio
async def test_anthropic_text_only_stream_roundtrip(mw: StreamConversionMiddleware):
    original_response = {
        "id": "msg_123",
        "model": "claude-3.5-sonnet",
        "content": [
            {"type": "text", "text": "Hello there from Claude."},
        ],
        "usage": {"input_tokens": 0, "output_tokens": 7},
        "stop_reason": "end_turn",
    }

    sse_chunks = await collect_sse(mw.anthropic_stream_generator(original_response))

    # sanity: stream contains lifecycle events
    assert any("event: message_start" in s for s in sse_chunks)
    assert any("event: message_stop" in s for s in sse_chunks)

    payloads = parse_anthropic_sse_to_json_payloads(sse_chunks)
    merged = merge_anthropic_streaming(iter(payloads))

    assert merged["role"] == "assistant"
    assert merged["content_text"] == "Hello there from Claude."
    assert "tool_calls" not in merged


@pytest.mark.asyncio
async def test_anthropic_tool_use_only_stream_roundtrip(mw: StreamConversionMiddleware):
    original_response = {
        "id": "msg_tool",
        "model": "claude-3.5-sonnet",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"location": "Boston"},
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "stop_reason": "end_turn",
    }

    sse_chunks = await collect_sse(mw.anthropic_stream_generator(original_response))
    payloads = parse_anthropic_sse_to_json_payloads(sse_chunks)

    merged = merge_anthropic_streaming(iter(payloads))

    assert merged["tool_calls"][0]["name"] == "get_weather"
    assert merged["tool_calls"][0]["id"] == "toolu_1"
    assert merged["tool_calls"][0]["input"] == {"location": "Boston"}


@pytest.mark.asyncio
async def test_anthropic_mixed_text_and_tool_use_roundtrip(mw: StreamConversionMiddleware):
    # tool input is long to ensure multiple input_json_delta chunks
    long_input = {
        "location": "Singapore",
        "units": "metric",
        "details": {"hourly": True, "with_forecast": True, "days": 5},
    }
    original_response = {
        "id": "msg_mixed",
        "model": "claude-3.5-sonnet",
        "content": [
            {"type": "text", "text": "I'll check the weather tool for you."},
            {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": long_input},
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "stop_reason": "end_turn",
    }

    sse_chunks = await collect_sse(mw.anthropic_stream_generator(original_response))
    payloads = parse_anthropic_sse_to_json_payloads(sse_chunks)

    # Verify we saw content_block_start/stop and deltas for both text and tool input
    types = [p.get("type") for p in payloads]
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert any(p.get("delta", {}).get("type") == "text_delta" for p in payloads)
    assert any(p.get("delta", {}).get("type") == "input_json_delta" for p in payloads)

    merged = merge_anthropic_streaming(iter(payloads))
    assert merged["content_text"].startswith("I'll check the weather tool")
    tool = merged["tool_calls"][0]
    assert tool["name"] == "get_weather"
    assert tool["id"] == "toolu_2"
    assert tool["input"] == long_input

# Copyright (c) Microsoft. All rights reserved.

"""Test the LLMProxy class. Still under development.

General TODOs:

1. Add tests for retries
2. Add tests for timeout
3. Add tests for multiple models in model list
4. Add tests for multi-modal models

There are some specific TODOs for each test function.
"""

import ast
import asyncio
import json
from typing import Any, cast

import anthropic
import openai
import pytest

from agentlightning.llm_proxy import LLMProxy, _reset_litellm_logging_worker  # pyright: ignore[reportPrivateUsage]
from agentlightning.store import LightningStore, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import LLM, Span

from ..common.network import get_free_port
from ..common.tracer import clear_tracer_provider
from ..common.vllm import VLLM_VERSION, RemoteOpenAIServer

try:
    import torch  # type: ignore

    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False  # type: ignore
    pytest.skip(reason="GPU not available", allow_module_level=True)


@pytest.fixture(scope="module")
def qwen25_model():
    with RemoteOpenAIServer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        vllm_serve_args=[
            "--gpu-memory-utilization",
            "0.7",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--port",
            str(get_free_port()),
        ],
    ) as server:
        yield server


def test_qwen25_model_sanity(qwen25_model: RemoteOpenAIServer):
    client = qwen25_model.get_client()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=False,
    )
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_basic_integration(qwen25_model: RemoteOpenAIServer):
    clear_tracer_provider()
    inmemory_store = InMemoryLightningStore()
    store = LightningStoreServer(store=inmemory_store, host="127.0.0.1", port=get_free_port())
    await store.start()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "hosted_vllm/" + qwen25_model.model,
                    "api_base": qwen25_model.url_for("v1"),
                },
            }
        ],
        store=store,
    )

    rollout = await store.start_rollout(None)

    await proxy.start()

    resource = proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id)

    client = openai.OpenAI(base_url=resource.endpoint, api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o-arbitrary",
        messages=[{"role": "user", "content": "Repeat after me: Hello, world!"}],
        stream=False,
    )
    assert response.choices[0].message.content is not None
    assert "hello, world" in response.choices[0].message.content.lower()

    await proxy.stop()

    spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)

    await store.stop()

    # Verify all spans have correct rollout_id, attempt_id, and sequence_id
    assert len(spans) > 0, "Should have captured spans"
    for span in spans:
        assert span.rollout_id == rollout.rollout_id, f"Span {span.name} has incorrect rollout_id"
        assert span.attempt_id == rollout.attempt.attempt_id, f"Span {span.name} has incorrect attempt_id"
        assert span.sequence_id == 1, f"Span {span.name} has incorrect sequence_id"

    # Find the raw_gen_ai_request span and verify token IDs
    raw_gen_ai_spans = [s for s in spans if s.name == "raw_gen_ai_request"]
    assert len(raw_gen_ai_spans) == 1, f"Expected 1 raw_gen_ai_request span, found {len(raw_gen_ai_spans)}"
    raw_span = raw_gen_ai_spans[0]

    # Verify prompt_token_ids is present and non-empty
    assert (
        "llm.hosted_vllm.prompt_token_ids" in raw_span.attributes
    ), "prompt_token_ids not found in raw_gen_ai_request span"
    prompt_token_ids: list[int] = ast.literal_eval(raw_span.attributes["llm.hosted_vllm.prompt_token_ids"])  # type: ignore
    assert isinstance(prompt_token_ids, list), "prompt_token_ids should be a list"
    assert len(prompt_token_ids) > 0, "prompt_token_ids should not be empty"
    assert all(isinstance(tid, int) for tid in prompt_token_ids), "All prompt token IDs should be integers"

    # Verify response token_ids is present in choices
    assert "llm.hosted_vllm.choices" in raw_span.attributes, "choices not found in raw_gen_ai_request span"
    choices: list[dict[str, Any]] = ast.literal_eval(raw_span.attributes["llm.hosted_vllm.choices"])  # type: ignore
    assert len(choices) > 0, "Should have at least one choice"
    if VLLM_VERSION >= (0, 10, 2):
        assert "token_ids" in choices[0], "token_ids not found in choice"
        response_token_ids: list[int] = choices[0]["token_ids"]
    else:
        assert (
            "llm.hosted_vllm.response_token_ids" in raw_span.attributes
        ), "response_token_ids not found in raw_gen_ai_request span"
        response_token_ids_list: list[list[int]] = ast.literal_eval(raw_span.attributes["llm.hosted_vllm.response_token_ids"])  # type: ignore
        assert isinstance(response_token_ids_list, list), "response_token_ids_list should be a list"
        assert len(response_token_ids_list) > 0, "response_token_ids_list should not be empty"
        assert all(
            isinstance(tid_list, list) for tid_list in response_token_ids_list
        ), "All response token IDs should be lists"
        assert all(
            isinstance(tid, int) for tid_list in response_token_ids_list for tid in tid_list
        ), "All response token IDs should be integers"
        response_token_ids = response_token_ids_list[0]
    assert isinstance(response_token_ids, list), "response token_ids should be a list"
    assert len(response_token_ids) > 0, "response token_ids should not be empty"
    assert all(isinstance(tid, int) for tid in response_token_ids), "All response token IDs should be integers"

    # Find the litellm_request span and verify gen_ai prompts/completions
    litellm_spans = [s for s in spans if s.name == "litellm_request"]
    assert len(litellm_spans) == 1, f"Expected 1 litellm_request span, found {len(litellm_spans)}"
    litellm_span = litellm_spans[0]

    # Verify gen_ai.prompt attributes
    assert "gen_ai.prompt.0.role" in litellm_span.attributes, "gen_ai.prompt.0.role not found"
    assert litellm_span.attributes["gen_ai.prompt.0.role"] == "user", "Expected user role in prompt"
    assert "gen_ai.prompt.0.content" in litellm_span.attributes, "gen_ai.prompt.0.content not found"
    assert litellm_span.attributes["gen_ai.prompt.0.content"] == "Repeat after me: Hello, world!"

    # Verify gen_ai.completion attributes
    assert "gen_ai.completion.0.role" in litellm_span.attributes, "gen_ai.completion.0.role not found"
    assert litellm_span.attributes["gen_ai.completion.0.role"] == "assistant", "Expected assistant role in completion"
    assert "gen_ai.completion.0.content" in litellm_span.attributes, "gen_ai.completion.0.content not found"
    assert "gen_ai.completion.0.finish_reason" in litellm_span.attributes, "gen_ai.completion.0.finish_reason not found"


async def _make_proxy_and_store(qwen25_model: RemoteOpenAIServer, *, retries: int = 0):
    clear_tracer_provider()
    _reset_litellm_logging_worker()  # type: ignore
    store = InMemoryLightningStore()
    store_server = LightningStoreServer(store=store, host="127.0.0.1", port=get_free_port())
    # When the server is forked into subprocess, it automatically becomes a client of the store
    await store_server.start()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "hosted_vllm/" + qwen25_model.model,
                    "api_base": qwen25_model.url_for("v1"),
                },
            }
        ],
        store=store_server,
        num_retries=retries,
    )
    await proxy.start()
    return proxy, store_server


async def _new_resource(proxy: LLMProxy, store: LightningStore):
    rollout = await store.start_rollout(None)
    return proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id), rollout


def _get_client_for_resource(resource: LLM):
    return openai.OpenAI(base_url=resource.endpoint, api_key="token-abc123", timeout=120, max_retries=0)


def _get_async_client_for_resource(resource: LLM):
    return openai.AsyncOpenAI(base_url=resource.endpoint, api_key="token-abc123", timeout=120, max_retries=0)


def _find_span(spans: list[Span], name: str):
    return [s for s in spans if s.name == name]


def _attr(s: Span, key: str, default: Any = None):  # type: ignore
    return s.attributes.get(key, default)


@pytest.mark.asyncio
async def test_multiple_requests_one_attempt(qwen25_model: RemoteOpenAIServer):
    proxy, store = await _make_proxy_and_store(qwen25_model)
    try:
        resource, rollout = await _new_resource(proxy, store)
        client = _get_client_for_resource(resource)

        for i in range(3):
            r = client.chat.completions.create(
                model="gpt-4o-arbitrary",
                messages=[{"role": "user", "content": f"Say ping {i}"}],
                stream=False,
            )
            assert r.choices[0].message.content

        spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
        assert len(spans) > 0
        # Different requests have different sequence_ids
        assert {s.sequence_id for s in spans} == {1, 2, 3}
        # At least 3 requests recorded
        assert len(_find_span(spans, "raw_gen_ai_request")) == 3
        # TODO: Check response contents and token ids for the 3 requests respectively
    finally:
        await proxy.stop()
        await store.stop()


@pytest.mark.asyncio
async def test_ten_concurrent_requests(qwen25_model: RemoteOpenAIServer):
    proxy, store = await _make_proxy_and_store(qwen25_model)
    try:
        resource, rollout = await _new_resource(proxy, store)
        aclient = _get_async_client_for_resource(resource)

        async def _one(i: int):
            r = await aclient.chat.completions.create(
                model="gpt-4o-arbitrary",
                messages=[{"role": "user", "content": f"Return #{i}"}],
                stream=False,
            )
            return r.choices[0].message.content

        outs = await asyncio.gather(*[_one(i) for i in range(10)])
        assert len([o for o in outs if o]) == 10

        spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
        assert len(_find_span(spans, "raw_gen_ai_request")) == 10
        assert {s.sequence_id for s in spans} == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        # TODO: Check whether the sequence ids get mixed up or not
    finally:
        await proxy.stop()
        await store.stop()


@pytest.mark.asyncio
async def test_anthropic_client_compat(qwen25_model: RemoteOpenAIServer):
    # litellm proxy accepts Anthropic schema and forwards to OpenAI backend
    proxy, store = await _make_proxy_and_store(qwen25_model)
    try:
        resource, rollout = await _new_resource(proxy, store)

        a = anthropic.Anthropic(base_url=resource.endpoint, api_key="token-abc123", timeout=120)
        msg = a.messages.create(
            model="gpt-4o-arbitrary",
            max_tokens=64,
            messages=[{"role": "user", "content": "Respond with the word: OK"}],
        )
        # Anthropic SDK returns content list
        txt = "".join([b.text for b in msg.content if b.type == "text"])
        assert "OK" in txt.upper()

        spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
        assert len(spans) > 0
    finally:
        await proxy.stop()
        await store.stop()


@pytest.mark.asyncio
async def test_tool_call_roundtrip(qwen25_model: RemoteOpenAIServer):
    proxy, store = await _make_proxy_and_store(qwen25_model)
    try:
        resource, rollout = await _new_resource(proxy, store)
        client = _get_client_for_resource(resource)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo a string",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                },
            }
        ]

        r1 = client.chat.completions.create(
            model="gpt-4o-arbitrary",
            messages=[{"role": "user", "content": "Call the echo tool with text=hello"}],
            tools=cast(Any, tools),
            tool_choice="auto",
            stream=False,
        )
        # If the small model does not tool-call, skip gracefully
        tool_calls = r1.choices[0].message.tool_calls or []
        if not tool_calls:
            pytest.skip("model did not emit tool calls in this environment")

        call = tool_calls[0]
        assert call.type == "function"
        assert call.function and call.function.name == "echo"
        args = json.loads(call.function.arguments)
        assert "text" in args

        r2 = client.chat.completions.create(
            model="gpt-4o-arbitrary",
            messages=cast(
                Any,
                [
                    {"role": "user", "content": "Call the echo tool with text=hello"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": "echo", "arguments": call.function.arguments},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": call.id, "name": "echo", "content": args["text"]},
                ],
            ),
            stream=False,
        )
        assert args["text"] in (r2.choices[0].message.content or "")

        spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
        assert len(_find_span(spans, "litellm_request")) == 2
        assert len(_find_span(spans, "raw_gen_ai_request")) == 2

        # TODO: Check response contents and token ids for the 2 requests respectively
    finally:
        await proxy.stop()
        await store.stop()


@pytest.mark.skip(reason="Streaming is not supported yet")
@pytest.mark.asyncio
async def test_streaming_chunks(qwen25_model: RemoteOpenAIServer):
    proxy, store = await _make_proxy_and_store(qwen25_model)
    try:
        resource, rollout = await _new_resource(proxy, store)
        client = _get_client_for_resource(resource)

        stream = client.chat.completions.create(
            model="gpt-4o-arbitrary",
            messages=[{"role": "user", "content": "Say the word 'apple'"}],
            stream=True,
        )
        collected: list[str] = []
        for evt in stream:
            for c in evt.choices:
                if c.delta and getattr(c.delta, "content", None):
                    assert isinstance(c.delta.content, str)
                    collected.append(c.delta.content)
        assert "apple" in "".join(collected).lower()

        spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
        assert len(spans) > 0
        # TODO: didn't test the token ids in streaming chunks here
    finally:
        await proxy.stop()
        await store.stop()

# Copyright (c) Microsoft. All rights reserved.

"""Test the LLMProxy class. Still under development.

General TODOs:

1. Add tests for update model list and server restart
2. Add tests for retries
3. Add tests for timeout
4. Add tests for multiple models in model list
5. Add tests for multi-modal models

There are some specific TODOs for each test function.
"""

import ast
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
from contextlib import closing
from typing import Any, List, Optional, cast

import anthropic
import httpx
import openai
import pytest
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.llm_proxy import LightningSpanExporter, LLMProxy
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.types import Span
from agentlightning.types import LLM
from tests.tracer.utils import clear_tracer_provider

try:
    import torch  # type: ignore

    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False  # type: ignore
    pytest.skip(reason="GPU not available")

VLLM_AVAILABLE = False
VLLM_UNAVAILABLE_REASON = ""

try:
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils import FlexibleArgumentParser

    VLLM_AVAILABLE = True  # type: ignore
    VLLM_VERSION = tuple(int(v) for v in vllm.__version__.split("."))
except ImportError as e:
    AsyncEngineArgs = None
    get_model_loader = None
    FlexibleArgumentParser = None
    ServeSubcommand = None
    VLLM_VERSION = (0, 0, 0)  # type: ignore
    VLLM_UNAVAILABLE_REASON = str(e)  # type: ignore


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RemoteOpenAIServer:
    """
    A context manager for launching and interacting with a remote vLLM-based
    OpenAI-compatible server instance.

    This class handles:
      - Preparing the environment and spawning the vLLM server process
      - Ensuring that the requested model is downloaded before server startup
      - Polling and health-checking the server until it is ready
      - Providing helper methods to construct URLs for API calls
      - Returning configured synchronous and asynchronous OpenAI clients
        that can communicate with the launched server

    Typical usage:
        with RemoteOpenAIServer(vllm_serve_args, port, model) as server:
            client = server.get_client()
            response = client.chat.completions.create(...)

    Attributes:
        DUMMY_API_KEY (str): A placeholder API key for compatibility
                             (vLLM does not require authentication).
        host (str): Host address of the server (default: "localhost").
        port (int): TCP port number for the server.
        proc (subprocess.Popen): Handle to the launched server process.
    """

    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, vllm_serve_args: list[str], env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # safer CUDA init
        if env_dict is not None:
            env.update(env_dict)

        if VLLM_VERSION >= (0, 10, 2):
            # Supports return_token_ids
            self.proc: subprocess.Popen[bytes] = subprocess.Popen(
                ["vllm", "serve", model, *vllm_serve_args],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        else:
            # Does not support return_token_ids
            self.proc = subprocess.Popen(
                ["python", "-m", "agentlightning.cli.vllm", "serve", model, *vllm_serve_args],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],  # should not include the model name
        env_dict: Optional[dict[str, str]] = None,
        seed: Optional[int] = 0,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        if (
            not VLLM_AVAILABLE
            or AsyncEngineArgs is None
            or get_model_loader is None
            or FlexibleArgumentParser is None
            or ServeSubcommand is None
        ):
            raise ImportError("vLLM is not available: " + VLLM_UNAVAILABLE_REASON)

        self.model = model

        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        parser = ServeSubcommand().subparser_init(subparsers)  # pyright: ignore[reportUnknownMemberType]
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        assert args is not None
        self.host = str(args.host or "localhost")
        self.port = int(args.port)

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"), timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _poll(self) -> Optional[int]:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def _wait_for_server(self, *, url: str, timeout: float):
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs: Any):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs: Any):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )


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
            str(_get_free_port()),
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


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    clear_tracer_provider()
    yield


@pytest.mark.asyncio
async def test_basic_integration(qwen25_model: RemoteOpenAIServer):
    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=_get_free_port(),
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

    proxy.start()

    resource = proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id)

    import openai

    client = openai.OpenAI(base_url=resource.endpoint, api_key="token-abc123")
    response = client.chat.completions.create(
        model="gpt-4o-arbitrary",
        messages=[{"role": "user", "content": "Repeat after me: Hello, world!"}],
        stream=False,
    )
    assert response.choices[0].message.content is not None
    assert "hello, world" in response.choices[0].message.content.lower()

    proxy.stop()

    spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)

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


def _make_proxy_and_store(qwen25_model: RemoteOpenAIServer, *, retries: int = 0):
    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=_get_free_port(),
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
        num_retries=retries,
    )
    proxy.start()
    return proxy, store


async def _new_resource(proxy: LLMProxy, store: InMemoryLightningStore):
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
    proxy, store = _make_proxy_and_store(qwen25_model)
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
        proxy.stop()


@pytest.mark.asyncio
async def test_ten_concurrent_requests(qwen25_model: RemoteOpenAIServer):
    proxy, store = _make_proxy_and_store(qwen25_model)
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
        proxy.stop()


@pytest.mark.asyncio
async def test_anthropic_client_compat(qwen25_model: RemoteOpenAIServer):
    # litellm proxy accepts Anthropic schema and forwards to OpenAI backend
    proxy, store = _make_proxy_and_store(qwen25_model)
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
        proxy.stop()


@pytest.mark.asyncio
async def test_tool_call_roundtrip(qwen25_model: RemoteOpenAIServer):
    proxy, store = _make_proxy_and_store(qwen25_model)
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
        proxy.stop()


@pytest.mark.asyncio
async def test_streaming_chunks(qwen25_model: RemoteOpenAIServer):
    proxy, store = _make_proxy_and_store(qwen25_model)
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
        proxy.stop()


class _FakeSpanContext:
    def __init__(self, span_id: int):
        self.span_id = span_id


class _FakeParent:
    def __init__(self, span_id: int):
        self.span_id = span_id


class _FakeReadableSpan:
    def __init__(self, span_id: int, parent_id: int | None, attrs: dict[str, str]):
        self._ctx = _FakeSpanContext(span_id)
        self.parent = None if parent_id is None else _FakeParent(parent_id)
        self.attributes = attrs
        self.name = f"span-{span_id}"

    def get_span_context(self):
        return self._ctx


class _FakeStore(InMemoryLightningStore):
    def __init__(self):
        super().__init__()
        self.added: list[tuple[str, str, int, _FakeReadableSpan]] = []

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        assert isinstance(sequence_id, int)
        assert isinstance(readable_span, _FakeReadableSpan)
        self.added.append((rollout_id, attempt_id, sequence_id, readable_span))
        return cast(Span, None)


@pytest.mark.asyncio
async def test_exporter_tree_and_flush_headers_parsing():
    store = _FakeStore()
    exporter = LightningSpanExporter(store)

    # Build a root and two children. Headers distributed across spans.
    root = _FakeReadableSpan(1, None, {"metadata.requester_custom_headers": "{'x-rollout-id': 'r1'}"})
    child_a = _FakeReadableSpan(2, 1, {"metadata.requester_custom_headers": "{'x-attempt-id': 'a9'}"})
    child_b = _FakeReadableSpan(3, 1, {"metadata.requester_custom_headers": "{'x-sequence-id': '7'}"})

    # Push to buffer and export
    res = exporter.export(cast(List[ReadableSpan], [root, child_a, child_b]))
    assert res.name == "SUCCESS"

    # Give event loop a moment to run exporter coroutine
    await asyncio.sleep(0.1)

    # Should have flushed all three with merged headers
    assert len(store.added) == 3
    for rid, aid, sid, sp in store.added:
        assert rid == "r1"
        assert aid == "a9"
        assert sid == 7
        assert isinstance(sp, _FakeReadableSpan)

    exporter.shutdown()


def test_exporter_helpers():
    store = _FakeStore()
    exporter = LightningSpanExporter(store)

    # Tree: 10(root) -> 11(child) -> 12(grandchild); 20(root2)
    s10 = _FakeReadableSpan(10, None, {})
    s11 = _FakeReadableSpan(11, 10, {})
    s12 = _FakeReadableSpan(12, 11, {})
    s20 = _FakeReadableSpan(20, None, {})

    for _ in range(10):
        exporter._buffer = cast(List[ReadableSpan], [s10, s11, s12, s20])  # pyright: ignore[reportPrivateUsage]
        random.shuffle(exporter._buffer)  # pyright: ignore[reportPrivateUsage]

        roots = list(exporter._get_root_span_ids())  # pyright: ignore[reportPrivateUsage]
        assert set(roots) == {10, 20}

        subtree_ids = set(exporter._get_subtrees(10))  # pyright: ignore[reportPrivateUsage]
        assert subtree_ids == {10, 11, 12}

        popped = exporter._pop_subtrees(10)  # pyright: ignore[reportPrivateUsage]
        assert {sp.get_span_context().span_id for sp in popped} == {  # pyright: ignore[reportOptionalMemberAccess]
            10,
            11,
            12,
        }
        # Remaining buffer has only s20
        assert {
            sp.get_span_context().span_id  # pyright: ignore[reportOptionalMemberAccess]
            for sp in exporter._buffer  # pyright: ignore[reportPrivateUsage]
        } == {20}

    exporter.shutdown()

    # TODO: add more complex tests for the exporter helper

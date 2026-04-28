# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import random
from typing import Any, List, cast

import litellm
import openai
import opentelemetry.trace as trace_api
import pytest
from agentops.sdk.core import BatchSpanProcessor
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse
from litellm.utils import custom_llm_setup
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agentlightning.llm_proxy import LightningSpanExporter, LLMProxy
from agentlightning.store import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.store.threading import LightningStoreThreaded
from agentlightning.types import Span
from agentlightning.utils.server_launcher import PythonServerLauncherArgs

from ..common.network import get_free_port
from ..common.tracer import clear_tracer_provider


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


@pytest.mark.asyncio
async def test_update_model_list():
    store = InMemoryLightningStore()
    proxy = LLMProxy(
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                },
            }
        ],
        launch_mode="asyncio",
        port=get_free_port(),
        store=store,
    )
    await proxy.start()
    assert proxy.is_running()
    assert proxy.model_list == [
        {
            "model_name": "gpt-4o-arbitrary",
            "litellm_params": {
                "model": "openai/gpt-4o",
            },
        }
    ]
    proxy.update_model_list(
        [
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                },
            }
        ]
    )
    assert proxy.model_list == [
        {
            "model_name": "gpt-4o-arbitrary",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
            },
        }
    ]
    assert proxy.is_running()
    await proxy.stop()


@pytest.mark.asyncio
async def test_restart_resets_litellm_logging_worker() -> None:
    """LLMProxy.start() should recreate LiteLLM's logging worker on each run."""

    try:
        from litellm.litellm_core_utils import logging_worker as litellm_logging_worker
    except ImportError:
        pytest.skip("LiteLLM logging worker not available")

    store = InMemoryLightningStore()
    proxy = LLMProxy(
        model_list=[
            {
                "model_name": "dummy-model",
                # The backend is never invoked; only the proxy lifecycle matters here.
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        store=store,
        launcher_args=PythonServerLauncherArgs(
            port=get_free_port(),
            launch_mode="asyncio",
            healthcheck_url="/health",
            startup_timeout=10.0,
            process_join_timeout=10.0,
        ),
    )

    try:
        await proxy.start()
        first_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
        await proxy.stop()

        await proxy.start()
        second_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
    finally:
        await proxy.stop()

    assert first_worker is not second_worker, "LiteLLM logging worker should be refreshed after restart"


class TestLLM(CustomLLM):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return litellm.completion(  # type: ignore
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response=self.content,
        )

    async def acompletion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return litellm.completion(  # type: ignore
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response=self.content,
        )


@pytest.mark.asyncio
async def test_custom_llm_restarted_multiple_times(caplog: pytest.LogCaptureFixture) -> None:
    clear_tracer_provider()

    restart_times: int = 30

    store = LightningStoreThreaded(InMemoryLightningStore())
    caplog.set_level(logging.WARNING)

    port = get_free_port()
    try:
        llm_proxy = LLMProxy(
            model_list=[
                {
                    "model_name": "gpt-4o-arbitrary",
                    "litellm_params": {
                        # NOTE: The model after "/" cannot be an openai model like gpt-4o
                        # This might be a bug with litellm
                        "model": "test-llm/any-llm",
                    },
                }
            ],
            launcher_args=PythonServerLauncherArgs(
                launch_mode="thread",
                healthcheck_url="/health",
                port=port,
            ),
            store=store,
        )
        for restart_idx in range(restart_times):
            llm_instance = TestLLM(f"Hi! {restart_idx}")
            litellm.custom_provider_map = [{"provider": "test-llm", "custom_handler": llm_instance}]
            custom_llm_setup()
            await llm_proxy.restart()
            assert llm_proxy.is_running()

            openai_client = openai.AsyncOpenAI(
                base_url=llm_proxy.server_launcher.access_endpoint,
                api_key="token-abc123",
                timeout=5,
                max_retries=0,
            )
            response = await openai_client.chat.completions.create(
                model="gpt-4o-arbitrary",
                messages=[{"role": "user", "content": "Hello world"}],
                stream=False,
            )
            assert response.choices[0].message.content == f"Hi! {restart_idx}"

            error_logs = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
            assert not error_logs, f"Found error logs: {error_logs}"
            assert not any("Cannot add callback" in record.message for record in caplog.records)

        await llm_proxy.stop()
    finally:
        litellm.custom_provider_map = []
        custom_llm_setup()


async def llm_proxy_span_exporter_loop(otlp_enabled: bool = False):
    store = LightningStoreThreaded(InMemoryLightningStore())

    if otlp_enabled:
        store = LightningStoreServer(store, "127.0.0.1", get_free_port())
        await store.start()

    llm_instance = TestLLM(f"Hi! I'm a test LLM")
    litellm.custom_provider_map = [{"provider": "test-llm", "custom_handler": llm_instance}]
    custom_llm_setup()
    proxy = LLMProxy(
        launcher_args=PythonServerLauncherArgs(
            launch_mode="thread",
            healthcheck_url="/health",
            port=get_free_port(),
        ),
        store=store,
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "test-llm/any-llm",
                },
            }
        ],
    )
    await proxy.start()

    rollout = await store.start_rollout(None)
    resource = proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id)

    client = openai.AsyncOpenAI(
        base_url=resource.endpoint,
        api_key="token-abc123",
        timeout=5,
        max_retries=0,
    )
    response = await client.chat.completions.create(
        model="gpt-4o-arbitrary",
        messages=[{"role": "user", "content": "Hello world"}],
        stream=False,
    )
    assert response.choices[0].message.content == "Hi! I'm a test LLM"

    spans = await store.query_spans(rollout.rollout_id, rollout.attempt.attempt_id)
    assert len(spans) > 0, "Should have captured spans"
    for span in spans:
        assert span.rollout_id == rollout.rollout_id, f"Span {span.name} has incorrect rollout_id"
        assert span.attempt_id == rollout.attempt.attempt_id, f"Span {span.name} has incorrect attempt_id"
        assert span.sequence_id == 1, f"Span {span.name} has incorrect sequence_id"

    tracer_provider = trace_api.get_tracer_provider()

    have_asserted_loop = False
    for span_processor in tracer_provider._active_span_processor._span_processors:  # type: ignore
        if isinstance(span_processor, (SimpleSpanProcessor, BatchSpanProcessor)):
            if isinstance(span_processor.span_exporter, LightningSpanExporter):
                if otlp_enabled:
                    assert span_processor.span_exporter._loop is None  # type: ignore
                else:
                    assert span_processor.span_exporter._loop is not None  # type: ignore
                have_asserted_loop = True
                break
    assert have_asserted_loop, f"LightningSpanExporter should be used with otlp_enabled={otlp_enabled}"

    await proxy.stop()

    if isinstance(store, LightningStoreServer):
        await store.stop()


def llm_proxy_span_exporter_loop_sync(otlp_enabled: bool = False):
    asyncio.run(llm_proxy_span_exporter_loop(otlp_enabled))


@pytest.mark.parametrize("otlp_enabled", [True, False])
def test_llm_proxy_span_exporter_loop(otlp_enabled: bool):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=llm_proxy_span_exporter_loop_sync, args=(otlp_enabled,))
    process.start()
    process.join(timeout=30.0)
    assert process.exitcode == 0

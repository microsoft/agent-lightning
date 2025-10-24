# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import random
from typing import Any, List, cast

import litellm
import openai
import pytest
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse
from litellm.utils import custom_llm_setup
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.llm_proxy import LightningSpanExporter, LLMProxy
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import Span

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


def test_update_model_list():
    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                },
            }
        ],
        store=store,
    )
    proxy.start()
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
    proxy.stop()


def test_restart_resets_litellm_logging_worker() -> None:
    """LLMProxy.start() should recreate LiteLLM's logging worker on each run."""

    try:
        from litellm.litellm_core_utils import logging_worker as litellm_logging_worker
    except ImportError:
        pytest.skip("LiteLLM logging worker not available")

    store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "dummy-model",
                # The backend is never invoked; only the proxy lifecycle matters here.
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        store=store,
    )

    try:
        proxy.start()
        first_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
        proxy.stop()

        proxy.start()
        second_worker = litellm_logging_worker.GLOBAL_LOGGING_WORKER
    finally:
        proxy.stop()

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


def test_custom_llm_restarted_multiple_times(caplog: pytest.LogCaptureFixture) -> None:
    clear_tracer_provider()

    restart_times: int = 30

    store = InMemoryLightningStore()
    caplog.set_level(logging.WARNING)

    port = get_free_port()
    try:
        llm_proxy = LLMProxy(
            port=port,
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
            store=store,
        )
        for restart_idx in range(restart_times):
            llm_instance = TestLLM(f"Hi! {restart_idx}")
            litellm.custom_provider_map = [{"provider": "test-llm", "custom_handler": llm_instance}]
            custom_llm_setup()
            llm_proxy.restart()
            assert llm_proxy.is_running()

            openai_client = openai.OpenAI(
                base_url=f"http://localhost:{port}",
                api_key="token-abc123",
                timeout=5,
                max_retries=0,
            )
            response = openai_client.chat.completions.create(
                model="gpt-4o-arbitrary",
                messages=[{"role": "user", "content": "Hello world"}],
                stream=False,
            )
            assert response.choices[0].message.content == f"Hi! {restart_idx}"

            error_logs = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
            assert not error_logs, f"Found error logs: {error_logs}"
            assert not any("Cannot add callback" in record.message for record in caplog.records)

        llm_proxy.stop()
    finally:
        litellm.custom_provider_map = []
        custom_llm_setup()

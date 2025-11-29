# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import multiprocessing
from typing import Any, Callable, Coroutine

import pytest

from agentlightning.store.base import LightningStore
from agentlightning.tracer.weave import WeaveTracer
from agentlightning.types import Span


class MockLightningStore(LightningStore):
    """A minimal stub-only LightningStore, only implements methods likely used in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.spans: list[Span] = []

    async def add_span(self, span: Span) -> Span:
        self.spans.append(span)
        return span  # 返回同类型

    def get_traces(self) -> list[Span]:
        return self.spans


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


@pytest.mark.skip(reason="Skipping this test temporarily")
def test_weave_trace_workable_store_valid():
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_async, args=(_test_weave_trace_workable_store_valid_async,))
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."


def _run_async(fuc: Callable[[], Coroutine[Any, Any, None]]):
    asyncio.run(fuc())


async def _test_weave_trace_workable_store_valid_async():
    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    store = MockLightningStore()

    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        spans = store.get_traces()
        assert len(spans) > 0
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()

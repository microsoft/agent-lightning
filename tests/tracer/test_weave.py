# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import datetime
import multiprocessing
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Sequence

import pytest

from agentlightning.store.base import LightningStore
from agentlightning.tracer.weave import WeaveTracer
from agentlightning.types import Span


class MockLightningStore(LightningStore):
    """A minimal stub-only LightningStore, only implements methods likely used in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.spans: list[Span] = []

    async def add_many_spans(self, spans: Sequence[Span]) -> Sequence[Span]:
        self.spans.extend(spans)
        return spans

    def get_traces(self) -> list[Span]:
        return self.spans


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


def _func_with_exception():
    """Function that always executed successfully to test success tracing."""
    raise ValueError("This is a test exception")


@pytest.mark.parametrize("with_exception", [True, False])
def test_weave_trace_workable_store_valid(with_exception: bool):

    if with_exception:
        func = _test_weave_trace_with_exception
    else:
        func = _test_weave_trace_without_exception

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_async, args=(func,))
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."


def _run_async(coro: Callable[[], Coroutine[Any, Any, Any]]) -> None:
    """Small wrapper: run async function inside multiprocessing target."""
    import asyncio

    asyncio.run(coro())


async def _test_weave_trace_without_exception():
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

        has_error = False
        for span in spans:
            has_error = getattr(span, "status", None) and span.status.status_code == "ERROR"

        assert not has_error
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


async def _test_weave_trace_with_exception():
    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    store = MockLightningStore()

    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_with_exception()
        spans = store.get_traces()
        assert len(spans) > 0

        has_error = False
        for span in spans:
            has_error = getattr(span, "status", None) and span.status.status_code == "ERROR"

        assert has_error
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


def test_weave_with_op():
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_async, args=(_test_weave_with_op_imp,))
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."


async def _test_weave_with_op_imp():
    import weave  # type: ignore

    @weave.op  # type: ignore
    def _func_with_op():
        """Function that always executed successfully to test success tracing."""
        pass

    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    try:
        store = MockLightningStore()
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_with_op()
        spans = store.get_traces()
        len_spans_with_op = len(spans)

        store = MockLightningStore()
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        spans = store.get_traces()
        len_spans_without_op = len(spans)

        assert len_spans_with_op > 0
        assert len_spans_without_op > 0
        assert len_spans_with_op > len_spans_without_op

    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


def test_weave_trace_call_to_span():
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_test_weave_trace_call_to_span)
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."


async def _test_weave_trace_call_to_span():
    child = SimpleNamespace(
        inputs={"child_input": "x"},
        output={"child_output": 42},
        summary={"status_counts": {"success": 1, "error": 0}},
        _children=[],
        started_at=None,
        ended_at=datetime.datetime(2025, 12, 1, 0, 0, 2, tzinfo=datetime.timezone.utc),
        trace_id="trace-1",
        id="span-2",
        parent_id="span-1",
        func_name="child_func",
    )

    parent = SimpleNamespace(
        inputs={"parent_input": "y"},
        output={"parent_output": 99},
        summary={"status_counts": {"success": 1, "error": 0}},
        _children=[child],
        started_at=datetime.datetime(2025, 12, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        ended_at=datetime.datetime(2025, 12, 1, 0, 0, 1, tzinfo=datetime.timezone.utc),
        trace_id="trace-1",
        id="span-1",
        parent_id=None,
        func_name="parent_func",
    )

    tracer = WeaveTracer()
    spans, _ = tracer.convert_call_to_spans(parent)  # type: ignore

    assert len(spans) == 2
    assert spans[0].sequence_id == 0
    assert spans[1].sequence_id == 1
    assert spans[1].parent_id == "span-1"
    assert spans[1].attributes["input.child_input"] == "x"
    assert spans[1].attributes["output.child_output"] == 42

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import datetime
import multiprocessing
from types import SimpleNamespace

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
        return span

    def get_traces(self) -> list[Span]:
        return self.spans


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


@pytest.mark.skip(reason="Skipping this test temporarily")
def test_weave_trace_workable_store_valid():
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_test_weave_trace_workable_store_valid_async)
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."


def _test_weave_trace_workable_store_valid_async():
    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    store = MockLightningStore()

    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        spans = store.get_traces()
        assert len(spans) > 0
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

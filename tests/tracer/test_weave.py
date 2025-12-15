# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import datetime
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

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return len(self.spans)

    async def add_span(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    def clear_spans(self) -> None:
        self.spans = []

    def get_traces(self) -> list[Span]:
        return self.spans


class FakeWeaveCall:
    def __init__(
        self,
        op: str,
        attributes: dict[str, object] | None = None,
        inputs: dict[str, object] | None = None,
    ):
        base_time = datetime.datetime.fromtimestamp(96400, tz=datetime.timezone.utc)
        self.op_name = op
        self.attributes = attributes or {}
        self.inputs = inputs or {}
        self.output = None
        self.summary: dict[str, object] | None = {}
        self.started_at = base_time
        self.ended_at = base_time
        self.exception: str | None = None


class FakeWeaveClient:
    def __init__(self, server: object):
        self.server = server
        self.project = "fake-project"
        self.created_calls: list[FakeWeaveCall] = []
        self.finished_calls: list[FakeWeaveCall] = []

    def create_call(
        self,
        *,
        op: str,
        attributes: dict[str, object] | None = None,
        inputs: dict[str, object] | None = None,
    ) -> FakeWeaveCall:
        call = FakeWeaveCall(op=op, attributes=attributes or {}, inputs=inputs or {})
        self.created_calls.append(call)
        return call

    def finish_call(self, call: FakeWeaveCall, exception: Exception | None = None) -> None:
        if exception is not None:
            call.exception = str(exception)
        call.ended_at = call.started_at + datetime.timedelta(seconds=1)
        self.finished_calls.append(call)

    def flush(self) -> None:
        pass


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


def _func_with_exception():
    """Function that always executed successfully to test success tracing."""
    raise ValueError("This is a test exception")


@pytest.mark.weave
def test_weave_tracer_create_span(monkeypatch: pytest.MonkeyPatch) -> None:
    import weave  # type: ignore

    tracer = WeaveTracer(instrument_managed=False)
    fake_client = FakeWeaveClient(server=tracer._server)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(weave, "get_client", lambda: fake_client)

    span = tracer.create_span("weave-span", attributes={"foo": "bar"})

    assert span.name == "weave-span"
    assert span.attributes == {"foo": "bar"}
    assert len(fake_client.created_calls) == 1
    assert len(fake_client.finished_calls) == 1


@pytest.mark.weave
def test_weave_tracer_operation_context_records_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import weave  # type: ignore

    tracer = WeaveTracer(instrument_managed=False)
    fake_client = FakeWeaveClient(server=tracer._server)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(weave, "get_client", lambda: fake_client)

    with pytest.raises(RuntimeError):
        with tracer.operation_context("weave-op", attributes={"foo": "bar"}) as ctx:
            raise RuntimeError("failed")

    recorded_span = ctx.get_recorded_span()  # type: ignore
    assert recorded_span.name == "weave-op"
    assert recorded_span.status.status_code == "ERROR"
    assert recorded_span.status.description == "failed"
    assert len(fake_client.finished_calls) == 1


@pytest.mark.weave
@pytest.mark.asyncio
async def test_weave_trace_without_exception():
    tracer = WeaveTracer()
    store = MockLightningStore()
    tracer.init()
    tracer.init_worker(0, store=store)

    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(name="weave_test", rollout_id="test_rollout_id", attempt_id="test_attempt_id"):
            _func_without_exception()
        print(tracer.get_last_trace())
        spans = store.get_traces()
        assert len(spans) > 0

        has_error = False
        for span in spans:
            has_error = getattr(span, "status", None) and span.status.status_code == "ERROR"

        assert not has_error
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


@pytest.mark.weave
@pytest.mark.asyncio
async def test_weave_trace_with_exception():
    tracer = WeaveTracer()
    store = MockLightningStore()
    tracer.init()
    tracer.init_worker(0, store=store)

    try:
        with pytest.raises(ValueError):
            # Case where store, rollout_id, and attempt_id are all non-none.
            async with tracer.trace_context(
                name="weave_test", rollout_id="test_rollout_id", attempt_id="test_attempt_id"
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


@pytest.mark.weave
@pytest.mark.asyncio
async def test_weave_with_op():
    import weave  # type: ignore

    @weave.op  # type: ignore
    def _func_with_op():
        """Function that always executed successfully to test success tracing."""
        pass

    tracer = WeaveTracer()
    store = MockLightningStore()
    tracer.init()
    tracer.init_worker(0, store=store)

    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(name="weave_test", rollout_id="test_rollout_id", attempt_id="test_attempt_id"):
            _func_with_op()
        spans = store.get_traces()
        len_spans_with_op = len(spans)

        store.clear_spans()
        async with tracer.trace_context(name="weave_test", rollout_id="test_rollout_id", attempt_id="test_attempt_id"):
            _func_without_exception()
        spans = store.get_traces()
        len_spans_without_op = len(spans)

        assert len_spans_with_op > 0
        assert len_spans_without_op > 0
        assert len_spans_with_op > len_spans_without_op

    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


@pytest.mark.weave
@pytest.mark.asyncio
async def test_weave_trace_call_to_span():
    child = SimpleNamespace(
        op_name="child_func",
        inputs={"child_input": "x"},
        output={"child_output": 42},
        attributes={"child_attribute": "z"},
        summary={"status_counts": {"success": 1, "error": 0}},
        _children=[],
        started_at=datetime.datetime(2025, 12, 1, 0, 0, 1, tzinfo=datetime.timezone.utc),
        ended_at=datetime.datetime(2025, 12, 1, 0, 0, 2, tzinfo=datetime.timezone.utc),
        trace_id="trace-1",
        id="span-2",
        parent_id="span-1",
        func_name="child_func",
        exception=None,
    )

    parent = SimpleNamespace(
        op_name="parent_func",
        inputs={"parent_input": "y"},
        output={"parent_output": 99},
        attributes={"parent_attribute": "y"},
        summary={"status_counts": {"success": 1, "error": 0}},
        _children=[child],
        started_at=datetime.datetime(2025, 12, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        ended_at=datetime.datetime(2025, 12, 1, 0, 0, 1, tzinfo=datetime.timezone.utc),
        trace_id="trace-1",
        id="span-1",
        parent_id=None,
        func_name="parent_func",
        exception=None,
    )

    tracer = WeaveTracer()
    parent_span = await tracer.convert_call_to_span(parent)  # type: ignore
    assert parent_span.attributes["agentlightning.operation.input.parent_input"] == "y"
    assert parent_span.attributes["agentlightning.operation.output.parent_output"] == 99

    child_span = await tracer.convert_call_to_span(child)  # type: ignore
    assert child_span.attributes["agentlightning.operation.input.child_input"] == "x"
    assert child_span.attributes["agentlightning.operation.output.child_output"] == 42
    assert child_span.parent_id == "span-1"

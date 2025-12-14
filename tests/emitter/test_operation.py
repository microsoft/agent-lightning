# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, ContextManager, Dict, Iterator, List, Optional, Tuple, Type

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import agentlightning.emitter.annotation as annotation_module
from agentlightning.emitter.annotation import OperationContext, emit_annotation, operation
from agentlightning.semconv import AGL_ANNOTATION, AGL_OPERATION, LightningSpanAttributes
from agentlightning.tracer.dummy import DummySpanRecordingContext, DummyTracer
from agentlightning.types import SpanCoreFields, TraceStatus
from agentlightning.types.tracer import Attributes
from agentlightning.utils.otel import (
    extract_links_from_attributes,
    filter_and_unflatten_attributes,
    make_link_attributes,
    query_linked_spans,
)


class RecordingTracer:
    def __init__(self) -> None:
        self._delegate = DummyTracer()
        self.recordings: List[DummySpanRecordingContext] = []

    def operation_context(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> ContextManager[DummySpanRecordingContext]:
        parent_ctx = self._delegate.operation_context(name, attributes, start_time, end_time)

        @contextmanager
        def _wrapper() -> Iterator[DummySpanRecordingContext]:
            with parent_ctx as recording:
                self.recordings.append(recording)
                yield recording

        return _wrapper()

    def create_span(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        return self._delegate.create_span(name, attributes, timestamp, status)


class OtelSpanRecordingContext:
    def __init__(self, span: trace_api.Span) -> None:
        self._span = span

    def record_exception(self, exception: BaseException) -> None:
        self._span.record_exception(exception)
        self.record_status("ERROR", str(exception))

    def record_attributes(self, attributes: Dict[str, Any]) -> None:
        for key, value in attributes.items():
            self._span.set_attribute(key, value)

    def record_status(self, status_code: str, description: Optional[str] = None) -> None:
        self._span.set_status(Status(StatusCode[status_code], description))  # type: ignore[index]

    def get_recorded_span(self) -> None:
        raise NotImplementedError()


class OtelTracerAdapter:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def operation_context(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        ctx = self._tracer.start_as_current_span(name, attributes=attributes)

        class _ContextManager:
            def __enter__(self) -> OtelSpanRecordingContext:
                span = ctx.__enter__()
                return OtelSpanRecordingContext(span)

            def __exit__(
                self,
                exc_type: Optional[Type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType],
            ) -> bool:
                result = ctx.__exit__(exc_type, exc_val, exc_tb)
                return bool(result)

        return _ContextManager()

    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        span = self._tracer.start_span(name, attributes=attributes)
        if status:
            span.set_status(Status(StatusCode[status.status_code], status.description))  # type: ignore[index]
        span.end()
        start = timestamp or time.time()
        return SpanCoreFields(
            name=name,
            attributes=attributes or {},
            start_time=start,
            end_time=start,
            status=status or TraceStatus(status_code="OK"),
        )


def _install_recording_tracer(monkeypatch: pytest.MonkeyPatch) -> RecordingTracer:
    tracer = RecordingTracer()

    def fake_get_active_tracer() -> RecordingTracer:
        return tracer

    monkeypatch.setattr(annotation_module, "get_active_tracer", fake_get_active_tracer)
    return tracer


def _resolve_attr(recording: DummySpanRecordingContext, key: str) -> Any:
    if key in recording.attributes:
        value = recording.attributes[key]
    else:
        value = filter_and_unflatten_attributes(recording.attributes, key)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass
class ComplexResult:
    values: Tuple[int, ...]
    marker: str


def test_operation_context_serializes_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    ctx = OperationContext("custom-span", {"meta": {"foo": 1}, "count": 2})

    with ctx as op:
        op.set_input({"payload": 1}, flag=True)
        op.set_output({"success": True})

    recording = tracer.recordings[-1]
    assert recording.name == "custom-span"
    assert recording.attributes["meta.foo"] == 1
    assert recording.attributes["count"] == 2
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert _resolve_attr(recording, f"{input_prefix}.args") == [{"payload": 1}]
    assert recording.attributes[f"{input_prefix}.flag"] is True
    assert _resolve_attr(recording, LightningSpanAttributes.OPERATION_OUTPUT.value) == {"success": True}


def test_operation_context_set_input_supports_multiple_values(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    ctx = OperationContext("ctx", {})

    with ctx as op:
        op.set_input(1, 2, data={"foo": ["bar"]}, flags=[True, False])

    recording = tracer.recordings[-1]
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert _resolve_attr(recording, f"{input_prefix}.args") == [1, 2]
    assert _resolve_attr(recording, f"{input_prefix}.data") == {"foo": ["bar"]}
    assert _resolve_attr(recording, f"{input_prefix}.flags") == [True, False]


def test_operation_context_set_input_expands_positional_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    ctx = OperationContext("ctx", {})

    with ctx as op:
        op.set_input("alpha", "beta")

    recording = tracer.recordings[-1]
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert recording.attributes[f"{input_prefix}.args.0"] == "alpha"
    assert recording.attributes[f"{input_prefix}.args.1"] == "beta"


def test_operation_context_rejects_non_serializable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_recording_tracer(monkeypatch)

    class Unserializable:
        pass

    ctx = OperationContext("ctx", {})

    with pytest.raises(ValueError, match="Object must be JSON serializable"):
        with ctx as op:
            op.set_output(Unserializable())


def test_operation_context_records_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    ctx = OperationContext("custom-span", {})

    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("boom")

    recording = tracer.recordings[-1]
    assert "exception.type" in recording.attributes
    assert recording.status.status_code == "ERROR"
    assert recording.status.description == "boom"


def test_operation_factory_context_records_inputs_and_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    with operation(tags=["one", "two"]) as ctx:
        ctx.set_input("alpha", meta={"score": 0.5})
        ctx.set_output(["beta", "gamma"])

    recording = tracer.recordings[-1]
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert recording.name == AGL_OPERATION
    assert recording.attributes["tags"] == ["one", "two"]
    assert _resolve_attr(recording, f"{input_prefix}.args") == ["alpha"]
    assert _resolve_attr(recording, f"{input_prefix}.meta") == {"score": 0.5}
    assert _resolve_attr(recording, LightningSpanAttributes.OPERATION_OUTPUT.value) == ["beta", "gamma"]


def test_operation_factory_uses_standard_span_name(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    with operation(user={"id": 5}) as ctx:
        ctx.set_output("done")

    recording = tracer.recordings[-1]
    assert recording.name == AGL_OPERATION
    assert recording.attributes["user.id"] == 5


def test_operation_rejects_custom_span_names() -> None:
    with pytest.raises(ValueError):
        operation("custom-name")  # type: ignore


def test_operation_decorator_sync_records_span_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    @operation(category={"kind": "combine"})
    def combine(data: Dict[str, int], *, meta: Dict[str, str]) -> Dict[str, Any]:
        return {"joined": {**data, **meta}}

    result = combine({"value": 1}, meta={"source": "unit"})

    assert result == {"joined": {"value": 1, "source": "unit"}}
    recording = tracer.recordings[-1]
    assert recording.name == AGL_OPERATION
    assert recording.attributes["category.kind"] == "combine"

    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert _resolve_attr(recording, f"{input_prefix}.data") == {"value": 1}
    assert _resolve_attr(recording, f"{input_prefix}.meta") == {"source": "unit"}
    assert recording.attributes[LightningSpanAttributes.OPERATION_NAME.value] == "combine"
    assert _resolve_attr(recording, LightningSpanAttributes.OPERATION_OUTPUT.value) == result


def test_operation_decorator_handles_complex_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    @operation()
    def complicated(
        first: int,
        /,
        required: str,
        default: int = 5,
        *extra: int,
        kwonly: str,
        kwdefault: str = "fallback",
        **rest: Any,
    ) -> ComplexResult:
        return ComplexResult(values=(first, len(extra), len(rest)), marker=kwonly + kwdefault + required)

    with pytest.raises(ValueError):
        complicated(1, "req", 7, 8, 9, kwonly="x", kwdefault="y", tag="value")

    recording = tracer.recordings[-1]
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value

    assert _resolve_attr(recording, f"{input_prefix}.first") == 1
    assert _resolve_attr(recording, f"{input_prefix}.required") == "req"
    assert _resolve_attr(recording, f"{input_prefix}.default") == 7
    assert _resolve_attr(recording, f"{input_prefix}.extra") == [8, 9]
    assert _resolve_attr(recording, f"{input_prefix}.kwonly") == "x"
    assert _resolve_attr(recording, f"{input_prefix}.kwdefault") == "y"
    assert _resolve_attr(recording, f"{input_prefix}.rest") == {"tag": "value"}
    assert recording.attributes[LightningSpanAttributes.OPERATION_NAME.value] == "complicated"
    assert recording.status.status_code == "ERROR"


def test_operation_decorator_records_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    @operation()
    def fail(value: int) -> int:
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        fail(1)

    recording = tracer.recordings[-1]
    assert recording.status.status_code == "ERROR"
    assert recording.status.description == "bad input"


@pytest.mark.asyncio()
async def test_operation_async_wrapper_records_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_recording_tracer(monkeypatch)

    @operation()
    async def echo(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"payload": payload}

    result = await echo({"value": 3})

    assert result == {"payload": {"value": 3}}
    recording = tracer.recordings[-1]
    prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert _resolve_attr(recording, f"{prefix}.payload") == {"value": 3}
    assert _resolve_attr(recording, LightningSpanAttributes.OPERATION_OUTPUT.value) == result


def test_operation_span_can_be_resolved_via_annotation_links(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracerAdapter(provider.get_tracer(__name__))

    monkeypatch.setattr(annotation_module, "get_active_tracer", lambda: tracer)

    @operation(conversation_id="conv-1")
    def decorated(value: int) -> int:
        return value + 1

    assert decorated(41) == 42

    spans = exporter.get_finished_spans()
    operation_span = next(span for span in spans if span.name == AGL_OPERATION)
    assert operation_span.attributes["conversation_id"] == "conv-1"  # type: ignore

    trace_id_hex = trace_api.format_trace_id(operation_span.context.trace_id)  # type: ignore
    span_id_hex = trace_api.format_span_id(operation_span.context.span_id)  # type: ignore
    link_attrs = make_link_attributes({"trace_id": trace_id_hex, "span_id": span_id_hex})

    emit_annotation({**link_attrs, "note": "operation-follow-up"})

    spans = exporter.get_finished_spans()
    annotation_span = next(span for span in spans if span.name == AGL_ANNOTATION)
    annotation_links = extract_links_from_attributes(dict(annotation_span.attributes or {}))

    matches = query_linked_spans([operation_span], annotation_links)
    assert matches == [operation_span]


def test_operation_honors_propagate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_active_tracer() -> RecordingTracer:
        raise AssertionError("get_active_tracer should not be called when propagate=False")

    monkeypatch.setattr(annotation_module, "get_active_tracer", fail_get_active_tracer)

    @operation(propagate=False)
    def decorated(value: int) -> int:
        return value

    assert decorated(7) == 7

    with operation(propagate=False, value=7) as op:
        with pytest.raises(RuntimeError):
            op.span()

    assert op.span() is not None
    assert op.span().name == AGL_OPERATION
    assert op.span().attributes == {"value": 7}
    assert op.span().status.status_code == "OK"
    assert op.span().status.description is None
    assert op.span().start_time is not None
    assert op.span().end_time is not None
    assert op.span().start_time < op.span().end_time  # type: ignore
    assert op.span().start_time < time.time()  # type: ignore
    assert op.span().end_time < time.time()  # type: ignore
    assert op.span().start_time < op.span().end_time  # type: ignore

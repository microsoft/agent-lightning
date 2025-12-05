# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import agentlightning.emitter.annotation as annotation_module
from agentlightning.emitter.annotation import _safe_json_dump  # pyright: ignore[reportPrivateUsage]
from agentlightning.emitter.annotation import (
    OperationContext,
    emit_annotation,
    operation,
)
from agentlightning.semconv import AGL_ANNOTATION, AGL_OPERATION, LightningSpanAttributes
from agentlightning.utils.otel import extract_links_from_attributes, make_link_attributes, query_linked_spans


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}
        self.recorded_exceptions: List[BaseException] = []
        self.statuses: List[Status] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.recorded_exceptions.append(exc)

    def set_status(self, status: Status) -> None:
        self.statuses.append(status)


class DummySpanContextManager:
    def __init__(self, span: RecordingSpan) -> None:
        self.span = span
        self.exit_calls: List[
            Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]]
        ] = []

    def __enter__(self) -> RecordingSpan:
        return self.span

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.exit_calls.append((exc_type, exc_val, exc_tb))
        return False


class DummyTracer:
    def __init__(self, start_span_instance: Optional[RecordingSpan] = None) -> None:
        self._start_span_instance = start_span_instance
        self.start_span_calls: List[Tuple[str, Dict[str, Any]]] = []
        self.start_as_current_span_calls: List[Tuple[str, Dict[str, Any], RecordingSpan]] = []

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> RecordingSpan:
        span = self._start_span_instance or RecordingSpan()
        self.start_span_calls.append((name, dict(attributes or {})))
        return span

    def start_as_current_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> DummySpanContextManager:
        span = RecordingSpan()
        self.start_as_current_span_calls.append((name, dict(attributes or {}), span))
        return DummySpanContextManager(span)


class DummyUseSpan:
    def __init__(self) -> None:
        self.calls: List[Tuple[RecordingSpan, bool]] = []
        self.exit_calls: List[
            Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]]
        ] = []

    def __call__(self, span: RecordingSpan, end_on_exit: bool) -> DummyUseSpan:
        self.calls.append((span, end_on_exit))
        self._span = span
        return self

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.exit_calls.append((exc_type, exc_val, exc_tb))
        return False


@dataclass
class ComplexResult:
    values: Tuple[int, ...]
    marker: str


def test_safe_json_dump_handles_recursive_structures() -> None:
    payload: List[Any] = []
    payload.append(payload)

    assert _safe_json_dump(payload) == "[[...]]"


def test_operation_context_records_inputs_and_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    ctx = OperationContext("custom-span", {"meta": {"foo": 1}, "count": 2})

    with ctx as op:
        op.set_input({"payload": 1}, flag=True)
        op.set_output({"success": True})

    assert tracer.start_span_calls
    start_name, start_attributes = tracer.start_span_calls[0]
    assert start_name == "custom-span"
    assert json.loads(start_attributes["meta"]) == {"foo": 1}
    assert start_attributes["count"] == 2

    assert json.loads(span.attributes["input.args"]) == [{"payload": 1}]
    assert span.attributes["input.flag"] == "true"
    assert json.loads(span.attributes["output"]) == {"success": True}
    assert use_span.calls == [(span, True)]


def test_operation_context_set_input_supports_multiple_values(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    ctx = OperationContext("ctx", {})

    with ctx as op:
        op.set_input(1, 2, data={"foo": ["bar"]}, flags=[True, False])

    assert json.loads(span.attributes["input.args"]) == [1, 2]
    assert json.loads(span.attributes["input.data"]) == {"foo": ["bar"]}
    assert json.loads(span.attributes["input.flags"]) == [True, False]


def test_operation_context_records_non_serializable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unserializable:
        def __str__(self) -> str:
            return "<Unserializable>"

    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    ctx = OperationContext("ctx", {})

    with ctx as op:
        op.set_output(Unserializable())

    assert json.loads(span.attributes["output"]) == "<Unserializable>"


def test_operation_context_records_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    ctx = OperationContext("custom-span", {})

    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("boom")

    assert isinstance(span.recorded_exceptions[0], RuntimeError)
    status = span.statuses[-1]
    assert status.status_code == StatusCode.ERROR
    assert status.description == "boom"
    assert use_span.exit_calls[-1][1].args == ("boom",)  # type: ignore


def test_operation_factory_context_records_inputs_and_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    with operation(tags=["one", "two"]) as ctx:
        ctx.set_input("alpha", meta={"score": 0.5})
        ctx.set_output(["beta", "gamma"])

    start_name, attrs = tracer.start_span_calls[0]
    assert start_name == AGL_OPERATION
    assert json.loads(attrs["tags"]) == ["one", "two"]
    assert json.loads(span.attributes["input.args"]) == ["alpha"]
    assert json.loads(span.attributes["input.meta"]) == {"score": 0.5}
    assert json.loads(span.attributes["output"]) == ["beta", "gamma"]


def test_operation_factory_uses_standard_span_name(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RecordingSpan()
    tracer = DummyTracer(start_span_instance=span)
    use_span = DummyUseSpan()

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    with operation(user={"id": 5}) as ctx:
        ctx.set_output("done")

    assert tracer.start_span_calls
    start_name, attrs = tracer.start_span_calls[0]
    assert start_name == AGL_OPERATION
    assert json.loads(attrs["user"]) == {"id": 5}


def test_operation_rejects_custom_span_names() -> None:
    with pytest.raises(ValueError):
        operation("custom-name")  # type: ignore


def test_operation_decorator_sync_records_span_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)

    @operation(category={"kind": "combine"})
    def combine(data: Dict[str, int], *, meta: Dict[str, str]) -> Dict[str, Any]:
        return {"joined": {**data, **meta}}

    result = combine({"value": 1}, meta={"source": "unit"})

    assert result == {"joined": {"value": 1, "source": "unit"}}
    assert tracer.start_as_current_span_calls
    span_name, span_attributes, span = tracer.start_as_current_span_calls[0]
    assert span_name == AGL_OPERATION
    assert json.loads(span_attributes["category"]) == {"kind": "combine"}

    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert json.loads(span.attributes[f"{input_prefix}.data"]) == {"value": 1}
    assert json.loads(span.attributes[f"{input_prefix}.meta"]) == {"source": "unit"}
    assert span.attributes[LightningSpanAttributes.OPERATION_NAME.value] == "combine"
    assert json.loads(span.attributes[LightningSpanAttributes.OPERATION_OUTPUT.value]) == result


def test_operation_decorator_handles_complex_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)

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

    result = complicated(1, "req", 7, 8, 9, kwonly="x", kwdefault="y", tag="value")

    span = tracer.start_as_current_span_calls[0][2]
    input_prefix = LightningSpanAttributes.OPERATION_INPUT.value

    assert json.loads(span.attributes[f"{input_prefix}.first"]) == 1
    assert json.loads(span.attributes[f"{input_prefix}.required"]) == "req"
    assert json.loads(span.attributes[f"{input_prefix}.default"]) == 7
    assert json.loads(span.attributes[f"{input_prefix}.extra"]) == [8, 9]
    assert json.loads(span.attributes[f"{input_prefix}.kwonly"]) == "x"
    assert json.loads(span.attributes[f"{input_prefix}.kwdefault"]) == "y"
    assert json.loads(span.attributes[f"{input_prefix}.rest"]) == {"tag": "value"}
    assert span.attributes[LightningSpanAttributes.OPERATION_NAME.value] == "complicated"
    assert json.loads(span.attributes[LightningSpanAttributes.OPERATION_OUTPUT.value]) == str(result)


def test_operation_decorator_records_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)

    @operation()
    def fail(value: int) -> int:
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        fail(1)

    span = tracer.start_as_current_span_calls[0][2]
    assert isinstance(span.recorded_exceptions[0], ValueError)
    status = span.statuses[-1]
    assert status.status_code == StatusCode.ERROR
    assert status.description == "bad input"


@pytest.mark.asyncio()
async def test_operation_async_wrapper_records_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)

    @operation()
    async def echo(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"payload": payload}

    result = await echo({"value": 3})

    assert result == {"payload": {"value": 3}}
    span = tracer.start_as_current_span_calls[0][2]
    prefix = LightningSpanAttributes.OPERATION_INPUT.value
    assert json.loads(span.attributes[f"{prefix}.payload"]) == {"value": 3}
    assert json.loads(span.attributes[LightningSpanAttributes.OPERATION_OUTPUT.value]) == result


def test_operation_span_can_be_resolved_via_annotation_links(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)
    monkeypatch.setattr(annotation_module, "get_tracer", lambda use_active_span_processor=True: tracer)

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
    tracer = DummyTracer()
    flags: List[bool] = []
    use_span = DummyUseSpan()

    def fake_get_tracer(use_active_span_processor: bool = True) -> DummyTracer:
        flags.append(use_active_span_processor)
        return tracer

    monkeypatch.setattr(annotation_module, "get_tracer", fake_get_tracer)
    monkeypatch.setattr(annotation_module.trace, "use_span", use_span)

    @operation(propagate=False)
    def decorated(value: int) -> int:
        return value

    assert decorated(7) == 7

    with operation(propagate=False):
        pass

    assert flags == [False, False]

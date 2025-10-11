# Copyright (c) Microsoft. All rights reserved.

import json
from typing import Any, Dict, Optional

import pytest
from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.emitter import emit_exception, emit_message, emit_object
from agentlightning.emitter import exception as exception_module
from agentlightning.emitter import message as message_module
from agentlightning.emitter import object as object_module
from agentlightning.types.tracer import SpanAttributeNames, SpanNames


class DummySpan:
    def __init__(self) -> None:
        self.recorded_exception: Optional[Exception] = None
        self.status: Optional[Any] = None

    def __enter__(self) -> "DummySpan":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def record_exception(self, exception: Exception) -> None:
        self.recorded_exception = exception

    def set_status(self, status: Any) -> None:
        self.status = status


class DummyTracer:
    def __init__(self, span: DummySpan) -> None:
        self._span = span
        self.last_name: Optional[str] = None
        self.last_attributes: Optional[Dict[str, Any]] = None

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> DummySpan:
        self.last_name = name
        self.last_attributes = attributes or {}
        return self._span


def test_emit_message_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = DummyTracer(span)
    monkeypatch.setattr(message_module, "get_tracer", lambda: tracer)

    emit_message("hello world")

    assert tracer.last_name == SpanNames.MESSAGE.value
    assert tracer.last_attributes == {SpanAttributeNames.MESSAGE.value: "hello world"}


def test_emit_message_requires_string(caplog: pytest.LogCaptureFixture) -> None:
    # emit_message logs an error but doesn't raise an exception for invalid input
    emit_message(123)  # type: ignore[arg-type]
    assert "Message must be a string" in caplog.text


def test_emit_object_serializes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = DummyTracer(span)
    monkeypatch.setattr(object_module, "get_tracer", lambda: tracer)

    payload = {"foo": "bar", "baz": [1, 2, 3]}
    emit_object(payload)

    assert tracer.last_name == SpanNames.OBJECT.value
    assert tracer.last_attributes is not None
    assert json.loads(tracer.last_attributes[SpanAttributeNames.OBJECT.value]) == payload


def test_emit_object_requires_json_serializable(caplog: pytest.LogCaptureFixture) -> None:
    # emit_object logs an error but doesn't raise an exception for non-serializable input
    emit_object(object())  # type: ignore[arg-type]
    assert "Object must be JSON serializable" in caplog.text


def test_emit_exception_records_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = DummyTracer(span)
    monkeypatch.setattr(exception_module, "get_tracer", lambda: tracer)

    exc: Optional[Exception] = None
    try:
        raise ValueError("boom")
    except ValueError as err:
        emit_exception(err)
        exc = err

    assert tracer.last_name == SpanNames.EXCEPTION.value
    assert tracer.last_attributes is not None
    assert tracer.last_attributes[exception_attributes.EXCEPTION_TYPE] == "ValueError"
    assert tracer.last_attributes[exception_attributes.EXCEPTION_MESSAGE] == "boom"
    assert tracer.last_attributes[exception_attributes.EXCEPTION_ESCAPED] is True
    assert span.recorded_exception is exc


def test_emit_exception_requires_exception_instance(caplog: pytest.LogCaptureFixture) -> None:
    # emit_exception logs an error but doesn't raise an exception for invalid input
    emit_exception("boom")  # type: ignore[arg-type]
    assert "Expected an BaseException instance" in caplog.text

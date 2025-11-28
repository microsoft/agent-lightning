# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.emitter import emit_exception
from agentlightning.emitter import exception as exception_module
from agentlightning.semconv import AGL_EXCEPTION


class DummySpan:
    def __init__(self) -> None:
        self.recorded_exception: Optional[Exception] = None

    def __enter__(self) -> "DummySpan":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def record_exception(self, exception: Exception) -> None:
        self.recorded_exception = exception


class DummyTracer:
    def __init__(self, span: DummySpan) -> None:
        self._span = span
        self.last_name: Optional[str] = None
        self.last_attributes: Optional[Dict[str, Any]] = None

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> DummySpan:
        self.last_name = name
        self.last_attributes = attributes or {}
        return self._span


def _stub_tracer(monkeypatch: pytest.MonkeyPatch, span: DummySpan) -> DummyTracer:
    tracer = DummyTracer(span)

    def fake_get_tracer(*_: Any, **__: Any) -> DummyTracer:
        return tracer

    monkeypatch.setattr(exception_module, "get_tracer", fake_get_tracer)
    return tracer


def test_emit_exception_records_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = _stub_tracer(monkeypatch, span)

    exc: Optional[Exception] = None
    try:
        raise ValueError("boom")
    except ValueError as err:
        emit_exception(err)
        exc = err

    assert tracer.last_name == AGL_EXCEPTION
    assert tracer.last_attributes is not None
    assert tracer.last_attributes[exception_attributes.EXCEPTION_TYPE] == "ValueError"
    assert tracer.last_attributes[exception_attributes.EXCEPTION_MESSAGE] == "boom"
    assert tracer.last_attributes[exception_attributes.EXCEPTION_ESCAPED] is True
    assert span.recorded_exception is exc


def test_emit_exception_requires_exception_instance() -> None:
    with pytest.raises(TypeError):
        emit_exception("boom")  # type: ignore[arg-type]

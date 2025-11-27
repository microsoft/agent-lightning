# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import pytest

from agentlightning.emitter import emit_message
from agentlightning.emitter import message as message_module
from agentlightning.emitter.message import get_message_value
from agentlightning.semconv import AGL_MESSAGE, LightningSpanAttributes
from agentlightning.types.tracer import SpanLike


@dataclass
class FakeSpan:
    attributes: Optional[Dict[str, Any]]


class DummySpan:
    def __enter__(self) -> "DummySpan":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False


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

    monkeypatch.setattr(message_module, "get_tracer", fake_get_tracer)
    return tracer


def test_get_message_value_returns_string() -> None:
    span = FakeSpan(attributes={LightningSpanAttributes.MESSAGE_BODY.value: "hello"})

    assert get_message_value(cast(SpanLike, span)) == "hello"


def test_get_message_value_returns_none_when_missing() -> None:
    span = FakeSpan(attributes={})

    assert get_message_value(cast(SpanLike, span)) is None


def test_get_message_value_rejects_non_string() -> None:
    span = FakeSpan(attributes={LightningSpanAttributes.MESSAGE_BODY.value: ["not", "string"]})

    with pytest.raises(TypeError):
        get_message_value(cast(SpanLike, span))


def test_emit_message_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = _stub_tracer(monkeypatch, span)

    emit_message("hello world")

    assert tracer.last_name == AGL_MESSAGE
    assert tracer.last_attributes == {LightningSpanAttributes.MESSAGE_BODY.value: "hello world"}


def test_emit_message_requires_string() -> None:
    with pytest.raises(TypeError):
        emit_message(123)  # type: ignore[arg-type]

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

from ..common.tracer import RecordingDummyTracer


@dataclass
class FakeSpan:
    attributes: Optional[Dict[str, Any]]


def _stub_tracer(monkeypatch: pytest.MonkeyPatch) -> RecordingDummyTracer:
    tracer = RecordingDummyTracer()
    monkeypatch.setattr(message_module, "get_active_tracer", lambda: tracer)
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
    tracer = _stub_tracer(monkeypatch)

    emit_message("hello world")

    assert tracer.last_span is not None
    assert tracer.last_span.name == AGL_MESSAGE
    assert tracer.last_span.attributes == {LightningSpanAttributes.MESSAGE_BODY.value: "hello world"}


def test_emit_message_requires_string() -> None:
    with pytest.raises(TypeError):
        emit_message(123)  # type: ignore[arg-type]


def test_emit_message_flattens_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _stub_tracer(monkeypatch)

    emit_message("hello", attributes={"meta": {"tag": "foo"}, "labels": ["a", "b"]})

    assert tracer.last_span is not None
    assert tracer.last_span.attributes["meta.tag"] == "foo"
    assert tracer.last_span.attributes["labels"] == ["a", "b"]


def test_emit_message_propagate_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_active_tracer() -> RecordingDummyTracer:
        raise AssertionError("Should not resolve tracer when propagate=False")

    monkeypatch.setattr(message_module, "get_active_tracer", fail_get_active_tracer)

    emit_message("local", propagate=False)

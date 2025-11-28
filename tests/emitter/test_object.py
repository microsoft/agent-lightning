# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import pytest

from agentlightning.emitter import emit_object
from agentlightning.emitter import object as object_module
from agentlightning.emitter.object import encode_object, get_object_value
from agentlightning.semconv import AGL_OBJECT, LightningSpanAttributes
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

    monkeypatch.setattr(object_module, "get_tracer", fake_get_tracer)
    return tracer


def test_encode_object_for_primitives() -> None:
    encoded = encode_object("hello")

    assert encoded == {
        LightningSpanAttributes.OBJECT_TYPE.value: "str",
        LightningSpanAttributes.OBJECT_LITERAL.value: "hello",
    }


def test_encode_object_for_bytes() -> None:
    payload = b"binary"
    encoded = encode_object(payload)
    expected_literal = base64.b64encode(payload).decode("utf-8")

    assert encoded == {
        LightningSpanAttributes.OBJECT_TYPE.value: "bytes",
        LightningSpanAttributes.OBJECT_LITERAL.value: expected_literal,
    }


def test_encode_object_for_dict_serializes_json() -> None:
    payload = {"key": 1, "nested": [1, 2]}
    encoded = encode_object(payload)

    assert encoded[LightningSpanAttributes.OBJECT_TYPE.value] == "dict"
    assert json.loads(encoded[LightningSpanAttributes.OBJECT_JSON.value]) == payload


def test_encode_object_raises_for_unserializable() -> None:
    with pytest.raises(RuntimeError):
        encode_object(object())


def test_get_object_value_from_json_attribute() -> None:
    payload = {"answer": 42}
    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_JSON.value: json.dumps(payload),
            LightningSpanAttributes.OBJECT_TYPE.value: "dict",
        }
    )

    assert get_object_value(cast(SpanLike, span)) == payload


def test_get_object_value_from_literal_types() -> None:
    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: "123",
            LightningSpanAttributes.OBJECT_TYPE.value: "int",
        }
    )
    assert get_object_value(cast(SpanLike, span)) == 123

    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: "3.14",
            LightningSpanAttributes.OBJECT_TYPE.value: "float",
        }
    )
    assert get_object_value(cast(SpanLike, span)) == pytest.approx(3.14)  # type: ignore

    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: "true",
            LightningSpanAttributes.OBJECT_TYPE.value: "bool",
        }
    )
    assert get_object_value(cast(SpanLike, span)) is True

    literal = base64.b64encode(b"hi").decode("utf-8")
    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: literal,
            LightningSpanAttributes.OBJECT_TYPE.value: "bytes",
        }
    )
    assert get_object_value(cast(SpanLike, span)) == b"hi"


def test_get_object_value_returns_none_when_missing() -> None:
    span = FakeSpan(attributes={})

    assert get_object_value(cast(SpanLike, span)) is None


def test_get_object_value_raises_for_invalid_json() -> None:
    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_JSON.value: "not json",
        }
    )

    with pytest.raises(RuntimeError):
        get_object_value(cast(SpanLike, span))


def test_emit_object_serializes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummySpan()
    tracer = _stub_tracer(monkeypatch, span)

    payload = {"foo": "bar", "baz": [1, 2, 3]}
    emit_object(payload)

    assert tracer.last_name == AGL_OBJECT
    assert tracer.last_attributes is not None
    assert json.loads(tracer.last_attributes[LightningSpanAttributes.OBJECT_JSON.value]) == payload


def test_emit_object_requires_json_serializable() -> None:
    with pytest.raises(RuntimeError):
        emit_object(object())  # type: ignore[arg-type]


def test_get_object_value_raises_for_unknown_literal_type() -> None:
    span = FakeSpan(
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: "value",
            LightningSpanAttributes.OBJECT_TYPE.value: "complex",
        }
    )

    with pytest.raises(RuntimeError):
        get_object_value(cast(SpanLike, span))

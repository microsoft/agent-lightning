# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import TraceFlags

from agentlightning.emitter import annotation as annotation_module
from agentlightning.emitter.annotation import emit_annotation
from agentlightning.semconv import AGL_ANNOTATION


class DummyReadableSpan(ReadableSpan):
    def __init__(self) -> None:
        super().__init__(
            name="dummy",
            context=trace_api.SpanContext(
                trace_id=0x1,
                span_id=0x2,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=trace_api.TraceState(),
            ),
            resource=Resource.create({}),
        )

    def __enter__(self) -> "DummyReadableSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class DummyTracer:
    def __init__(self, span: DummyReadableSpan) -> None:
        self._span = span
        self.last_name: str | None = None
        self.last_attributes: Dict[str, Any] | None = None

    def start_span(self, name: str, attributes: Dict[str, Any] | None = None) -> DummyReadableSpan:
        self.last_name = name
        self.last_attributes = attributes or {}
        return self._span


def test_emit_annotation_flattens_and_respects_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    span = DummyReadableSpan()
    tracer = DummyTracer(span)
    captured: Dict[str, Any] = {}

    def fake_get_tracer(*_: Any, **kwargs: Any) -> DummyTracer:
        captured["propagate"] = kwargs.get("use_active_span_processor")
        return tracer

    monkeypatch.setattr(annotation_module, "get_tracer", fake_get_tracer)

    result = emit_annotation({"meta": {"tag": "foo"}, "score": 1.5}, propagate=False)

    assert result is span
    assert captured["propagate"] is False
    assert tracer.last_name == AGL_ANNOTATION
    assert tracer.last_attributes == {"meta.tag": "foo", "score": 1.5}


def test_emit_annotation_rejects_non_primitive_values() -> None:
    with pytest.raises(TypeError):
        emit_annotation({"bad": {"set": {1}}})

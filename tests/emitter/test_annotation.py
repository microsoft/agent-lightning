# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Dict

import pytest

from agentlightning.emitter import annotation as annotation_module
from agentlightning.emitter.annotation import emit_annotation
from agentlightning.semconv import AGL_ANNOTATION

from ..common.tracer import RecordingDummyTracer


def _install_tracer(monkeypatch: pytest.MonkeyPatch) -> RecordingDummyTracer:
    tracer = RecordingDummyTracer()
    monkeypatch.setattr(annotation_module, "get_active_tracer", lambda: tracer)
    return tracer


def test_emit_annotation_flattens_and_sanitizes_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_tracer(monkeypatch)

    result = emit_annotation({"meta": {"tag": "foo"}, "score": 1.5})

    assert result.name == AGL_ANNOTATION
    assert tracer.last_span is not None
    assert tracer.last_span.attributes == {"meta.tag": "foo", "score": 1.5}


def test_emit_annotation_propagate_false_bypasses_active_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, bool] = {"called": False}

    def fail_get_active_tracer() -> RecordingDummyTracer:
        captured["called"] = True
        raise AssertionError("Should not resolve active tracer when propagate is False")

    monkeypatch.setattr(annotation_module, "get_active_tracer", fail_get_active_tracer)

    result = emit_annotation({"score": 1}, propagate=False)

    assert result.name == AGL_ANNOTATION
    assert captured["called"] is False


def test_emit_annotation_rejects_non_primitive_values() -> None:
    with pytest.raises(ValueError):
        emit_annotation({"bad": {"set": {1}}})

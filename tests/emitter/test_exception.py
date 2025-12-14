# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest
from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.emitter import emit_exception
from agentlightning.emitter import exception as exception_module
from agentlightning.semconv import AGL_EXCEPTION

from ..common.tracer import RecordingDummyTracer


def _install_tracer(monkeypatch: pytest.MonkeyPatch) -> RecordingDummyTracer:
    tracer = RecordingDummyTracer()
    monkeypatch.setattr(exception_module, "get_active_tracer", lambda: tracer)
    return tracer


def test_emit_exception_records_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _install_tracer(monkeypatch)
    err = ValueError("boom")

    emit_exception(err)

    assert tracer.last_span is not None
    assert tracer.last_span.name == AGL_EXCEPTION
    assert tracer.last_span.attributes[exception_attributes.EXCEPTION_TYPE] == "ValueError"
    assert tracer.last_span.attributes[exception_attributes.EXCEPTION_MESSAGE] == "boom"
    assert tracer.last_span.attributes[exception_attributes.EXCEPTION_ESCAPED] is True


def test_emit_exception_requires_exception_instance() -> None:
    with pytest.raises(TypeError):
        emit_exception("boom")  # type: ignore[arg-type]

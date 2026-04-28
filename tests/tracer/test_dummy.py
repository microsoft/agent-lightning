# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import itertools

import pytest

from agentlightning.tracer import dummy as dummy_module
from agentlightning.tracer.base import clear_active_tracer, get_active_tracer, set_active_tracer
from agentlightning.tracer.dummy import DummyTracer


def _fake_time_generator(start: float) -> itertools.count[float]:
    return itertools.count(start=start, step=1)


def test_dummy_tracer_create_span_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    fake_clock = _fake_time_generator(42.0)
    monkeypatch.setattr(dummy_module.time, "time", lambda: next(fake_clock))

    span = tracer.create_span("dummy-span", attributes={"foo": "bar"})

    assert span.name == "dummy-span"
    assert span.attributes == {"foo": "bar"}
    assert span.start_time == 42.0
    assert span.end_time == 42.0
    assert span.status.status_code == "OK"


def test_dummy_tracer_operation_context_records_span(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    fake_clock = _fake_time_generator(100.0)
    monkeypatch.setattr(dummy_module.time, "time", lambda: next(fake_clock))

    with tracer.operation_context("dummy-op", attributes={"foo": "bar"}) as ctx:
        ctx.record_attributes({"bar": "baz"})
        ctx.record_status("OK")

    recorded_span = ctx.get_recorded_span()
    assert recorded_span.name == "dummy-op"
    assert recorded_span.attributes == {"foo": "bar", "bar": "baz"}
    assert recorded_span.start_time == 100.0
    assert recorded_span.end_time == 101.0
    assert recorded_span.status.status_code == "OK"


def test_dummy_tracer_operation_context_records_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = DummyTracer()
    fake_clock = _fake_time_generator(200.0)
    monkeypatch.setattr(dummy_module.time, "time", lambda: next(fake_clock))

    with pytest.raises(RuntimeError):
        with tracer.operation_context("dummy-error") as ctx:
            raise RuntimeError("boom")

    recorded_span = ctx.get_recorded_span()  # type: ignore
    assert recorded_span.status.status_code == "ERROR"
    assert recorded_span.status.description == "boom"
    assert recorded_span.end_time == 201.0


@pytest.fixture(autouse=True)
def reset_active_tracer():
    clear_active_tracer()
    yield
    clear_active_tracer()


def test_set_active_tracer_returns_same_instance() -> None:
    tracer = DummyTracer()
    set_active_tracer(tracer)
    assert get_active_tracer() is tracer


def test_set_active_tracer_raises_when_existing() -> None:
    set_active_tracer(DummyTracer())
    with pytest.raises(ValueError):
        set_active_tracer(DummyTracer())


def test_clear_active_tracer_removes_current() -> None:
    tracer = DummyTracer()
    set_active_tracer(tracer)
    clear_active_tracer()
    assert get_active_tracer() is None

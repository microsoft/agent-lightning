"""Tests for InMemoryStore — event operations."""

import time

import pytest

from agl_lite.schemas.api import EnqueueRolloutRequest, PatchRolloutRequest
from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.rollout import RolloutConfig, RolloutStatus
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _enqueue(store: InMemoryStore, **kwargs):
    kwargs.setdefault("input", {})
    kwargs.setdefault("config", RolloutConfig(image="agent:v1"))
    return store.enqueue_rollout(EnqueueRolloutRequest(**kwargs))


def _patch(store: InMemoryStore, rollout_id: str, **kwargs):
    return store.update_rollout(rollout_id, PatchRolloutRequest(**kwargs))


class TestAddEvent:
    def test_basic(self, store: InMemoryStore):
        r = _enqueue(store)
        e = store.add_event(r.rollout_id, "pod-1", "model_request", {"model": "gpt-4"})
        assert e.event_type == "model_request"
        assert e.rollout_id == r.rollout_id
        assert e.attempt_id == "pod-1"
        assert e.data["model"] == "gpt-4"
        assert e.timestamp > 0

    def test_appends_in_order(self, store: InMemoryStore):
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"seq": 1})
        store.add_event(r.rollout_id, "pod-1", "model_request", {"seq": 2})
        store.add_event(r.rollout_id, "pod-1", "reward", {"value": 1.0})
        events = store.query_events(r.rollout_id, attempt_id="pod-1")
        assert len(events) == 3
        assert events[0].data["seq"] == 1
        assert events[1].data["seq"] == 2
        assert events[2].data["value"] == 1.0

    def test_separate_attempts(self, store: InMemoryStore):
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"attempt": 1})
        store.add_event(r.rollout_id, "pod-2", "model_request", {"attempt": 2})
        events_1 = store.query_events(r.rollout_id, attempt_id="pod-1")
        events_2 = store.query_events(r.rollout_id, attempt_id="pod-2")
        assert len(events_1) == 1
        assert len(events_2) == 1
        assert events_1[0].data["attempt"] == 1
        assert events_2[0].data["attempt"] == 2

    def test_nonexistent_rollout(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.add_event("nonexistent", "pod-1", "reward", {"value": 1.0})


class TestAddEvents:
    def test_batch(self, store: InMemoryStore):
        r = _enqueue(store)
        events = store.add_events(
            [
                {"rollout_id": r.rollout_id, "attempt_id": "pod-1", "event_type": "model_request", "data": {"i": 0}},
                {"rollout_id": r.rollout_id, "attempt_id": "pod-1", "event_type": "reward", "data": {"value": 1.0}},
            ]
        )
        assert len(events) == 2
        assert events[0].event_type == "model_request"
        assert events[1].event_type == "reward"


class TestQueryEvents:
    def test_filter_by_event_type(self, store: InMemoryStore):
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"model": "gpt-4"})
        store.add_event(r.rollout_id, "pod-1", "tool_result", {"output": "hello"})
        store.add_event(r.rollout_id, "pod-1", "model_request", {"model": "gpt-4"})
        store.add_event(r.rollout_id, "pod-1", "reward", {"value": 0.5})
        results = store.query_events(r.rollout_id, attempt_id="pod-1", event_type="model_request")
        assert len(results) == 2

    def test_pagination(self, store: InMemoryStore):
        r = _enqueue(store)
        for i in range(10):
            store.add_event(r.rollout_id, "pod-1", "model_request", {"i": i})
        page = store.query_events(r.rollout_id, attempt_id="pod-1", limit=3, offset=2)
        assert len(page) == 3
        assert page[0].data["i"] == 2

    def test_nonexistent_rollout(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.query_events("nonexistent")

    def test_nonexistent_attempt(self, store: InMemoryStore):
        r = _enqueue(store)
        events = store.query_events(r.rollout_id, attempt_id="nonexistent-pod")
        assert events == []

    def test_no_events(self, store: InMemoryStore):
        r = _enqueue(store)
        events = store.query_events(r.rollout_id)
        assert events == []


class TestSmartAttemptResolution:
    def test_uses_succeeded_attempt(self, store: InMemoryStore):
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"attempt": "failed"})
        store.add_event(r.rollout_id, "pod-2", "model_request", {"attempt": "succeeded"})
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        _patch(store, r.rollout_id, status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="pod-2")
        events = store.query_events(r.rollout_id)  # no attempt_id
        assert len(events) == 1
        assert events[0].data["attempt"] == "succeeded"

    def test_falls_back_to_latest_attempt(self, store: InMemoryStore):
        """When no succeeded_attempt_id, uses attempt with latest first event timestamp."""
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"attempt": "first"})
        time.sleep(0.01)
        store.add_event(r.rollout_id, "pod-2", "model_request", {"attempt": "second"})
        events = store.query_events(r.rollout_id)  # no attempt_id, no succeeded
        assert len(events) == 1
        assert events[0].data["attempt"] == "second"  # latest

    def test_empty_when_no_events(self, store: InMemoryStore):
        r = _enqueue(store)
        events = store.query_events(r.rollout_id)
        assert events == []


class TestListAttempts:
    def test_ordered_by_first_event(self, store: InMemoryStore):
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {})
        time.sleep(0.01)
        store.add_event(r.rollout_id, "pod-2", "model_request", {})
        attempts = store.list_attempts(r.rollout_id)
        assert attempts == ["pod-1", "pod-2"]

    def test_empty(self, store: InMemoryStore):
        r = _enqueue(store)
        assert store.list_attempts(r.rollout_id) == []

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.list_attempts("nonexistent")

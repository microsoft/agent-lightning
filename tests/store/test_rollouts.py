"""Tests for InMemoryStore — rollout operations."""

import pytest

from agl_lite.schemas.api import EnqueueRolloutRequest, PatchRolloutRequest
from agl_lite.schemas.errors import InvalidTransitionError, NotFoundError
from agl_lite.schemas.rollout import TERMINAL_STATUSES, RolloutConfig, RolloutStatus
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _enqueue(store: InMemoryStore, **kwargs):
    """Helper: enqueue with defaults."""
    kwargs.setdefault("input", {})
    kwargs.setdefault("config", RolloutConfig())
    return store.enqueue_rollouts([EnqueueRolloutRequest(**kwargs)])[0]


def _patch(store: InMemoryStore, rollout_id: str, **kwargs):
    """Helper: partial update."""
    return store.update_rollout(rollout_id, PatchRolloutRequest(**kwargs))


class TestEnqueueRollout:
    def test_creates_rollout_in_queuing(self, store: InMemoryStore):
        r = _enqueue(store, input={"prompt": "hello"})
        assert r.status == RolloutStatus.QUEUING
        assert r.input == {"prompt": "hello"}
        assert r.config.pod_spec is None
        assert r.version == 1
        assert r.cancel_requested is False
        assert r.rollout_id  # non-empty

    def test_unique_ids(self, store: InMemoryStore):
        r1 = _enqueue(store, input={"a": 1})
        r2 = _enqueue(store, input={"a": 2})
        assert r1.rollout_id != r2.rollout_id

    def test_with_resources_id(self, store: InMemoryStore):
        r = _enqueue(store, resources_id="res-1")
        assert r.resources_id == "res-1"

    def test_initializes_event_dict(self, store: InMemoryStore):
        r = _enqueue(store)
        assert store._events[r.rollout_id] == {}

    def test_default_config_when_none(self, store: InMemoryStore):
        r = store.enqueue_rollouts([EnqueueRolloutRequest(input={"x": 1})])[0]
        assert r.config is not None


class TestGetRollout:
    def test_found(self, store: InMemoryStore):
        r = _enqueue(store)
        found = store.get_rollout(r.rollout_id)
        assert found.rollout_id == r.rollout_id

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError, match="not found"):
            store.get_rollout("nonexistent")


class TestRolloutExists:
    def test_exists(self, store: InMemoryStore):
        r = _enqueue(store)
        assert store.rollout_exists(r.rollout_id) is True

    def test_not_exists(self, store: InMemoryStore):
        assert store.rollout_exists("nonexistent") is False


class TestUpdateRollout:
    def test_queuing_to_running(self, store: InMemoryStore):
        r = _enqueue(store)
        updated = _patch(store, r.rollout_id, status=RolloutStatus.RUNNING, job_name="agl-rollout-x")
        assert updated.status == RolloutStatus.RUNNING
        assert updated.version == 2
        assert updated.job_name == "agl-rollout-x"

    def test_running_to_succeeded(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        updated = _patch(store, r.rollout_id, status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="pod-uid-1")
        assert updated.status == RolloutStatus.SUCCEEDED
        assert updated.succeeded_attempt_id == "pod-uid-1"
        assert updated.version == 3

    def test_running_to_terminal_failed(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        updated = _patch(
            store, r.rollout_id, status=RolloutStatus.TERMINAL_FAILED, error_message="BackoffLimitExceeded"
        )
        assert updated.status == RolloutStatus.TERMINAL_FAILED
        assert updated.error_message == "BackoffLimitExceeded"

    def test_running_to_cancelled(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        updated = _patch(store, r.rollout_id, status=RolloutStatus.CANCELLED)
        assert updated.status == RolloutStatus.CANCELLED

    def test_queuing_to_terminal_failed(self, store: InMemoryStore):
        r = _enqueue(store)
        updated = _patch(store, r.rollout_id, status=RolloutStatus.TERMINAL_FAILED, error_message="Job creation failed")
        assert updated.status == RolloutStatus.TERMINAL_FAILED

    def test_queuing_to_cancelled(self, store: InMemoryStore):
        r = _enqueue(store)
        updated = _patch(store, r.rollout_id, status=RolloutStatus.CANCELLED)
        assert updated.status == RolloutStatus.CANCELLED

    # --- Invalid transitions ---

    def test_queuing_to_succeeded_rejected(self, store: InMemoryStore):
        r = _enqueue(store)
        with pytest.raises(InvalidTransitionError):
            _patch(store, r.rollout_id, status=RolloutStatus.SUCCEEDED)

    def test_running_to_queuing_rejected(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        with pytest.raises(InvalidTransitionError):
            _patch(store, r.rollout_id, status=RolloutStatus.QUEUING)

    def test_terminal_to_anything_rejected(self, store: InMemoryStore):
        for terminal in TERMINAL_STATUSES:
            s = InMemoryStore()
            r = _enqueue(s)
            if terminal == RolloutStatus.SUCCEEDED:
                _patch(s, r.rollout_id, status=RolloutStatus.RUNNING)
                _patch(s, r.rollout_id, status=terminal)
            else:
                _patch(s, r.rollout_id, status=terminal)

            for target in RolloutStatus:
                with pytest.raises(InvalidTransitionError):
                    _patch(s, r.rollout_id, status=target)

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            _patch(store, "nonexistent", status=RolloutStatus.RUNNING)

    # --- Partial update semantics ---

    def test_only_set_fields_applied(self, store: InMemoryStore):
        """Fields not in request body are untouched."""
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING, job_name="my-job")
        # Patch status only — job_name should be preserved.
        updated = _patch(store, r.rollout_id, status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="pod-1")
        assert updated.job_name == "my-job"  # preserved
        assert updated.succeeded_attempt_id == "pod-1"

    def test_update_without_status(self, store: InMemoryStore):
        """Can update non-status fields without changing status."""
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        updated = _patch(store, r.rollout_id, job_name="agl-rollout-xyz")
        assert updated.status == RolloutStatus.RUNNING  # unchanged
        assert updated.job_name == "agl-rollout-xyz"
        assert updated.version == 3

    def test_empty_patch_is_noop(self, store: InMemoryStore):
        """Empty body = no changes, no version bump."""
        r = _enqueue(store)
        updated = store.update_rollout(r.rollout_id, PatchRolloutRequest())
        assert updated.version == 1  # no bump

    def test_explicit_null_clears_field(self, store: InMemoryStore):
        """Explicitly sending null should set the field to None."""
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING, job_name="my-job")
        # Explicitly clear job_name by constructing request with job_name=None set.
        req = PatchRolloutRequest.model_validate({"job_name": None})
        updated = store.update_rollout(r.rollout_id, req)
        assert updated.job_name is None
        assert updated.version == 3


class TestCancelRollout:
    def test_cancel_queuing(self, store: InMemoryStore):
        r = _enqueue(store)
        updated = store.cancel_rollout(r.rollout_id)
        assert updated.cancel_requested is True
        assert updated.version == 2

    def test_cancel_running(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.RUNNING)
        updated = store.cancel_rollout(r.rollout_id)
        assert updated.cancel_requested is True

    def test_cancel_idempotent(self, store: InMemoryStore):
        r = _enqueue(store)
        first = store.cancel_rollout(r.rollout_id)
        second = store.cancel_rollout(r.rollout_id)
        assert second.version == first.version  # no version bump on idempotent call

    def test_cancel_terminal_rejected(self, store: InMemoryStore):
        r = _enqueue(store)
        _patch(store, r.rollout_id, status=RolloutStatus.TERMINAL_FAILED)
        with pytest.raises(InvalidTransitionError, match="cancel_requested"):
            store.cancel_rollout(r.rollout_id)

    def test_cancel_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.cancel_rollout("nonexistent")


class TestQueryRollouts:
    def test_all(self, store: InMemoryStore):
        _enqueue(store, input={"a": 1})
        _enqueue(store, input={"a": 2})
        results = store.query_rollouts()
        assert len(results) == 2

    def test_by_ids(self, store: InMemoryStore):
        r1 = _enqueue(store)
        _enqueue(store)
        r3 = _enqueue(store)
        results = store.query_rollouts(ids=[r1.rollout_id, r3.rollout_id])
        assert len(results) == 2
        assert {r.rollout_id for r in results} == {r1.rollout_id, r3.rollout_id}

    def test_by_ids_missing(self, store: InMemoryStore):
        r = _enqueue(store)
        results = store.query_rollouts(ids=[r.rollout_id, "nonexistent"])
        assert len(results) == 1

    def test_filter_status(self, store: InMemoryStore):
        r1 = _enqueue(store)
        _enqueue(store)
        _patch(store, r1.rollout_id, status=RolloutStatus.RUNNING)
        results = store.query_rollouts(status_in=[RolloutStatus.RUNNING])
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_filter_cancel_requested(self, store: InMemoryStore):
        r1 = _enqueue(store)
        _enqueue(store)
        store.cancel_rollout(r1.rollout_id)
        results = store.query_rollouts(cancel_requested=True)
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_combined_filters(self, store: InMemoryStore):
        r1 = _enqueue(store)
        r2 = _enqueue(store)
        store.cancel_rollout(r1.rollout_id)
        store.cancel_rollout(r2.rollout_id)
        _patch(store, r1.rollout_id, status=RolloutStatus.RUNNING)
        results = store.query_rollouts(status_in=[RolloutStatus.RUNNING], cancel_requested=True)
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_pagination(self, store: InMemoryStore):
        for i in range(10):
            _enqueue(store, input={"i": i})
        page1 = store.query_rollouts(limit=3, offset=0)
        page2 = store.query_rollouts(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert {r.rollout_id for r in page1}.isdisjoint({r.rollout_id for r in page2})

    def test_empty(self, store: InMemoryStore):
        assert store.query_rollouts() == []

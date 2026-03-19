"""Tests for InMemoryStore — rollout operations."""

import pytest

from agl_lite.schemas.errors import ConflictError, InvalidTransitionError, NotFoundError
from agl_lite.schemas.rollout import TERMINAL_STATUSES, RolloutConfig, RolloutStatus
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def config() -> RolloutConfig:
    return RolloutConfig(image="agent:v1")


class TestEnqueueRollout:
    def test_creates_rollout_in_queuing(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={"prompt": "hello"}, config=config)
        assert r.status == RolloutStatus.QUEUING
        assert r.input == {"prompt": "hello"}
        assert r.config.image == "agent:v1"
        assert r.version == 1
        assert r.cancel_requested is False
        assert r.rollout_id  # non-empty

    def test_unique_ids(self, store: InMemoryStore, config: RolloutConfig):
        r1 = store.enqueue_rollout(input={"a": 1}, config=config)
        r2 = store.enqueue_rollout(input={"a": 2}, config=config)
        assert r1.rollout_id != r2.rollout_id

    def test_with_resources_id(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config, resources_id="res-1")
        assert r.resources_id == "res-1"

    def test_initializes_event_dict(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        assert store._events[r.rollout_id] == {}


class TestGetRollout:
    def test_found(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        found = store.get_rollout(r.rollout_id)
        assert found.rollout_id == r.rollout_id

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError, match="not found"):
            store.get_rollout("nonexistent")


class TestRolloutExists:
    def test_exists(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        assert store.rollout_exists(r.rollout_id) is True

    def test_not_exists(self, store: InMemoryStore):
        assert store.rollout_exists("nonexistent") is False


class TestUpdateRollout:
    def test_queuing_to_running(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        updated = store.update_rollout(
            r.rollout_id, RolloutStatus.RUNNING, expected_version=1, job_name="agl-rollout-x"
        )
        assert updated.status == RolloutStatus.RUNNING
        assert updated.version == 2
        assert updated.job_name == "agl-rollout-x"

    def test_running_to_succeeded(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        updated = store.update_rollout(
            r.rollout_id, RolloutStatus.SUCCEEDED, expected_version=2, succeeded_attempt_id="pod-uid-1"
        )
        assert updated.status == RolloutStatus.SUCCEEDED
        assert updated.succeeded_attempt_id == "pod-uid-1"
        assert updated.version == 3

    def test_running_to_terminal_failed(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        updated = store.update_rollout(
            r.rollout_id, RolloutStatus.TERMINAL_FAILED, expected_version=2, error_message="BackoffLimitExceeded"
        )
        assert updated.status == RolloutStatus.TERMINAL_FAILED
        assert updated.error_message == "BackoffLimitExceeded"

    def test_running_to_cancelled(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        updated = store.update_rollout(r.rollout_id, RolloutStatus.CANCELLED, expected_version=2)
        assert updated.status == RolloutStatus.CANCELLED

    def test_queuing_to_terminal_failed(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        updated = store.update_rollout(
            r.rollout_id, RolloutStatus.TERMINAL_FAILED, expected_version=1, error_message="Job creation failed"
        )
        assert updated.status == RolloutStatus.TERMINAL_FAILED

    def test_queuing_to_cancelled(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        updated = store.update_rollout(r.rollout_id, RolloutStatus.CANCELLED, expected_version=1)
        assert updated.status == RolloutStatus.CANCELLED

    # --- Invalid transitions ---

    def test_queuing_to_succeeded_rejected(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        with pytest.raises(InvalidTransitionError):
            store.update_rollout(r.rollout_id, RolloutStatus.SUCCEEDED, expected_version=1)

    def test_running_to_queuing_rejected(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        with pytest.raises(InvalidTransitionError):
            store.update_rollout(r.rollout_id, RolloutStatus.QUEUING, expected_version=2)

    def test_terminal_to_anything_rejected(self, store: InMemoryStore, config: RolloutConfig):
        for terminal in TERMINAL_STATUSES:
            s = InMemoryStore()
            r = s.enqueue_rollout(input={}, config=config)
            if terminal == RolloutStatus.SUCCEEDED:
                s.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
                s.update_rollout(r.rollout_id, terminal, expected_version=2)
                version = 3
            else:
                s.update_rollout(r.rollout_id, terminal, expected_version=1)
                version = 2

            for target in RolloutStatus:
                with pytest.raises(InvalidTransitionError):
                    s.update_rollout(r.rollout_id, target, expected_version=version)

    # --- Optimistic locking ---

    def test_version_mismatch(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        with pytest.raises(ConflictError, match="expected version 99"):
            store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=99)

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.update_rollout("nonexistent", RolloutStatus.RUNNING, expected_version=1)

    # --- Optional fields preserved ---

    def test_optional_fields_not_overwritten(self, store: InMemoryStore, config: RolloutConfig):
        """Passing None for optional fields should not overwrite existing values."""
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1, job_name="my-job")
        # Update to succeeded without re-specifying job_name.
        updated = store.update_rollout(
            r.rollout_id, RolloutStatus.SUCCEEDED, expected_version=2, succeeded_attempt_id="pod-1"
        )
        assert updated.job_name == "my-job"  # preserved
        assert updated.succeeded_attempt_id == "pod-1"


class TestCancelRollout:
    def test_cancel_queuing(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        updated = store.cancel_rollout(r.rollout_id)
        assert updated.cancel_requested is True
        assert updated.version == 2

    def test_cancel_running(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        updated = store.cancel_rollout(r.rollout_id)
        assert updated.cancel_requested is True

    def test_cancel_idempotent(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        first = store.cancel_rollout(r.rollout_id)
        second = store.cancel_rollout(r.rollout_id)
        assert second.version == first.version  # no version bump on idempotent call

    def test_cancel_terminal_rejected(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r.rollout_id, RolloutStatus.TERMINAL_FAILED, expected_version=1)
        with pytest.raises(InvalidTransitionError, match="cancel_requested"):
            store.cancel_rollout(r.rollout_id)

    def test_cancel_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.cancel_rollout("nonexistent")


class TestQueryRollouts:
    def test_all(self, store: InMemoryStore, config: RolloutConfig):
        store.enqueue_rollout(input={"a": 1}, config=config)
        store.enqueue_rollout(input={"a": 2}, config=config)
        results = store.query_rollouts()
        assert len(results) == 2

    def test_by_ids(self, store: InMemoryStore, config: RolloutConfig):
        r1 = store.enqueue_rollout(input={}, config=config)
        store.enqueue_rollout(input={}, config=config)
        r3 = store.enqueue_rollout(input={}, config=config)
        results = store.query_rollouts(ids=[r1.rollout_id, r3.rollout_id])
        assert len(results) == 2
        assert {r.rollout_id for r in results} == {r1.rollout_id, r3.rollout_id}

    def test_by_ids_missing(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        results = store.query_rollouts(ids=[r.rollout_id, "nonexistent"])
        assert len(results) == 1

    def test_filter_status(self, store: InMemoryStore, config: RolloutConfig):
        r1 = store.enqueue_rollout(input={}, config=config)
        store.enqueue_rollout(input={}, config=config)
        store.update_rollout(r1.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        results = store.query_rollouts(status_in=[RolloutStatus.RUNNING])
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_filter_cancel_requested(self, store: InMemoryStore, config: RolloutConfig):
        r1 = store.enqueue_rollout(input={}, config=config)
        store.enqueue_rollout(input={}, config=config)
        store.cancel_rollout(r1.rollout_id)
        results = store.query_rollouts(cancel_requested=True)
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_combined_filters(self, store: InMemoryStore, config: RolloutConfig):
        r1 = store.enqueue_rollout(input={}, config=config)
        r2 = store.enqueue_rollout(input={}, config=config)
        store.cancel_rollout(r1.rollout_id)
        store.cancel_rollout(r2.rollout_id)
        store.update_rollout(r1.rollout_id, RolloutStatus.RUNNING, expected_version=2)
        results = store.query_rollouts(status_in=[RolloutStatus.RUNNING], cancel_requested=True)
        assert len(results) == 1
        assert results[0].rollout_id == r1.rollout_id

    def test_pagination(self, store: InMemoryStore, config: RolloutConfig):
        for i in range(10):
            store.enqueue_rollout(input={"i": i}, config=config)
        page1 = store.query_rollouts(limit=3, offset=0)
        page2 = store.query_rollouts(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert {r.rollout_id for r in page1}.isdisjoint({r.rollout_id for r in page2})

    def test_empty(self, store: InMemoryStore):
        assert store.query_rollouts() == []

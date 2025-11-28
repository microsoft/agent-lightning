# Copyright (c) Microsoft. All rights reserved.

"""
Comprehensive tests for LightningStore.

Test categories:
- Core CRUD operations
- Queue operations (FIFO behavior)
- Resource versioning
- Span tracking and sequencing
- Rollout lifecycle and status transitions
- Concurrent access patterns
- Error handling and edge cases

It should work for multiple store implementations (InMemory, SQL, etc.).
"""

import asyncio
import logging
import sys
import time
from typing import List, Optional, Sequence, cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from agentlightning.store.base import UNSET, LightningStore
from agentlightning.store.memory import InMemoryLightningStore, estimate_model_size
from agentlightning.types import (
    LLM,
    AttemptedRollout,
    Event,
    Link,
    OtelResource,
    PaginatedResult,
    PromptTemplate,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    Span,
    SpanContext,
    TraceStatus,
)

# Typing tests


def test_paginated_result_behaves_like_sequence() -> None:
    result = PaginatedResult(items=["a", "b", "c"], limit=2, offset=1, total=5)

    assert isinstance(result, Sequence)
    assert len(result) == 3
    assert result[0] == "a"
    assert result[1:] == ["b", "c"]
    assert list(result) == ["a", "b", "c"]

    assert repr(result) == "<PaginatedResult (1:3 of 5) ['a', ...]>"

    result2 = PaginatedResult(items=["a", "b", "c"], limit=-1, offset=1, total=5)
    assert repr(result2) == "<PaginatedResult (1: of 5) ['a', ...]>"


# Core CRUD Operations Tests


@pytest.mark.asyncio
async def test_enqueue_rollout_creates_rollout(store_fixture: LightningStore) -> None:
    """Test that enqueue_rollout creates a properly initialized rollout."""
    sample = {"input": "test_data"}
    metadata = {"key": "value", "number": 42}

    rollout = await store_fixture.enqueue_rollout(input=sample, mode="train", resources_id="res-123", metadata=metadata)

    assert rollout.rollout_id.startswith("ro-")
    assert rollout.input == sample
    assert rollout.mode == "train"
    assert rollout.resources_id == "res-123"
    assert rollout.metadata == metadata
    assert rollout.status == "queuing"
    assert rollout.start_time is not None


@pytest.mark.asyncio
async def test_enqueue_rollout_accepts_config(store_fixture: LightningStore) -> None:
    """Rollout-specific configs can be provided when enqueuing tasks."""
    config = RolloutConfig(timeout_seconds=12.0, max_attempts=3, retry_condition=["timeout"])

    rollout = await store_fixture.enqueue_rollout(input={"sample": True}, config=config)

    assert rollout.config.timeout_seconds == 12.0
    assert rollout.config.max_attempts == 3
    assert rollout.config.retry_condition == ["timeout"]

    stored = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert stored is not None
    assert stored.config.timeout_seconds == 12.0
    assert stored.config.max_attempts == 3
    assert stored.config.retry_condition == ["timeout"]


@pytest.mark.asyncio
async def test_add_rollout_initializes_attempt(store_fixture: LightningStore) -> None:
    """Test that add_rollout immediately tracks a preparing attempt."""
    sample = {"payload": "value"}

    attempt_rollout = await store_fixture.start_rollout(input=sample, mode="val", resources_id="res-add")

    assert attempt_rollout.status == "preparing"
    assert attempt_rollout.rollout_id.startswith("ro-")
    assert attempt_rollout.attempt.attempt_id.startswith("at-")
    assert attempt_rollout.attempt.sequence_id == 1
    assert attempt_rollout.attempt.status == "preparing"

    stored = await store_fixture.query_rollouts(status=["preparing"])
    assert len(stored) == 1
    assert stored[0].rollout_id == attempt_rollout.rollout_id
    assert stored[0].resources_id == "res-add"

    attempts = await store_fixture.query_attempts(attempt_rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempt_rollout.attempt.attempt_id

    latest_attempt = await store_fixture.get_latest_attempt(attempt_rollout.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == attempt_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_start_rollout_accepts_config(store_fixture: LightningStore) -> None:
    """Custom rollout config is preserved for started rollouts."""
    config = RolloutConfig(unresponsive_seconds=5.0, max_attempts=2, retry_condition=["unresponsive"])

    attempt_rollout = await store_fixture.start_rollout(input={"payload": "value"}, config=config)

    assert attempt_rollout.config.unresponsive_seconds == 5.0
    assert attempt_rollout.config.max_attempts == 2
    assert attempt_rollout.config.retry_condition == ["unresponsive"]

    stored = await store_fixture.get_rollout_by_id(attempt_rollout.rollout_id)
    assert stored is not None
    assert stored.config.unresponsive_seconds == 5.0
    assert stored.config.max_attempts == 2
    assert stored.config.retry_condition == ["unresponsive"]


@pytest.mark.asyncio
async def test_query_rollouts_by_status(store_fixture: LightningStore) -> None:
    """Test querying rollouts filtered by status."""
    # Create rollouts with different statuses
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    # Modify statuses
    await store_fixture.dequeue_rollout()  # r1 becomes "preparing"
    await store_fixture.update_rollout(rollout_id=r2.rollout_id, status="failed")
    # r3 remains "queuing"

    # Test various queries
    all_rollouts = await store_fixture.query_rollouts()
    assert len(all_rollouts) == 3

    queuing = await store_fixture.query_rollouts(status=["queuing"])
    assert len(queuing) == 1
    assert queuing[0].rollout_id == r3.rollout_id

    preparing = await store_fixture.query_rollouts(status=["preparing"])
    assert len(preparing) == 1
    assert preparing[0].rollout_id == r1.rollout_id

    finished = await store_fixture.query_rollouts(status=["failed", "succeeded"])
    assert len(finished) == 1
    assert finished[0].rollout_id == r2.rollout_id

    # Empty status list
    none = await store_fixture.query_rollouts(status=[])
    assert len(none) == 0


@pytest.mark.asyncio
async def test_query_rollouts_returns_latest_attempt(store_fixture: LightningStore) -> None:
    """Querying rollouts should attach the most recent attempt when present."""
    attempted = await store_fixture.start_rollout(input={"sample": "latest"})
    latest_attempt = await store_fixture.start_attempt(attempted.rollout_id)

    results = await store_fixture.query_rollouts(rollout_ids=[attempted.rollout_id])
    assert len(results) == 1

    retrieved = results[0]
    assert type(retrieved) is AttemptedRollout
    assert retrieved.attempt.attempt_id == latest_attempt.attempt.attempt_id
    assert retrieved.attempt.sequence_id == latest_attempt.attempt.sequence_id


@pytest.mark.asyncio
async def test_query_rollouts_supports_new_filters(store_fixture: LightningStore) -> None:
    """The expanded query interface should honor filtering, sorting, and pagination."""
    rollouts = [await store_fixture.enqueue_rollout(input={"idx": idx}) for idx in range(3)]
    await store_fixture.update_rollout(rollout_id=rollouts[2].rollout_id, status="failed")

    failed = await store_fixture.query_rollouts(status_in=["failed"])
    assert [r.rollout_id for r in failed] == [rollouts[2].rollout_id]

    sorted_desc = sorted([r.rollout_id for r in rollouts], reverse=True)
    paged = await store_fixture.query_rollouts(sort_by="rollout_id", sort_order="desc", limit=2)
    assert [r.rollout_id for r in paged] == sorted_desc[:2]

    offset_item = await store_fixture.query_rollouts(sort_by="rollout_id", sort_order="desc", limit=1, offset=1)
    assert [r.rollout_id for r in offset_item] == sorted_desc[1:2]

    contains = await store_fixture.query_rollouts(
        rollout_id_contains=rollouts[0].rollout_id[-4:],
        status_in=["queuing"],
    )
    assert any(r.rollout_id == rollouts[0].rollout_id for r in contains)

    or_filtered = await store_fixture.query_rollouts(
        status_in=["succeeded"],
        rollout_id_in=[rollouts[1].rollout_id],
        filter_logic="or",
    )
    assert [r.rollout_id for r in or_filtered] == [rollouts[1].rollout_id]


@pytest.mark.asyncio
async def test_query_rollouts_status_in_takes_precedence(store_fixture: LightningStore) -> None:
    """status_in should override the legacy status parameter when both are provided."""
    failed = await store_fixture.enqueue_rollout(input={"kind": "failed"})
    pending = await store_fixture.enqueue_rollout(input={"kind": "pending"})

    await store_fixture.update_rollout(rollout_id=failed.rollout_id, status="failed")

    results = await store_fixture.query_rollouts(status=["queuing"], status_in=["failed"])
    assert [rollout.rollout_id for rollout in results] == [failed.rollout_id]

    legacy = await store_fixture.query_rollouts(status=["queuing"])
    assert any(rollout.rollout_id == pending.rollout_id for rollout in legacy)


@pytest.mark.asyncio
async def test_query_rollouts_rollout_id_in_takes_precedence(store_fixture: LightningStore) -> None:
    """rollout_id_in should override the legacy rollout_ids parameter."""
    keep = await store_fixture.enqueue_rollout(input={"kind": "keep"})
    ignored = await store_fixture.enqueue_rollout(input={"kind": "ignored"})

    results = await store_fixture.query_rollouts(rollout_ids=[ignored.rollout_id], rollout_id_in=[keep.rollout_id])
    assert [rollout.rollout_id for rollout in results] == [keep.rollout_id]


@pytest.mark.asyncio
async def test_query_rollouts_filter_logic_controls_contains_behavior(store_fixture: LightningStore) -> None:
    """Changing filter_logic should alter how status and substring filters are combined."""
    failed = await store_fixture.enqueue_rollout(input={"kind": "failed"})
    still_queueing = await store_fixture.enqueue_rollout(input={"kind": "queue"})
    await store_fixture.update_rollout(rollout_id=failed.rollout_id, status="failed")

    substring = still_queueing.rollout_id[-6:]
    and_results = await store_fixture.query_rollouts(
        status_in=["failed"],
        rollout_id_contains=substring,
    )
    assert len(and_results) == 0

    or_results = await store_fixture.query_rollouts(
        status_in=["failed"],
        rollout_id_contains=substring,
        filter_logic="or",
        sort_by="rollout_id",
    )
    returned_ids = {rollout.rollout_id for rollout in or_results}
    assert returned_ids == {failed.rollout_id, still_queueing.rollout_id}


@pytest.mark.asyncio
async def test_query_rollouts_combined_filters_with_sort_and_pagination(store_fixture: LightningStore) -> None:
    """Complex combinations of filters should work with explicit sorting and pagination."""
    rollouts = [await store_fixture.enqueue_rollout(input={"idx": idx}) for idx in range(4)]
    await store_fixture.update_rollout(rollout_id=rollouts[0].rollout_id, status="failed")
    await store_fixture.update_rollout(rollout_id=rollouts[3].rollout_id, status="failed")

    substring = rollouts[2].rollout_id[-5:]
    filtered = await store_fixture.query_rollouts(
        status_in=["failed"],
        rollout_id_contains=substring,
        filter_logic="or",
        sort_by="rollout_id",
        sort_order="asc",
        limit=2,
        offset=1,
    )

    expected_ids = sorted({rollouts[0].rollout_id, rollouts[2].rollout_id, rollouts[3].rollout_id})
    assert [item.rollout_id for item in filtered] == expected_ids[1:3]


@pytest.mark.asyncio
async def test_query_rollouts_reports_pagination_metadata(store_fixture: LightningStore) -> None:
    """Paginated rollouts should expose limit/offset/total values."""
    rollouts = [await store_fixture.enqueue_rollout(input={"idx": idx}) for idx in range(3)]

    paginated = await store_fixture.query_rollouts(sort_by="rollout_id", sort_order="asc", limit=1, offset=1)
    assert isinstance(paginated, PaginatedResult)
    assert paginated.limit == 1
    assert paginated.offset == 1
    assert paginated.total == len(rollouts)


@pytest.mark.asyncio
async def test_query_attempts_supports_sort_and_limit(store_fixture: LightningStore) -> None:
    """Attempt queries should respect sorting and pagination controls."""
    attempted = await store_fixture.start_rollout(input={"payload": "attempt-filters"})
    await store_fixture.start_attempt(attempted.rollout_id)
    await store_fixture.start_attempt(attempted.rollout_id)

    attempts_desc = await store_fixture.query_attempts(attempted.rollout_id, sort_by="sequence_id", sort_order="desc")
    assert [attempt.sequence_id for attempt in attempts_desc] == [3, 2, 1]

    middle_attempt = await store_fixture.query_attempts(
        attempted.rollout_id,
        sort_by="sequence_id",
        sort_order="desc",
        limit=1,
        offset=1,
    )
    assert [attempt.sequence_id for attempt in middle_attempt] == [2]


@pytest.mark.asyncio
async def test_query_attempts_offset_past_end_returns_empty(store_fixture: LightningStore) -> None:
    """Offsets beyond the result size should return an empty list."""
    attempted = await store_fixture.start_rollout(input={"payload": "attempt-offset"})
    await store_fixture.start_attempt(attempted.rollout_id)
    await store_fixture.start_attempt(attempted.rollout_id)

    results = await store_fixture.query_attempts(
        attempted.rollout_id,
        sort_by="sequence_id",
        sort_order="asc",
        limit=1,
        offset=10,
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_query_attempts_zero_limit_returns_no_items(store_fixture: LightningStore) -> None:
    """A zero limit should be treated as 'return nothing' even when attempts exist."""
    attempted = await store_fixture.start_rollout(input={"payload": "attempt-limit"})
    await store_fixture.start_attempt(attempted.rollout_id)

    results = await store_fixture.query_attempts(attempted.rollout_id, limit=0)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_query_attempts_reports_pagination_metadata(store_fixture: LightningStore) -> None:
    """Attempt pagination should retain metadata."""
    attempted = await store_fixture.start_rollout(input={"payload": "attempt-pagination"})
    await store_fixture.start_attempt(attempted.rollout_id)
    await store_fixture.start_attempt(attempted.rollout_id)

    paginated = await store_fixture.query_attempts(
        attempted.rollout_id,
        sort_by="sequence_id",
        sort_order="asc",
        limit=1,
        offset=1,
    )
    assert isinstance(paginated, PaginatedResult)
    assert paginated.limit == 1
    assert paginated.offset == 1
    assert paginated.total == 3


@pytest.mark.asyncio
async def test_get_rollout_by_id_returns_latest_attempt(store_fixture: LightningStore) -> None:
    """Fetching a rollout by ID should include the latest attempt when available."""
    attempted = await store_fixture.start_rollout(input={"foo": "bar"})
    second_attempt = await store_fixture.start_attempt(attempted.rollout_id)

    retrieved = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert retrieved is not None
    assert type(retrieved) is AttemptedRollout
    assert retrieved.attempt.attempt_id == second_attempt.attempt.attempt_id
    assert retrieved.attempt.sequence_id == second_attempt.attempt.sequence_id


@pytest.mark.asyncio
async def test_get_rollout_by_id_without_attempt_returns_rollout(
    store_fixture: LightningStore,
) -> None:
    """Rollouts with no attempts should be returned without the Attempt wrapper."""
    queued = await store_fixture.enqueue_rollout(input={"foo": "bar"})

    retrieved = await store_fixture.get_rollout_by_id(queued.rollout_id)
    assert retrieved is not None
    assert type(retrieved) is Rollout
    assert not hasattr(retrieved, "attempt")


@pytest.mark.asyncio
async def test_get_rollout_by_id(store_fixture: LightningStore) -> None:
    """Test retrieving rollouts by their ID."""
    # Test getting non-existent rollout
    rollout = await store_fixture.get_rollout_by_id("nonexistent")
    assert rollout is None

    # Create a rollout
    created = await store_fixture.enqueue_rollout(input={"test": "data"}, mode="train")

    # Retrieve by ID
    retrieved = await store_fixture.get_rollout_by_id(created.rollout_id)
    assert retrieved is not None
    assert retrieved.rollout_id == created.rollout_id
    assert retrieved.input == created.input
    assert retrieved.mode == created.mode
    assert retrieved.status == created.status

    # Update rollout and verify changes are reflected
    await store_fixture.update_rollout(rollout_id=created.rollout_id, status="running")
    updated = await store_fixture.get_rollout_by_id(created.rollout_id)
    assert updated is not None
    assert updated.status == "running"


@pytest.mark.asyncio
async def test_store_lock_rebinds_to_new_event_loop(
    store_fixture: LightningStore,
) -> None:
    """The in-memory store can be reused after switching to a new event loop."""

    rollout = await store_fixture.enqueue_rollout(input={"foo": "bar"})

    def run_in_new_loop() -> Optional[Rollout]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(store_fixture.get_rollout_by_id(rollout.rollout_id))
        finally:
            loop.close()

    retrieved = await asyncio.to_thread(run_in_new_loop)

    assert retrieved is not None
    assert retrieved.rollout_id == rollout.rollout_id


@pytest.mark.asyncio
async def test_query_rollouts_by_rollout_ids(store_fixture: LightningStore) -> None:
    """Test querying rollouts filtered by rollout IDs."""
    # Create multiple rollouts
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    # Query by specific IDs
    selected = await store_fixture.query_rollouts(rollout_ids=[r1.rollout_id, r3.rollout_id])
    assert len(selected) == 2
    selected_ids = {r.rollout_id for r in selected}
    assert selected_ids == {r1.rollout_id, r3.rollout_id}

    # Query by single ID
    single = await store_fixture.query_rollouts(rollout_ids=[r2.rollout_id])
    assert len(single) == 1
    assert single[0].rollout_id == r2.rollout_id

    # Query by non-existent ID
    none = await store_fixture.query_rollouts(rollout_ids=["nonexistent"])
    assert len(none) == 0

    # Combine with status filter
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    await store_fixture.update_rollout(rollout_id=r2.rollout_id, status="failed")

    filtered = await store_fixture.query_rollouts(
        rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], status=["succeeded", "queuing"]
    )
    assert len(filtered) == 2
    filtered_ids = {r.rollout_id for r in filtered}
    assert filtered_ids == {r1.rollout_id, r3.rollout_id}  # r1 succeeded, r3 still queuing


@pytest.mark.asyncio
async def test_update_rollout_fields(store_fixture: LightningStore) -> None:
    """Test updating various rollout fields."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # Update multiple fields at once including config
    config = RolloutConfig(
        timeout_seconds=60.0, unresponsive_seconds=30.0, max_attempts=3, retry_condition=["timeout", "unresponsive"]
    )
    await store_fixture.update_rollout(
        rollout_id=rollout.rollout_id,
        status="running",
        mode="train",
        resources_id="new-resources",
        config=config,
        metadata={"custom_field": "custom_value"},
    )

    # Verify all updates
    updated_rollouts = await store_fixture.query_rollouts()
    updated = updated_rollouts[0]
    assert updated.status == "running"
    assert updated.mode == "train"
    assert updated.resources_id == "new-resources"
    assert updated.config.timeout_seconds == 60.0
    assert updated.config.unresponsive_seconds == 30.0
    assert updated.config.max_attempts == 3
    assert updated.config.retry_condition == ["timeout", "unresponsive"]
    assert updated.metadata is not None
    assert updated.metadata["custom_field"] == "custom_value"


@pytest.mark.asyncio
async def test_rollout_config_functionality(store_fixture: LightningStore) -> None:
    """Test RolloutConfig controls retry and timeout behavior."""
    # Create rollout with specific retry configuration
    config = RolloutConfig(
        timeout_seconds=30.0,
        unresponsive_seconds=15.0,
        max_attempts=2,
        retry_condition=["timeout", "unresponsive", "failed"],
    )

    rollout = await store_fixture.enqueue_rollout(input={"test": "retry"})
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Verify config is stored
    stored = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert stored is not None
    assert stored.config.timeout_seconds == 30.0
    assert stored.config.max_attempts == 2
    assert "failed" in stored.config.retry_condition

    # Test that different rollouts can have different configs
    config2 = RolloutConfig(timeout_seconds=120.0, max_attempts=5, retry_condition=["timeout"])

    rollout2 = await store_fixture.enqueue_rollout(input={"test": "different_config"})
    await store_fixture.update_rollout(rollout_id=rollout2.rollout_id, config=config2)

    stored2 = await store_fixture.get_rollout_by_id(rollout2.rollout_id)
    assert stored2 is not None
    assert stored2.config.timeout_seconds == 120.0
    assert stored2.config.max_attempts == 5
    assert stored2.config.retry_condition == ["timeout"]

    # Verify first rollout config unchanged
    stored1_again = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert stored1_again is not None
    assert stored1_again.config.timeout_seconds == 30.0


# Queue Operations Tests


@pytest.mark.asyncio
async def test_dequeue_rollout_skips_non_queuing_status(store_fixture: LightningStore) -> None:
    """Test that dequeue_rollout skips rollouts that have been updated to non-queuing status."""
    # Add multiple rollouts to the queue
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    # Update r1 to succeeded status while it's still in the queue
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Update r2 to failed status
    await store_fixture.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # r3 should still be in queuing status

    # Pop should skip r1 and r2 (both non-queuing) and return r3
    popped = await store_fixture.dequeue_rollout()
    assert popped is not None
    assert popped.rollout_id == r3.rollout_id
    assert popped.status == "preparing"
    assert popped.input["id"] == 3

    # Second pop should return None since no queuing rollouts remain
    popped2 = await store_fixture.dequeue_rollout()
    assert popped2 is None

    # Verify r1 and r2 are still in their non-queuing states
    all_rollouts = await store_fixture.query_rollouts()
    rollout_statuses = {r.rollout_id: r.status for r in all_rollouts}
    assert rollout_statuses[r1.rollout_id] == "succeeded"
    assert rollout_statuses[r2.rollout_id] == "failed"
    assert rollout_statuses[r3.rollout_id] == "preparing"


@pytest.mark.asyncio
async def test_fifo_ordering(store_fixture: LightningStore) -> None:
    """Test that queue maintains FIFO order."""
    rollouts: List[Rollout] = []
    for i in range(5):
        r = await store_fixture.enqueue_rollout(input={"order": i})
        rollouts.append(r)

    # Pop all and verify order
    for i in range(5):
        popped = await store_fixture.dequeue_rollout()
        assert popped is not None
        assert popped.rollout_id == rollouts[i].rollout_id
        assert popped.input["order"] == i
        assert popped.status == "preparing"


@pytest.mark.asyncio
async def test_pop_empty_queue(store_fixture: LightningStore) -> None:
    """Test popping from empty queue returns None."""
    result = await store_fixture.dequeue_rollout()
    assert result is None

    # Multiple pops should all return None
    for _ in range(3):
        assert await store_fixture.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeue_mechanism(store_fixture: LightningStore) -> None:
    """Test requeuing puts rollout back in queue."""
    rollout = await store_fixture.enqueue_rollout(input={"data": "test"})
    original_id = rollout.rollout_id

    # Pop and verify it's not in queue
    popped = await store_fixture.dequeue_rollout()
    assert popped is not None
    assert await store_fixture.dequeue_rollout() is None

    # Requeue it
    await store_fixture.update_rollout(rollout_id=original_id, status="requeuing")

    # Should be back in queue
    requeued = await store_fixture.dequeue_rollout()
    assert requeued is not None
    assert requeued.rollout_id == original_id
    assert requeued.status == "preparing"  # Changes when popped
    # Check that a new attempt was created
    attempts = await store_fixture.query_attempts(requeued.rollout_id)
    assert len(attempts) == 2  # First attempt plus requeued attempt

    latest_attempt = await store_fixture.get_latest_attempt(requeued.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "preparing"
    assert latest_attempt.sequence_id == 2


@pytest.mark.asyncio
async def test_update_and_query_workers(store_fixture: LightningStore) -> None:
    """Workers can be created, heartbeats recorded, and telemetry auto-updated."""
    first = await store_fixture.update_worker("worker-1", heartbeat_stats={"cpu": 0.5})
    assert first.worker_id == "worker-1"
    assert first.heartbeat_stats == {"cpu": 0.5}
    assert isinstance(first.last_heartbeat_time, float)
    assert first.status == "unknown"

    rollout = await store_fixture.enqueue_rollout(input={"task": "work"})
    claimed = await store_fixture.dequeue_rollout(worker_id="worker-1")
    assert claimed is not None
    assert claimed.rollout_id == rollout.rollout_id

    await store_fixture.update_attempt(claimed.rollout_id, claimed.attempt.attempt_id, worker_id="worker-1")
    busy = await store_fixture.get_worker_by_id("worker-1")
    assert busy is not None
    assert busy.status == "busy"
    assert busy.current_rollout_id == claimed.rollout_id
    assert busy.current_attempt_id == claimed.attempt.attempt_id
    assert isinstance(busy.last_dequeue_time, float)
    assert isinstance(busy.last_busy_time, float)

    heartbeat = await store_fixture.update_worker("worker-1")
    assert heartbeat.last_heartbeat_time is not None
    assert heartbeat.last_heartbeat_time >= first.last_heartbeat_time

    await store_fixture.update_attempt(claimed.rollout_id, claimed.attempt.attempt_id, status="succeeded")
    idle = await store_fixture.get_worker_by_id("worker-1")
    assert idle is not None
    assert idle.status == "idle"
    assert idle.current_rollout_id is None
    assert idle.current_attempt_id is None
    assert isinstance(idle.last_idle_time, float)

    workers = await store_fixture.query_workers()
    assert any(w.worker_id == "worker-1" for w in workers)
    assert await store_fixture.get_worker_by_id("missing") is None

    with pytest.raises(TypeError):
        await store_fixture.update_worker("worker-1", heartbeat_stats=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_workers_supports_filters(store_fixture: LightningStore) -> None:
    """Worker queries should support filtering, sorting, and pagination."""
    await store_fixture.update_worker("alpha-worker", heartbeat_stats={"cpu": 0.2})
    await store_fixture.update_worker("beta-worker", heartbeat_stats={"cpu": 0.8})

    busy_rollout = await store_fixture.start_rollout(input={"worker": "alpha"})
    await store_fixture.update_attempt(
        busy_rollout.rollout_id,
        busy_rollout.attempt.attempt_id,
        worker_id="alpha-worker",
        status="running",
    )

    idle_rollout = await store_fixture.start_rollout(input={"worker": "beta"})
    await store_fixture.update_attempt(
        idle_rollout.rollout_id,
        idle_rollout.attempt.attempt_id,
        worker_id="beta-worker",
        status="succeeded",
    )

    busy_workers = await store_fixture.query_workers(status_in=["busy"])
    assert [worker.worker_id for worker in busy_workers] == ["alpha-worker"]

    contains_beta = await store_fixture.query_workers(worker_id_contains="beta")
    assert [worker.worker_id for worker in contains_beta] == ["beta-worker"]

    sorted_workers = await store_fixture.query_workers(sort_by="worker_id", sort_order="desc")
    assert [worker.worker_id for worker in sorted_workers] == ["beta-worker", "alpha-worker"]

    paged = await store_fixture.query_workers(sort_by="worker_id", sort_order="desc", limit=1, offset=1)
    assert [worker.worker_id for worker in paged] == ["alpha-worker"]


@pytest.mark.asyncio
async def test_query_workers_filter_logic_or_combines_conditions(store_fixture: LightningStore) -> None:
    """filter_logic should dictate whether worker filters act as AND or OR."""
    await store_fixture.update_worker("cpu-worker", heartbeat_stats={"cpu": 0.1})

    busy_rollout = await store_fixture.start_rollout(input={"task": "busy"})
    await store_fixture.update_attempt(
        busy_rollout.rollout_id,
        busy_rollout.attempt.attempt_id,
        worker_id="busy-worker",
        status="running",
    )

    and_results = await store_fixture.query_workers(status_in=["busy"], worker_id_contains="cpu")
    assert len(and_results) == 0

    or_results = await store_fixture.query_workers(
        status_in=["busy"],
        worker_id_contains="cpu",
        filter_logic="or",
        sort_by="worker_id",
    )
    assert [worker.worker_id for worker in or_results] == ["busy-worker", "cpu-worker"]


@pytest.mark.asyncio
async def test_query_workers_filter_logic_and_with_sort(store_fixture: LightningStore) -> None:
    """Explicit AND logic should combine substring filters with status filters and respect sorting."""
    await store_fixture.update_worker("alpha-worker", heartbeat_stats={"cpu": 0.1})
    await store_fixture.update_worker("beta-worker", heartbeat_stats={"cpu": 0.1})

    busy_rollout = await store_fixture.start_rollout(input={"task": "alpha"})
    await store_fixture.update_attempt(
        busy_rollout.rollout_id,
        busy_rollout.attempt.attempt_id,
        worker_id="alpha-worker",
        status="running",
    )
    idle_rollout = await store_fixture.start_rollout(input={"task": "beta"})
    await store_fixture.update_attempt(
        idle_rollout.rollout_id,
        idle_rollout.attempt.attempt_id,
        worker_id="beta-worker",
        status="succeeded",
    )

    filtered = await store_fixture.query_workers(
        status_in=["busy"],
        worker_id_contains="alpha",
        filter_logic="and",
        sort_by="worker_id",
        sort_order="asc",
        limit=1,
    )
    assert [worker.worker_id for worker in filtered] == ["alpha-worker"]


@pytest.mark.asyncio
async def test_query_workers_reports_pagination_metadata(store_fixture: LightningStore) -> None:
    """Worker pagination should expose metadata for callers."""
    for worker_id in ["worker-a", "worker-b", "worker-c"]:
        await store_fixture.update_worker(worker_id, heartbeat_stats={"cpu": 0.1})

    paginated = await store_fixture.query_workers(sort_by="worker_id", sort_order="asc", limit=1, offset=1)
    assert isinstance(paginated, PaginatedResult)
    assert paginated.limit == 1
    assert paginated.offset == 1
    assert paginated.total == 3


# Resource Management Tests


@pytest.mark.asyncio
async def test_add_resources_generates_id_and_stores(store_fixture: LightningStore) -> None:
    """Test that add_resources generates a resources_id and stores the resources."""
    # Initially no resources
    assert await store_fixture.get_latest_resources() is None

    # Add resources using add_resources (auto-generates ID)
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")

    resources_update = await store_fixture.add_resources({"main_llm": llm, "greeting": prompt})

    # Verify resources_id was auto-generated with correct prefix
    assert resources_update.resources_id.startswith("rs-")
    assert len(resources_update.resources_id) == 15  # "rs-" + 12 char hash

    # Verify resources were stored correctly
    assert isinstance(resources_update.resources["main_llm"], LLM)
    assert resources_update.resources["main_llm"].model == "test-model"
    assert isinstance(resources_update.resources["greeting"], PromptTemplate)
    assert resources_update.resources["greeting"].template == "Hello {name}!"

    # Verify it's set as latest
    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == resources_update.resources_id
    assert latest.resources["main_llm"].model == "test-model"  # type: ignore

    # Verify we can retrieve by ID
    retrieved = await store_fixture.get_resources_by_id(resources_update.resources_id)
    assert retrieved is not None
    assert retrieved.resources_id == resources_update.resources_id


@pytest.mark.asyncio
async def test_add_resources_multiple_times_generates_unique_ids(store_fixture: LightningStore) -> None:
    """Test that multiple calls to add_resources generate unique IDs."""
    llm1 = LLM(resource_type="llm", endpoint="http://localhost:8080", model="model-v1")
    llm2 = LLM(resource_type="llm", endpoint="http://localhost:8080", model="model-v2")

    update1 = await store_fixture.add_resources({"llm": llm1})
    update2 = await store_fixture.add_resources({"llm": llm2})

    # IDs should be different
    assert update1.resources_id != update2.resources_id
    assert update1.resources_id.startswith("rs-")
    assert update2.resources_id.startswith("rs-")

    # Both should be retrievable
    retrieved1 = await store_fixture.get_resources_by_id(update1.resources_id)
    retrieved2 = await store_fixture.get_resources_by_id(update2.resources_id)
    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.resources["llm"].model == "model-v1"  # type: ignore
    assert retrieved2.resources["llm"].model == "model-v2"  # type: ignore

    # Latest should be the second one
    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == update2.resources_id


@pytest.mark.asyncio
async def test_query_resources_returns_history(store_fixture: LightningStore) -> None:
    """query_resources should list snapshots in the order they were stored."""
    assert len(await store_fixture.query_resources()) == 0

    first = await store_fixture.add_resources(
        {
            "llm": LLM(resource_type="llm", endpoint="http://localhost:8080", model="model-v1"),
        }
    )
    second = await store_fixture.update_resources(
        "custom-snapshot",
        {
            "prompt": PromptTemplate(resource_type="prompt_template", template="Hi {name}", engine="f-string"),
        },
    )

    history = await store_fixture.query_resources()
    assert set([item.resources_id for item in history]) == {first.resources_id, second.resources_id}
    assert isinstance(history[0], ResourcesUpdate)
    assert isinstance(history[1], ResourcesUpdate)


@pytest.mark.asyncio
async def test_query_resources_supports_filters(store_fixture: LightningStore) -> None:
    """Resource queries should support substring filters and pagination."""
    alpha = PromptTemplate(resource_type="prompt_template", template="alpha", engine="jinja")
    beta = PromptTemplate(resource_type="prompt_template", template="beta", engine="jinja")

    await store_fixture.update_resources("manual-alpha", {"prompt": alpha})
    await store_fixture.update_resources("manual-beta", {"prompt": beta})

    contains_beta = await store_fixture.query_resources(resources_id_contains="beta")
    assert [item.resources_id for item in contains_beta] == ["manual-beta"]

    sorted_ids = sorted(["manual-alpha", "manual-beta"], reverse=True)
    paged = await store_fixture.query_resources(sort_by="resources_id", sort_order="desc", limit=1)
    assert [item.resources_id for item in paged] == sorted_ids[:1]

    offset_item = await store_fixture.query_resources(sort_by="resources_id", sort_order="desc", limit=1, offset=1)
    assert [item.resources_id for item in offset_item] == sorted_ids[1:2]


@pytest.mark.asyncio
async def test_query_resources_combines_exact_and_contains_filters(store_fixture: LightningStore) -> None:
    """Exact and substring filters should be usable together."""
    alpha = PromptTemplate(resource_type="prompt_template", template="alpha", engine="jinja")
    beta = PromptTemplate(resource_type="prompt_template", template="beta", engine="jinja")

    await store_fixture.update_resources("manual-alpha", {"prompt": alpha})
    await store_fixture.update_resources("manual-beta", {"prompt": beta})

    results = await store_fixture.query_resources(resources_id="manual-alpha", resources_id_contains="manual")
    assert [item.resources_id for item in results] == ["manual-alpha"]


@pytest.mark.asyncio
async def test_query_resources_offset_beyond_range_returns_empty(store_fixture: LightningStore) -> None:
    """Large offsets should simply return an empty result."""
    await store_fixture.update_resources(
        "snapshot-a",
        {"prompt": PromptTemplate(resource_type="prompt_template", template="a", engine="jinja")},
    )
    await store_fixture.update_resources(
        "snapshot-b",
        {"prompt": PromptTemplate(resource_type="prompt_template", template="b", engine="jinja")},
    )

    results = await store_fixture.query_resources(sort_by="resources_id", limit=1, offset=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_query_resources_contains_with_sort_and_pagination(store_fixture: LightningStore) -> None:
    """Substring filters should combine with sort order, limit, and offset."""
    for suffix in ["alpha", "beta", "gamma"]:
        await store_fixture.update_resources(
            f"manual-{suffix}",
            {"prompt": PromptTemplate(resource_type="prompt_template", template=suffix, engine="jinja")},
        )

    filtered = await store_fixture.query_resources(
        resources_id_contains="manual-",
        sort_by="resources_id",
        sort_order="asc",
        limit=2,
        offset=1,
    )

    expected_ids = sorted([f"manual-{suffix}" for suffix in ["alpha", "beta", "gamma"]])
    assert [item.resources_id for item in filtered] == expected_ids[1:3]


@pytest.mark.asyncio
async def test_query_resources_reports_pagination_metadata(store_fixture: LightningStore) -> None:
    """Resource pagination should expose metadata fields."""
    await store_fixture.update_resources(
        "snapshot-a",
        {"prompt": PromptTemplate(resource_type="prompt_template", template="a", engine="jinja")},
    )
    await store_fixture.update_resources(
        "snapshot-b",
        {"prompt": PromptTemplate(resource_type="prompt_template", template="b", engine="jinja")},
    )

    paginated = await store_fixture.query_resources(sort_by="resources_id", limit=1, offset=1)
    assert isinstance(paginated, PaginatedResult)
    assert paginated.limit == 1
    assert paginated.offset == 1
    assert paginated.total == 2


@pytest.mark.asyncio
async def test_resource_lifecycle(store_fixture: LightningStore) -> None:
    """Test adding, updating, and retrieving resources."""
    # Initially no resources
    assert await store_fixture.get_latest_resources() is None
    assert await store_fixture.get_resources_by_id("any-id") is None

    # Add first version with proper LLM resource
    llm_v1 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model-v1",
        sampling_parameters={"temperature": 0.7},
    )
    update = await store_fixture.update_resources("v1", {"main_llm": llm_v1})
    assert update.resources_id == "v1"

    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v1"
    assert isinstance(latest.resources["main_llm"], LLM)
    assert latest.resources["main_llm"].model == "test-model-v1"

    # Add second version with different LLM
    llm_v2 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v2",
        model="test-model-v2",
        sampling_parameters={"temperature": 0.8},
    )
    v2 = await store_fixture.update_resources("v2", {"main_llm": llm_v2})
    assert v2.resources_id == "v2"
    assert isinstance(v2.resources["main_llm"], LLM)
    assert v2.resources["main_llm"].model == "test-model-v2"

    # Latest should be v2
    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v2"

    # Can still retrieve v1
    old = await store_fixture.get_resources_by_id("v1")
    assert old is not None
    assert isinstance(old.resources["main_llm"], LLM)
    assert old.resources["main_llm"].model == "test-model-v1"


@pytest.mark.asyncio
async def test_latest_resources_rehydrates_cache(store_fixture: LightningStore) -> None:
    """get_latest_resources should consult storage even if the cache is unset."""
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="cache-model",
        sampling_parameters={"temperature": 0.1},
    )
    update = await store_fixture.update_resources("cache-test", {"main_llm": llm})

    # Simulate a fresh process by clearing the cache.
    store_fixture._latest_resources_id = UNSET  # type: ignore[attr-defined]

    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == update.resources_id
    assert latest.resources["main_llm"].model == "cache-model"  # type: ignore


@pytest.mark.asyncio
async def test_task_inherits_latest_resources(store_fixture: LightningStore) -> None:
    """Test that new tasks inherit latest resources_id if not specified."""
    # Set up resources with proper PromptTemplate
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")
    update = ResourcesUpdate(
        resources_id="current",
        resources={"greeting": prompt},
        create_time=time.time(),
        update_time=time.time(),
        version=1,
    )
    await store_fixture.update_resources(update.resources_id, update.resources)

    # Task without explicit resources_id
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    assert r1.resources_id == "current"

    # Task with explicit resources_id
    r2 = await store_fixture.enqueue_rollout(input={"id": 2}, resources_id="override")
    assert r2.resources_id == "override"

    # Update resources
    new_prompt = PromptTemplate(resource_type="prompt_template", template="Hi {name}!", engine="f-string")
    update2 = ResourcesUpdate(
        resources_id="new",
        resources={"greeting": new_prompt},
        create_time=time.time(),
        update_time=time.time(),
        version=1,
    )
    await store_fixture.update_resources(update2.resources_id, update2.resources)

    # New task gets new resources
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})
    assert r3.resources_id == "new"


# Span Management Tests


@pytest.mark.asyncio
async def test_span_sequence_generation(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test automatic sequence ID generation for spans."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # First span gets sequence_id 1
    seq_id = await store_fixture.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 1

    span1 = await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span1.sequence_id == 2

    # Next span gets sequence_id 3
    seq_id = await store_fixture.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 3

    span2 = await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span2.sequence_id == 4

    # Different attempt reuses the same rollout_id
    seq_id = await store_fixture.get_next_span_sequence_id(rollout.rollout_id, "attempt-does-not-exist")
    assert seq_id == 5


@pytest.mark.asyncio
async def test_span_updates_attempt_status(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Adding a span should persist attempt heartbeat and transition status to running."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "attempt-status"})
    await store_fixture.dequeue_rollout()

    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert attempts
    attempt_id = attempts[0].attempt_id
    assert attempts[0].status == "preparing"
    assert attempts[0].last_heartbeat_time is None

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    updated_attempt = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert updated_attempt.status == "running"
    assert updated_attempt.last_heartbeat_time is not None


@pytest.mark.asyncio
async def test_unresponsive_attempt_recovers_after_span(
    store_fixture: LightningStore, mock_readable_span: Mock
) -> None:
    """Spans arriving for an unresponsive attempt should mark it running again."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "unresponsive"})
    dequeued = await store_fixture.dequeue_rollout()
    assert dequeued is not None
    attempt_id = dequeued.attempt.attempt_id

    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt_id,
        status="unresponsive",
    )

    attempt_before = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert attempt_before.status == "unresponsive"

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    attempt_after = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert attempt_after.status == "running"
    assert attempt_after.last_heartbeat_time is not None


@pytest.mark.asyncio
async def test_running_attempt_updates_heartbeat(
    store_fixture: LightningStore, mock_readable_span: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding spans to an already running attempt should advance its heartbeat."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "running-heartbeat"})
    await store_fixture.dequeue_rollout()
    attempt_id = (await store_fixture.query_attempts(rollout.rollout_id))[0].attempt_id

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    attempt_after_first = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert attempt_after_first.status == "running"
    first_heartbeat = attempt_after_first.last_heartbeat_time
    assert first_heartbeat is not None

    monkeypatch.setattr("agentlightning.store.collection_based.time.time", lambda: first_heartbeat + 100.0)

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    attempt_after_second = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert attempt_after_second.last_heartbeat_time == first_heartbeat + 100.0


@pytest.mark.asyncio
async def test_duplicate_span_id_error(
    store_fixture: LightningStore, mock_readable_span: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    """Adding two spans with the same span_id should raise a ValueError."""
    caplog.set_level(logging.ERROR)
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Force the mock to reuse the same span context for every call.
    fixed_context = mock_readable_span.get_span_context()
    mock_readable_span.get_span_context = Mock(return_value=fixed_context)

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert "Duplicated span added" in caplog.text


@pytest.mark.asyncio
async def test_span_with_explicit_sequence_id(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test providing explicit sequence_id to spans."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add span with explicit sequence_id
    span = await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span, sequence_id=100)
    assert span.sequence_id == 100

    next_seq = await store_fixture.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert next_seq == 101


@pytest.mark.asyncio
async def test_query_spans_by_attempt(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test querying spans filtered by attempt_id."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    # Pop to create first attempt
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt1_id = attempts[0].attempt_id

    # Add spans for first attempt
    for _ in range(2):
        await store_fixture.add_otel_span(rollout.rollout_id, attempt1_id, mock_readable_span)

    # Simulate requeue and create second attempt
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt2_id = attempts[1].attempt_id

    # Add spans for second attempt
    for _ in range(3):
        await store_fixture.add_otel_span(rollout.rollout_id, attempt2_id, mock_readable_span)

    # Query all spans
    all_spans = await store_fixture.query_spans(rollout.rollout_id)
    assert len(all_spans) == 5

    # Query specific attempt
    attempt1_spans = await store_fixture.query_spans(rollout.rollout_id, attempt_id=attempt1_id)
    assert len(attempt1_spans) == 2
    assert all(s.attempt_id == attempt1_id for s in attempt1_spans)

    # Query latest attempt
    latest_spans = await store_fixture.query_spans(rollout.rollout_id, attempt_id="latest")
    assert len(latest_spans) == 3
    assert all(s.attempt_id == attempt2_id for s in latest_spans)

    # Query non-existent attempt
    no_spans = await store_fixture.query_spans(rollout.rollout_id, attempt_id="nonexistent")
    assert len(no_spans) == 0


@pytest.mark.asyncio
async def test_query_spans_supports_filters(store_fixture: LightningStore) -> None:
    """Span queries should honor filtering logic and pagination."""
    attempted = await store_fixture.start_rollout(input={"payload": "span-filters"})
    attempt_id = attempted.attempt.attempt_id

    def build_span(idx: int, *, name: str, parent: Optional[str]) -> Span:
        trace_hex = f"{idx:032x}"
        span_hex = f"{idx:016x}"
        return Span(
            rollout_id=attempted.rollout_id,
            attempt_id=attempt_id,
            sequence_id=idx,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=parent,
            name=name,
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[Event(name=f"event-{idx}", attributes={})],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    created_spans = [
        build_span(1, name="reward", parent=None),
        build_span(2, name="planner", parent=f"{1:016x}"),
        build_span(3, name="tool-call", parent=f"{2:016x}"),
    ]
    for span in created_spans:
        await store_fixture.add_span(span)

    trace_filtered = await store_fixture.query_spans(attempted.rollout_id, trace_id=created_spans[1].trace_id)
    assert [s.span_id for s in trace_filtered] == [created_spans[1].span_id]

    or_filtered = await store_fixture.query_spans(
        attempted.rollout_id,
        span_id=created_spans[0].span_id,
        trace_id_contains=created_spans[2].trace_id[-4:],
        filter_logic="or",
    )
    assert {s.span_id for s in or_filtered} == {created_spans[0].span_id, created_spans[2].span_id}

    parent_filtered = await store_fixture.query_spans(
        attempted.rollout_id,
        parent_id_contains=created_spans[1].span_id[-4:],
    )
    assert [s.span_id for s in parent_filtered] == [created_spans[2].span_id]

    sorted_ids = sorted([span.span_id for span in created_spans], reverse=True)
    paged = await store_fixture.query_spans(
        attempted.rollout_id,
        sort_by="span_id",
        sort_order="desc",
        limit=1,
        offset=1,
    )
    assert [span.span_id for span in paged] == sorted_ids[1:2]


@pytest.mark.asyncio
async def test_query_spans_filter_logic_respects_rollout_scope(store_fixture: LightningStore) -> None:
    """Even with OR logic, query_spans should not leak spans from other rollouts."""
    first = await store_fixture.start_rollout(input={"payload": "first-span"})
    second = await store_fixture.start_rollout(input={"payload": "second-span"})

    def make_span(rollout_id: str, attempt_id: str, idx: int, name: str) -> Span:
        span_hex = f"{idx:016x}"
        trace_hex = f"{idx:032x}"
        return Span(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=idx,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=None,
            name=name,
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    span_first = make_span(first.rollout_id, first.attempt.attempt_id, 1, "alpha")
    span_second = make_span(second.rollout_id, second.attempt.attempt_id, 2, "beta")
    await store_fixture.add_span(span_first)
    await store_fixture.add_span(span_second)

    results = await store_fixture.query_spans(
        first.rollout_id,
        attempt_id=first.attempt.attempt_id,
        name_contains="alpha",
        trace_id=span_second.trace_id,
        filter_logic="or",
    )
    assert [span.span_id for span in results] == [span_first.span_id]


@pytest.mark.asyncio
async def test_query_spans_supports_name_contains_filter(store_fixture: LightningStore) -> None:
    """name_contains filtering should happen before pagination."""
    attempted = await store_fixture.start_rollout(input={"payload": "span-names"})
    attempt_id = attempted.attempt.attempt_id

    def build_span(idx: int, name: str) -> Span:
        span_hex = f"{idx:016x}"
        trace_hex = f"{idx:032x}"
        return Span(
            rollout_id=attempted.rollout_id,
            attempt_id=attempt_id,
            sequence_id=idx,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=None,
            name=name,
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    matching = build_span(1, "planner-step")
    non_matching = build_span(2, "tool-call")
    await store_fixture.add_span(matching)
    await store_fixture.add_span(non_matching)

    results = await store_fixture.query_spans(
        attempted.rollout_id,
        name_contains="plan",
        sort_by="sequence_id",
        limit=1,
    )
    assert [span.span_id for span in results] == [matching.span_id]


@pytest.mark.asyncio
async def test_query_spans_multiple_filters_require_all(store_fixture: LightningStore) -> None:
    """Using multiple exact/substring filters together should narrow down to a single span."""
    attempted = await store_fixture.start_rollout(input={"payload": "span-multi"})
    attempt_id = attempted.attempt.attempt_id

    def build_span(seq: int, parent: Optional[str], name: str) -> Span:
        span_hex = f"{seq:016x}"
        trace_hex = f"{(seq * 10):032x}"
        return Span(
            rollout_id=attempted.rollout_id,
            attempt_id=attempt_id,
            sequence_id=seq,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=parent,
            name=name,
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    spans = [
        build_span(1, None, "phase-plan"),
        build_span(2, f"{1:016x}", "phase-run"),
        build_span(3, f"{2:016x}", "tool-call"),
    ]
    for span in spans:
        await store_fixture.add_span(span)

    target = spans[2]
    filtered = await store_fixture.query_spans(
        attempted.rollout_id,
        trace_id_contains=target.trace_id[-4:],
        span_id_contains=target.span_id[-4:],
        parent_id=target.parent_id,
        name_contains="tool",
    )
    assert [span.span_id for span in filtered] == [target.span_id]


@pytest.mark.asyncio
async def test_query_spans_reports_pagination_metadata(store_fixture: LightningStore) -> None:
    """Span pagination should return limit/offset/total values."""
    attempted = await store_fixture.start_rollout(input={"payload": "span-pagination"})
    attempt_id = attempted.attempt.attempt_id

    def build_span(idx: int) -> Span:
        span_hex = f"{idx:016x}"
        trace_hex = f"{idx:032x}"
        return Span(
            rollout_id=attempted.rollout_id,
            attempt_id=attempt_id,
            sequence_id=idx,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=None,
            name=f"span-{idx}",
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    for idx in range(1, 4):
        await store_fixture.add_span(build_span(idx))

    paginated = await store_fixture.query_spans(
        attempted.rollout_id,
        attempt_id=attempt_id,
        sort_by="sequence_id",
        sort_order="asc",
        limit=1,
        offset=1,
    )
    assert isinstance(paginated, PaginatedResult)
    assert paginated.limit == 1
    assert paginated.offset == 1
    assert paginated.total == 3


@pytest.mark.asyncio
async def test_query_spans_or_filters_with_sort_and_offset(store_fixture: LightningStore) -> None:
    """OR logic combined with sort + pagination should return deterministic slices."""
    attempted = await store_fixture.start_rollout(input={"payload": "span-or"})
    attempt_id = attempted.attempt.attempt_id

    def build_span(seq: int, name: str, parent: Optional[str]) -> Span:
        span_hex = f"{seq:016x}"
        trace_hex = f"{(seq * 3):032x}"
        return Span(
            rollout_id=attempted.rollout_id,
            attempt_id=attempt_id,
            sequence_id=seq,
            trace_id=trace_hex,
            span_id=span_hex,
            parent_id=parent,
            name=name,
            status=TraceStatus(status_code="OK"),
            attributes={},
            events=[],
            links=[],
            start_time=None,
            end_time=None,
            context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        )

    span_plan = build_span(10, "phase-plan", None)
    span_run = build_span(20, "phase-run", span_plan.span_id)
    span_tool = build_span(30, "tool-call", span_run.span_id)
    for span in [span_plan, span_run, span_tool]:
        await store_fixture.add_span(span)

    filtered = await store_fixture.query_spans(
        attempted.rollout_id,
        name_contains="phase",
        parent_id_contains=span_run.span_id[-4:],
        span_id=span_tool.span_id,
        filter_logic="or",
        sort_by="sequence_id",
        sort_order="desc",
        limit=1,
        offset=1,
    )

    assert [span.span_id for span in filtered] == [span_run.span_id]


@pytest.mark.asyncio
async def test_span_eviction_removes_oldest_rollouts(mock_readable_span: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentlightning.store.memory._detect_total_memory_bytes", lambda: 100)
    store = InMemoryLightningStore(
        eviction_memory_threshold=0.5,
        safe_memory_threshold=0.05,
        span_size_estimator=lambda span: 20,
    )

    attempted_rollouts: List[AttemptedRollout] = []
    for index in range(4):
        attempted = await store.start_rollout(input={"index": index})
        attempted_rollouts.append(attempted)
        await store.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    for attempted in attempted_rollouts[:3]:
        with pytest.raises(RuntimeError):
            await store.query_spans(attempted.rollout_id)

    remaining_spans = await store.query_spans(attempted_rollouts[3].rollout_id)
    assert len(remaining_spans) == 1
    assert remaining_spans[0].rollout_id == attempted_rollouts[3].rollout_id


def test_memory_threshold_accepts_byte_values() -> None:
    store = InMemoryLightningStore(
        eviction_memory_threshold=150,
        safe_memory_threshold=20,
    )

    assert store._eviction_threshold_bytes == 150  # pyright: ignore[reportPrivateUsage]
    assert store._safe_threshold_bytes == 20  # pyright: ignore[reportPrivateUsage]


def test_memory_threshold_accepts_ratios_with_zero_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentlightning.store.memory._detect_total_memory_bytes", lambda: 200)
    store = InMemoryLightningStore(
        eviction_memory_threshold=0.6,
        safe_memory_threshold=0.0,
    )

    assert store._eviction_threshold_bytes == int(200 * 0.6)  # pyright: ignore[reportPrivateUsage]
    assert store._safe_threshold_bytes == 0  # pyright: ignore[reportPrivateUsage]


def test_invalid_safe_threshold_raises_value_error() -> None:
    with pytest.raises(ValueError):
        InMemoryLightningStore(
            eviction_memory_threshold=50,
            safe_memory_threshold=100,
        )


def test_estimate_model_size_counts_nested_models() -> None:
    class Inner(BaseModel):
        value: int
        data: List[int]

    class Outer(BaseModel):
        inner: Inner
        mapping: dict[str, str]
        tags: List[str]

    inner = Inner(value=7, data=[1, 2, 3])
    outer = Outer(inner=inner, mapping={"alpha": "beta"}, tags=["x", "yz"])

    inner_expected = (
        sys.getsizeof(inner)
        + sys.getsizeof(inner.value)
        + sys.getsizeof(inner.data)
        + sum(sys.getsizeof(item) for item in inner.data)
    )
    assert estimate_model_size(inner) == inner_expected

    mapping_expected = sys.getsizeof(outer.mapping) + sum(sys.getsizeof(v) for v in outer.mapping.values())
    tags_expected = sys.getsizeof(outer.tags) + sum(sys.getsizeof(tag) for tag in outer.tags)
    outer_expected = sys.getsizeof(outer) + inner_expected + mapping_expected + tags_expected
    assert estimate_model_size(outer) == outer_expected


def test_estimate_model_size_handles_span_objects() -> None:
    status = TraceStatus(status_code="OK", description="fine")
    context = SpanContext(trace_id="trace", span_id="parent", is_remote=False, trace_state={"foo": "bar"})
    event = Event(name="step", attributes={"detail": "value"}, timestamp=1.0)
    link = Link(context=context, attributes=None)
    resource = OtelResource(attributes={"service.name": "unit"}, schema_url="schema")

    span = Span(
        rollout_id="ro-1",
        attempt_id="at-1",
        sequence_id=1,
        trace_id="trace",
        span_id="span",
        parent_id=None,
        name="operation",
        status=status,
        attributes={"foo": "bar", "answer": 42},
        events=[event],
        links=[link],
        start_time=1.0,
        end_time=2.0,
        context=None,
        parent=None,
        resource=resource,
    )

    status_expected = sys.getsizeof(status) + sys.getsizeof(status.status_code) + sys.getsizeof(status.description)

    trace_state_values = context.trace_state.values()
    context_expected = (
        sys.getsizeof(context)
        + sys.getsizeof(context.trace_id)
        + sys.getsizeof(context.span_id)
        + sys.getsizeof(context.is_remote)
        + sys.getsizeof(context.trace_state)
        + sum(sys.getsizeof(v) for v in trace_state_values)
    )

    event_attributes_expected = sys.getsizeof(event.attributes) + sys.getsizeof("value")
    event_expected = (
        sys.getsizeof(event) + sys.getsizeof(event.name) + event_attributes_expected + sys.getsizeof(event.timestamp)
    )
    events_expected = sys.getsizeof(span.events) + event_expected

    link_attributes = cast(Optional[dict[str, str]], link.attributes)
    link_attribute_values = link_attributes.values() if link_attributes is not None else ()
    link_attributes_expected = sys.getsizeof(link_attributes if link_attributes is not None else None) + sum(
        sys.getsizeof(v) for v in link_attribute_values
    )
    link_expected = sys.getsizeof(link) + context_expected + link_attributes_expected
    links_expected = sys.getsizeof(span.links) + link_expected

    attributes_expected = (
        sys.getsizeof(span.attributes) + sys.getsizeof("bar") + sys.getsizeof(span.attributes["answer"])
    )

    resource_expected = (
        sys.getsizeof(resource)
        + sys.getsizeof(resource.attributes)
        + sum(sys.getsizeof(v) for v in resource.attributes.values())
        + sys.getsizeof(resource.schema_url)
    )

    expected_size = (
        sys.getsizeof(span)
        + sys.getsizeof(span.rollout_id)
        + sys.getsizeof(span.attempt_id)
        + sys.getsizeof(span.sequence_id)
        + sys.getsizeof(span.trace_id)
        + sys.getsizeof(span.span_id)
        + sys.getsizeof(span.parent_id)
        + sys.getsizeof(span.name)
        + status_expected
        + attributes_expected
        + events_expected
        + links_expected
        + sys.getsizeof(span.start_time)
        + sys.getsizeof(span.end_time)
        + sys.getsizeof(span.context)
        + sys.getsizeof(span.parent)
        + resource_expected
    )

    assert estimate_model_size(span) == expected_size


@pytest.mark.asyncio
async def test_span_triggers_status_transition(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test that adding first span transitions rollout from preparing to running."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # Pop to set status to preparing and create attempt
    popped = await store_fixture.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Verify status in store
    rollouts = await store_fixture.query_rollouts(status=["preparing"])
    assert len(rollouts) == 1

    # Get the attempt
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add first span
    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    # Status should transition to running
    rollouts = await store_fixture.query_rollouts(status=["running"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id


# Rollout Lifecycle Tests


@pytest.mark.asyncio
async def test_span_does_not_reset_timeout_attempt(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Adding a span to a timed-out attempt should not mark it running again."""

    rollout = await store_fixture.enqueue_rollout(input={"test": "timeout-span"})

    # Create the first attempt
    dequeued = await store_fixture.dequeue_rollout()
    assert dequeued is not None
    attempt_id = dequeued.attempt.attempt_id

    # Simulate the attempt timing out
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt_id,
        status="timeout",
    )

    attempts_before = await store_fixture.query_attempts(rollout.rollout_id)
    assert attempts_before[0].status == "timeout"

    rollout_before = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert rollout_before is not None
    assert rollout_before.status != "running"

    # Adding a new span should keep the attempt in timeout state
    await store_fixture.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    attempts_after = await store_fixture.query_attempts(rollout.rollout_id)
    assert attempts_after[0].status == "timeout"
    assert attempts_after[0].last_heartbeat_time is not None

    rollout_after = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == rollout_before.status


@pytest.mark.asyncio
async def test_completion_sets_end_time(store_fixture: LightningStore) -> None:
    """Test that completing a rollout sets end_time."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # Initially no end_time
    assert rollout.end_time is None

    # Complete as succeeded
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    completed_rollouts = await store_fixture.query_rollouts()
    completed = completed_rollouts[0]
    assert completed.status == "succeeded"
    assert completed.end_time is not None
    assert completed.end_time > completed.start_time


@pytest.mark.asyncio
async def test_wait_for_rollouts(store_fixture: LightningStore) -> None:
    """Test waiting for rollout completion."""
    # Add multiple rollouts
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    _r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    # Start waiting for r1 and r2
    async def wait_for_completion() -> List[Rollout]:
        return await store_fixture.wait_for_rollouts(rollout_ids=[r1.rollout_id, r2.rollout_id], timeout=5.0)

    wait_task = asyncio.create_task(wait_for_completion())
    await asyncio.sleep(0.01)  # Let wait task start

    # Complete r1
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Complete r2
    await store_fixture.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # Get results
    completed = await wait_task
    assert len(completed) == 2
    assert {r.rollout_id for r in completed} == {r1.rollout_id, r2.rollout_id}
    assert {r.status for r in completed} == {"succeeded", "failed"}


@pytest.mark.asyncio
async def test_wait_timeout(store_fixture: LightningStore) -> None:
    """Test wait_for_rollouts timeout behavior."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    start = time.time()
    completed = await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0.1)
    elapsed = time.time() - start

    assert elapsed < 0.2  # Should timeout quickly
    assert len(completed) == 0  # No completions


@pytest.mark.asyncio
async def test_wait_with_timeout_none_polling(store_fixture: LightningStore) -> None:
    """Test wait_for_rollouts with timeout=None uses polling and can be cancelled."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "indefinite"})

    async def wait_indefinitely():
        return await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None)

    # Start waiting with timeout=None
    wait_task = asyncio.create_task(wait_indefinitely())

    # Give it a moment to start polling
    await asyncio.sleep(0.1)

    # Complete the rollout
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # The wait should complete now
    timeout = 1.0 if isinstance(store_fixture, InMemoryLightningStore) else 11.0
    completed = await asyncio.wait_for(wait_task, timeout=timeout)
    assert len(completed) == 1
    assert completed[0].rollout_id == rollout.rollout_id
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_with_timeout_none_can_be_cancelled(store_fixture: LightningStore) -> None:
    """Test that wait_for_rollouts with timeout=None can be cancelled cleanly."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "cancel"})

    async def wait_indefinitely():
        return await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None)

    # Start waiting with timeout=None
    wait_task = asyncio.create_task(wait_indefinitely())

    # Give it time to start polling
    await asyncio.sleep(0.15)  # Wait for at least one poll cycle

    # Cancel the task
    wait_task.cancel()

    # Should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await wait_task

    # Task should be cancelled, no hanging threads
    assert wait_task.cancelled()


@pytest.mark.asyncio
async def test_wait_with_timeout_zero(store_fixture: LightningStore) -> None:
    """Test wait_for_rollouts with timeout=0 returns immediately."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "zero"})

    start = time.time()
    completed = await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0)
    elapsed = time.time() - start

    # Should return almost immediately
    assert elapsed < 0.05
    assert len(completed) == 0


@pytest.mark.asyncio
async def test_wait_with_already_completed_rollout(store_fixture: LightningStore) -> None:
    """Test wait_for_rollouts returns immediately for already completed rollouts."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "already_done"})

    # Complete it first
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Wait should return immediately without blocking
    start = time.time()
    completed = await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=5.0)
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be instant
    assert len(completed) == 1
    assert completed[0].rollout_id == rollout.rollout_id
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_multiple_rollouts_different_completion_times(store_fixture: LightningStore) -> None:
    """Test waiting for multiple rollouts that complete at different times."""
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    async def wait_for_all():
        return await store_fixture.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], timeout=2.0
        )

    wait_task = asyncio.create_task(wait_for_all())

    # Complete them at different times
    await asyncio.sleep(0.05)
    await store_fixture.update_rollout(rollout_id=r2.rollout_id, status="succeeded")

    await asyncio.sleep(0.05)
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="failed")

    await asyncio.sleep(0.05)
    await store_fixture.update_rollout(rollout_id=r3.rollout_id, status="succeeded")

    # All should be collected
    completed = await wait_task
    assert len(completed) == 3
    completed_ids = {r.rollout_id for r in completed}
    assert completed_ids == {r1.rollout_id, r2.rollout_id, r3.rollout_id}


@pytest.mark.asyncio
async def test_wait_partial_completion_on_timeout(store_fixture: LightningStore) -> None:
    """Test that wait_for_rollouts returns partial results when timeout occurs."""
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})
    r2 = await store_fixture.enqueue_rollout(input={"id": 2})
    r3 = await store_fixture.enqueue_rollout(input={"id": 3})

    async def wait_with_short_timeout():
        return await store_fixture.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], timeout=0.2
        )

    wait_task = asyncio.create_task(wait_with_short_timeout())

    # Only complete one before timeout
    await asyncio.sleep(0.05)
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Wait for timeout
    completed = await wait_task

    # Should only get r1
    assert len(completed) == 1
    assert completed[0].rollout_id == r1.rollout_id


@pytest.mark.asyncio
async def test_wait_concurrent_waiters_on_same_rollout(store_fixture: LightningStore) -> None:
    """Test multiple concurrent waiters on the same rollout."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "concurrent"})

    async def wait_for_completion():
        return await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=2.0)

    # Start multiple waiters concurrently
    wait_tasks = [asyncio.create_task(wait_for_completion()) for _ in range(5)]

    await asyncio.sleep(0.05)

    # Complete the rollout
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # All waiters should complete
    results = await asyncio.gather(*wait_tasks)

    # Each waiter should get the completed rollout
    for completed in results:
        assert len(completed) == 1
        assert completed[0].rollout_id == rollout.rollout_id
        assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_nonexistent_rollout_with_finite_timeout(store_fixture: LightningStore) -> None:
    """Test waiting for non-existent rollout with finite timeout."""
    start = time.time()
    completed = await store_fixture.wait_for_rollouts(rollout_ids=["nonexistent"], timeout=0.1)
    elapsed = time.time() - start

    if isinstance(store_fixture, InMemoryLightningStore):
        # Should timeout quickly (not wait indefinitely)
        assert elapsed < 0.2
    else:
        # Should be slower, but not too slow
        assert elapsed < 2.0
    assert len(completed) == 0


@pytest.mark.asyncio
async def test_wait_mixed_existing_and_nonexistent_rollouts(store_fixture: LightningStore) -> None:
    """Test waiting for mix of existing and non-existent rollouts."""
    r1 = await store_fixture.enqueue_rollout(input={"id": 1})

    async def wait_for_mixed():
        return await store_fixture.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, "nonexistent1", "nonexistent2"], timeout=0.5
        )

    wait_task = asyncio.create_task(wait_for_mixed())

    await asyncio.sleep(0.05)
    await store_fixture.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    completed = await wait_task

    # Should only get the existing, completed rollout
    assert len(completed) == 1
    assert completed[0].rollout_id == r1.rollout_id


@pytest.mark.asyncio
async def test_wait_event_set_before_wait_starts(store_fixture: LightningStore) -> None:
    """Test that waiting on an already-set event returns immediately."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "early_complete"})

    # Complete it before waiting
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Now start waiting - should return immediately
    start = time.time()
    completed = await store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=10.0)
    elapsed = time.time() - start

    assert elapsed < 0.05  # Should be instant
    assert len(completed) == 1
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_polling_interval_with_timeout_none(store_fixture: LightningStore) -> None:
    """Test that timeout=None polling doesn't busy-wait (uses reasonable intervals)."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "polling"})

    start = time.time()

    async def wait_and_complete():
        # Start waiting with timeout=None
        wait_task = asyncio.create_task(store_fixture.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None))

        # Wait for 0.5 seconds to let polling happen
        await asyncio.sleep(0.5)

        # Complete the rollout
        await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

        return await wait_task

    completed = await wait_and_complete()
    elapsed = time.time() - start

    if isinstance(store_fixture, InMemoryLightningStore):
        # Should complete after ~0.5s (when we set the event)
        assert 0.4 < elapsed < 0.7
    else:
        # Should be more than 5 seconds
        assert 5 < elapsed < 15
    assert len(completed) == 1
    assert completed[0].status == "succeeded"


# Concurrent Access Tests


@pytest.mark.asyncio
async def test_concurrent_task_addition(store_fixture: LightningStore) -> None:
    """Test adding tasks concurrently."""

    async def enqueue_rollout(index: int) -> Rollout:
        return await store_fixture.enqueue_rollout(input={"index": index})

    # Add 50 tasks concurrently
    tasks = [enqueue_rollout(i) for i in range(50)]
    rollouts = await asyncio.gather(*tasks)

    # All should succeed with unique IDs
    assert len(rollouts) == 50
    ids = [r.rollout_id for r in rollouts]
    assert len(set(ids)) == 50

    # All should be in store
    all_rollouts = await store_fixture.query_rollouts()
    assert len(all_rollouts) == 50


@pytest.mark.asyncio
async def test_concurrent_pop_operations(store_fixture: LightningStore) -> None:
    """Test concurrent popping ensures each rollout is popped once."""
    # Add 20 tasks
    for i in range(20):
        await store_fixture.enqueue_rollout(input={"index": i})

    async def pop_task() -> Rollout | None:
        return await store_fixture.dequeue_rollout()

    # Pop concurrently (more attempts than available)
    tasks = [pop_task() for _ in range(30)]
    results = await asyncio.gather(*tasks)

    # Should get exactly 20 rollouts and 10 None
    valid = [r for r in results if r is not None]
    none_results = [r for r in results if r is None]

    assert len(valid) == 20
    assert len(none_results) == 10

    # Each rollout popped exactly once
    ids = [r.rollout_id for r in valid]
    assert len(set(ids)) == 20


@pytest.mark.asyncio
async def test_concurrent_span_additions(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test concurrent span additions maintain consistency."""
    await store_fixture.enqueue_rollout(input={"test": "data"})
    rollout = await store_fixture.dequeue_rollout()  # Create an attempt
    assert rollout is not None

    async def add_span(index: int) -> Span:
        return await store_fixture.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, mock_readable_span)

    # Add 30 spans concurrently
    tasks = [add_span(i) for i in range(30)]
    spans = await asyncio.gather(*tasks)

    # All should have unique sequence IDs
    seq_ids = [s.sequence_id for s in spans]
    assert len(set(seq_ids)) == 30
    assert set(seq_ids) == set(range(1, 31))


@pytest.mark.asyncio
async def test_concurrent_resource_updates(store_fixture: LightningStore) -> None:
    """Test concurrent resource updates are atomic."""

    async def update_resource(ver: int) -> None:
        llm = LLM(
            resource_type="llm",
            endpoint=f"http://localhost:808{ver % 10}",
            model=f"model-v{ver}",
            sampling_parameters={"temperature": 0.5 + ver * 0.01},
        )
        update = ResourcesUpdate(
            resources_id=f"v{ver}", resources={"llm": llm}, create_time=time.time(), update_time=time.time(), version=1
        )
        await store_fixture.update_resources(update.resources_id, update.resources)

    # Update concurrently
    tasks = [update_resource(i) for i in range(50)]
    await asyncio.gather(*tasks)

    # Latest should be one of the versions
    latest = await store_fixture.get_latest_resources()
    assert latest is not None
    assert latest.resources_id.startswith("v")

    # All versions should be stored
    for i in range(50):
        res = await store_fixture.get_resources_by_id(f"v{i}")
        assert res is not None
        assert isinstance(res.resources["llm"], LLM)
        assert res.resources["llm"].model == f"model-v{i}"


# Error Handling Tests


@pytest.mark.asyncio
async def test_update_nonexistent_rollout(store_fixture: LightningStore) -> None:
    """Test updating non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store_fixture.update_rollout(rollout_id="nonexistent", status="failed")


@pytest.mark.asyncio
async def test_add_span_without_rollout(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test adding span to non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store_fixture.add_otel_span("nonexistent", "attempt-1", mock_readable_span)


@pytest.mark.asyncio
async def test_add_span_with_missing_attempt(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test adding span with an unknown attempt_id raises a helpful error."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    # Create a valid attempt to ensure rollout exists in store
    await store_fixture.dequeue_rollout()

    invalid_span = Span.from_opentelemetry(
        mock_readable_span,
        rollout_id=rollout.rollout_id,
        attempt_id="attempt-missing",
        sequence_id=1,
    )

    with pytest.raises(ValueError, match="Attempt attempt-missing not found"):
        await store_fixture.add_span(invalid_span)


@pytest.mark.asyncio
async def test_query_empty_spans(store_fixture: LightningStore) -> None:
    """Test querying spans for non-existent rollout returns empty."""
    spans = await store_fixture.query_spans("nonexistent")
    assert len(spans) == 0

    # With attempt_id
    spans = await store_fixture.query_spans("nonexistent", attempt_id="attempt-1")
    assert len(spans) == 0

    # With latest
    spans = await store_fixture.query_spans("nonexistent", attempt_id="latest")
    assert len(spans) == 0


@pytest.mark.asyncio
async def test_query_latest_with_no_spans(store_fixture: LightningStore) -> None:
    """Test querying 'latest' attempt when no spans exist."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    spans = await store_fixture.query_spans(rollout.rollout_id, attempt_id="latest")
    assert len(spans) == 0


@pytest.mark.asyncio
async def test_wait_for_nonexistent_rollout(store_fixture: LightningStore) -> None:
    """Test waiting for non-existent rollout handles gracefully."""
    completed = await store_fixture.wait_for_rollouts(rollout_ids=["nonexistent"], timeout=0.1)
    assert len(completed) == 0


# Attempt Management Tests


@pytest.mark.asyncio
async def test_query_attempts(store_fixture: LightningStore) -> None:
    """Test querying attempts for a rollout."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # Initially no attempts
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 0

    # Pop creates first attempt
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].sequence_id == 1
    assert attempts[0].status == "preparing"

    # Requeue and pop creates second attempt
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store_fixture.dequeue_rollout()
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 2
    assert attempts[0].sequence_id == 1
    assert attempts[1].sequence_id == 2


@pytest.mark.asyncio
async def test_get_latest_attempt(store_fixture: LightningStore) -> None:
    """Test getting the latest attempt."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # No attempts initially
    latest = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert latest is None

    # Create first attempt
    await store_fixture.dequeue_rollout()
    latest = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 1

    # Create second attempt
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store_fixture.dequeue_rollout()
    latest = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 2


@pytest.mark.asyncio
async def test_update_attempt_fields(store_fixture: LightningStore) -> None:
    """Test updating attempt fields."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    await store_fixture.dequeue_rollout()

    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Update various fields
    updated = await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="running",
        worker_id="worker-123",
        last_heartbeat_time=time.time(),
        metadata={"custom": "value"},
    )

    assert updated.status == "running"
    assert updated.worker_id == "worker-123"
    assert updated.last_heartbeat_time is not None
    assert updated.metadata is not None
    assert updated.metadata["custom"] == "value"


@pytest.mark.asyncio
async def test_update_latest_attempt(store_fixture: LightningStore) -> None:
    """Test updating latest attempt using 'latest' identifier."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    await store_fixture.dequeue_rollout()

    # Update using 'latest'
    updated = await store_fixture.update_attempt(rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded")

    assert updated.status == "succeeded"
    assert updated.end_time is not None  # Should auto-set end_time


@pytest.mark.asyncio
async def test_update_attempt_sets_end_time_for_terminal_status(store_fixture: LightningStore) -> None:
    """Terminal attempt statuses set end_time while in-progress statuses don't."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    await store_fixture.dequeue_rollout()

    attempt = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    assert attempt.end_time is None

    running = await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="running",
    )
    assert running.status == "running"
    assert running.end_time is None

    failed = await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="failed",
    )
    assert failed.status == "failed"
    assert failed.end_time is not None
    assert failed.end_time >= failed.start_time

    rollout = await store_fixture.get_rollout_by_id(rollout_id=rollout.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"
    assert rollout.end_time is not None
    assert rollout.end_time >= failed.end_time


@pytest.mark.asyncio
async def test_rollout_retry_lifecycle_updates_statuses(
    store_fixture: LightningStore, mock_readable_span: Mock
) -> None:
    """Rollout retry creates new attempts and updates statuses via spans and completions."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    first_attempted = await store_fixture.dequeue_rollout()
    assert first_attempted is not None
    assert first_attempted.status == "preparing"

    first_attempt = (await store_fixture.query_attempts(rollout.rollout_id))[0]
    await store_fixture.add_otel_span(rollout.rollout_id, first_attempt.attempt_id, mock_readable_span)

    # Status should reflect running state after span is recorded
    running_rollout = await store_fixture.query_rollouts(status=["running"])
    assert running_rollout and running_rollout[0].rollout_id == rollout.rollout_id

    running_attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"

    # Mark first attempt as failed and requeue rollout
    failed_attempt = await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=first_attempt.attempt_id,
        status="failed",
    )
    assert failed_attempt.end_time is not None
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")

    attempts_after_failure = await store_fixture.query_attempts(rollout.rollout_id)
    assert [a.status for a in attempts_after_failure] == ["failed"]

    retry_attempted = await store_fixture.dequeue_rollout()
    assert retry_attempted is not None
    assert retry_attempted.status == "preparing"
    assert retry_attempted.attempt.sequence_id == 2

    latest_pre_span = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert latest_pre_span is not None and latest_pre_span.sequence_id == 2
    assert latest_pre_span.status == "preparing"

    await store_fixture.add_otel_span(rollout.rollout_id, retry_attempted.attempt.attempt_id, mock_readable_span)

    latest_running = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert latest_running is not None
    assert latest_running.sequence_id == 2
    assert latest_running.status == "running"

    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=retry_attempted.attempt.attempt_id,
        status="succeeded",
    )
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    final_rollout = await store_fixture.query_rollouts(status=["succeeded"])
    assert final_rollout and final_rollout[0].rollout_id == rollout.rollout_id

    final_attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert [a.status for a in final_attempts] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_update_nonexistent_attempt(store_fixture: LightningStore) -> None:
    """Test updating non-existent attempt raises error."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    with pytest.raises(ValueError, match="No attempts found"):
        await store_fixture.update_attempt(rollout_id=rollout.rollout_id, attempt_id="nonexistent", status="failed")


# Add Attempt Tests


@pytest.mark.asyncio
async def test_add_attempt_creates_new_attempt(store_fixture: LightningStore) -> None:
    """Test add_attempt creates a new attempt for existing rollout."""
    # Create a rollout
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})

    # Add first manual attempt
    attempted_rollout = await store_fixture.start_attempt(rollout.rollout_id)

    assert attempted_rollout.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.sequence_id == 1
    assert attempted_rollout.attempt.status == "preparing"
    assert attempted_rollout.attempt.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.attempt_id.startswith("at-")

    # Verify attempt is stored
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempted_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_add_attempt_increments_sequence_id(store_fixture: LightningStore) -> None:
    """Test add_attempt correctly increments sequence_id."""
    # Create a rollout and dequeue to create first attempt
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    await store_fixture.dequeue_rollout()  # Creates attempt with sequence_id=1

    # Add second attempt manually
    attempted_rollout2 = await store_fixture.start_attempt(rollout.rollout_id)
    assert attempted_rollout2.attempt.sequence_id == 2

    # Add third attempt manually
    attempted_rollout3 = await store_fixture.start_attempt(rollout.rollout_id)
    assert attempted_rollout3.attempt.sequence_id == 3

    # Verify all attempts exist
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3
    assert [a.sequence_id for a in attempts] == [1, 2, 3]


@pytest.mark.asyncio
async def test_add_attempt_nonexistent_rollout(store_fixture: LightningStore) -> None:
    """Test add_attempt raises error for nonexistent rollout."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store_fixture.start_attempt("nonexistent")


@pytest.mark.asyncio
async def test_add_attempt_ignores_max_attempts(store_fixture: LightningStore) -> None:
    """Test add_attempt ignores max_attempts configuration."""
    # Create rollout with max_attempts=2
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"})
    config = RolloutConfig(max_attempts=2)
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Add attempts beyond max_attempts
    attempt1 = await store_fixture.start_attempt(rollout.rollout_id)
    attempt2 = await store_fixture.start_attempt(rollout.rollout_id)
    attempt3 = await store_fixture.start_attempt(rollout.rollout_id)  # Should succeed despite max_attempts=2

    assert attempt1.attempt.sequence_id == 1
    assert attempt2.attempt.sequence_id == 2
    assert attempt3.attempt.sequence_id == 3

    # All attempts should exist
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3


# Latest Attempt Status Propagation Tests


@pytest.mark.asyncio
async def test_status_propagation_only_for_latest_attempt(store_fixture: LightningStore) -> None:
    """Test that status changes only propagate to rollout when updating latest attempt."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "propagation"})

    # Create multiple attempts
    attempt1 = await store_fixture.start_attempt(rollout.rollout_id)
    _attempt2 = await store_fixture.start_attempt(rollout.rollout_id)
    attempt3 = await store_fixture.start_attempt(rollout.rollout_id)  # This is the latest

    # Update attempt1 (not latest) to succeeded
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout status should NOT change since attempt1 is not the latest
    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "preparing"  # Should be status of attempt 2

    # Update attempt3 (latest) to succeeded
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt3.attempt.attempt_id, status="succeeded"
    )

    # Now rollout status should change since we updated the latest attempt
    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_status_propagation_with_retry_for_latest_attempt(store_fixture: LightningStore) -> None:
    """Test retry logic only applies when updating latest attempt."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "retry"})
    config = RolloutConfig(max_attempts=3, retry_condition=["failed"])
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Create multiple attempts
    attempt1 = await store_fixture.start_attempt(rollout.rollout_id)  # sequence_id=1
    attempt2 = await store_fixture.start_attempt(rollout.rollout_id)  # sequence_id=2 (latest)

    # Fail attempt1 (not latest) - should NOT trigger retry
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed"
    )

    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "preparing"  # Should be status of attempt 2

    # Fail attempt2 (latest) - should trigger retry since sequence_id=2 < max_attempts=3
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed"
    )

    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "requeuing"  # Should be requeued for retry


@pytest.mark.asyncio
async def test_status_propagation_latest_changes_when_new_attempt_added(store_fixture: LightningStore) -> None:
    """Test that the 'latest attempt' changes as new attempts are added."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "latest_changes"})

    # Create first attempt and update it to succeeded
    attempt1 = await store_fixture.start_attempt(rollout.rollout_id)
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout should be succeeded since attempt1 is latest
    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"

    # Add second attempt (now this becomes latest)
    attempt2 = await store_fixture.start_attempt(rollout.rollout_id)

    # Update attempt1 to failed - should NOT affect rollout since it's no longer latest
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed"
    )

    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "preparing"  # Should be the status of attempt 2

    # Update attempt2 (now latest) to failed
    await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed"
    )

    # Now rollout should change since we updated the new latest attempt
    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "failed"


@pytest.mark.asyncio
async def test_status_propagation_update_latest_by_reference(store_fixture: LightningStore) -> None:
    """Test status propagation when updating latest attempt using 'latest' reference."""
    rollout = await store_fixture.enqueue_rollout(input={"test": "latest_ref"})

    # Create multiple attempts
    await store_fixture.start_attempt(rollout.rollout_id)
    await store_fixture.start_attempt(rollout.rollout_id)
    attempt3 = await store_fixture.start_attempt(rollout.rollout_id)  # This is latest

    # Update using "latest" reference
    updated_attempt = await store_fixture.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded"
    )

    # Should have updated attempt3
    assert updated_attempt.attempt_id == attempt3.attempt.attempt_id
    assert updated_attempt.status == "succeeded"

    # Rollout should be updated since we updated the latest attempt
    updated_rollout = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_healthcheck_timeout_behavior(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test that healthcheck detects and handles timeout conditions."""
    # Create rollout with short timeout configuration
    config = RolloutConfig(
        timeout_seconds=0.1, max_attempts=2, retry_condition=["timeout"]  # Very short timeout for testing
    )

    rollout = await store_fixture.enqueue_rollout(input={"test": "timeout"})
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue to create an attempt and add span to make it running
    attempted = await store_fixture.dequeue_rollout()
    assert attempted is not None
    await store_fixture.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running
    running_rollouts = await store_fixture.query_rollouts(status=["running"])
    assert len(running_rollouts) == 1

    # Wait for timeout to occur
    await asyncio.sleep(0.15)  # Wait longer than timeout_seconds

    # Trigger healthcheck by calling any decorated method
    # Verify the attempt was marked as timeout and rollout was requeued
    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].status == "timeout"

    # Since retry_condition includes "timeout" and max_attempts=2, should requeue
    rollout_after = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "requeuing"


@pytest.mark.asyncio
async def test_healthcheck_unresponsive_behavior(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test that healthcheck detects and handles unresponsive conditions."""
    # Create rollout with short unresponsive timeout but no retry for unresponsive
    config = RolloutConfig(
        unresponsive_seconds=0.1,  # Very short unresponsive timeout
        max_attempts=3,
        retry_condition=["timeout"],  # Note: "unresponsive" not in retry_condition
    )

    rollout = await store_fixture.enqueue_rollout(input={"test": "unresponsive"})
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue and add span to make it running (this sets last_heartbeat_time)
    attempted = await store_fixture.dequeue_rollout()
    assert attempted is not None
    await store_fixture.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running and has heartbeat
    running_attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"
    assert running_attempts[0].last_heartbeat_time is not None

    # Wait for unresponsive timeout
    await asyncio.sleep(0.15)  # Wait longer than unresponsive_seconds

    # Verify attempt was marked as unresponsive
    attempts_after = await store_fixture.query_attempts(rollout.rollout_id)
    assert attempts_after[0].status == "unresponsive"

    # Since "unresponsive" not in retry_condition, rollout should be failed
    rollout_after = await store_fixture.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "failed"


# Full Lifecycle Integration Tests


@pytest.mark.asyncio
async def test_full_lifecycle_success(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """Test successful rollout lifecycle: queue -> prepare -> run -> succeed."""
    # 1. Create task
    rollout = await store_fixture.enqueue_rollout(input={"test": "data"}, mode="train")
    assert rollout.status == "queuing"

    # 2. Pop to start processing (creates attempt)
    popped = await store_fixture.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.status == "preparing"

    # 3. Add span (transitions to running)
    span = await store_fixture.add_otel_span(rollout.rollout_id, attempt.attempt_id, mock_readable_span)
    assert span.sequence_id == 1

    # Check status transitions
    rollouts = await store_fixture.query_rollouts(status=["running"])
    assert len(rollouts) == 1

    attempts = await store_fixture.query_attempts(rollout.rollout_id)
    assert attempts[0].status == "running"
    assert attempts[0].last_heartbeat_time is not None

    # 4. Complete successfully
    await store_fixture.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt.attempt_id, status="succeeded")
    await store_fixture.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Verify final state
    final = (await store_fixture.query_rollouts())[0]
    assert final.status == "succeeded"
    assert final.end_time is not None

    final_attempt = await store_fixture.get_latest_attempt(rollout.rollout_id)
    assert final_attempt is not None
    assert final_attempt.status == "succeeded"
    assert final_attempt.end_time is not None


# Retry and requeue interactions


def _retry_config() -> RolloutConfig:
    """Helper to create a rollout config that retries unresponsive attempts."""

    return RolloutConfig(max_attempts=2, retry_condition=["unresponsive"])


@pytest.mark.asyncio
async def test_requeued_attempt_recovers_before_retry(store_fixture: LightningStore, mock_readable_span: Mock) -> None:
    """A requeued attempt that resumes should be removed from the queue."""

    attempted = await store_fixture.start_rollout(input={"foo": "bar"})
    await store_fixture.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    rollout = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "requeuing"

    await store_fixture.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    latest_attempt = await store_fixture.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == attempted.attempt.attempt_id
    assert latest_attempt.status == "running"

    rollout = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "running"

    # Queue should no longer return the rollout for retry.
    assert await store_fixture.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_succeeds_without_new_attempt(
    store_fixture: LightningStore, mock_readable_span: Mock
) -> None:
    """Recovered attempts can finish successfully without spawning a retry."""

    attempted = await store_fixture.start_rollout(input={"foo": "bar"})
    await store_fixture.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    await store_fixture.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="succeeded"
    )

    rollout = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "succeeded"

    latest_attempt = await store_fixture.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "succeeded"
    assert latest_attempt.end_time is not None

    assert await store_fixture.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_fails_without_new_attempt(
    store_fixture: LightningStore, mock_readable_span: Mock
) -> None:
    """Recovered attempts that fail should mark the rollout failed without retries."""

    attempted = await store_fixture.start_rollout(input={"foo": "bar"})
    await store_fixture.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    await store_fixture.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="failed"
    )

    rollout = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"

    latest_attempt = await store_fixture.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "failed"
    assert latest_attempt.end_time is not None

    assert await store_fixture.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_recovers_after_retry_started(
    store_fixture: LightningStore, mock_readable_span: Mock
) -> None:
    """Data from an old attempt should not disrupt a newly started retry."""

    attempted = await store_fixture.start_rollout(input={"foo": "bar"})
    await store_fixture.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await store_fixture.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    # Start a new attempt by dequeuing the rollout from the queue.
    retried = await store_fixture.dequeue_rollout()
    assert retried is not None
    assert retried.attempt.sequence_id == 2

    await store_fixture.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    latest_attempt = await store_fixture.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == retried.attempt.attempt_id
    assert latest_attempt.sequence_id == 2

    # The old attempt is still marked running but does not change the rollout state.
    first_attempts = await store_fixture.query_attempts(attempted.rollout_id)
    assert first_attempts[0].status == "running"
    rollout = await store_fixture.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "preparing"

    assert await store_fixture.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_resources_update_tracks_create_and_update_times(store_fixture: LightningStore) -> None:
    """Test that ResourcesUpdate tracks create_time and update_time correctly."""
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )

    # Add initial resource
    start_time = time.time()
    update1 = await store_fixture.add_resources({"main_llm": llm})

    # Verify create_time is set and reasonable
    assert update1.create_time >= start_time
    assert update1.create_time <= time.time()

    # Initially, update_time should equal create_time
    assert update1.update_time == update1.create_time
    assert update1.version == 1

    # Wait a bit and update the same resource
    await asyncio.sleep(0.01)
    llm_v2 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="test-model-v2",
        sampling_parameters={"temperature": 0.8},
    )
    update2 = await store_fixture.update_resources(update1.resources_id, {"main_llm": llm_v2})

    # Verify update_time changed but create_time stayed the same
    assert update2.resources_id == update1.resources_id
    assert update2.create_time == update1.create_time  # create_time should not change
    assert update2.update_time > update1.update_time  # update_time should be newer
    assert update2.version == 2  # version should increment


@pytest.mark.asyncio
async def test_resources_update_version_increments(store_fixture: LightningStore) -> None:
    """Test that ResourcesUpdate version increments correctly with each update."""
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )

    # Add initial resource
    update1 = await store_fixture.add_resources({"main_llm": llm})
    assert update1.version == 1

    # Update it multiple times
    for i in range(2, 6):
        llm_updated = LLM(
            resource_type="llm",
            endpoint="http://localhost:8080",
            model=f"test-model-v{i}",
            sampling_parameters={"temperature": 0.7},
        )
        update = await store_fixture.update_resources(update1.resources_id, {"main_llm": llm_updated})
        assert update.version == i
        assert update.resources_id == update1.resources_id
        assert update.create_time == update1.create_time


@pytest.mark.asyncio
async def test_resources_different_ids_have_independent_versions(store_fixture: LightningStore) -> None:
    """Test that different resources_ids have independent version counters."""
    llm1 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="model-1",
        sampling_parameters={"temperature": 0.7},
    )
    llm2 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="model-2",
        sampling_parameters={"temperature": 0.8},
    )

    # Add two different resources
    res1 = await store_fixture.add_resources({"llm": llm1})
    res2 = await store_fixture.add_resources({"llm": llm2})

    # Both should start at version 1
    assert res1.version == 1
    assert res2.version == 1
    assert res1.resources_id != res2.resources_id

    # Update res1 twice
    for i in range(2):
        llm_updated = LLM(
            resource_type="llm",
            endpoint="http://localhost:8080",
            model=f"model-1-v{i+2}",
            sampling_parameters={"temperature": 0.7},
        )
        res1 = await store_fixture.update_resources(res1.resources_id, {"llm": llm_updated})

    # res1 should be at version 3, res2 should still be at version 1
    assert res1.version == 3
    retrieved_res2 = await store_fixture.get_resources_by_id(res2.resources_id)
    assert retrieved_res2 is not None
    assert retrieved_res2.version == 1


@pytest.mark.asyncio
async def test_query_resources_returns_all_fields(store_fixture: LightningStore) -> None:
    """Test that query_resources returns all ResourcesUpdate fields."""
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )

    # Add multiple resources
    await store_fixture.add_resources({"llm": llm})
    await asyncio.sleep(0.01)
    await store_fixture.add_resources({"llm": llm})

    # Query all resources
    all_resources = await store_fixture.query_resources()

    assert len(all_resources) == 2
    for res in all_resources:
        assert res.resources_id is not None
        assert res.create_time > 0
        assert res.update_time > 0
        assert res.version >= 1
        assert res.resources is not None

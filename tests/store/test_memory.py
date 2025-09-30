# Copyright (c) Microsoft. All rights reserved.

"""
Comprehensive tests for InMemoryLightningStore.

Test categories:
- Core CRUD operations
- Queue operations (FIFO behavior)
- Resource versioning
- Span tracking and sequencing
- Rollout lifecycle and status transitions
- Concurrent access patterns
- Error handling and edge cases
"""

import asyncio
import time
from typing import List
from unittest.mock import Mock

import pytest

from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import Span
from agentlightning.types import (
    LLM,
    PromptTemplate,
    ResourcesUpdate,
    RolloutConfig,
    RolloutV2,
)

# Core CRUD Operations Tests


@pytest.mark.asyncio
async def test_enqueue_rollout_creates_rollout(store: InMemoryLightningStore) -> None:
    """Test that enqueue_rollout creates a properly initialized rollout."""
    sample = {"input": "test_data"}
    metadata = {"key": "value", "number": 42}

    rollout = await store.enqueue_rollout(input=sample, mode="train", resources_id="res-123", metadata=metadata)

    assert rollout.rollout_id.startswith("rollout-")
    assert rollout.input == sample
    assert rollout.mode == "train"
    assert rollout.resources_id == "res-123"
    assert rollout.metadata == metadata
    assert rollout.status == "queuing"
    assert rollout.start_time is not None


@pytest.mark.asyncio
async def test_add_rollout_initializes_attempt(store: InMemoryLightningStore) -> None:
    """Test that add_rollout immediately tracks a preparing attempt."""
    sample = {"payload": "value"}

    attempt_rollout = await store.start_rollout(input=sample, mode="val", resources_id="res-add")

    assert attempt_rollout.status == "preparing"
    assert attempt_rollout.rollout_id.startswith("rollout-")
    assert attempt_rollout.attempt.attempt_id.startswith("attempt-")
    assert attempt_rollout.attempt.sequence_id == 1
    assert attempt_rollout.attempt.status == "preparing"

    stored = await store.query_rollouts(status=["preparing"])
    assert len(stored) == 1
    assert stored[0].rollout_id == attempt_rollout.rollout_id
    assert stored[0].resources_id == "res-add"

    attempts = await store.query_attempts(attempt_rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempt_rollout.attempt.attempt_id

    latest_attempt = await store.get_latest_attempt(attempt_rollout.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == attempt_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_query_rollouts_by_status(store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by status."""
    # Create rollouts with different statuses
    r1 = await store.enqueue_rollout(input={"id": 1})
    r2 = await store.enqueue_rollout(input={"id": 2})
    r3 = await store.enqueue_rollout(input={"id": 3})

    # Modify statuses
    await store.dequeue_rollout()  # r1 becomes "preparing"
    await store.update_rollout(rollout_id=r2.rollout_id, status="failed")
    # r3 remains "queuing"

    # Test various queries
    all_rollouts = await store.query_rollouts()
    assert len(all_rollouts) == 3

    queuing = await store.query_rollouts(status=["queuing"])
    assert len(queuing) == 1
    assert queuing[0].rollout_id == r3.rollout_id

    preparing = await store.query_rollouts(status=["preparing"])
    assert len(preparing) == 1
    assert preparing[0].rollout_id == r1.rollout_id

    finished = await store.query_rollouts(status=["failed", "succeeded"])
    assert len(finished) == 1
    assert finished[0].rollout_id == r2.rollout_id

    # Empty status list
    none = await store.query_rollouts(status=[])
    assert len(none) == 0


@pytest.mark.asyncio
async def test_get_rollout_by_id(store: InMemoryLightningStore) -> None:
    """Test retrieving rollouts by their ID."""
    # Test getting non-existent rollout
    rollout = await store.get_rollout_by_id("nonexistent")
    assert rollout is None

    # Create a rollout
    created = await store.enqueue_rollout(input={"test": "data"}, mode="train")

    # Retrieve by ID
    retrieved = await store.get_rollout_by_id(created.rollout_id)
    assert retrieved is not None
    assert retrieved.rollout_id == created.rollout_id
    assert retrieved.input == created.input
    assert retrieved.mode == created.mode
    assert retrieved.status == created.status

    # Update rollout and verify changes are reflected
    await store.update_rollout(rollout_id=created.rollout_id, status="running")
    updated = await store.get_rollout_by_id(created.rollout_id)
    assert updated is not None
    assert updated.status == "running"


@pytest.mark.asyncio
async def test_query_rollouts_by_rollout_ids(store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by rollout IDs."""
    # Create multiple rollouts
    r1 = await store.enqueue_rollout(input={"id": 1})
    r2 = await store.enqueue_rollout(input={"id": 2})
    r3 = await store.enqueue_rollout(input={"id": 3})

    # Query by specific IDs
    selected = await store.query_rollouts(rollout_ids=[r1.rollout_id, r3.rollout_id])
    assert len(selected) == 2
    selected_ids = {r.rollout_id for r in selected}
    assert selected_ids == {r1.rollout_id, r3.rollout_id}

    # Query by single ID
    single = await store.query_rollouts(rollout_ids=[r2.rollout_id])
    assert len(single) == 1
    assert single[0].rollout_id == r2.rollout_id

    # Query by non-existent ID
    none = await store.query_rollouts(rollout_ids=["nonexistent"])
    assert len(none) == 0

    # Combine with status filter
    await store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    await store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    filtered = await store.query_rollouts(
        rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], status=["succeeded", "queuing"]
    )
    assert len(filtered) == 2
    filtered_ids = {r.rollout_id for r in filtered}
    assert filtered_ids == {r1.rollout_id, r3.rollout_id}  # r1 succeeded, r3 still queuing


@pytest.mark.asyncio
async def test_update_rollout_fields(store: InMemoryLightningStore) -> None:
    """Test updating various rollout fields."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # Update multiple fields at once including config
    config = RolloutConfig(
        timeout_seconds=60.0, unresponsive_seconds=30.0, max_attempts=3, retry_condition=["timeout", "unresponsive"]
    )
    await store.update_rollout(
        rollout_id=rollout.rollout_id,
        status="running",
        mode="train",
        resources_id="new-resources",
        config=config,
        metadata={"custom_field": "custom_value"},
    )

    # Verify all updates
    updated_rollouts = await store.query_rollouts()
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
async def test_rollout_config_functionality(store: InMemoryLightningStore) -> None:
    """Test RolloutConfig controls retry and timeout behavior."""
    # Create rollout with specific retry configuration
    config = RolloutConfig(
        timeout_seconds=30.0,
        unresponsive_seconds=15.0,
        max_attempts=2,
        retry_condition=["timeout", "unresponsive", "failed"],
    )

    rollout = await store.enqueue_rollout(input={"test": "retry"})
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Verify config is stored
    stored = await store.get_rollout_by_id(rollout.rollout_id)
    assert stored is not None
    assert stored.config.timeout_seconds == 30.0
    assert stored.config.max_attempts == 2
    assert "failed" in stored.config.retry_condition

    # Test that different rollouts can have different configs
    config2 = RolloutConfig(timeout_seconds=120.0, max_attempts=5, retry_condition=["timeout"])

    rollout2 = await store.enqueue_rollout(input={"test": "different_config"})
    await store.update_rollout(rollout_id=rollout2.rollout_id, config=config2)

    stored2 = await store.get_rollout_by_id(rollout2.rollout_id)
    assert stored2 is not None
    assert stored2.config.timeout_seconds == 120.0
    assert stored2.config.max_attempts == 5
    assert stored2.config.retry_condition == ["timeout"]

    # Verify first rollout config unchanged
    stored1_again = await store.get_rollout_by_id(rollout.rollout_id)
    assert stored1_again is not None
    assert stored1_again.config.timeout_seconds == 30.0


# Queue Operations Tests


@pytest.mark.asyncio
async def test_dequeue_rollout_skips_non_queuing_status(store: InMemoryLightningStore) -> None:
    """Test that dequeue_rollout skips rollouts that have been updated to non-queuing status."""
    # Add multiple rollouts to the queue
    r1 = await store.enqueue_rollout(input={"id": 1})
    r2 = await store.enqueue_rollout(input={"id": 2})
    r3 = await store.enqueue_rollout(input={"id": 3})

    # Update r1 to succeeded status while it's still in the queue
    await store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Update r2 to failed status
    await store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # r3 should still be in queuing status

    # Pop should skip r1 and r2 (both non-queuing) and return r3
    popped = await store.dequeue_rollout()
    assert popped is not None
    assert popped.rollout_id == r3.rollout_id
    assert popped.status == "preparing"
    assert popped.input["id"] == 3

    # Second pop should return None since no queuing rollouts remain
    popped2 = await store.dequeue_rollout()
    assert popped2 is None

    # Verify r1 and r2 are still in their non-queuing states
    all_rollouts = await store.query_rollouts()
    rollout_statuses = {r.rollout_id: r.status for r in all_rollouts}
    assert rollout_statuses[r1.rollout_id] == "succeeded"
    assert rollout_statuses[r2.rollout_id] == "failed"
    assert rollout_statuses[r3.rollout_id] == "preparing"


@pytest.mark.asyncio
async def test_fifo_ordering(store: InMemoryLightningStore) -> None:
    """Test that queue maintains FIFO order."""
    rollouts: List[RolloutV2] = []
    for i in range(5):
        r = await store.enqueue_rollout(input={"order": i})
        rollouts.append(r)

    # Pop all and verify order
    for i in range(5):
        popped = await store.dequeue_rollout()
        assert popped is not None
        assert popped.rollout_id == rollouts[i].rollout_id
        assert popped.input["order"] == i
        assert popped.status == "preparing"


@pytest.mark.asyncio
async def test_pop_empty_queue(store: InMemoryLightningStore) -> None:
    """Test popping from empty queue returns None."""
    result = await store.dequeue_rollout()
    assert result is None

    # Multiple pops should all return None
    for _ in range(3):
        assert await store.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeue_mechanism(store: InMemoryLightningStore) -> None:
    """Test requeuing puts rollout back in queue."""
    rollout = await store.enqueue_rollout(input={"data": "test"})
    original_id = rollout.rollout_id

    # Pop and verify it's not in queue
    popped = await store.dequeue_rollout()
    assert popped is not None
    assert await store.dequeue_rollout() is None

    # Requeue it
    await store.update_rollout(rollout_id=original_id, status="requeuing")

    # Should be back in queue
    requeued = await store.dequeue_rollout()
    assert requeued is not None
    assert requeued.rollout_id == original_id
    assert requeued.status == "preparing"  # Changes when popped
    # Check that a new attempt was created
    attempts = await store.query_attempts(requeued.rollout_id)
    assert len(attempts) == 2  # First attempt plus requeued attempt

    latest_attempt = await store.get_latest_attempt(requeued.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "preparing"
    assert latest_attempt.sequence_id == 2


# Resource Management Tests


@pytest.mark.asyncio
async def test_resource_lifecycle(store: InMemoryLightningStore) -> None:
    """Test adding, updating, and retrieving resources."""
    # Initially no resources
    assert await store.get_latest_resources() is None
    assert await store.get_resources_by_id("any-id") is None

    # Add first version with proper LLM resource
    llm_v1 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model-v1",
        sampling_parameters={"temperature": 0.7},
    )
    update = await store.update_resources("v1", {"main_llm": llm_v1})
    assert update.resources_id == "v1"

    latest = await store.get_latest_resources()
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
    v2 = await store.update_resources("v2", {"main_llm": llm_v2})
    assert v2.resources_id == "v2"
    assert isinstance(v2.resources["main_llm"], LLM)
    assert v2.resources["main_llm"].model == "test-model-v2"

    # Latest should be v2
    latest = await store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v2"

    # Can still retrieve v1
    old = await store.get_resources_by_id("v1")
    assert old is not None
    assert isinstance(old.resources["main_llm"], LLM)
    assert old.resources["main_llm"].model == "test-model-v1"


@pytest.mark.asyncio
async def test_task_inherits_latest_resources(store: InMemoryLightningStore) -> None:
    """Test that new tasks inherit latest resources_id if not specified."""
    # Set up resources with proper PromptTemplate
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")
    update = ResourcesUpdate(resources_id="current", resources={"greeting": prompt})
    await store.update_resources(update.resources_id, update.resources)

    # Task without explicit resources_id
    r1 = await store.enqueue_rollout(input={"id": 1})
    assert r1.resources_id == "current"

    # Task with explicit resources_id
    r2 = await store.enqueue_rollout(input={"id": 2}, resources_id="override")
    assert r2.resources_id == "override"

    # Update resources
    new_prompt = PromptTemplate(resource_type="prompt_template", template="Hi {name}!", engine="f-string")
    update2 = ResourcesUpdate(resources_id="new", resources={"greeting": new_prompt})
    await store.update_resources(update2.resources_id, update2.resources)

    # New task gets new resources
    r3 = await store.enqueue_rollout(input={"id": 3})
    assert r3.resources_id == "new"


# Span Management Tests


@pytest.mark.asyncio
async def test_span_sequence_generation(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test automatic sequence ID generation for spans."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # First span gets sequence_id 1
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 1

    span1 = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span1.sequence_id == 2

    # Next span gets sequence_id 3
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 3

    span2 = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span2.sequence_id == 4

    # Different attempt reuses the same rollout_id
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-2")
    assert seq_id == 5


@pytest.mark.asyncio
async def test_span_with_explicit_sequence_id(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test providing explicit sequence_id to spans."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add span with explicit sequence_id
    span = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span, sequence_id=100)
    assert span.sequence_id == 100

    next_seq = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert next_seq == 101


@pytest.mark.asyncio
async def test_query_spans_by_attempt(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test querying spans filtered by attempt_id."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    # Pop to create first attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt1_id = attempts[0].attempt_id

    # Add spans for first attempt
    for _ in range(2):
        await store.add_otel_span(rollout.rollout_id, attempt1_id, mock_readable_span)

    # Simulate requeue and create second attempt
    await store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt2_id = attempts[1].attempt_id

    # Add spans for second attempt
    for _ in range(3):
        await store.add_otel_span(rollout.rollout_id, attempt2_id, mock_readable_span)

    # Query all spans
    all_spans = await store.query_spans(rollout.rollout_id)
    assert len(all_spans) == 5

    # Query specific attempt
    attempt1_spans = await store.query_spans(rollout.rollout_id, attempt_id=attempt1_id)
    assert len(attempt1_spans) == 2
    assert all(s.attempt_id == attempt1_id for s in attempt1_spans)

    # Query latest attempt
    latest_spans = await store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert len(latest_spans) == 3
    assert all(s.attempt_id == attempt2_id for s in latest_spans)

    # Query non-existent attempt
    no_spans = await store.query_spans(rollout.rollout_id, attempt_id="nonexistent")
    assert len(no_spans) == 0


@pytest.mark.asyncio
async def test_span_triggers_status_transition(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test that adding first span transitions rollout from preparing to running."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # Pop to set status to preparing and create attempt
    popped = await store.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Verify status in store
    rollouts = await store.query_rollouts(status=["preparing"])
    assert len(rollouts) == 1

    # Get the attempt
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add first span
    await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    # Status should transition to running
    rollouts = await store.query_rollouts(status=["running"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id


# Rollout Lifecycle Tests


@pytest.mark.asyncio
async def test_completion_sets_end_time(store: InMemoryLightningStore) -> None:
    """Test that completing a rollout sets end_time."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # Initially no end_time
    assert rollout.end_time is None

    # Complete as succeeded
    await store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    completed_rollouts = await store.query_rollouts()
    completed = completed_rollouts[0]
    assert completed.status == "succeeded"
    assert completed.end_time is not None
    assert completed.end_time > completed.start_time


@pytest.mark.asyncio
async def test_wait_for_rollouts(store: InMemoryLightningStore) -> None:
    """Test waiting for rollout completion."""
    # Add multiple rollouts
    r1 = await store.enqueue_rollout(input={"id": 1})
    r2 = await store.enqueue_rollout(input={"id": 2})
    _r3 = await store.enqueue_rollout(input={"id": 3})

    # Start waiting for r1 and r2
    async def wait_for_completion() -> List[RolloutV2]:
        return await store.wait_for_rollouts(rollout_ids=[r1.rollout_id, r2.rollout_id], timeout=5.0)

    wait_task = asyncio.create_task(wait_for_completion())
    await asyncio.sleep(0.01)  # Let wait task start

    # Complete r1
    await store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Complete r2
    await store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # Get results
    completed = await wait_task
    assert len(completed) == 2
    assert {r.rollout_id for r in completed} == {r1.rollout_id, r2.rollout_id}
    assert {r.status for r in completed} == {"succeeded", "failed"}


@pytest.mark.asyncio
async def test_wait_timeout(store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts timeout behavior."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    start = time.time()
    completed = await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0.1)
    elapsed = time.time() - start

    assert elapsed < 0.2  # Should timeout quickly
    assert len(completed) == 0  # No completions


# Concurrent Access Tests


@pytest.mark.asyncio
async def test_concurrent_task_addition(store: InMemoryLightningStore) -> None:
    """Test adding tasks concurrently."""

    async def enqueue_rollout(index: int) -> RolloutV2:
        return await store.enqueue_rollout(input={"index": index})

    # Add 50 tasks concurrently
    tasks = [enqueue_rollout(i) for i in range(50)]
    rollouts = await asyncio.gather(*tasks)

    # All should succeed with unique IDs
    assert len(rollouts) == 50
    ids = [r.rollout_id for r in rollouts]
    assert len(set(ids)) == 50

    # All should be in store
    all_rollouts = await store.query_rollouts()
    assert len(all_rollouts) == 50


@pytest.mark.asyncio
async def test_concurrent_pop_operations(store: InMemoryLightningStore) -> None:
    """Test concurrent popping ensures each rollout is popped once."""
    # Add 20 tasks
    for i in range(20):
        await store.enqueue_rollout(input={"index": i})

    async def pop_task() -> RolloutV2 | None:
        return await store.dequeue_rollout()

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
async def test_concurrent_span_additions(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test concurrent span additions maintain consistency."""
    await store.enqueue_rollout(input={"test": "data"})
    rollout = await store.dequeue_rollout()  # Create an attempt
    assert rollout is not None

    async def add_span(index: int) -> Span:
        return await store.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, mock_readable_span)

    # Add 30 spans concurrently
    tasks = [add_span(i) for i in range(30)]
    spans = await asyncio.gather(*tasks)

    # All should have unique sequence IDs
    seq_ids = [s.sequence_id for s in spans]
    assert len(set(seq_ids)) == 30
    assert set(seq_ids) == set(range(1, 31))


@pytest.mark.asyncio
async def test_concurrent_resource_updates(store: InMemoryLightningStore) -> None:
    """Test concurrent resource updates are atomic."""

    async def update_resource(ver: int) -> None:
        llm = LLM(
            resource_type="llm",
            endpoint=f"http://localhost:808{ver % 10}",
            model=f"model-v{ver}",
            sampling_parameters={"temperature": 0.5 + ver * 0.01},
        )
        update = ResourcesUpdate(resources_id=f"v{ver}", resources={"llm": llm})
        await store.update_resources(update.resources_id, update.resources)

    # Update concurrently
    tasks = [update_resource(i) for i in range(50)]
    await asyncio.gather(*tasks)

    # Latest should be one of the versions
    latest = await store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id.startswith("v")

    # All versions should be stored
    for i in range(50):
        res = await store.get_resources_by_id(f"v{i}")
        assert res is not None
        assert isinstance(res.resources["llm"], LLM)
        assert res.resources["llm"].model == f"model-v{i}"


# Error Handling Tests


@pytest.mark.asyncio
async def test_update_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test updating non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store.update_rollout(rollout_id="nonexistent", status="failed")


@pytest.mark.asyncio
async def test_add_span_without_rollout(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test adding span to non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store.add_otel_span("nonexistent", "attempt-1", mock_readable_span)


@pytest.mark.asyncio
async def test_add_span_with_missing_attempt(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test adding span with an unknown attempt_id raises a helpful error."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    # Create a valid attempt to ensure rollout exists in store
    await store.dequeue_rollout()

    invalid_span = Span.from_opentelemetry(
        mock_readable_span,
        rollout_id=rollout.rollout_id,
        attempt_id="attempt-missing",
        sequence_id=1,
    )

    with pytest.raises(ValueError, match="Attempt attempt-missing not found"):
        await store.add_span(invalid_span)


@pytest.mark.asyncio
async def test_query_empty_spans(store: InMemoryLightningStore) -> None:
    """Test querying spans for non-existent rollout returns empty."""
    spans = await store.query_spans("nonexistent")
    assert spans == []

    # With attempt_id
    spans = await store.query_spans("nonexistent", attempt_id="attempt-1")
    assert spans == []

    # With latest
    spans = await store.query_spans("nonexistent", attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_query_latest_with_no_spans(store: InMemoryLightningStore) -> None:
    """Test querying 'latest' attempt when no spans exist."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    spans = await store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_wait_for_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test waiting for non-existent rollout handles gracefully."""
    completed = await store.wait_for_rollouts(rollout_ids=["nonexistent"], timeout=0.1)
    assert len(completed) == 0


# Attempt Management Tests


@pytest.mark.asyncio
async def test_query_attempts(store: InMemoryLightningStore) -> None:
    """Test querying attempts for a rollout."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # Initially no attempts
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 0

    # Pop creates first attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].sequence_id == 1
    assert attempts[0].status == "preparing"

    # Requeue and pop creates second attempt
    await store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 2
    assert attempts[0].sequence_id == 1
    assert attempts[1].sequence_id == 2


@pytest.mark.asyncio
async def test_get_latest_attempt(store: InMemoryLightningStore) -> None:
    """Test getting the latest attempt."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # No attempts initially
    latest = await store.get_latest_attempt(rollout.rollout_id)
    assert latest is None

    # Create first attempt
    await store.dequeue_rollout()
    latest = await store.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 1

    # Create second attempt
    await store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await store.dequeue_rollout()
    latest = await store.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 2


@pytest.mark.asyncio
async def test_update_attempt_fields(store: InMemoryLightningStore) -> None:
    """Test updating attempt fields."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    await store.dequeue_rollout()

    attempts = await store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Update various fields
    updated = await store.update_attempt(
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
async def test_update_latest_attempt(store: InMemoryLightningStore) -> None:
    """Test updating latest attempt using 'latest' identifier."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    await store.dequeue_rollout()

    # Update using 'latest'
    updated = await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded")

    assert updated.status == "succeeded"
    assert updated.end_time is not None  # Should auto-set end_time


@pytest.mark.asyncio
async def test_update_attempt_sets_end_time_for_terminal_status(store: InMemoryLightningStore) -> None:
    """Terminal attempt statuses set end_time while in-progress statuses don't."""
    rollout = await store.enqueue_rollout(input={"test": "data"})
    await store.dequeue_rollout()

    attempt = (await store.query_attempts(rollout.rollout_id))[0]
    assert attempt.end_time is None

    running = await store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="running",
    )
    assert running.status == "running"
    assert running.end_time is None

    failed = await store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="failed",
    )
    assert failed.status == "failed"
    assert failed.end_time is not None
    assert failed.end_time >= failed.start_time

    rollout = await store.get_rollout_by_id(rollout_id=rollout.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"
    assert rollout.end_time is not None
    assert rollout.end_time >= failed.end_time


@pytest.mark.asyncio
async def test_rollout_retry_lifecycle_updates_statuses(
    store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Rollout retry creates new attempts and updates statuses via spans and completions."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    first_attempted = await store.dequeue_rollout()
    assert first_attempted is not None
    assert first_attempted.status == "preparing"

    first_attempt = (await store.query_attempts(rollout.rollout_id))[0]
    await store.add_otel_span(rollout.rollout_id, first_attempt.attempt_id, mock_readable_span)

    # Status should reflect running state after span is recorded
    running_rollout = await store.query_rollouts(status=["running"])
    assert running_rollout and running_rollout[0].rollout_id == rollout.rollout_id

    running_attempts = await store.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"

    # Mark first attempt as failed and requeue rollout
    failed_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=first_attempt.attempt_id,
        status="failed",
    )
    assert failed_attempt.end_time is not None
    await store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")

    attempts_after_failure = await store.query_attempts(rollout.rollout_id)
    assert [a.status for a in attempts_after_failure] == ["failed"]

    retry_attempted = await store.dequeue_rollout()
    assert retry_attempted is not None
    assert retry_attempted.status == "preparing"
    assert retry_attempted.attempt.sequence_id == 2

    latest_pre_span = await store.get_latest_attempt(rollout.rollout_id)
    assert latest_pre_span is not None and latest_pre_span.sequence_id == 2
    assert latest_pre_span.status == "preparing"

    await store.add_otel_span(rollout.rollout_id, retry_attempted.attempt.attempt_id, mock_readable_span)

    latest_running = await store.get_latest_attempt(rollout.rollout_id)
    assert latest_running is not None
    assert latest_running.sequence_id == 2
    assert latest_running.status == "running"

    await store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=retry_attempted.attempt.attempt_id,
        status="succeeded",
    )
    await store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    final_rollout = await store.query_rollouts(status=["succeeded"])
    assert final_rollout and final_rollout[0].rollout_id == rollout.rollout_id

    final_attempts = await store.query_attempts(rollout.rollout_id)
    assert [a.status for a in final_attempts] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_update_nonexistent_attempt(store: InMemoryLightningStore) -> None:
    """Test updating non-existent attempt raises error."""
    rollout = await store.enqueue_rollout(input={"test": "data"})

    with pytest.raises(ValueError, match="No attempts found"):
        await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="nonexistent", status="failed")


# Add Attempt Tests


@pytest.mark.asyncio
async def test_add_attempt_creates_new_attempt(store: InMemoryLightningStore) -> None:
    """Test add_attempt creates a new attempt for existing rollout."""
    # Create a rollout
    rollout = await store.enqueue_rollout(input={"test": "data"})

    # Add first manual attempt
    attempted_rollout = await store.start_attempt(rollout.rollout_id)

    assert attempted_rollout.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.sequence_id == 1
    assert attempted_rollout.attempt.status == "preparing"
    assert attempted_rollout.attempt.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.attempt_id.startswith("attempt-")

    # Verify attempt is stored
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempted_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_add_attempt_increments_sequence_id(store: InMemoryLightningStore) -> None:
    """Test add_attempt correctly increments sequence_id."""
    # Create a rollout and dequeue to create first attempt
    rollout = await store.enqueue_rollout(input={"test": "data"})
    await store.dequeue_rollout()  # Creates attempt with sequence_id=1

    # Add second attempt manually
    attempted_rollout2 = await store.start_attempt(rollout.rollout_id)
    assert attempted_rollout2.attempt.sequence_id == 2

    # Add third attempt manually
    attempted_rollout3 = await store.start_attempt(rollout.rollout_id)
    assert attempted_rollout3.attempt.sequence_id == 3

    # Verify all attempts exist
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3
    assert [a.sequence_id for a in attempts] == [1, 2, 3]


@pytest.mark.asyncio
async def test_add_attempt_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test add_attempt raises error for nonexistent rollout."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store.start_attempt("nonexistent")


@pytest.mark.asyncio
async def test_add_attempt_ignores_max_attempts(store: InMemoryLightningStore) -> None:
    """Test add_attempt ignores max_attempts configuration."""
    # Create rollout with max_attempts=2
    rollout = await store.enqueue_rollout(input={"test": "data"})
    config = RolloutConfig(max_attempts=2)
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Add attempts beyond max_attempts
    attempt1 = await store.start_attempt(rollout.rollout_id)
    attempt2 = await store.start_attempt(rollout.rollout_id)
    attempt3 = await store.start_attempt(rollout.rollout_id)  # Should succeed despite max_attempts=2

    assert attempt1.attempt.sequence_id == 1
    assert attempt2.attempt.sequence_id == 2
    assert attempt3.attempt.sequence_id == 3

    # All attempts should exist
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3


# Latest Attempt Status Propagation Tests


@pytest.mark.asyncio
async def test_status_propagation_only_for_latest_attempt(store: InMemoryLightningStore) -> None:
    """Test that status changes only propagate to rollout when updating latest attempt."""
    rollout = await store.enqueue_rollout(input={"test": "propagation"})

    # Create multiple attempts
    attempt1 = await store.start_attempt(rollout.rollout_id)
    _attempt2 = await store.start_attempt(rollout.rollout_id)
    attempt3 = await store.start_attempt(rollout.rollout_id)  # This is the latest

    # Update attempt1 (not latest) to succeeded
    await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout status should NOT change since attempt1 is not the latest
    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "queuing"  # Should remain unchanged

    # Update attempt3 (latest) to succeeded
    await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt3.attempt.attempt_id, status="succeeded"
    )

    # Now rollout status should change since we updated the latest attempt
    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_status_propagation_with_retry_for_latest_attempt(store: InMemoryLightningStore) -> None:
    """Test retry logic only applies when updating latest attempt."""
    rollout = await store.enqueue_rollout(input={"test": "retry"})
    config = RolloutConfig(max_attempts=3, retry_condition=["failed"])
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Create multiple attempts
    attempt1 = await store.start_attempt(rollout.rollout_id)  # sequence_id=1
    attempt2 = await store.start_attempt(rollout.rollout_id)  # sequence_id=2 (latest)

    # Fail attempt1 (not latest) - should NOT trigger retry
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed")

    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "queuing"  # Should remain unchanged

    # Fail attempt2 (latest) - should trigger retry since sequence_id=2 < max_attempts=3
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed")

    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "requeuing"  # Should be requeued for retry


@pytest.mark.asyncio
async def test_status_propagation_latest_changes_when_new_attempt_added(store: InMemoryLightningStore) -> None:
    """Test that the 'latest attempt' changes as new attempts are added."""
    rollout = await store.enqueue_rollout(input={"test": "latest_changes"})

    # Create first attempt and update it to succeeded
    attempt1 = await store.start_attempt(rollout.rollout_id)
    await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout should be succeeded since attempt1 is latest
    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"

    # Add second attempt (now this becomes latest)
    attempt2 = await store.start_attempt(rollout.rollout_id)

    # Update attempt1 to failed - should NOT affect rollout since it's no longer latest
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed")

    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"  # Should remain unchanged

    # Update attempt2 (now latest) to failed
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed")

    # Now rollout should change since we updated the new latest attempt
    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "failed"


@pytest.mark.asyncio
async def test_status_propagation_update_latest_by_reference(store: InMemoryLightningStore) -> None:
    """Test status propagation when updating latest attempt using 'latest' reference."""
    rollout = await store.enqueue_rollout(input={"test": "latest_ref"})

    # Create multiple attempts
    await store.start_attempt(rollout.rollout_id)
    await store.start_attempt(rollout.rollout_id)
    attempt3 = await store.start_attempt(rollout.rollout_id)  # This is latest

    # Update using "latest" reference
    updated_attempt = await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded")

    # Should have updated attempt3
    assert updated_attempt.attempt_id == attempt3.attempt.attempt_id
    assert updated_attempt.status == "succeeded"

    # Rollout should be updated since we updated the latest attempt
    updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_healthcheck_timeout_behavior(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test that healthcheck detects and handles timeout conditions."""
    # Create rollout with short timeout configuration
    config = RolloutConfig(
        timeout_seconds=0.1, max_attempts=2, retry_condition=["timeout"]  # Very short timeout for testing
    )

    rollout = await store.enqueue_rollout(input={"test": "timeout"})
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue to create an attempt and add span to make it running
    attempted = await store.dequeue_rollout()
    assert attempted is not None
    await store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running
    running_rollouts = await store.query_rollouts(status=["running"])
    assert len(running_rollouts) == 1

    # Wait for timeout to occur
    await asyncio.sleep(0.15)  # Wait longer than timeout_seconds

    # Trigger healthcheck by calling any decorated method
    # Verify the attempt was marked as timeout and rollout was requeued
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].status == "timeout"

    # Since retry_condition includes "timeout" and max_attempts=2, should requeue
    rollout_after = await store.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "requeuing"


@pytest.mark.asyncio
async def test_healthcheck_unresponsive_behavior(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test that healthcheck detects and handles unresponsive conditions."""
    # Create rollout with short unresponsive timeout but no retry for unresponsive
    config = RolloutConfig(
        unresponsive_seconds=0.1,  # Very short unresponsive timeout
        max_attempts=3,
        retry_condition=["timeout"],  # Note: "unresponsive" not in retry_condition
    )

    rollout = await store.enqueue_rollout(input={"test": "unresponsive"})
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue and add span to make it running (this sets last_heartbeat_time)
    attempted = await store.dequeue_rollout()
    assert attempted is not None
    await store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running and has heartbeat
    running_attempts = await store.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"
    assert running_attempts[0].last_heartbeat_time is not None

    # Wait for unresponsive timeout
    await asyncio.sleep(0.15)  # Wait longer than unresponsive_seconds

    # Verify attempt was marked as unresponsive
    attempts_after = await store.query_attempts(rollout.rollout_id)
    assert attempts_after[0].status == "unresponsive"

    # Since "unresponsive" not in retry_condition, rollout should be failed
    rollout_after = await store.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "failed"


# Full Lifecycle Integration Tests


@pytest.mark.asyncio
async def test_full_lifecycle_success(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test successful rollout lifecycle: queue -> prepare -> run -> succeed."""
    # 1. Create task
    rollout = await store.enqueue_rollout(input={"test": "data"}, mode="train")
    assert rollout.status == "queuing"

    # 2. Pop to start processing (creates attempt)
    popped = await store.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.status == "preparing"

    # 3. Add span (transitions to running)
    span = await store.add_otel_span(rollout.rollout_id, attempt.attempt_id, mock_readable_span)
    assert span.sequence_id == 1

    # Check status transitions
    rollouts = await store.query_rollouts(status=["running"])
    assert len(rollouts) == 1

    attempts = await store.query_attempts(rollout.rollout_id)
    assert attempts[0].status == "running"
    assert attempts[0].last_heartbeat_time is not None

    # 4. Complete successfully
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt.attempt_id, status="succeeded")
    await store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Verify final state
    final = (await store.query_rollouts())[0]
    assert final.status == "succeeded"
    assert final.end_time is not None

    final_attempt = await store.get_latest_attempt(rollout.rollout_id)
    assert final_attempt is not None
    assert final_attempt.status == "succeeded"
    assert final_attempt.end_time is not None

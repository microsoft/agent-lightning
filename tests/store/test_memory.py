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
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentlightning.store.base import LightningStoreWatchDog
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import Span
from agentlightning.types import LLM, Attempt, AttemptStatus, PromptTemplate, ResourcesUpdate, RolloutStatus, RolloutV2

# Core CRUD Operations Tests


@pytest.mark.asyncio
async def test_enqueue_rollout_creates_rollout(store: InMemoryLightningStore) -> None:
    """Test that enqueue_rollout creates a properly initialized rollout."""
    sample = {"input": "test_data"}
    metadata = {"key": "value", "number": 42}

    rollout = await store.enqueue_rollout(sample=sample, mode="train", resources_id="res-123", metadata=metadata)

    assert rollout.rollout_id.startswith("rollout-")
    assert rollout.input == sample
    assert rollout.mode == "train"
    assert rollout.resources_id == "res-123"
    assert rollout.metadata == metadata
    assert rollout.status == "queuing"
    assert rollout.start_time is not None


@pytest.mark.asyncio
async def test_query_rollouts_by_status(store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by status."""
    # Create rollouts with different statuses
    r1 = await store.enqueue_rollout(sample={"id": 1})
    r2 = await store.enqueue_rollout(sample={"id": 2})
    r3 = await store.enqueue_rollout(sample={"id": 3})

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
async def test_update_rollout_fields(store: InMemoryLightningStore) -> None:
    """Test updating various rollout fields."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    # Update multiple fields at once
    await store.update_rollout(
        rollout_id=rollout.rollout_id,
        status="running",
        mode="train",
        resources_id="new-resources",
        metadata={"custom_field": "custom_value"},
    )

    # Verify all updates
    updated_rollouts = await store.query_rollouts()
    updated = updated_rollouts[0]
    assert updated.status == "running"
    assert updated.mode == "train"
    assert updated.resources_id == "new-resources"
    assert updated.metadata["custom_field"] == "custom_value"


# Queue Operations Tests


@pytest.mark.asyncio
async def test_dequeue_rollout_skips_non_queuing_status(store: InMemoryLightningStore) -> None:
    """Test that dequeue_rollout skips rollouts that have been updated to non-queuing status."""
    # Add multiple rollouts to the queue
    r1 = await store.enqueue_rollout(sample={"id": 1})
    r2 = await store.enqueue_rollout(sample={"id": 2})
    r3 = await store.enqueue_rollout(sample={"id": 3})

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
        r = await store.enqueue_rollout(sample={"order": i})
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
    rollout = await store.enqueue_rollout(sample={"data": "test"})
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
    r1 = await store.enqueue_rollout(sample={"id": 1})
    assert r1.resources_id == "current"

    # Task with explicit resources_id
    r2 = await store.enqueue_rollout(sample={"id": 2}, resources_id="override")
    assert r2.resources_id == "override"

    # Update resources
    new_prompt = PromptTemplate(resource_type="prompt_template", template="Hi {name}!", engine="f-string")
    update2 = ResourcesUpdate(resources_id="new", resources={"greeting": new_prompt})
    await store.update_resources(update2.resources_id, update2.resources)

    # New task gets new resources
    r3 = await store.enqueue_rollout(sample={"id": 3})
    assert r3.resources_id == "new"


# Span Management Tests


@pytest.mark.asyncio
async def test_span_sequence_generation(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test automatic sequence ID generation for spans."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    # Pop to create an attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # First span gets sequence_id 1
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 1

    span1 = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span1.sequence_id == 1

    # Next span gets sequence_id 2
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 2

    span2 = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span2.sequence_id == 2

    # Different attempt starts from 1
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-2")
    assert seq_id == 1


@pytest.mark.asyncio
async def test_span_with_explicit_sequence_id(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test providing explicit sequence_id to spans."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    # Pop to create an attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add span with explicit sequence_id
    span = await store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span, sequence_id=100)
    assert span.sequence_id == 100

    # Next auto-generated should not be affected by explicit IDs
    next_seq = await store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert next_seq == 2  # Only one span added so far


@pytest.mark.asyncio
async def test_query_spans_by_attempt(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test querying spans filtered by attempt_id."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
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
    rollout = await store.enqueue_rollout(sample={"test": "data"})

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
    rollout = await store.enqueue_rollout(sample={"test": "data"})

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
    r1 = await store.enqueue_rollout(sample={"id": 1})
    r2 = await store.enqueue_rollout(sample={"id": 2})
    _r3 = await store.enqueue_rollout(sample={"id": 3})

    # Start waiting for r1 and r2
    async def wait_for_completion() -> List[RolloutV2]:
        return await store.wait_for_rollouts([r1.rollout_id, r2.rollout_id], timeout=5.0)

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
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    start = time.time()
    completed = await store.wait_for_rollouts([rollout.rollout_id], timeout=0.1)
    elapsed = time.time() - start

    assert elapsed < 0.2  # Should timeout quickly
    assert len(completed) == 0  # No completions


# Concurrent Access Tests


@pytest.mark.asyncio
async def test_concurrent_task_addition(store: InMemoryLightningStore) -> None:
    """Test adding tasks concurrently."""

    async def enqueue_rollout(index: int) -> RolloutV2:
        return await store.enqueue_rollout(sample={"index": index})

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
        await store.enqueue_rollout(sample={"index": i})

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
    await store.enqueue_rollout(sample={"test": "data"})
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
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    spans = await store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_wait_for_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test waiting for non-existent rollout handles gracefully."""
    completed = await store.wait_for_rollouts(["nonexistent"], timeout=0.1)
    assert len(completed) == 0


# Attempt Management Tests


@pytest.mark.asyncio
async def test_query_attempts(store: InMemoryLightningStore) -> None:
    """Test querying attempts for a rollout."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})

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
    rollout = await store.enqueue_rollout(sample={"test": "data"})

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
    rollout = await store.enqueue_rollout(sample={"test": "data"})
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
    assert updated.metadata["custom"] == "value"


@pytest.mark.asyncio
async def test_update_latest_attempt(store: InMemoryLightningStore) -> None:
    """Test updating latest attempt using 'latest' identifier."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    await store.dequeue_rollout()

    # Update using 'latest'
    updated = await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded")

    assert updated.status == "succeeded"
    assert updated.end_time is not None  # Should auto-set end_time


@pytest.mark.asyncio
async def test_update_nonexistent_attempt(store: InMemoryLightningStore) -> None:
    """Test updating non-existent attempt raises error."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    with pytest.raises(ValueError, match="No attempts found"):
        await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="nonexistent", status="failed")


# Full Lifecycle Integration Tests


@pytest.mark.asyncio
async def test_full_lifecycle_success(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test successful rollout lifecycle: queue -> prepare -> run -> succeed."""
    # 1. Create task
    rollout = await store.enqueue_rollout(sample={"test": "data"}, mode="train")
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

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
import sys
import time
from typing import List, Optional, cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from agentlightning.store.memory import InMemoryLightningStore, estimate_model_size
from agentlightning.types import (
    LLM,
    AttemptedRollout,
    Event,
    Link,
    PromptTemplate,
    Resource,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    Span,
    SpanContext,
    TraceStatus,
)

# Core CRUD Operations Tests


@pytest.mark.asyncio
async def test_enqueue_rollout_creates_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test that enqueue_rollout creates a properly initialized rollout."""
    sample = {"input": "test_data"}
    metadata = {"key": "value", "number": 42}

    rollout = await inmemory_store.enqueue_rollout(
        input=sample, mode="train", resources_id="res-123", metadata=metadata
    )

    assert rollout.rollout_id.startswith("ro-")
    assert rollout.input == sample
    assert rollout.mode == "train"
    assert rollout.resources_id == "res-123"
    assert rollout.metadata == metadata
    assert rollout.status == "queuing"
    assert rollout.start_time is not None


@pytest.mark.asyncio
async def test_add_rollout_initializes_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test that add_rollout immediately tracks a preparing attempt."""
    sample = {"payload": "value"}

    attempt_rollout = await inmemory_store.start_rollout(input=sample, mode="val", resources_id="res-add")

    assert attempt_rollout.status == "preparing"
    assert attempt_rollout.rollout_id.startswith("ro-")
    assert attempt_rollout.attempt.attempt_id.startswith("at-")
    assert attempt_rollout.attempt.sequence_id == 1
    assert attempt_rollout.attempt.status == "preparing"

    stored = await inmemory_store.query_rollouts(status=["preparing"])
    assert len(stored) == 1
    assert stored[0].rollout_id == attempt_rollout.rollout_id
    assert stored[0].resources_id == "res-add"

    attempts = await inmemory_store.query_attempts(attempt_rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempt_rollout.attempt.attempt_id

    latest_attempt = await inmemory_store.get_latest_attempt(attempt_rollout.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == attempt_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_query_rollouts_by_status(inmemory_store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by status."""
    # Create rollouts with different statuses
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    # Modify statuses
    await inmemory_store.dequeue_rollout()  # r1 becomes "preparing"
    await inmemory_store.update_rollout(rollout_id=r2.rollout_id, status="failed")
    # r3 remains "queuing"

    # Test various queries
    all_rollouts = await inmemory_store.query_rollouts()
    assert len(all_rollouts) == 3

    queuing = await inmemory_store.query_rollouts(status=["queuing"])
    assert len(queuing) == 1
    assert queuing[0].rollout_id == r3.rollout_id

    preparing = await inmemory_store.query_rollouts(status=["preparing"])
    assert len(preparing) == 1
    assert preparing[0].rollout_id == r1.rollout_id

    finished = await inmemory_store.query_rollouts(status=["failed", "succeeded"])
    assert len(finished) == 1
    assert finished[0].rollout_id == r2.rollout_id

    # Empty status list
    none = await inmemory_store.query_rollouts(status=[])
    assert len(none) == 0


@pytest.mark.asyncio
async def test_get_rollout_by_id(inmemory_store: InMemoryLightningStore) -> None:
    """Test retrieving rollouts by their ID."""
    # Test getting non-existent rollout
    rollout = await inmemory_store.get_rollout_by_id("nonexistent")
    assert rollout is None

    # Create a rollout
    created = await inmemory_store.enqueue_rollout(input={"test": "data"}, mode="train")

    # Retrieve by ID
    retrieved = await inmemory_store.get_rollout_by_id(created.rollout_id)
    assert retrieved is not None
    assert retrieved.rollout_id == created.rollout_id
    assert retrieved.input == created.input
    assert retrieved.mode == created.mode
    assert retrieved.status == created.status

    # Update rollout and verify changes are reflected
    await inmemory_store.update_rollout(rollout_id=created.rollout_id, status="running")
    updated = await inmemory_store.get_rollout_by_id(created.rollout_id)
    assert updated is not None
    assert updated.status == "running"


@pytest.mark.asyncio
async def test_query_rollouts_by_rollout_ids(inmemory_store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by rollout IDs."""
    # Create multiple rollouts
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    # Query by specific IDs
    selected = await inmemory_store.query_rollouts(rollout_ids=[r1.rollout_id, r3.rollout_id])
    assert len(selected) == 2
    selected_ids = {r.rollout_id for r in selected}
    assert selected_ids == {r1.rollout_id, r3.rollout_id}

    # Query by single ID
    single = await inmemory_store.query_rollouts(rollout_ids=[r2.rollout_id])
    assert len(single) == 1
    assert single[0].rollout_id == r2.rollout_id

    # Query by non-existent ID
    none = await inmemory_store.query_rollouts(rollout_ids=["nonexistent"])
    assert len(none) == 0

    # Combine with status filter
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    await inmemory_store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    filtered = await inmemory_store.query_rollouts(
        rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], status=["succeeded", "queuing"]
    )
    assert len(filtered) == 2
    filtered_ids = {r.rollout_id for r in filtered}
    assert filtered_ids == {r1.rollout_id, r3.rollout_id}  # r1 succeeded, r3 still queuing


@pytest.mark.asyncio
async def test_update_rollout_fields(inmemory_store: InMemoryLightningStore) -> None:
    """Test updating various rollout fields."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # Update multiple fields at once including config
    config = RolloutConfig(
        timeout_seconds=60.0, unresponsive_seconds=30.0, max_attempts=3, retry_condition=["timeout", "unresponsive"]
    )
    await inmemory_store.update_rollout(
        rollout_id=rollout.rollout_id,
        status="running",
        mode="train",
        resources_id="new-resources",
        config=config,
        metadata={"custom_field": "custom_value"},
    )

    # Verify all updates
    updated_rollouts = await inmemory_store.query_rollouts()
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
async def test_rollout_config_functionality(inmemory_store: InMemoryLightningStore) -> None:
    """Test RolloutConfig controls retry and timeout behavior."""
    # Create rollout with specific retry configuration
    config = RolloutConfig(
        timeout_seconds=30.0,
        unresponsive_seconds=15.0,
        max_attempts=2,
        retry_condition=["timeout", "unresponsive", "failed"],
    )

    rollout = await inmemory_store.enqueue_rollout(input={"test": "retry"})
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Verify config is stored
    stored = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert stored is not None
    assert stored.config.timeout_seconds == 30.0
    assert stored.config.max_attempts == 2
    assert "failed" in stored.config.retry_condition

    # Test that different rollouts can have different configs
    config2 = RolloutConfig(timeout_seconds=120.0, max_attempts=5, retry_condition=["timeout"])

    rollout2 = await inmemory_store.enqueue_rollout(input={"test": "different_config"})
    await inmemory_store.update_rollout(rollout_id=rollout2.rollout_id, config=config2)

    stored2 = await inmemory_store.get_rollout_by_id(rollout2.rollout_id)
    assert stored2 is not None
    assert stored2.config.timeout_seconds == 120.0
    assert stored2.config.max_attempts == 5
    assert stored2.config.retry_condition == ["timeout"]

    # Verify first rollout config unchanged
    stored1_again = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert stored1_again is not None
    assert stored1_again.config.timeout_seconds == 30.0


# Queue Operations Tests


@pytest.mark.asyncio
async def test_dequeue_rollout_skips_non_queuing_status(inmemory_store: InMemoryLightningStore) -> None:
    """Test that dequeue_rollout skips rollouts that have been updated to non-queuing status."""
    # Add multiple rollouts to the queue
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    # Update r1 to succeeded status while it's still in the queue
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Update r2 to failed status
    await inmemory_store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # r3 should still be in queuing status

    # Pop should skip r1 and r2 (both non-queuing) and return r3
    popped = await inmemory_store.dequeue_rollout()
    assert popped is not None
    assert popped.rollout_id == r3.rollout_id
    assert popped.status == "preparing"
    assert popped.input["id"] == 3

    # Second pop should return None since no queuing rollouts remain
    popped2 = await inmemory_store.dequeue_rollout()
    assert popped2 is None

    # Verify r1 and r2 are still in their non-queuing states
    all_rollouts = await inmemory_store.query_rollouts()
    rollout_statuses = {r.rollout_id: r.status for r in all_rollouts}
    assert rollout_statuses[r1.rollout_id] == "succeeded"
    assert rollout_statuses[r2.rollout_id] == "failed"
    assert rollout_statuses[r3.rollout_id] == "preparing"


@pytest.mark.asyncio
async def test_fifo_ordering(inmemory_store: InMemoryLightningStore) -> None:
    """Test that queue maintains FIFO order."""
    rollouts: List[Rollout] = []
    for i in range(5):
        r = await inmemory_store.enqueue_rollout(input={"order": i})
        rollouts.append(r)

    # Pop all and verify order
    for i in range(5):
        popped = await inmemory_store.dequeue_rollout()
        assert popped is not None
        assert popped.rollout_id == rollouts[i].rollout_id
        assert popped.input["order"] == i
        assert popped.status == "preparing"


@pytest.mark.asyncio
async def test_pop_empty_queue(inmemory_store: InMemoryLightningStore) -> None:
    """Test popping from empty queue returns None."""
    result = await inmemory_store.dequeue_rollout()
    assert result is None

    # Multiple pops should all return None
    for _ in range(3):
        assert await inmemory_store.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeue_mechanism(inmemory_store: InMemoryLightningStore) -> None:
    """Test requeuing puts rollout back in queue."""
    rollout = await inmemory_store.enqueue_rollout(input={"data": "test"})
    original_id = rollout.rollout_id

    # Pop and verify it's not in queue
    popped = await inmemory_store.dequeue_rollout()
    assert popped is not None
    assert await inmemory_store.dequeue_rollout() is None

    # Requeue it
    await inmemory_store.update_rollout(rollout_id=original_id, status="requeuing")

    # Should be back in queue
    requeued = await inmemory_store.dequeue_rollout()
    assert requeued is not None
    assert requeued.rollout_id == original_id
    assert requeued.status == "preparing"  # Changes when popped
    # Check that a new attempt was created
    attempts = await inmemory_store.query_attempts(requeued.rollout_id)
    assert len(attempts) == 2  # First attempt plus requeued attempt

    latest_attempt = await inmemory_store.get_latest_attempt(requeued.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "preparing"
    assert latest_attempt.sequence_id == 2


# Resource Management Tests


@pytest.mark.asyncio
async def test_add_resources_generates_id_and_stores(inmemory_store: InMemoryLightningStore) -> None:
    """Test that add_resources generates a resources_id and stores the resources."""
    # Initially no resources
    assert await inmemory_store.get_latest_resources() is None

    # Add resources using add_resources (auto-generates ID)
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")

    resources_update = await inmemory_store.add_resources({"main_llm": llm, "greeting": prompt})

    # Verify resources_id was auto-generated with correct prefix
    assert resources_update.resources_id.startswith("rs-")
    assert len(resources_update.resources_id) == 15  # "rs-" + 12 char hash

    # Verify resources were stored correctly
    assert isinstance(resources_update.resources["main_llm"], LLM)
    assert resources_update.resources["main_llm"].model == "test-model"
    assert isinstance(resources_update.resources["greeting"], PromptTemplate)
    assert resources_update.resources["greeting"].template == "Hello {name}!"

    # Verify it's set as latest
    latest = await inmemory_store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == resources_update.resources_id
    assert latest.resources["main_llm"].model == "test-model"  # type: ignore

    # Verify we can retrieve by ID
    retrieved = await inmemory_store.get_resources_by_id(resources_update.resources_id)
    assert retrieved is not None
    assert retrieved.resources_id == resources_update.resources_id


@pytest.mark.asyncio
async def test_add_resources_multiple_times_generates_unique_ids(inmemory_store: InMemoryLightningStore) -> None:
    """Test that multiple calls to add_resources generate unique IDs."""
    llm1 = LLM(resource_type="llm", endpoint="http://localhost:8080", model="model-v1")
    llm2 = LLM(resource_type="llm", endpoint="http://localhost:8080", model="model-v2")

    update1 = await inmemory_store.add_resources({"llm": llm1})
    update2 = await inmemory_store.add_resources({"llm": llm2})

    # IDs should be different
    assert update1.resources_id != update2.resources_id
    assert update1.resources_id.startswith("rs-")
    assert update2.resources_id.startswith("rs-")

    # Both should be retrievable
    retrieved1 = await inmemory_store.get_resources_by_id(update1.resources_id)
    retrieved2 = await inmemory_store.get_resources_by_id(update2.resources_id)
    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.resources["llm"].model == "model-v1"  # type: ignore
    assert retrieved2.resources["llm"].model == "model-v2"  # type: ignore

    # Latest should be the second one
    latest = await inmemory_store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == update2.resources_id


@pytest.mark.asyncio
async def test_resource_lifecycle(inmemory_store: InMemoryLightningStore) -> None:
    """Test adding, updating, and retrieving resources."""
    # Initially no resources
    assert await inmemory_store.get_latest_resources() is None
    assert await inmemory_store.get_resources_by_id("any-id") is None

    # Add first version with proper LLM resource
    llm_v1 = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model-v1",
        sampling_parameters={"temperature": 0.7},
    )
    update = await inmemory_store.update_resources("v1", {"main_llm": llm_v1})
    assert update.resources_id == "v1"

    latest = await inmemory_store.get_latest_resources()
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
    v2 = await inmemory_store.update_resources("v2", {"main_llm": llm_v2})
    assert v2.resources_id == "v2"
    assert isinstance(v2.resources["main_llm"], LLM)
    assert v2.resources["main_llm"].model == "test-model-v2"

    # Latest should be v2
    latest = await inmemory_store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v2"

    # Can still retrieve v1
    old = await inmemory_store.get_resources_by_id("v1")
    assert old is not None
    assert isinstance(old.resources["main_llm"], LLM)
    assert old.resources["main_llm"].model == "test-model-v1"


@pytest.mark.asyncio
async def test_task_inherits_latest_resources(inmemory_store: InMemoryLightningStore) -> None:
    """Test that new tasks inherit latest resources_id if not specified."""
    # Set up resources with proper PromptTemplate
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")
    update = ResourcesUpdate(resources_id="current", resources={"greeting": prompt})
    await inmemory_store.update_resources(update.resources_id, update.resources)

    # Task without explicit resources_id
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    assert r1.resources_id == "current"

    # Task with explicit resources_id
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2}, resources_id="override")
    assert r2.resources_id == "override"

    # Update resources
    new_prompt = PromptTemplate(resource_type="prompt_template", template="Hi {name}!", engine="f-string")
    update2 = ResourcesUpdate(resources_id="new", resources={"greeting": new_prompt})
    await inmemory_store.update_resources(update2.resources_id, update2.resources)

    # New task gets new resources
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})
    assert r3.resources_id == "new"


# Span Management Tests


@pytest.mark.asyncio
async def test_span_sequence_generation(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test automatic sequence ID generation for spans."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # First span gets sequence_id 1
    seq_id = await inmemory_store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 1

    span1 = await inmemory_store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span1.sequence_id == 2

    # Next span gets sequence_id 3
    seq_id = await inmemory_store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert seq_id == 3

    span2 = await inmemory_store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)
    assert span2.sequence_id == 4

    # Different attempt reuses the same rollout_id
    seq_id = await inmemory_store.get_next_span_sequence_id(rollout.rollout_id, "attempt-does-not-exist")
    assert seq_id == 5


@pytest.mark.asyncio
async def test_span_with_explicit_sequence_id(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test providing explicit sequence_id to spans."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    # Pop to create an attempt
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add span with explicit sequence_id
    span = await inmemory_store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span, sequence_id=100)
    assert span.sequence_id == 100

    next_seq = await inmemory_store.get_next_span_sequence_id(rollout.rollout_id, attempt_id)
    assert next_seq == 101


@pytest.mark.asyncio
async def test_query_spans_by_attempt(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test querying spans filtered by attempt_id."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    # Pop to create first attempt
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt1_id = attempts[0].attempt_id

    # Add spans for first attempt
    for _ in range(2):
        await inmemory_store.add_otel_span(rollout.rollout_id, attempt1_id, mock_readable_span)

    # Simulate requeue and create second attempt
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt2_id = attempts[1].attempt_id

    # Add spans for second attempt
    for _ in range(3):
        await inmemory_store.add_otel_span(rollout.rollout_id, attempt2_id, mock_readable_span)

    # Query all spans
    all_spans = await inmemory_store.query_spans(rollout.rollout_id)
    assert len(all_spans) == 5

    # Query specific attempt
    attempt1_spans = await inmemory_store.query_spans(rollout.rollout_id, attempt_id=attempt1_id)
    assert len(attempt1_spans) == 2
    assert all(s.attempt_id == attempt1_id for s in attempt1_spans)

    # Query latest attempt
    latest_spans = await inmemory_store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert len(latest_spans) == 3
    assert all(s.attempt_id == attempt2_id for s in latest_spans)

    # Query non-existent attempt
    no_spans = await inmemory_store.query_spans(rollout.rollout_id, attempt_id="nonexistent")
    assert len(no_spans) == 0


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
    context = SpanContext(trace_id="trace", span_id="parent", is_remote=False, trace_state={})
    event = Event(name="step", attributes={"detail": "value"}, timestamp=1.0)
    link = Link(context=context, attributes=None)
    resource = Resource(attributes={"service.name": "unit"}, schema_url="schema")

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

    trace_state_values = context.trace_state.values() if context.trace_state is not None else ()
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
async def test_span_triggers_status_transition(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Test that adding first span transitions rollout from preparing to running."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # Pop to set status to preparing and create attempt
    popped = await inmemory_store.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Verify status in store
    rollouts = await inmemory_store.query_rollouts(status=["preparing"])
    assert len(rollouts) == 1

    # Get the attempt
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt_id = attempts[0].attempt_id

    # Add first span
    await inmemory_store.add_otel_span(rollout.rollout_id, attempt_id, mock_readable_span)

    # Status should transition to running
    rollouts = await inmemory_store.query_rollouts(status=["running"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id


# Rollout Lifecycle Tests


@pytest.mark.asyncio
async def test_completion_sets_end_time(inmemory_store: InMemoryLightningStore) -> None:
    """Test that completing a rollout sets end_time."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # Initially no end_time
    assert rollout.end_time is None

    # Complete as succeeded
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    completed_rollouts = await inmemory_store.query_rollouts()
    completed = completed_rollouts[0]
    assert completed.status == "succeeded"
    assert completed.end_time is not None
    assert completed.end_time > completed.start_time


@pytest.mark.asyncio
async def test_wait_for_rollouts(inmemory_store: InMemoryLightningStore) -> None:
    """Test waiting for rollout completion."""
    # Add multiple rollouts
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    _r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    # Start waiting for r1 and r2
    async def wait_for_completion() -> List[Rollout]:
        return await inmemory_store.wait_for_rollouts(rollout_ids=[r1.rollout_id, r2.rollout_id], timeout=5.0)

    wait_task = asyncio.create_task(wait_for_completion())
    await asyncio.sleep(0.01)  # Let wait task start

    # Complete r1
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Complete r2
    await inmemory_store.update_rollout(rollout_id=r2.rollout_id, status="failed")

    # Get results
    completed = await wait_task
    assert len(completed) == 2
    assert {r.rollout_id for r in completed} == {r1.rollout_id, r2.rollout_id}
    assert {r.status for r in completed} == {"succeeded", "failed"}


@pytest.mark.asyncio
async def test_wait_timeout(inmemory_store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts timeout behavior."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    start = time.time()
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0.1)
    elapsed = time.time() - start

    assert elapsed < 0.2  # Should timeout quickly
    assert len(completed) == 0  # No completions


@pytest.mark.asyncio
async def test_wait_with_timeout_none_polling(inmemory_store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts with timeout=None uses polling and can be cancelled."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "indefinite"})

    async def wait_indefinitely():
        return await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None)

    # Start waiting with timeout=None
    wait_task = asyncio.create_task(wait_indefinitely())

    # Give it a moment to start polling
    await asyncio.sleep(0.1)

    # Complete the rollout
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # The wait should complete now
    completed = await asyncio.wait_for(wait_task, timeout=1.0)
    assert len(completed) == 1
    assert completed[0].rollout_id == rollout.rollout_id
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_with_timeout_none_can_be_cancelled(inmemory_store: InMemoryLightningStore) -> None:
    """Test that wait_for_rollouts with timeout=None can be cancelled cleanly."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "cancel"})

    async def wait_indefinitely():
        return await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None)

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
async def test_wait_with_timeout_zero(inmemory_store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts with timeout=0 returns immediately."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "zero"})

    start = time.time()
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0)
    elapsed = time.time() - start

    # Should return almost immediately
    assert elapsed < 0.05
    assert len(completed) == 0


@pytest.mark.asyncio
async def test_wait_with_already_completed_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts returns immediately for already completed rollouts."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "already_done"})

    # Complete it first
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Wait should return immediately without blocking
    start = time.time()
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=5.0)
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be instant
    assert len(completed) == 1
    assert completed[0].rollout_id == rollout.rollout_id
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_multiple_rollouts_different_completion_times(inmemory_store: InMemoryLightningStore) -> None:
    """Test waiting for multiple rollouts that complete at different times."""
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    async def wait_for_all():
        return await inmemory_store.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], timeout=2.0
        )

    wait_task = asyncio.create_task(wait_for_all())

    # Complete them at different times
    await asyncio.sleep(0.05)
    await inmemory_store.update_rollout(rollout_id=r2.rollout_id, status="succeeded")

    await asyncio.sleep(0.05)
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="failed")

    await asyncio.sleep(0.05)
    await inmemory_store.update_rollout(rollout_id=r3.rollout_id, status="succeeded")

    # All should be collected
    completed = await wait_task
    assert len(completed) == 3
    completed_ids = {r.rollout_id for r in completed}
    assert completed_ids == {r1.rollout_id, r2.rollout_id, r3.rollout_id}


@pytest.mark.asyncio
async def test_wait_partial_completion_on_timeout(inmemory_store: InMemoryLightningStore) -> None:
    """Test that wait_for_rollouts returns partial results when timeout occurs."""
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})
    r2 = await inmemory_store.enqueue_rollout(input={"id": 2})
    r3 = await inmemory_store.enqueue_rollout(input={"id": 3})

    async def wait_with_short_timeout():
        return await inmemory_store.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, r2.rollout_id, r3.rollout_id], timeout=0.2
        )

    wait_task = asyncio.create_task(wait_with_short_timeout())

    # Only complete one before timeout
    await asyncio.sleep(0.05)
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Wait for timeout
    completed = await wait_task

    # Should only get r1
    assert len(completed) == 1
    assert completed[0].rollout_id == r1.rollout_id


@pytest.mark.asyncio
async def test_wait_concurrent_waiters_on_same_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test multiple concurrent waiters on the same rollout."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "concurrent"})

    async def wait_for_completion():
        return await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=2.0)

    # Start multiple waiters concurrently
    wait_tasks = [asyncio.create_task(wait_for_completion()) for _ in range(5)]

    await asyncio.sleep(0.05)

    # Complete the rollout
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # All waiters should complete
    results = await asyncio.gather(*wait_tasks)

    # Each waiter should get the completed rollout
    for completed in results:
        assert len(completed) == 1
        assert completed[0].rollout_id == rollout.rollout_id
        assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_nonexistent_rollout_with_finite_timeout(inmemory_store: InMemoryLightningStore) -> None:
    """Test waiting for non-existent rollout with finite timeout."""
    start = time.time()
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=["nonexistent"], timeout=0.1)
    elapsed = time.time() - start

    # Should timeout quickly (not wait indefinitely)
    assert elapsed < 0.2
    assert len(completed) == 0


@pytest.mark.asyncio
async def test_wait_mixed_existing_and_nonexistent_rollouts(inmemory_store: InMemoryLightningStore) -> None:
    """Test waiting for mix of existing and non-existent rollouts."""
    r1 = await inmemory_store.enqueue_rollout(input={"id": 1})

    async def wait_for_mixed():
        return await inmemory_store.wait_for_rollouts(
            rollout_ids=[r1.rollout_id, "nonexistent1", "nonexistent2"], timeout=0.5
        )

    wait_task = asyncio.create_task(wait_for_mixed())

    await asyncio.sleep(0.05)
    await inmemory_store.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    completed = await wait_task

    # Should only get the existing, completed rollout
    assert len(completed) == 1
    assert completed[0].rollout_id == r1.rollout_id


@pytest.mark.asyncio
async def test_wait_event_set_before_wait_starts(inmemory_store: InMemoryLightningStore) -> None:
    """Test that waiting on an already-set event returns immediately."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "early_complete"})

    # Complete it before waiting
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Now start waiting - should return immediately
    start = time.time()
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=10.0)
    elapsed = time.time() - start

    assert elapsed < 0.05  # Should be instant
    assert len(completed) == 1
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_wait_polling_interval_with_timeout_none(inmemory_store: InMemoryLightningStore) -> None:
    """Test that timeout=None polling doesn't busy-wait (uses reasonable intervals)."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "polling"})

    start = time.time()

    async def wait_and_complete():
        # Start waiting with timeout=None
        wait_task = asyncio.create_task(
            inmemory_store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=None)
        )

        # Wait for 0.5 seconds to let polling happen
        await asyncio.sleep(0.5)

        # Complete the rollout
        await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

        return await wait_task

    completed = await wait_and_complete()
    elapsed = time.time() - start

    # Should complete after ~0.5s (when we set the event)
    assert 0.4 < elapsed < 0.7
    assert len(completed) == 1
    assert completed[0].status == "succeeded"


# Concurrent Access Tests


@pytest.mark.asyncio
async def test_concurrent_task_addition(inmemory_store: InMemoryLightningStore) -> None:
    """Test adding tasks concurrently."""

    async def enqueue_rollout(index: int) -> Rollout:
        return await inmemory_store.enqueue_rollout(input={"index": index})

    # Add 50 tasks concurrently
    tasks = [enqueue_rollout(i) for i in range(50)]
    rollouts = await asyncio.gather(*tasks)

    # All should succeed with unique IDs
    assert len(rollouts) == 50
    ids = [r.rollout_id for r in rollouts]
    assert len(set(ids)) == 50

    # All should be in store
    all_rollouts = await inmemory_store.query_rollouts()
    assert len(all_rollouts) == 50


@pytest.mark.asyncio
async def test_concurrent_pop_operations(inmemory_store: InMemoryLightningStore) -> None:
    """Test concurrent popping ensures each rollout is popped once."""
    # Add 20 tasks
    for i in range(20):
        await inmemory_store.enqueue_rollout(input={"index": i})

    async def pop_task() -> Rollout | None:
        return await inmemory_store.dequeue_rollout()

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
async def test_concurrent_span_additions(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test concurrent span additions maintain consistency."""
    await inmemory_store.enqueue_rollout(input={"test": "data"})
    rollout = await inmemory_store.dequeue_rollout()  # Create an attempt
    assert rollout is not None

    async def add_span(index: int) -> Span:
        return await inmemory_store.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, mock_readable_span)

    # Add 30 spans concurrently
    tasks = [add_span(i) for i in range(30)]
    spans = await asyncio.gather(*tasks)

    # All should have unique sequence IDs
    seq_ids = [s.sequence_id for s in spans]
    assert len(set(seq_ids)) == 30
    assert set(seq_ids) == set(range(1, 31))


@pytest.mark.asyncio
async def test_concurrent_resource_updates(inmemory_store: InMemoryLightningStore) -> None:
    """Test concurrent resource updates are atomic."""

    async def update_resource(ver: int) -> None:
        llm = LLM(
            resource_type="llm",
            endpoint=f"http://localhost:808{ver % 10}",
            model=f"model-v{ver}",
            sampling_parameters={"temperature": 0.5 + ver * 0.01},
        )
        update = ResourcesUpdate(resources_id=f"v{ver}", resources={"llm": llm})
        await inmemory_store.update_resources(update.resources_id, update.resources)

    # Update concurrently
    tasks = [update_resource(i) for i in range(50)]
    await asyncio.gather(*tasks)

    # Latest should be one of the versions
    latest = await inmemory_store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id.startswith("v")

    # All versions should be stored
    for i in range(50):
        res = await inmemory_store.get_resources_by_id(f"v{i}")
        assert res is not None
        assert isinstance(res.resources["llm"], LLM)
        assert res.resources["llm"].model == f"model-v{i}"


# Error Handling Tests


@pytest.mark.asyncio
async def test_update_nonexistent_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test updating non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await inmemory_store.update_rollout(rollout_id="nonexistent", status="failed")


@pytest.mark.asyncio
async def test_add_span_without_rollout(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test adding span to non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await inmemory_store.add_otel_span("nonexistent", "attempt-1", mock_readable_span)


@pytest.mark.asyncio
async def test_add_span_with_missing_attempt(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test adding span with an unknown attempt_id raises a helpful error."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    # Create a valid attempt to ensure rollout exists in store
    await inmemory_store.dequeue_rollout()

    invalid_span = Span.from_opentelemetry(
        mock_readable_span,
        rollout_id=rollout.rollout_id,
        attempt_id="attempt-missing",
        sequence_id=1,
    )

    with pytest.raises(ValueError, match="Attempt attempt-missing not found"):
        await inmemory_store.add_span(invalid_span)


@pytest.mark.asyncio
async def test_query_empty_spans(inmemory_store: InMemoryLightningStore) -> None:
    """Test querying spans for non-existent rollout returns empty."""
    spans = await inmemory_store.query_spans("nonexistent")
    assert spans == []

    # With attempt_id
    spans = await inmemory_store.query_spans("nonexistent", attempt_id="attempt-1")
    assert spans == []

    # With latest
    spans = await inmemory_store.query_spans("nonexistent", attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_query_latest_with_no_spans(inmemory_store: InMemoryLightningStore) -> None:
    """Test querying 'latest' attempt when no spans exist."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    spans = await inmemory_store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_wait_for_nonexistent_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test waiting for non-existent rollout handles gracefully."""
    completed = await inmemory_store.wait_for_rollouts(rollout_ids=["nonexistent"], timeout=0.1)
    assert len(completed) == 0


# Attempt Management Tests


@pytest.mark.asyncio
async def test_query_attempts(inmemory_store: InMemoryLightningStore) -> None:
    """Test querying attempts for a rollout."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # Initially no attempts
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 0

    # Pop creates first attempt
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].sequence_id == 1
    assert attempts[0].status == "preparing"

    # Requeue and pop creates second attempt
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await inmemory_store.dequeue_rollout()
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 2
    assert attempts[0].sequence_id == 1
    assert attempts[1].sequence_id == 2


@pytest.mark.asyncio
async def test_get_latest_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test getting the latest attempt."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # No attempts initially
    latest = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert latest is None

    # Create first attempt
    await inmemory_store.dequeue_rollout()
    latest = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 1

    # Create second attempt
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")
    await inmemory_store.dequeue_rollout()
    latest = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert latest is not None
    assert latest.sequence_id == 2


@pytest.mark.asyncio
async def test_update_attempt_fields(inmemory_store: InMemoryLightningStore) -> None:
    """Test updating attempt fields."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    await inmemory_store.dequeue_rollout()

    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Update various fields
    updated = await inmemory_store.update_attempt(
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
async def test_update_latest_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test updating latest attempt using 'latest' identifier."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    await inmemory_store.dequeue_rollout()

    # Update using 'latest'
    updated = await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded"
    )

    assert updated.status == "succeeded"
    assert updated.end_time is not None  # Should auto-set end_time


@pytest.mark.asyncio
async def test_update_attempt_sets_end_time_for_terminal_status(inmemory_store: InMemoryLightningStore) -> None:
    """Terminal attempt statuses set end_time while in-progress statuses don't."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    await inmemory_store.dequeue_rollout()

    attempt = (await inmemory_store.query_attempts(rollout.rollout_id))[0]
    assert attempt.end_time is None

    running = await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="running",
    )
    assert running.status == "running"
    assert running.end_time is None

    failed = await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=attempt.attempt_id,
        status="failed",
    )
    assert failed.status == "failed"
    assert failed.end_time is not None
    assert failed.end_time >= failed.start_time

    rollout = await inmemory_store.get_rollout_by_id(rollout_id=rollout.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"
    assert rollout.end_time is not None
    assert rollout.end_time >= failed.end_time


@pytest.mark.asyncio
async def test_rollout_retry_lifecycle_updates_statuses(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Rollout retry creates new attempts and updates statuses via spans and completions."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    first_attempted = await inmemory_store.dequeue_rollout()
    assert first_attempted is not None
    assert first_attempted.status == "preparing"

    first_attempt = (await inmemory_store.query_attempts(rollout.rollout_id))[0]
    await inmemory_store.add_otel_span(rollout.rollout_id, first_attempt.attempt_id, mock_readable_span)

    # Status should reflect running state after span is recorded
    running_rollout = await inmemory_store.query_rollouts(status=["running"])
    assert running_rollout and running_rollout[0].rollout_id == rollout.rollout_id

    running_attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"

    # Mark first attempt as failed and requeue rollout
    failed_attempt = await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=first_attempt.attempt_id,
        status="failed",
    )
    assert failed_attempt.end_time is not None
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="requeuing")

    attempts_after_failure = await inmemory_store.query_attempts(rollout.rollout_id)
    assert [a.status for a in attempts_after_failure] == ["failed"]

    retry_attempted = await inmemory_store.dequeue_rollout()
    assert retry_attempted is not None
    assert retry_attempted.status == "preparing"
    assert retry_attempted.attempt.sequence_id == 2

    latest_pre_span = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert latest_pre_span is not None and latest_pre_span.sequence_id == 2
    assert latest_pre_span.status == "preparing"

    await inmemory_store.add_otel_span(rollout.rollout_id, retry_attempted.attempt.attempt_id, mock_readable_span)

    latest_running = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert latest_running is not None
    assert latest_running.sequence_id == 2
    assert latest_running.status == "running"

    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id,
        attempt_id=retry_attempted.attempt.attempt_id,
        status="succeeded",
    )
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    final_rollout = await inmemory_store.query_rollouts(status=["succeeded"])
    assert final_rollout and final_rollout[0].rollout_id == rollout.rollout_id

    final_attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert [a.status for a in final_attempts] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_update_nonexistent_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test updating non-existent attempt raises error."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    with pytest.raises(ValueError, match="No attempts found"):
        await inmemory_store.update_attempt(rollout_id=rollout.rollout_id, attempt_id="nonexistent", status="failed")


# Add Attempt Tests


@pytest.mark.asyncio
async def test_add_attempt_creates_new_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test add_attempt creates a new attempt for existing rollout."""
    # Create a rollout
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})

    # Add first manual attempt
    attempted_rollout = await inmemory_store.start_attempt(rollout.rollout_id)

    assert attempted_rollout.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.sequence_id == 1
    assert attempted_rollout.attempt.status == "preparing"
    assert attempted_rollout.attempt.rollout_id == rollout.rollout_id
    assert attempted_rollout.attempt.attempt_id.startswith("at-")

    # Verify attempt is stored
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempted_rollout.attempt.attempt_id


@pytest.mark.asyncio
async def test_add_attempt_increments_sequence_id(inmemory_store: InMemoryLightningStore) -> None:
    """Test add_attempt correctly increments sequence_id."""
    # Create a rollout and dequeue to create first attempt
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    await inmemory_store.dequeue_rollout()  # Creates attempt with sequence_id=1

    # Add second attempt manually
    attempted_rollout2 = await inmemory_store.start_attempt(rollout.rollout_id)
    assert attempted_rollout2.attempt.sequence_id == 2

    # Add third attempt manually
    attempted_rollout3 = await inmemory_store.start_attempt(rollout.rollout_id)
    assert attempted_rollout3.attempt.sequence_id == 3

    # Verify all attempts exist
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3
    assert [a.sequence_id for a in attempts] == [1, 2, 3]


@pytest.mark.asyncio
async def test_add_attempt_nonexistent_rollout(inmemory_store: InMemoryLightningStore) -> None:
    """Test add_attempt raises error for nonexistent rollout."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await inmemory_store.start_attempt("nonexistent")


@pytest.mark.asyncio
async def test_add_attempt_ignores_max_attempts(inmemory_store: InMemoryLightningStore) -> None:
    """Test add_attempt ignores max_attempts configuration."""
    # Create rollout with max_attempts=2
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"})
    config = RolloutConfig(max_attempts=2)
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Add attempts beyond max_attempts
    attempt1 = await inmemory_store.start_attempt(rollout.rollout_id)
    attempt2 = await inmemory_store.start_attempt(rollout.rollout_id)
    attempt3 = await inmemory_store.start_attempt(rollout.rollout_id)  # Should succeed despite max_attempts=2

    assert attempt1.attempt.sequence_id == 1
    assert attempt2.attempt.sequence_id == 2
    assert attempt3.attempt.sequence_id == 3

    # All attempts should exist
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 3


# Latest Attempt Status Propagation Tests


@pytest.mark.asyncio
async def test_status_propagation_only_for_latest_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test that status changes only propagate to rollout when updating latest attempt."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "propagation"})

    # Create multiple attempts
    attempt1 = await inmemory_store.start_attempt(rollout.rollout_id)
    _attempt2 = await inmemory_store.start_attempt(rollout.rollout_id)
    attempt3 = await inmemory_store.start_attempt(rollout.rollout_id)  # This is the latest

    # Update attempt1 (not latest) to succeeded
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout status should NOT change since attempt1 is not the latest
    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "queuing"  # Should remain unchanged

    # Update attempt3 (latest) to succeeded
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt3.attempt.attempt_id, status="succeeded"
    )

    # Now rollout status should change since we updated the latest attempt
    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_status_propagation_with_retry_for_latest_attempt(inmemory_store: InMemoryLightningStore) -> None:
    """Test retry logic only applies when updating latest attempt."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "retry"})
    config = RolloutConfig(max_attempts=3, retry_condition=["failed"])
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Create multiple attempts
    attempt1 = await inmemory_store.start_attempt(rollout.rollout_id)  # sequence_id=1
    attempt2 = await inmemory_store.start_attempt(rollout.rollout_id)  # sequence_id=2 (latest)

    # Fail attempt1 (not latest) - should NOT trigger retry
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed"
    )

    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "queuing"  # Should remain unchanged

    # Fail attempt2 (latest) - should trigger retry since sequence_id=2 < max_attempts=3
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed"
    )

    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "requeuing"  # Should be requeued for retry


@pytest.mark.asyncio
async def test_status_propagation_latest_changes_when_new_attempt_added(inmemory_store: InMemoryLightningStore) -> None:
    """Test that the 'latest attempt' changes as new attempts are added."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "latest_changes"})

    # Create first attempt and update it to succeeded
    attempt1 = await inmemory_store.start_attempt(rollout.rollout_id)
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="succeeded"
    )

    # Rollout should be succeeded since attempt1 is latest
    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"

    # Add second attempt (now this becomes latest)
    attempt2 = await inmemory_store.start_attempt(rollout.rollout_id)

    # Update attempt1 to failed - should NOT affect rollout since it's no longer latest
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt1.attempt.attempt_id, status="failed"
    )

    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"  # Should remain unchanged

    # Update attempt2 (now latest) to failed
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt2.attempt.attempt_id, status="failed"
    )

    # Now rollout should change since we updated the new latest attempt
    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "failed"


@pytest.mark.asyncio
async def test_status_propagation_update_latest_by_reference(inmemory_store: InMemoryLightningStore) -> None:
    """Test status propagation when updating latest attempt using 'latest' reference."""
    rollout = await inmemory_store.enqueue_rollout(input={"test": "latest_ref"})

    # Create multiple attempts
    await inmemory_store.start_attempt(rollout.rollout_id)
    await inmemory_store.start_attempt(rollout.rollout_id)
    attempt3 = await inmemory_store.start_attempt(rollout.rollout_id)  # This is latest

    # Update using "latest" reference
    updated_attempt = await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id="latest", status="succeeded"
    )

    # Should have updated attempt3
    assert updated_attempt.attempt_id == attempt3.attempt.attempt_id
    assert updated_attempt.status == "succeeded"

    # Rollout should be updated since we updated the latest attempt
    updated_rollout = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert updated_rollout is not None
    assert updated_rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_healthcheck_timeout_behavior(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test that healthcheck detects and handles timeout conditions."""
    # Create rollout with short timeout configuration
    config = RolloutConfig(
        timeout_seconds=0.1, max_attempts=2, retry_condition=["timeout"]  # Very short timeout for testing
    )

    rollout = await inmemory_store.enqueue_rollout(input={"test": "timeout"})
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue to create an attempt and add span to make it running
    attempted = await inmemory_store.dequeue_rollout()
    assert attempted is not None
    await inmemory_store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running
    running_rollouts = await inmemory_store.query_rollouts(status=["running"])
    assert len(running_rollouts) == 1

    # Wait for timeout to occur
    await asyncio.sleep(0.15)  # Wait longer than timeout_seconds

    # Trigger healthcheck by calling any decorated method
    # Verify the attempt was marked as timeout and rollout was requeued
    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].status == "timeout"

    # Since retry_condition includes "timeout" and max_attempts=2, should requeue
    rollout_after = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "requeuing"


@pytest.mark.asyncio
async def test_healthcheck_unresponsive_behavior(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Test that healthcheck detects and handles unresponsive conditions."""
    # Create rollout with short unresponsive timeout but no retry for unresponsive
    config = RolloutConfig(
        unresponsive_seconds=0.1,  # Very short unresponsive timeout
        max_attempts=3,
        retry_condition=["timeout"],  # Note: "unresponsive" not in retry_condition
    )

    rollout = await inmemory_store.enqueue_rollout(input={"test": "unresponsive"})
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    # Dequeue and add span to make it running (this sets last_heartbeat_time)
    attempted = await inmemory_store.dequeue_rollout()
    assert attempted is not None
    await inmemory_store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Verify it's running and has heartbeat
    running_attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert running_attempts[0].status == "running"
    assert running_attempts[0].last_heartbeat_time is not None

    # Wait for unresponsive timeout
    await asyncio.sleep(0.15)  # Wait longer than unresponsive_seconds

    # Verify attempt was marked as unresponsive
    attempts_after = await inmemory_store.query_attempts(rollout.rollout_id)
    assert attempts_after[0].status == "unresponsive"

    # Since "unresponsive" not in retry_condition, rollout should be failed
    rollout_after = await inmemory_store.get_rollout_by_id(rollout.rollout_id)
    assert rollout_after is not None
    assert rollout_after.status == "failed"


# Full Lifecycle Integration Tests


@pytest.mark.asyncio
async def test_full_lifecycle_success(inmemory_store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test successful rollout lifecycle: queue -> prepare -> run -> succeed."""
    # 1. Create task
    rollout = await inmemory_store.enqueue_rollout(input={"test": "data"}, mode="train")
    assert rollout.status == "queuing"

    # 2. Pop to start processing (creates attempt)
    popped = await inmemory_store.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.status == "preparing"

    # 3. Add span (transitions to running)
    span = await inmemory_store.add_otel_span(rollout.rollout_id, attempt.attempt_id, mock_readable_span)
    assert span.sequence_id == 1

    # Check status transitions
    rollouts = await inmemory_store.query_rollouts(status=["running"])
    assert len(rollouts) == 1

    attempts = await inmemory_store.query_attempts(rollout.rollout_id)
    assert attempts[0].status == "running"
    assert attempts[0].last_heartbeat_time is not None

    # 4. Complete successfully
    await inmemory_store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempt.attempt_id, status="succeeded"
    )
    await inmemory_store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Verify final state
    final = (await inmemory_store.query_rollouts())[0]
    assert final.status == "succeeded"
    assert final.end_time is not None

    final_attempt = await inmemory_store.get_latest_attempt(rollout.rollout_id)
    assert final_attempt is not None
    assert final_attempt.status == "succeeded"
    assert final_attempt.end_time is not None


# Retry and requeue interactions


def _retry_config() -> RolloutConfig:
    """Helper to create a rollout config that retries unresponsive attempts."""

    return RolloutConfig(max_attempts=2, retry_condition=["unresponsive"])


@pytest.mark.asyncio
async def test_requeued_attempt_recovers_before_retry(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """A requeued attempt that resumes should be removed from the queue."""

    attempted = await inmemory_store.start_rollout(input={"foo": "bar"})
    await inmemory_store.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    rollout = await inmemory_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "requeuing"

    await inmemory_store.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    latest_attempt = await inmemory_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == attempted.attempt.attempt_id
    assert latest_attempt.status == "running"

    rollout = await inmemory_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "running"

    # Queue should no longer return the rollout for retry.
    assert await inmemory_store.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_succeeds_without_new_attempt(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Recovered attempts can finish successfully without spawning a retry."""

    attempted = await inmemory_store.start_rollout(input={"foo": "bar"})
    await inmemory_store.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    await inmemory_store.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="succeeded"
    )

    rollout = await inmemory_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "succeeded"

    latest_attempt = await inmemory_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "succeeded"
    assert latest_attempt.end_time is not None

    assert await inmemory_store.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_fails_without_new_attempt(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Recovered attempts that fail should mark the rollout failed without retries."""

    attempted = await inmemory_store.start_rollout(input={"foo": "bar"})
    await inmemory_store.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    await inmemory_store.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="failed"
    )

    rollout = await inmemory_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"

    latest_attempt = await inmemory_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "failed"
    assert latest_attempt.end_time is not None

    assert await inmemory_store.dequeue_rollout() is None


@pytest.mark.asyncio
async def test_requeued_attempt_recovers_after_retry_started(
    inmemory_store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Data from an old attempt should not disrupt a newly started retry."""

    attempted = await inmemory_store.start_rollout(input={"foo": "bar"})
    await inmemory_store.update_rollout(rollout_id=attempted.rollout_id, config=_retry_config())

    await inmemory_store.update_attempt(
        rollout_id=attempted.rollout_id, attempt_id=attempted.attempt.attempt_id, status="unresponsive"
    )

    # Start a new attempt by dequeuing the rollout from the queue.
    retried = await inmemory_store.dequeue_rollout()
    assert retried is not None
    assert retried.attempt.sequence_id == 2

    await inmemory_store.add_otel_span(attempted.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    latest_attempt = await inmemory_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id == retried.attempt.attempt_id
    assert latest_attempt.sequence_id == 2

    # The old attempt is still marked running but does not change the rollout state.
    first_attempts = await inmemory_store.query_attempts(attempted.rollout_id)
    assert first_attempts[0].status == "running"
    rollout = await inmemory_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "preparing"

    assert await inmemory_store.dequeue_rollout() is None

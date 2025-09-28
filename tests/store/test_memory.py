# Copyright (c) Microsoft. All rights reserved.

# type: ignore

"""
Comprehensive tests for InMemoryLightningStore.

Test categories:
- Core CRUD operations
- Queue operations (FIFO behavior)
- Resource versioning
- Span tracking and sequencing
- Rollout lifecycle and status transitions
- Watchdog behavior with time simulation
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
from agentlightning.types import LLM, PromptTemplate, ResourcesUpdate, RolloutV2


@pytest.fixture
def store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


@pytest.fixture
def store_with_watchdog() -> InMemoryLightningStore:
    """Create a store with watchdog configured."""
    watchdog = LightningStoreWatchDog(
        timeout_seconds=5.0,
        unresponsive_seconds=2.0,
        max_attempts=3,
        retry_condition=["unresponsive", "timeout"],
    )
    return InMemoryLightningStore(watchdog=watchdog)


@pytest.fixture
def mock_readable_span() -> Mock:
    """Create a mock ReadableSpan for testing."""
    span = Mock()
    span.name = "test_span"

    # Mock context
    context = Mock()
    context.trace_id = 111111
    context.span_id = 222222
    context.is_remote = False
    context.trace_state = {}  # Make it an empty dict instead of Mock
    span.get_span_context = Mock(return_value=context)

    # Mock other attributes
    span.parent = None
    # Fix mock status to return proper string values
    status_code_mock = Mock()
    status_code_mock.name = "OK"
    span.status = Mock(status_code=status_code_mock, description=None)
    span.attributes = {"test": "value"}
    span.events = []
    span.links = []
    span.start_time = time.time_ns()
    span.end_time = time.time_ns() + 1000000
    span.resource = Mock(attributes={}, schema_url="")

    return span


# Core CRUD Operations Tests


@pytest.mark.asyncio
async def test_add_task_creates_rollout(store: InMemoryLightningStore) -> None:
    """Test that add_task creates a properly initialized rollout."""
    sample = {"input": "test_data"}
    metadata = {"key": "value", "number": 42}

    rollout = await store.add_task(sample=sample, mode="train", resources_id="res-123", metadata=metadata)

    assert rollout.rollout_id.startswith("rollout-")
    assert rollout.input == sample
    assert rollout.mode == "train"
    assert rollout.resources_id == "res-123"
    assert rollout.metadata == metadata
    assert rollout.status == "queuing"
    assert rollout.attempt_sequence_id == 1
    assert rollout.start_time is not None


@pytest.mark.asyncio
async def test_add_existing_rollout(store: InMemoryLightningStore) -> None:
    """Test adding a pre-created rollout to the store."""
    rollout = RolloutV2(
        rollout_id="custom-id-123",
        input={"custom": "data"},
        status="queuing",
        start_time=time.time(),
        metadata={"source": "external"},
    )

    await store.add_rollout(rollout)

    # Verify it's in the store
    all_rollouts = await store.query_rollouts()
    assert len(all_rollouts) == 1
    assert all_rollouts[0].rollout_id == "custom-id-123"
    assert all_rollouts[0].metadata["source"] == "external"

    # Verify it's in the queue
    popped = await store.pop_rollout()
    assert popped is not None
    assert popped.rollout_id == "custom-id-123"


@pytest.mark.asyncio
async def test_query_rollouts_by_status(store: InMemoryLightningStore) -> None:
    """Test querying rollouts filtered by status."""
    # Create rollouts with different statuses
    r1 = await store.add_task(sample={"id": 1})
    r2 = await store.add_task(sample={"id": 2})
    r3 = await store.add_task(sample={"id": 3})

    # Modify statuses
    await store.pop_rollout()  # r1 becomes "preparing"
    await store.update_rollout(rollout_id=r2.rollout_id, status="error")
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

    finished = await store.query_rollouts(status=["error", "success"])
    assert len(finished) == 1
    assert finished[0].rollout_id == r2.rollout_id

    # Empty status list
    none = await store.query_rollouts(status=[])
    assert len(none) == 0


@pytest.mark.asyncio
async def test_update_rollout_fields(store: InMemoryLightningStore) -> None:
    """Test updating various rollout fields."""
    rollout = await store.add_task(sample={"test": "data"})

    # Update multiple fields at once
    await store.update_rollout(
        rollout_id=rollout.rollout_id,
        status="running",
        worker_id="worker-abc",
        attempt_id="attempt-1",
        attempt_sequence_id=2,
        attempt_start_time=time.time(),
        last_attempt_status="timeout",
        metadata=dict(custom_field="custom_value"),  # Extra field goes to metadata
    )

    # Verify all updates
    updated_rollouts = await store.query_rollouts()
    updated = updated_rollouts[0]
    assert updated.status == "running"
    assert updated.worker_id == "worker-abc"
    assert updated.attempt_id == "attempt-1"
    assert updated.attempt_sequence_id == 2
    assert updated.attempt_start_time is not None
    assert updated.last_attempt_status == "timeout"
    assert updated.metadata["custom_field"] == "custom_value"


# Queue Operations Tests


@pytest.mark.asyncio
async def test_pop_rollout_skips_non_queuing_status(store: InMemoryLightningStore) -> None:
    """Test that pop_rollout skips rollouts that have been updated to non-queuing status."""
    # Add multiple rollouts to the queue
    r1 = await store.add_task(sample={"id": 1})
    r2 = await store.add_task(sample={"id": 2})
    r3 = await store.add_task(sample={"id": 3})

    # Update r1 to success status while it's still in the queue
    await store.update_rollout(rollout_id=r1.rollout_id, status="success")

    # Update r2 to have a span
    await store.update_rollout(rollout_id=r2.rollout_id, status="error")

    # r3 should still be in queuing status

    # Pop should skip r1 and r2 (both non-queuing) and return r3
    popped = await store.pop_rollout()
    assert popped is not None
    assert popped.rollout_id == r3.rollout_id
    assert popped.status == "preparing"
    assert popped.input["id"] == 3

    # Second pop should return None since no queuing rollouts remain
    popped2 = await store.pop_rollout()
    assert popped2 is None

    # Verify r1 and r2 are still in their non-queuing states
    all_rollouts = await store.query_rollouts()
    rollout_statuses = {r.rollout_id: r.status for r in all_rollouts}
    assert rollout_statuses[r1.rollout_id] == "success"
    assert rollout_statuses[r2.rollout_id] == "error"
    assert rollout_statuses[r3.rollout_id] == "preparing"


@pytest.mark.asyncio
async def test_fifo_ordering(store: InMemoryLightningStore) -> None:
    """Test that queue maintains FIFO order."""
    rollouts = []
    for i in range(5):
        r = await store.add_task(sample={"order": i})
        rollouts.append(r)

    # Pop all and verify order
    for i in range(5):
        popped = await store.pop_rollout()
        assert popped is not None
        assert popped.rollout_id == rollouts[i].rollout_id
        assert popped.input["order"] == i
        assert popped.status == "preparing"
        assert popped.attempt_start_time is not None


@pytest.mark.asyncio
async def test_pop_empty_queue(store: InMemoryLightningStore) -> None:
    """Test popping from empty queue returns None."""
    result = await store.pop_rollout()
    assert result is None

    # Multiple pops should all return None
    for _ in range(3):
        assert await store.pop_rollout() is None


@pytest.mark.asyncio
async def test_requeue_mechanism(store: InMemoryLightningStore) -> None:
    """Test requeuing puts rollout back in queue."""
    rollout = await store.add_task(sample={"data": "test"})
    original_id = rollout.rollout_id

    # Pop and verify it's not in queue
    popped = await store.pop_rollout()
    assert popped is not None
    assert await store.pop_rollout() is None

    # Requeue it
    await store.update_rollout(
        rollout_id=original_id, status="requeuing", last_attempt_status="timeout", attempt_sequence_id=2
    )

    # Should be back in queue with updated metadata
    requeued = await store.pop_rollout()
    assert requeued is not None
    assert requeued.rollout_id == original_id
    assert requeued.status == "preparing"  # Changes when popped
    assert requeued.last_attempt_status == "timeout"
    assert requeued.attempt_sequence_id == 2


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
    v1 = ResourcesUpdate(resources_id="v1", resources={"main_llm": llm_v1})
    update = await store.update_resources("v1", {"main_llm": llm_v1})
    assert update.resources_id == "v1"

    latest = await store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v1"
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
    assert v2.resources["main_llm"].model == "test-model-v2"

    # Latest should be v2
    latest = await store.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "v2"

    # Can still retrieve v1
    old = await store.get_resources_by_id("v1")
    assert old is not None
    assert old.resources["main_llm"].model == "test-model-v1"


@pytest.mark.asyncio
async def test_task_inherits_latest_resources(store: InMemoryLightningStore) -> None:
    """Test that new tasks inherit latest resources_id if not specified."""
    # Set up resources with proper PromptTemplate
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")
    update = ResourcesUpdate(resources_id="current", resources={"greeting": prompt})
    await store.update_resources(update.resources_id, update.resources)

    # Task without explicit resources_id
    r1 = await store.add_task(sample={"id": 1})
    assert r1.resources_id == "current"

    # Task with explicit resources_id
    r2 = await store.add_task(sample={"id": 2}, resources_id="override")
    assert r2.resources_id == "override"

    # Update resources
    new_prompt = PromptTemplate(resource_type="prompt_template", template="Hi {name}!", engine="f-string")
    update2 = ResourcesUpdate(resources_id="new", resources={"greeting": new_prompt})
    await store.update_resources(update2.resources_id, update2.resources)

    # New task gets new resources
    r3 = await store.add_task(sample={"id": 3})
    assert r3.resources_id == "new"


# Span Management Tests


@pytest.mark.asyncio
async def test_span_sequence_generation(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test automatic sequence ID generation for spans."""
    rollout = await store.add_task(sample={"test": "data"})

    # First span gets sequence_id 1
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-1")
    assert seq_id == 1

    span1 = await store.add_otel_span(rollout.rollout_id, "attempt-1", mock_readable_span)
    assert span1.sequence_id == 1

    # Next span gets sequence_id 2
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-1")
    assert seq_id == 2

    span2 = await store.add_otel_span(rollout.rollout_id, "attempt-1", mock_readable_span)
    assert span2.sequence_id == 2

    # Different attempt starts from 1
    seq_id = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-2")
    assert seq_id == 1


@pytest.mark.asyncio
async def test_span_with_explicit_sequence_id(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test providing explicit sequence_id to spans."""
    rollout = await store.add_task(sample={"test": "data"})

    # Add span with explicit sequence_id
    span = await store.add_otel_span(rollout.rollout_id, "attempt-1", mock_readable_span, sequence_id=100)
    assert span.sequence_id == 100

    # Next auto-generated should not be affected by explicit IDs
    next_seq = await store.get_next_span_sequence_id(rollout.rollout_id, "attempt-1")
    assert next_seq == 2  # Only one span added so far


@pytest.mark.asyncio
async def test_query_spans_by_attempt(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test querying spans filtered by attempt_id."""
    rollout = await store.add_task(sample={"test": "data"})

    # Add spans for multiple attempts
    for attempt in ["attempt-1", "attempt-2", "attempt-3"]:
        for _ in range(2):
            await store.add_otel_span(rollout.rollout_id, attempt, mock_readable_span)

    # Query all spans
    all_spans = await store.query_spans(rollout.rollout_id)
    assert len(all_spans) == 6

    # Query specific attempt
    attempt1_spans = await store.query_spans(rollout.rollout_id, attempt_id="attempt-1")
    assert len(attempt1_spans) == 2
    assert all(s.attempt_id == "attempt-1" for s in attempt1_spans)

    # Query latest attempt (lexicographically)
    latest_spans = await store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert len(latest_spans) == 2
    assert all(s.attempt_id == "attempt-3" for s in latest_spans)

    # Query non-existent attempt
    no_spans = await store.query_spans(rollout.rollout_id, attempt_id="nonexistent")
    assert len(no_spans) == 0


@pytest.mark.asyncio
async def test_span_triggers_status_transition(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test that adding first span transitions rollout from preparing to running."""
    rollout = await store.add_task(sample={"test": "data"})

    # Pop to set status to preparing
    popped = await store.pop_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Verify status in store
    rollouts = await store.query_rollouts(status=["preparing"])
    assert len(rollouts) == 1

    # Add first span
    await store.add_otel_span(rollout.rollout_id, "attempt-1", mock_readable_span)

    # Status should transition to running
    rollouts = await store.query_rollouts(status=["running"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id


# Rollout Lifecycle Tests


@pytest.mark.asyncio
async def test_completion_sets_end_time(store: InMemoryLightningStore) -> None:
    """Test that completing a rollout sets end_time."""
    rollout = await store.add_task(sample={"test": "data"})

    # Initially no end_time
    assert rollout.end_time is None

    # Complete as success
    await store.update_rollout(rollout_id=rollout.rollout_id, status="success")

    completed_rollouts = await store.query_rollouts()
    completed = completed_rollouts[0]
    assert completed.status == "success"
    assert completed.end_time is not None
    assert completed.end_time > completed.start_time


@pytest.mark.asyncio
async def test_wait_for_rollouts(store: InMemoryLightningStore) -> None:
    """Test waiting for rollout completion."""
    # Add multiple rollouts
    r1 = await store.add_task(sample={"id": 1})
    r2 = await store.add_task(sample={"id": 2})
    r3 = await store.add_task(sample={"id": 3})

    # Start waiting for r1 and r2
    async def wait_for_completion() -> List[RolloutV2]:
        return await store.wait_for_rollouts([r1.rollout_id, r2.rollout_id], timeout=5.0)

    wait_task = asyncio.create_task(wait_for_completion())
    await asyncio.sleep(0.01)  # Let wait task start

    # Complete r1
    await store.update_rollout(rollout_id=r1.rollout_id, status="success")

    # Complete r2
    await store.update_rollout(rollout_id=r2.rollout_id, status="error")

    # Get results
    completed = await wait_task
    assert len(completed) == 2
    assert {r.rollout_id for r in completed} == {r1.rollout_id, r2.rollout_id}
    assert {r.status for r in completed} == {"success", "error"}


@pytest.mark.asyncio
async def test_wait_timeout(store: InMemoryLightningStore) -> None:
    """Test wait_for_rollouts timeout behavior."""
    rollout = await store.add_task(sample={"test": "data"})

    start = time.time()
    completed = await store.wait_for_rollouts([rollout.rollout_id], timeout=0.1)
    elapsed = time.time() - start

    assert elapsed < 0.2  # Should timeout quickly
    assert len(completed) == 0  # No completions


# Watchdog Tests


@pytest.mark.asyncio
async def test_watchdog_healthcheck_queries(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test that missing @healthcheck decorators prevent watchdog from detecting stuck rollouts."""
    store = store_with_watchdog
    rollout = await store.add_task(sample={"test": "data"})

    # Start processing by popping the rollout
    popped = await store.pop_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Simulate timeout condition by advancing time
    with patch("time.time") as mock_time:
        mock_time.return_value = rollout.start_time + 10.0  # Past timeout threshold

        # The rollout should have timed out
        rollouts = await store.query_rollouts(status=["requeuing"])
        assert len(rollouts) == 1
        assert rollouts[0].last_attempt_status == "timeout"

        spans = await store.query_spans(rollout.rollout_id)
        assert len(spans) == 0

        await store.update_rollout(rollout_id=rollout.rollout_id, status="running")

    # Now call a method that HAS @healthcheck decorator
    await store.add_task(sample={"test2": "data2"})

    rollouts = await store.query_rollouts(status=["running"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id


@pytest.mark.asyncio
async def test_watchdog_detects_timeout(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog detecting and handling timeouts."""
    store = store_with_watchdog
    rollout = await store.add_task(sample={"test": "data"})

    # Start processing
    await store.pop_rollout()

    # Simulate timeout
    with patch("time.time") as mock_time:
        mock_time.return_value = rollout.start_time + 10.0  # Past timeout
        if store.watchdog:
            await store.watchdog.healthcheck(store)

    # Should be requeued
    rollouts = await store.query_rollouts(status=["requeuing"])
    assert len(rollouts) == 1
    assert rollouts[0].last_attempt_status == "timeout"
    assert rollouts[0].attempt_sequence_id == 2


@pytest.mark.asyncio
async def test_watchdog_detects_unresponsive(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog detecting unresponsive workers."""
    store = store_with_watchdog
    rollout = await store.add_task(sample={"test": "data"})
    popped = await store.pop_rollout()
    assert popped is not None

    # Simulate unresponsiveness (no spans for too long)
    with patch("time.time") as mock_time:
        mock_time.return_value = popped.attempt_start_time + 3.0  # Past unresponsive threshold
        if store.watchdog:
            await store.watchdog.healthcheck(store)

    # Should be requeued
    rollouts = await store.query_rollouts(status=["requeuing"])
    assert len(rollouts) == 1
    assert rollouts[0].last_attempt_status == "unresponsive"


@pytest.mark.asyncio
async def test_watchdog_respects_max_attempts(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog stops retrying after max attempts."""
    store = store_with_watchdog
    rollout = await store.add_task(sample={"test": "data"})

    # Simulate multiple failures
    for attempt in range(3):  # max_attempts = 3
        await store.pop_rollout()
        with patch("time.time") as mock_time:
            mock_time.return_value = rollout.start_time + (attempt + 1) * 10
            if store.watchdog:
                await store.watchdog.healthcheck(store)

        if attempt < 2:
            # Should be requeued
            rollouts = await store.query_rollouts(status=["requeuing"])
            assert len(rollouts) == 1
        else:
            # Should be marked as timeout (not requeued)
            rollouts = await store.query_rollouts(status=["timeout"])
            assert len(rollouts) == 1


@pytest.mark.asyncio
async def test_healthcheck_decorator(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test healthcheck decorator calls watchdog."""
    store = store_with_watchdog

    if store.watchdog:
        with patch.object(store.watchdog, "healthcheck", new_callable=AsyncMock) as mock_check:
            # Call decorated methods
            await store.add_task(sample={"test": "data"})
            await store.get_latest_resources()
            await store.wait_for_rollouts([], timeout=0.01)

            # Verify healthcheck was called
            assert mock_check.call_count == 3
            for call in mock_check.call_args_list:
                assert call[0][0] == store


@pytest.mark.asyncio
async def test_healthcheck_recursive_prevention(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test that healthcheck decorator prevents recursive calls."""
    store = store_with_watchdog

    if store.watchdog:
        # Create a mock healthcheck that tries to call another decorated method
        async def recursive_healthcheck(store_arg):
            # This should NOT trigger another healthcheck due to the flag
            await store_arg.query_rollouts(status=["preparing"])

        with patch.object(store.watchdog, "healthcheck", new_callable=AsyncMock) as mock_check:
            mock_check.side_effect = recursive_healthcheck

            # Call a decorated method - this should only trigger healthcheck once
            await store.add_task(sample={"test": "data"})

            # Verify healthcheck was called exactly once despite the recursive call
            assert mock_check.call_count == 1

            # Verify the flag is properly cleared after execution
            assert not getattr(store, "_healthcheck_running", False)


@pytest.mark.asyncio
async def test_watchdog_no_recent_span_activity_logic(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test the _no_recent_span_activity logic with different span scenarios."""
    store = store_with_watchdog
    rollout = await store.add_task(sample={"test": "data"})

    if store.watchdog:
        current_time = time.time()

        # Test 1: No spans at all should return False (handled separately)
        assert store.watchdog._no_recent_span_activity([], current_time) is False

        # Test 2: Spans with recent activity should return False
        recent_span = Mock()
        recent_span.end_time = current_time - 1.0  # 1 second ago
        recent_span.start_time = current_time - 2.0

        spans_with_recent_activity = [recent_span]
        assert store.watchdog._no_recent_span_activity(spans_with_recent_activity, current_time) is False

        # Test 3: Spans with old activity should return True
        old_span = Mock()
        old_span.end_time = current_time - 10.0  # 10 seconds ago (past unresponsive threshold of 2.0)
        old_span.start_time = current_time - 11.0

        spans_with_old_activity = [old_span]
        assert store.watchdog._no_recent_span_activity(spans_with_old_activity, current_time) is True

        # Test 4: Spans with no end_time but recent start_time should return False
        running_span = Mock()
        running_span.end_time = None
        running_span.start_time = current_time - 1.0  # 1 second ago

        spans_still_running = [running_span]
        assert store.watchdog._no_recent_span_activity(spans_still_running, current_time) is False

        # Test 5: Spans with no times at all should return False (latest_span_time = 0.0)
        span_no_times = Mock()
        span_no_times.end_time = None
        span_no_times.start_time = None

        spans_no_times = [span_no_times]
        assert store.watchdog._no_recent_span_activity(spans_no_times, current_time) is False


# Concurrent Access Tests


@pytest.mark.asyncio
async def test_concurrent_task_addition(store: InMemoryLightningStore) -> None:
    """Test adding tasks concurrently."""

    async def add_task(index: int) -> RolloutV2:
        return await store.add_task(sample={"index": index})

    # Add 50 tasks concurrently
    tasks = [add_task(i) for i in range(50)]
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
        await store.add_task(sample={"index": i})

    async def pop_task() -> RolloutV2 | None:
        return await store.pop_rollout()

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
    rollout = await store.add_task(sample={"test": "data"})

    async def add_span(index: int) -> Span:
        return await store.add_otel_span(rollout.rollout_id, "attempt-1", mock_readable_span)

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

    async def update_resource(version: int) -> None:
        llm = LLM(
            resource_type="llm",
            endpoint=f"http://localhost:808{version % 10}",
            model=f"model-v{version}",
            sampling_parameters={"temperature": 0.5 + version * 0.01},
        )
        update = ResourcesUpdate(resources_id=f"v{version}", resources={"llm": llm})
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
        assert res.resources["llm"].model == f"model-v{i}"


# Error Handling Tests


@pytest.mark.asyncio
async def test_update_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test updating non-existent rollout raises error."""
    with pytest.raises(ValueError, match="Rollout nonexistent not found"):
        await store.update_rollout(rollout_id="nonexistent", status="error")


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
    rollout = await store.add_task(sample={"test": "data"})

    spans = await store.query_spans(rollout.rollout_id, attempt_id="latest")
    assert spans == []


@pytest.mark.asyncio
async def test_wait_for_nonexistent_rollout(store: InMemoryLightningStore) -> None:
    """Test waiting for non-existent rollout handles gracefully."""
    completed = await store.wait_for_rollouts(["nonexistent"], timeout=0.1)
    assert len(completed) == 0

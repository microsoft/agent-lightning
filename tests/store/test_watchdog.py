# Copyright (c) Microsoft. All rights reserved.

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentlightning.store.base import LightningStoreWatchDog
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import Span
from agentlightning.types import LLM, Attempt, AttemptStatus, PromptTemplate, ResourcesUpdate, RolloutStatus, RolloutV2


@pytest.mark.asyncio
async def test_watchdog_healthcheck_queries(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog healthcheck behavior with new attempt system."""
    store = store_with_watchdog
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    # Start processing by popping the rollout (creates first attempt)
    popped = await store.dequeue_rollout()
    assert popped is not None
    assert popped.status == "preparing"

    # Get the created attempt
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    attempt = attempts[0]

    # Simulate timeout condition by advancing time
    with patch("time.time") as mock_time:
        mock_time.return_value = attempt.start_time + 10.0  # Past timeout threshold

        # Call healthcheck
        if store.watchdog:
            await store.watchdog.healthcheck(store)

        # The rollout should be requeued due to timeout
        rollouts = await store.query_rollouts(status=["requeuing"])
        assert len(rollouts) == 1

        # The attempt should be marked as timed out
        attempts = await store.query_attempts(rollout.rollout_id)
        assert attempts[0].status == "timeout"


@pytest.mark.asyncio
async def test_watchdog_detects_timeout(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog detecting and handling timeouts."""
    store = store_with_watchdog
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    # Start processing
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Simulate timeout
    with patch("time.time") as mock_time:
        mock_time.return_value = attempt.start_time + 10.0  # Past timeout
        if store.watchdog:
            await store.watchdog.healthcheck(store)

    # Should be requeued
    rollouts = await store.query_rollouts(status=["requeuing"])
    assert len(rollouts) == 1

    # Check attempts - should still have one marked as timeout
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 1
    assert attempts[0].status == "timeout"


@pytest.mark.asyncio
async def test_watchdog_detects_unresponsive(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog detecting unresponsive workers."""
    store = store_with_watchdog
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    popped = await store.dequeue_rollout()
    assert popped is not None

    attempts = await store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Simulate unresponsiveness (no heartbeat for too long)
    with patch("time.time") as mock_time:
        mock_time.return_value = attempt.start_time + 3.0  # Past unresponsive threshold
        if store.watchdog:
            await store.watchdog.healthcheck(store)

    # Should be requeued
    rollouts = await store.query_rollouts(status=["requeuing"])
    assert len(rollouts) == 1

    # Attempt should be marked unresponsive
    attempts = await store.query_attempts(rollout.rollout_id)
    assert attempts[0].status == "unresponsive"


@pytest.mark.asyncio
async def test_watchdog_respects_max_attempts(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog stops retrying after max attempts."""
    store = store_with_watchdog
    await store.enqueue_rollout(sample={"test": "data"})

    # Simulate multiple failures
    for attempt_num in range(3):  # max_attempts = 3
        rollout = await store.dequeue_rollout()
        attempts = await store.query_attempts(rollout.rollout_id)
        current_attempt = attempts[-1]  # Get the latest attempt

        with patch("time.time") as mock_time:
            mock_time.return_value = current_attempt.start_time + 10  # Trigger timeout
            if store.watchdog:
                await store.watchdog.healthcheck(store)

        if attempt_num < 2:
            # Should be requeued
            rollouts = await store.query_rollouts(status=["requeuing"])
            assert len(rollouts) == 1
        else:
            # Should be marked as failed (not requeued)
            rollouts = await store.query_rollouts(status=["failed"])
            assert len(rollouts) == 1


@pytest.mark.asyncio
async def test_healthcheck_decorator(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test healthcheck decorator calls watchdog."""
    store = store_with_watchdog

    if store.watchdog:
        with patch.object(store.watchdog, "healthcheck", new_callable=AsyncMock) as mock_check:
            # Call decorated methods
            await store.enqueue_rollout(sample={"test": "data"})
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
            await store.enqueue_rollout(sample={"test": "data"})

            # Verify healthcheck was called exactly once despite the recursive call
            assert mock_check.call_count == 1

            # Verify the flag is properly cleared after execution
            assert not getattr(store, "_healthcheck_running", False)


@pytest.mark.asyncio
async def test_watchdog_heartbeat_based_unresponsiveness(
    store_with_watchdog: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Test that watchdog detects unresponsiveness based on heartbeat times."""
    store = store_with_watchdog
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    # Start processing
    popped = await store.dequeue_rollout()
    assert popped is not None
    attempts = await store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    if store.watchdog:
        # Test 1: No heartbeat at all - should be marked unresponsive after threshold
        with patch("time.time") as mock_time:
            mock_time.return_value = attempt.start_time + 3.0  # Past unresponsive threshold of 2.0
            await store.watchdog.healthcheck(store)

            rollouts = await store.query_rollouts(status=["requeuing"])
            assert len(rollouts) == 1
            attempts = await store.query_attempts(rollout.rollout_id)
            assert attempts[0].status == "unresponsive"

        # Reset for next test
        await store.update_rollout(rollout_id=rollout.rollout_id, status="queuing")
        popped = await store.dequeue_rollout()
        attempts = await store.query_attempts(rollout.rollout_id)
        attempt = attempts[1]  # New attempt

        # Test 2: Add a span (sets heartbeat) - should NOT be unresponsive
        await store.add_otel_span(rollout.rollout_id, attempt.attempt_id, mock_readable_span)

        with patch("time.time") as mock_time:
            # Even after some time, if within threshold, should be fine
            mock_time.return_value = attempt.start_time + 1.5  # Within unresponsive threshold
            await store.watchdog.healthcheck(store)

            # Should still be running
            rollouts = await store.query_rollouts(status=["running"])
            assert len(rollouts) == 1
            attempts = await store.query_attempts(rollout.rollout_id)
            assert attempts[1].status == "running"

        # Test 3: Old heartbeat - should be marked unresponsive
        # Get current heartbeat time before patching
        current_attempt = await store.get_latest_attempt(rollout.rollout_id)
        assert current_attempt.last_heartbeat_time is not None

        with patch("time.time") as mock_time:
            # Simulate time passing beyond unresponsive threshold
            mock_time.return_value = current_attempt.last_heartbeat_time + 3.0  # Past threshold
            await store.watchdog.healthcheck(store)

            rollouts = await store.query_rollouts(status=["requeuing"])
            assert len(rollouts) == 1
            attempts = await store.query_attempts(rollout.rollout_id)
            assert attempts[1].status == "unresponsive"


@pytest.mark.asyncio
async def test_full_lifecycle_with_retry(store_with_watchdog: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test rollout lifecycle with failure and retry."""
    store = store_with_watchdog

    # 1. Create and start task
    rollout = await store.enqueue_rollout(sample={"retry": "test"})
    await store.dequeue_rollout()

    attempts = await store.query_attempts(rollout.rollout_id)
    first_attempt = attempts[0]

    # 2. Simulate timeout
    with patch("time.time") as mock_time:
        mock_time.return_value = first_attempt.start_time + 10  # Past timeout
        if store.watchdog:
            await store.watchdog.healthcheck(store)

    # Should be requeuing
    rollouts = await store.query_rollouts(status=["requeuing"])
    assert len(rollouts) == 1

    # First attempt should be marked as timeout
    attempts = await store.query_attempts(rollout.rollout_id)
    assert attempts[0].status == "timeout"

    # 3. Process retry attempt
    await store.dequeue_rollout()
    attempts = await store.query_attempts(rollout.rollout_id)
    assert len(attempts) == 2
    second_attempt = attempts[1]
    assert second_attempt.sequence_id == 2

    # 4. Add span to second attempt
    await store.add_otel_span(rollout.rollout_id, second_attempt.attempt_id, mock_readable_span)

    # 5. Complete successfully
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=second_attempt.attempt_id, status="succeeded")
    await store.update_rollout(rollout_id=rollout.rollout_id, status="succeeded")

    # Verify final state
    final = (await store.query_rollouts())[0]
    assert final.status == "succeeded"

    all_attempts = await store.query_attempts(rollout.rollout_id)
    assert len(all_attempts) == 2
    assert all_attempts[0].status == "timeout"  # First failed
    assert all_attempts[1].status == "succeeded"  # Second succeeded


@pytest.mark.asyncio
async def test_watchdog_handles_succeeded_attempt(store_with_watchdog: InMemoryLightningStore) -> None:
    """Test watchdog properly handles succeeded attempts."""
    store = store_with_watchdog
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    await store.dequeue_rollout()

    attempts = await store.query_attempts(rollout.rollout_id)
    attempt = attempts[0]

    # Mark attempt as succeeded
    await store.update_attempt(rollout_id=rollout.rollout_id, attempt_id=attempt.attempt_id, status="succeeded")

    # Run watchdog healthcheck
    if store.watchdog:
        await store.watchdog.healthcheck(store)

    # Rollout should be marked as succeeded
    rollouts = await store.query_rollouts(status=["succeeded"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == rollout.rollout_id

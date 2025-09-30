# Copyright (c) Microsoft. All rights reserved.

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.store.utils import healthcheck, propagate_status
from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    RolloutConfig,
)


@pytest.mark.asyncio
async def test_propagate_status_succeeds_rollout(store: InMemoryLightningStore) -> None:
    """Test propagate_status correctly handles succeeded attempts."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Create succeeded attempt
    succeeded_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempted.attempt.attempt_id, status="succeeded"
    )

    config = RolloutConfig(max_attempts=3, retry_condition=["timeout", "unresponsive"])

    # Mock update function
    update_rollout_mock = AsyncMock(return_value=rollout)

    # Test propagate_status
    result = await propagate_status(update_rollout_mock, succeeded_attempt, config)

    # Should call update with succeeded status
    update_rollout_mock.assert_called_once_with(rollout.rollout_id, "succeeded")
    assert result == rollout


@pytest.mark.asyncio
async def test_propagate_status_retries_failed_attempt(store: InMemoryLightningStore) -> None:
    """Test propagate_status retries failed attempts when configured."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Create failed attempt (first attempt, sequence_id=1)
    failed_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempted.attempt.attempt_id, status="failed"
    )

    # Config allows retry for failed with max_attempts=3
    config = RolloutConfig(max_attempts=3, retry_condition=["failed", "timeout"])

    update_rollout_mock = AsyncMock(return_value=rollout)

    # Test propagate_status - should retry since sequence_id=1 < max_attempts=3
    result = await propagate_status(update_rollout_mock, failed_attempt, config)

    # Should call update with requeuing status
    update_rollout_mock.assert_called_once_with(rollout.rollout_id, "requeuing")
    assert result == rollout


@pytest.mark.asyncio
async def test_propagate_status_no_retry_when_max_attempts_reached(store: InMemoryLightningStore) -> None:
    """Test propagate_status marks rollout as failed when max attempts reached."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})

    # Simulate third attempt (sequence_id=3) with max_attempts=3
    attempt = Attempt(
        rollout_id=rollout.rollout_id, attempt_id="attempt-3", sequence_id=3, start_time=time.time(), status="failed"
    )

    config = RolloutConfig(max_attempts=3, retry_condition=["failed"])
    update_rollout_mock = AsyncMock(return_value=rollout)

    # Should mark as failed, not retry, since sequence_id >= max_attempts
    result = await propagate_status(update_rollout_mock, attempt, config)

    update_rollout_mock.assert_called_once_with(rollout.rollout_id, "failed")
    assert result == rollout


@pytest.mark.asyncio
async def test_propagate_status_no_retry_when_not_in_retry_condition(store: InMemoryLightningStore) -> None:
    """Test propagate_status doesn't retry when status not in retry_condition."""
    rollout = await store.enqueue_rollout(sample={"test": "data"})
    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Create failed attempt but config doesn't allow retry for "failed"
    failed_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempted.attempt.attempt_id, status="failed"
    )

    config = RolloutConfig(max_attempts=3, retry_condition=["timeout", "unresponsive"])  # No "failed"
    update_rollout_mock = AsyncMock(return_value=rollout)

    result = await propagate_status(update_rollout_mock, failed_attempt, config)

    # Should mark as failed, not retry
    update_rollout_mock.assert_called_once_with(rollout.rollout_id, "failed")
    assert result == rollout


@pytest.mark.asyncio
async def test_healthcheck_detects_timeout(store: InMemoryLightningStore, mock_readable_span: Mock) -> None:
    """Test healthcheck function detects and handles timeouts."""
    # Create rollout with short timeout
    rollout = await store.enqueue_rollout(sample={"test": "timeout"})
    config = RolloutConfig(timeout_seconds=1.0, max_attempts=2, retry_condition=["timeout"])
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    attempted = await store.dequeue_rollout()
    assert attempted is not None
    await store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Create mocks for the callback functions
    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    # Simulate time passing beyond timeout
    with patch("time.time") as mock_time:
        mock_time.return_value = attempted.attempt.start_time + 2.0  # Past timeout

        # Call healthcheck with the running rollout
        await healthcheck([attempted], update_rollout_mock, update_attempt_mock)

    # Should have called update_attempt with timeout status
    update_attempt_mock.assert_called_once_with(attempted.rollout_id, attempted.attempt.attempt_id, "timeout")


@pytest.mark.asyncio
async def test_healthcheck_detects_unresponsive_no_heartbeat(store: InMemoryLightningStore) -> None:
    """Test healthcheck detects unresponsive attempts with no heartbeat."""
    rollout = await store.enqueue_rollout(sample={"test": "unresponsive"})
    config = RolloutConfig(unresponsive_seconds=1.0, max_attempts=2, retry_condition=["unresponsive"])
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Create mocks
    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    # Simulate time passing beyond unresponsive threshold (no heartbeat)
    with patch("time.time") as mock_time:
        mock_time.return_value = attempted.attempt.start_time + 2.0  # Past unresponsive threshold

        await healthcheck([attempted], update_rollout_mock, update_attempt_mock)

    # Should mark as unresponsive since no heartbeat was ever recorded
    update_attempt_mock.assert_called_once_with(attempted.rollout_id, attempted.attempt.attempt_id, "unresponsive")


@pytest.mark.asyncio
async def test_healthcheck_detects_unresponsive_old_heartbeat(
    store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Test healthcheck detects unresponsive attempts with old heartbeat."""
    rollout = await store.enqueue_rollout(sample={"test": "unresponsive"})
    config = RolloutConfig(unresponsive_seconds=1.0, max_attempts=2, retry_condition=["unresponsive"])
    await store.update_rollout(rollout_id=rollout.rollout_id, config=config)

    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Add span to set heartbeat
    await store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Get updated attempt with heartbeat
    updated_attempt = await store.get_latest_attempt(rollout.rollout_id)
    assert updated_attempt is not None
    assert updated_attempt.last_heartbeat_time is not None

    attempted_rollout = AttemptedRollout(**rollout.model_dump(), attempt=updated_attempt)

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    # Simulate time passing beyond unresponsive threshold from last heartbeat
    with patch("time.time") as mock_time:
        mock_time.return_value = updated_attempt.last_heartbeat_time + 2.0  # Past threshold

        await healthcheck([attempted_rollout], update_rollout_mock, update_attempt_mock)

    # Should mark as unresponsive
    update_attempt_mock.assert_called_once_with(
        attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id, "unresponsive"
    )


@pytest.mark.asyncio
async def test_healthcheck_promotes_preparing_to_running(
    store: InMemoryLightningStore, mock_readable_span: Mock
) -> None:
    """Test healthcheck promotes preparing attempts with heartbeat to running."""
    rollout = await store.enqueue_rollout(sample={"test": "promote"})
    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Add span to set heartbeat but keep status as preparing
    await store.add_otel_span(rollout.rollout_id, attempted.attempt.attempt_id, mock_readable_span)

    # Manually reset status to preparing to test promotion
    preparing_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempted.attempt.attempt_id, status="preparing"
    )

    attempted_rollout = AttemptedRollout(**rollout.model_dump(), attempt=preparing_attempt)

    update_rollout_mock = AsyncMock(return_value=preparing_attempt)
    update_attempt_mock = AsyncMock(return_value=preparing_attempt)

    await healthcheck([attempted_rollout], update_rollout_mock, update_attempt_mock)

    # Should promote to running
    update_attempt_mock.assert_called_once_with(
        attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id, "running"
    )


@pytest.mark.asyncio
async def test_healthcheck_propagates_completed_status(store: InMemoryLightningStore) -> None:
    """Test healthcheck propagates already completed attempt statuses."""
    rollout = await store.enqueue_rollout(sample={"test": "completed"})
    attempted = await store.dequeue_rollout()
    assert attempted is not None

    # Mark attempt as succeeded
    succeeded_attempt = await store.update_attempt(
        rollout_id=rollout.rollout_id, attempt_id=attempted.attempt.attempt_id, status="succeeded"
    )

    attempted_rollout = AttemptedRollout(**rollout.model_dump(), attempt=succeeded_attempt)

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    await healthcheck([attempted_rollout], update_rollout_mock, update_attempt_mock)

    # Should not call update_attempt, but should call propagate_status
    update_attempt_mock.assert_not_called()
    # The propagate_status is called internally and would call update_rollout_mock

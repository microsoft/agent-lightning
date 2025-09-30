# Copyright (c) Microsoft. All rights reserved.

import time
from typing import List, Optional, cast
from unittest.mock import AsyncMock, patch

import pytest

from agentlightning.store.utils import healthcheck, propagate_status
from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    RolloutConfig,
)

# Tests for propagate_status function


@pytest.mark.parametrize(
    "status,expected_call",
    [
        ("preparing", "preparing"),
        ("running", "running"),
        ("succeeded", "succeeded"),
    ],
)
@pytest.mark.asyncio
async def test_propagate_status_direct_statuses(status: AttemptStatus, expected_call: AttemptStatus) -> None:
    """Test propagate_status directly propagates preparing/running/succeeded statuses."""
    attempt = Attempt(
        rollout_id="test-rollout", attempt_id="test-attempt", sequence_id=1, start_time=time.time(), status=status
    )
    config = RolloutConfig()
    update_rollout_mock = AsyncMock()

    await propagate_status(update_rollout_mock, attempt, config)

    update_rollout_mock.assert_called_once_with("test-rollout", expected_call)


@pytest.mark.parametrize(
    "status,in_retry_condition,sequence_id,max_attempts,expected_call",
    [
        ("failed", True, 1, 3, "requeuing"),  # Should retry
        ("failed", True, 3, 3, "failed"),  # Max attempts reached
        ("failed", False, 1, 3, "failed"),  # Not in retry condition
        ("timeout", True, 2, 3, "requeuing"),  # Should retry
        ("timeout", False, 1, 3, "failed"),  # Not in retry condition
        ("unresponsive", True, 1, 2, "requeuing"),  # Should retry
    ],
)
@pytest.mark.asyncio
async def test_propagate_status_retry_logic(
    status: AttemptStatus, in_retry_condition: bool, sequence_id: int, max_attempts: int, expected_call: AttemptStatus
) -> None:
    """Test propagate_status retry logic for different combinations."""
    attempt = Attempt(
        rollout_id="test-rollout",
        attempt_id="test-attempt",
        sequence_id=sequence_id,
        start_time=time.time(),
        status=status,
    )

    retry_condition: List[AttemptStatus] = [status] if in_retry_condition else []
    config = RolloutConfig(max_attempts=max_attempts, retry_condition=retry_condition)
    update_rollout_mock = AsyncMock()

    await propagate_status(update_rollout_mock, attempt, config)

    update_rollout_mock.assert_called_once_with("test-rollout", expected_call)


@pytest.mark.asyncio
async def test_propagate_status_invalid_status() -> None:
    """Test propagate_status raises error for invalid status."""
    # Create a valid attempt first, then modify its status
    attempt = Attempt(
        rollout_id="test-rollout", attempt_id="test-attempt", sequence_id=1, start_time=time.time(), status="failed"
    )
    # Bypass Pydantic validation by directly setting the attribute
    attempt.status = cast(AttemptStatus, "invalid_status")  # Invalid status

    config = RolloutConfig()
    update_rollout_mock = AsyncMock()

    with pytest.raises(ValueError, match="Invalid attempt status: invalid_status"):
        await propagate_status(update_rollout_mock, attempt, config)


# Tests for healthcheck function


@pytest.mark.asyncio
async def test_healthcheck_empty_rollouts_list() -> None:
    """Test healthcheck handles empty rollouts list gracefully."""
    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    await healthcheck([], update_rollout_mock, update_attempt_mock)

    # Should not call any updates
    update_rollout_mock.assert_not_called()
    update_attempt_mock.assert_not_called()


@pytest.mark.asyncio
async def test_healthcheck_multiple_rollouts_different_timeouts() -> None:
    """Test healthcheck handles multiple rollouts with different timeout configs."""
    current_time = time.time()

    # Rollout 1: Short timeout, should timeout
    config1 = RolloutConfig(timeout_seconds=1.0)
    attempt1 = Attempt(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        sequence_id=1,
        start_time=current_time - 2.0,
        status="running",  # 2 seconds ago
    )
    rollout1 = AttemptedRollout(
        rollout_id="rollout-1",
        input={"test": 1},
        status="running",
        start_time=current_time,
        config=config1,
        attempt=attempt1,
    )

    # Rollout 2: Long timeout, should not timeout
    config2 = RolloutConfig(timeout_seconds=10.0)
    attempt2 = Attempt(
        rollout_id="rollout-2",
        attempt_id="attempt-2",
        sequence_id=1,
        start_time=current_time - 2.0,
        status="running",  # 2 seconds ago
    )
    rollout2 = AttemptedRollout(
        rollout_id="rollout-2",
        input={"test": 2},
        status="running",
        start_time=current_time,
        config=config2,
        attempt=attempt2,
    )

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    with patch("time.time", return_value=current_time):
        await healthcheck([rollout1, rollout2], update_rollout_mock, update_attempt_mock)

    # Only rollout1 should be marked as timeout
    update_attempt_mock.assert_called_once_with("rollout-1", "attempt-1", "timeout")


@pytest.mark.parametrize(
    "timeout_seconds,unresponsive_seconds,should_timeout,should_unresponsive",
    [
        (None, None, False, False),  # No timeouts configured
        (None, 1.0, False, True),  # Only unresponsive timeout
        (1.0, None, True, False),  # Only regular timeout
        (0.5, 1.0, True, False),  # Timeout triggers first
        (2.0, 0.5, False, True),  # Unresponsive triggers first
    ],
)
@pytest.mark.asyncio
async def test_healthcheck_timeout_configurations(
    timeout_seconds: Optional[float],
    unresponsive_seconds: Optional[float],
    should_timeout: bool,
    should_unresponsive: bool,
) -> None:
    """Test healthcheck with various timeout configurations."""
    current_time = time.time()

    config = RolloutConfig(timeout_seconds=timeout_seconds, unresponsive_seconds=unresponsive_seconds)

    attempt = Attempt(
        rollout_id="test-rollout",
        attempt_id="test-attempt",
        sequence_id=1,
        start_time=current_time - 1.5,
        status="running",  # 1.5 seconds ago
        last_heartbeat_time=None,  # No heartbeat for unresponsive detection
    )

    rollout = AttemptedRollout(
        rollout_id="test-rollout",
        input={"test": 1},
        status="running",
        start_time=current_time,
        config=config,
        attempt=attempt,
    )

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    with patch("time.time", return_value=current_time):
        await healthcheck([rollout], update_rollout_mock, update_attempt_mock)

    if should_timeout:
        update_attempt_mock.assert_called_once_with("test-rollout", "test-attempt", "timeout")
    elif should_unresponsive:
        update_attempt_mock.assert_called_once_with("test-rollout", "test-attempt", "unresponsive")
    else:
        update_attempt_mock.assert_not_called()


@pytest.mark.asyncio
async def test_healthcheck_unresponsive_with_heartbeat_timing() -> None:
    """Test unresponsive detection considers heartbeat timing correctly."""
    current_time = time.time()
    config = RolloutConfig(unresponsive_seconds=1.0)

    # Case 1: Recent heartbeat - should not be unresponsive
    attempt_recent = Attempt(
        rollout_id="rollout-recent",
        attempt_id="attempt-recent",
        sequence_id=1,
        start_time=current_time - 5.0,
        status="running",
        last_heartbeat_time=current_time - 0.5,  # Recent heartbeat
    )
    rollout_recent = AttemptedRollout(
        rollout_id="rollout-recent",
        input={"test": 1},
        status="running",
        start_time=current_time,
        config=config,
        attempt=attempt_recent,
    )

    # Case 2: Old heartbeat - should be unresponsive
    attempt_old = Attempt(
        rollout_id="rollout-old",
        attempt_id="attempt-old",
        sequence_id=1,
        start_time=current_time - 5.0,
        status="running",
        last_heartbeat_time=current_time - 2.0,  # Old heartbeat
    )
    rollout_old = AttemptedRollout(
        rollout_id="rollout-old",
        input={"test": 2},
        status="running",
        start_time=current_time,
        config=config,
        attempt=attempt_old,
    )

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    with patch("time.time", return_value=current_time):
        await healthcheck([rollout_recent, rollout_old], update_rollout_mock, update_attempt_mock)

    # Only the old heartbeat should trigger unresponsive
    update_attempt_mock.assert_called_once_with("rollout-old", "attempt-old", "unresponsive")


@pytest.mark.asyncio
async def test_healthcheck_preparing_with_heartbeat_promotion() -> None:
    """Test healthcheck promotes preparing attempts with heartbeat to running."""
    current_time = time.time()

    config = RolloutConfig()
    attempt = Attempt(
        rollout_id="test-rollout",
        attempt_id="test-attempt",
        sequence_id=1,
        start_time=current_time,
        status="preparing",
        last_heartbeat_time=current_time,  # Has heartbeat
    )
    rollout = AttemptedRollout(
        rollout_id="test-rollout",
        input={"test": 1},
        status="preparing",
        start_time=current_time,
        config=config,
        attempt=attempt,
    )

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    await healthcheck([rollout], update_rollout_mock, update_attempt_mock)

    # Should promote to running
    update_attempt_mock.assert_called_once_with("test-rollout", "test-attempt", "running")


@pytest.mark.asyncio
async def test_healthcheck_skips_rollouts_without_attempts() -> None:
    """Test healthcheck gracefully skips rollouts with no attempts."""
    config = RolloutConfig()

    # Create a valid attempt first, then set it to None
    attempt = Attempt(
        rollout_id="test-rollout", attempt_id="test-attempt", sequence_id=1, start_time=time.time(), status="running"
    )
    rollout = AttemptedRollout(
        rollout_id="test-rollout",
        input={"test": 1},
        status="running",
        start_time=time.time(),
        config=config,
        attempt=attempt,
    )

    # Bypass Pydantic validation by directly setting the attribute
    rollout.attempt = cast(Attempt, None)  # No attempt

    update_rollout_mock = AsyncMock()
    update_attempt_mock = AsyncMock()

    await healthcheck([rollout], update_rollout_mock, update_attempt_mock)

    # Should not call any updates
    update_rollout_mock.assert_not_called()
    update_attempt_mock.assert_not_called()

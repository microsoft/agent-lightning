# Copyright (c) Microsoft. All rights reserved.

import time
from typing import List, Optional, cast
from unittest.mock import patch

import pytest

from agentlightning.store.utils import rollout_status_from_attempt, scan_unhealthy_rollouts
from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    RolloutConfig,
)

# Tests for rollout_status_from_attempt function


@pytest.mark.parametrize(
    "status,expected_status",
    [
        ("preparing", "preparing"),
        ("running", "running"),
        ("succeeded", "succeeded"),
    ],
)
@pytest.mark.asyncio
async def test_rollout_status_from_attempt_direct_statuses(
    status: AttemptStatus, expected_status: AttemptStatus
) -> None:
    """Test rollout_status_from_attempt directly propagates preparing/running/succeeded statuses."""
    attempt = Attempt(
        rollout_id="test-rollout", attempt_id="test-attempt", sequence_id=1, start_time=time.time(), status=status
    )
    config = RolloutConfig()

    result = await rollout_status_from_attempt(attempt, config)

    assert result == expected_status


@pytest.mark.parametrize(
    "status,in_retry_condition,sequence_id,max_attempts,expected_status",
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
async def test_rollout_status_from_attempt_retry_logic(
    status: AttemptStatus, in_retry_condition: bool, sequence_id: int, max_attempts: int, expected_status: AttemptStatus
) -> None:
    """Test rollout_status_from_attempt retry logic for different combinations."""
    attempt = Attempt(
        rollout_id="test-rollout",
        attempt_id="test-attempt",
        sequence_id=sequence_id,
        start_time=time.time(),
        status=status,
    )

    retry_condition: List[AttemptStatus] = [status] if in_retry_condition else []
    config = RolloutConfig(max_attempts=max_attempts, retry_condition=retry_condition)

    result = await rollout_status_from_attempt(attempt, config)

    assert result == expected_status


@pytest.mark.asyncio
async def test_rollout_status_from_attempt_invalid_status() -> None:
    """Test rollout_status_from_attempt raises error for invalid status."""
    # Create a valid attempt first, then modify its status
    attempt = Attempt(
        rollout_id="test-rollout", attempt_id="test-attempt", sequence_id=1, start_time=time.time(), status="failed"
    )
    # Bypass Pydantic validation by directly setting the attribute
    attempt.status = cast(AttemptStatus, "invalid_status")  # Invalid status

    config = RolloutConfig()

    with pytest.raises(ValueError, match="Invalid attempt status: invalid_status"):
        await rollout_status_from_attempt(attempt, config)


# Tests for scan_unhealthy_rollouts function


@pytest.mark.asyncio
async def test_scan_unhealthy_rollouts_empty_list() -> None:
    """Test scan_unhealthy_rollouts handles empty rollouts list gracefully."""
    updates = await scan_unhealthy_rollouts([])

    assert updates == {}


@pytest.mark.asyncio
async def test_scan_unhealthy_rollouts_multiple_rollouts_different_timeouts() -> None:
    """Test scan_unhealthy_rollouts handles multiple rollouts with different timeout configs."""
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

    with patch("time.time", return_value=current_time):
        updates = await scan_unhealthy_rollouts([rollout1, rollout2])

    # Only rollout1 should be marked as timeout
    assert updates == {("rollout-1", "attempt-1"): "timeout"}


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
async def test_scan_unhealthy_rollouts_timeout_configurations(
    timeout_seconds: Optional[float],
    unresponsive_seconds: Optional[float],
    should_timeout: bool,
    should_unresponsive: bool,
) -> None:
    """Test scan_unhealthy_rollouts with various timeout configurations."""
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

    with patch("time.time", return_value=current_time):
        updates = await scan_unhealthy_rollouts([rollout])

    expected_updates = {}
    if should_timeout:
        expected_updates[("test-rollout", "test-attempt")] = "timeout"
    elif should_unresponsive:
        expected_updates[("test-rollout", "test-attempt")] = "unresponsive"

    assert updates == expected_updates


@pytest.mark.asyncio
async def test_scan_unhealthy_rollouts_unresponsive_with_heartbeat_timing() -> None:
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

    with patch("time.time", return_value=current_time):
        updates = await scan_unhealthy_rollouts([rollout_recent, rollout_old])

    # Only the old heartbeat should trigger unresponsive
    assert updates == {("rollout-old", "attempt-old"): "unresponsive"}


@pytest.mark.asyncio
async def test_scan_unhealthy_rollouts_skips_rollouts_without_attempts() -> None:
    """Test scan_unhealthy_rollouts gracefully skips rollouts with no attempts."""
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

    updates = await scan_unhealthy_rollouts([rollout])

    # Should not include rollout without attempts
    assert updates == {}

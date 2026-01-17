# Copyright (c) Microsoft. All rights reserved.

import time
from typing import Awaitable, Callable, Dict, List, Tuple

from agentlightning.types import Attempt, AttemptedRollout, AttemptStatus, Rollout, RolloutConfig, RolloutStatus

UpdateRolloutStatus = Callable[[str, RolloutStatus], Awaitable[Rollout]]
UpdateAttemptStatus = Callable[[str, str, AttemptStatus], Awaitable[Attempt]]


LATENCY_BUCKETS = [
    0.000001,
    0.000002,
    0.000005,
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    60.0,
    90.0,
    120.0,
    180.0,
    240.0,
    300.0,
]


async def rollout_status_from_attempt(
    attempt: Attempt,
    config: RolloutConfig,
) -> RolloutStatus:
    """
    Propagate the status of an attempt to the rollout.

    Returns:
        The status of the rollout from the perspective of the attempt.
    """
    # Propagate the status directly to the rollout
    if attempt.status == "preparing" or attempt.status == "running" or attempt.status == "succeeded":
        return attempt.status

    if attempt.status == "failed" or attempt.status == "timeout" or attempt.status == "unresponsive":
        # Check if this status should trigger a retry
        if attempt.status in config.retry_condition:
            # If we haven't exceeded max attempts, retry
            if attempt.sequence_id < config.max_attempts:
                return "requeuing"

        # If we can't retry or shouldn't retry, mark as failed
        return "failed"

    raise ValueError(f"Invalid attempt status: {attempt.status}")


async def scan_unhealthy_rollouts(
    rollouts: List[AttemptedRollout],
) -> Dict[Tuple[str, str], AttemptStatus]:
    """
    Perform health check on all running rollouts in the store.

    This method should be called periodically to:

    1. Check for unresponsive attempts (no heartbeat or spans for a while)
    2. Check for timed-out rollouts (running too long since start_time)

    This operation is completely unlocked. The caller is responsible for locking the store.

    Args:
        rollouts: The list of running rollouts to check.

    Returns:
        A dictionary of updates to the rollouts.
    """
    current_time = time.time()
    updates: Dict[Tuple[str, str], AttemptStatus] = {}

    for rollout in rollouts:
        config = rollout.config  # policy for retry and timeout

        # Get the latest attempt for this rollout
        latest_attempt = rollout.attempt
        if not latest_attempt:
            # This should not happen
            continue

        # Check for timeout condition (based on attempt start_time, instead of rollout start_time)
        if config.timeout_seconds is not None and current_time - latest_attempt.start_time > config.timeout_seconds:
            updates[(latest_attempt.rollout_id, latest_attempt.attempt_id)] = "timeout"
            continue

        # Check for unresponsive condition (based on last heartbeat)
        # (1) Haven't received heartbeat for a while
        if (
            latest_attempt.last_heartbeat_time
            and config.unresponsive_seconds is not None
            and current_time - latest_attempt.last_heartbeat_time > config.unresponsive_seconds
        ):
            updates[(latest_attempt.rollout_id, latest_attempt.attempt_id)] = "unresponsive"
            continue

        # (2) Check if there's no last heartbeat (no spans) at all
        if (
            latest_attempt.last_heartbeat_time is None
            and config.unresponsive_seconds is not None
            and current_time - latest_attempt.start_time > config.unresponsive_seconds
        ):
            updates[(latest_attempt.rollout_id, latest_attempt.attempt_id)] = "unresponsive"
            continue

    return updates

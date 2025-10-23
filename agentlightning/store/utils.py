# Copyright (c) Microsoft. All rights reserved.

import time
from typing import Awaitable, Callable, List, cast

from agentlightning.types import Attempt, AttemptedRollout, AttemptStatus, Rollout, RolloutConfig, RolloutStatus

UpdateRolloutStatus = Callable[[str, RolloutStatus], Awaitable[Rollout]]
UpdateAttemptStatus = Callable[[str, str, AttemptStatus], Awaitable[Attempt]]


async def propagate_status(
    update_rollout_status: UpdateRolloutStatus,  # this should be unlocked
    attempt: Attempt,
    config: RolloutConfig,
) -> Rollout:
    """
    Propagate the status of an attempt to the rollout.

    The rollout should be made sure in a state to be outdated.
    Requeue the rollout if it should be retried.

    This operation is completely unlocked. The caller is responsible for locking the store.
    """
    # Propagate the status directly to the rollout
    if attempt.status == "preparing" or attempt.status == "running" or attempt.status == "succeeded":
        return await update_rollout_status(
            attempt.rollout_id,
            attempt.status,
        )

    if attempt.status == "failed" or attempt.status == "timeout" or attempt.status == "unresponsive":
        # Check if this status should trigger a retry
        if attempt.status in config.retry_condition:
            # If we haven't exceeded max attempts, retry
            if attempt.sequence_id < config.max_attempts:
                return await update_rollout_status(
                    attempt.rollout_id,
                    "requeuing",
                )

        # If we can't retry or shouldn't retry, mark as failed
        return await update_rollout_status(
            attempt.rollout_id,
            "failed",
        )

    raise ValueError(f"Invalid attempt status: {attempt.status}")


async def healthcheck(
    rollouts: List[AttemptedRollout],
    update_rollout_status: UpdateRolloutStatus,
    update_attempt_status: UpdateAttemptStatus,
) -> None:
    """
    Perform health check on all running rollouts in the store.

    This method should be called periodically to:

    1. Update rollout status to failed to succeeded when the attempt is done
    2. Check for unresponsive attempts (no heartbeat or spans for a while)
    3. Check for timed-out rollouts (running too long since start_time)
    4. Update attempt/rollout status accordingly

    This operation is completely unlocked. The caller is responsible for locking the store.

    Args:
        store: The LightningStore instance to check rollouts from
    """
    current_time = time.time()

    for rollout in rollouts:
        config = rollout.config  # policy for retry and timeout

        # Get the latest attempt for this rollout
        latest_attempt = rollout.attempt
        if not latest_attempt:
            continue

        # Check if the attempt has already failed or succeeded
        if latest_attempt.status == "failed" or latest_attempt.status == "succeeded":
            await propagate_status(update_rollout_status, latest_attempt, config)
            continue

        # Check for timeout condition (based on attempt start_time, instead of rollout start_time)
        if config.timeout_seconds is not None and current_time - latest_attempt.start_time > config.timeout_seconds:
            await update_attempt_status(
                latest_attempt.rollout_id,
                latest_attempt.attempt_id,
                "timeout",
            )
            continue

        # Check for unresponsive condition (based on last heartbeat)
        if latest_attempt.last_heartbeat_time:
            if latest_attempt.status == "preparing":
                # If still preparing, mark it as running
                latest_attempt = await update_attempt_status(
                    latest_attempt.rollout_id,
                    latest_attempt.attempt_id,
                    "running",
                )

            # Haven't received heartbeat for a while
            if (
                config.unresponsive_seconds is not None
                and current_time - cast(float, latest_attempt.last_heartbeat_time) > config.unresponsive_seconds
            ):
                await update_attempt_status(
                    latest_attempt.rollout_id,
                    latest_attempt.attempt_id,
                    "unresponsive",
                )
                continue

        # Check if there's no last heartbeat (no spans) at all
        if (
            latest_attempt.last_heartbeat_time is None
            and config.unresponsive_seconds is not None
            and current_time - latest_attempt.start_time > config.unresponsive_seconds
        ):
            await update_attempt_status(
                latest_attempt.rollout_id,
                latest_attempt.attempt_id,
                "unresponsive",
            )

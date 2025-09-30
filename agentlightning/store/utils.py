import functools
import time
from typing import Any, Callable, TypeVar, cast

from agentlightning.types import Attempt, RolloutConfig, RolloutV2

from .base import LightningStore


async def propagate_status(store: LightningStore, attempt: Attempt, config: RolloutConfig) -> RolloutV2:
    """
    Propagate the status of an attempt to the rollout.

    The rollout should be made sure in a state to be outdated.
    Requeue the rollout if it should be retried.
    """
    # Propagate the status directly to the rollout
    if attempt.status == "preparing" or attempt.status == "running" or attempt.status == "succeeded":
        return await store.update_rollout(
            rollout_id=attempt.rollout_id,
            status=attempt.status,
        )

    if attempt.status == "failed" or attempt.status == "timeout" or attempt.status == "unresponsive":
        # Check if this status should trigger a retry
        if attempt.status in config.retry_condition:
            # If we haven't exceeded max attempts, retry
            if attempt.sequence_id < config.max_attempts:
                return await store.update_rollout(
                    rollout_id=attempt.rollout_id,
                    status="requeuing",
                )

        # If we can't retry or shouldn't retry, mark as failed
        return await store.update_rollout(
            rollout_id=attempt.rollout_id,
            status="failed",
        )

    raise ValueError(f"Invalid attempt status: {attempt.status}")


async def healthcheck(store: LightningStore) -> None:
    """
    Perform health check on all running rollouts in the store.

    This method should be called periodically to:
    1. Update rollout status to failed to succeeded when the attempt is done
    2. Check for unresponsive attempts (no heartbeat or spans for a while)
    3. Check for timed-out rollouts (running too long since start_time)
    4. Update attempt/rollout status accordingly

    Args:
        store: The LightningStore instance to check rollouts from
    """
    current_time = time.time()

    # Get all running rollouts from the store
    running_rollouts = await store.query_rollouts(status=["preparing", "running"])

    for rollout in running_rollouts:
        config = rollout.config  # policy for retry and timeout

        # Get the latest attempt for this rollout
        latest_attempt = await store.get_latest_attempt(rollout.rollout_id)
        if not latest_attempt:
            continue

        # Check if the attempt has already failed or succeeded
        if latest_attempt.status == "failed" or latest_attempt.status == "succeeded":
            await propagate_status(store, latest_attempt, config)
            continue

        # Check for timeout condition (based on attempt start_time, instead of rollout start_time)
        if config.timeout_seconds is not None and current_time - latest_attempt.start_time > config.timeout_seconds:
            await store.update_attempt(
                rollout_id=latest_attempt.rollout_id,
                attempt_id=latest_attempt.attempt_id,
                status="timeout",
            )
            continue

        # Check for unresponsive condition (based on last heartbeat)
        if latest_attempt.last_heartbeat_time:
            if latest_attempt.status == "preparing":
                # If still preparing, mark it as running
                latest_attempt = await store.update_attempt(
                    rollout_id=latest_attempt.rollout_id,
                    attempt_id=latest_attempt.attempt_id,
                    status="running",
                )

            # Haven't received heartbeat for a while
            if (
                config.unresponsive_seconds is not None
                and current_time - cast(float, latest_attempt.last_heartbeat_time) > config.unresponsive_seconds
            ):
                await store.update_attempt(
                    rollout_id=latest_attempt.rollout_id,
                    attempt_id=latest_attempt.attempt_id,
                    status="unresponsive",
                )
                continue

        # Check if there's no last heartbeat (no spans) at all
        if (
            latest_attempt.last_heartbeat_time is None
            and config.unresponsive_seconds is not None
            and current_time - latest_attempt.start_time > config.unresponsive_seconds
        ):
            await store.update_attempt(
                rollout_id=latest_attempt.rollout_id,
                attempt_id=latest_attempt.attempt_id,
                status="unresponsive",
            )


T_callable = TypeVar("T_callable", bound=Callable[..., Any])


def healthcheck_wrapper(func: T_callable) -> T_callable:
    """
    Decorator to run the watchdog healthcheck **before** executing the decorated method.
    Only runs if the store has a watchdog configured.
    Prevents recursive healthcheck execution using a flag on the store instance.
    """

    @functools.wraps(func)
    async def wrapper(self: LightningStore, *args: Any, **kwargs: Any) -> Any:
        # Check if healthcheck is already running to prevent recursion
        if getattr(self, "_healthcheck_running", False):
            # Skip healthcheck if already running
            return await func(self, *args, **kwargs)

        # Set flag to prevent recursive healthcheck calls
        # This flag is not asyncio/thread-safe, but it doesn't matter
        self._healthcheck_running = True  # type: ignore
        try:
            await healthcheck(self)
        finally:
            # Always clear the flag, even if healthcheck fails
            self._healthcheck_running = False  # type: ignore

        # Execute the original method
        return await func(self, *args, **kwargs)

    return cast(T_callable, wrapper)

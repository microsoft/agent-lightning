# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import (
    Attempt,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    RolloutStatus,
    RolloutV2,
    TaskInput,
)


def is_queuing(rollout: RolloutV2) -> bool:
    return rollout.status == "queuing" or rollout.status == "requeuing"


def is_running(rollout: RolloutV2) -> bool:
    return rollout.status == "preparing" or rollout.status == "running"


def is_finished(rollout: RolloutV2) -> bool:
    return rollout.status == "failed" or rollout.status == "succeeded" or rollout.status == "cancelled"


T_callable = TypeVar("T_callable", bound=Callable[..., Any])


class _UnsetType:
    """A sentinel type to indicate an unset value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __reduce__(self):
        return (_get_unset, ())


def _get_unset() -> _UnsetType:
    return UNSET


UNSET = _UnsetType()
Unset = _UnsetType  # Alias for convenience


def healthcheck(func: T_callable) -> T_callable:
    """
    Decorator to run the watchdog healthcheck before executing the decorated method.
    Only runs if the store has a watchdog configured.
    Prevents recursive healthcheck execution using a flag on the store instance.
    """

    @functools.wraps(func)
    async def wrapper(self: LightningStore, *args: Any, **kwargs: Any) -> Any:
        # Check if healthcheck is already running to prevent recursion
        if getattr(self, "_healthcheck_running", False):
            # Skip healthcheck if already running
            return await func(self, *args, **kwargs)

        # Run watchdog healthcheck if available and not already running
        if self.watchdog is not None:
            # Set flag to prevent recursive healthcheck calls
            # This flag is not asyncio/thread-safe, but it doesn't matter
            self._healthcheck_running = True  # type: ignore
            try:
                await self.watchdog.healthcheck(self)
            finally:
                # Always clear the flag, even if healthcheck fails
                self._healthcheck_running = False  # type: ignore

        # Execute the original method
        return await func(self, *args, **kwargs)

    return cast(T_callable, wrapper)


class LightningStore:
    """
    A centralized, thread-safe, async, data store for the lightning's state.
    This holds the task queue, versioned resources, and completed rollouts.
    """

    def __init__(self, watchdog: LightningStoreWatchDog | None = None):
        self.watchdog = watchdog

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        raise NotImplementedError()

    async def add_rollout(self, rollout: RolloutV2) -> RolloutV2:
        """
        Add a rollout to the store.
        """
        raise NotImplementedError()

    async def pop_rollout(self) -> Optional[RolloutV2]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.

        Will set the rollout status to preparing.
        """
        raise NotImplementedError()

    async def add_span(self, span: Span) -> Span:
        """
        Add a span to the store.

        This method is responsible for updating the rollout/attempt status to "running" if needed.
        """
        raise NotImplementedError()

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        """
        Add an opentelemetry span to the store.

        If sequence_id is not provided, it will be fetched from `get_next_span_sequence_id` and assigned automatically.
        """
        raise NotImplementedError()

    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        """
        Query and retrieve rollouts filtered by their status.
        If no status is provided, returns all rollouts.
        """
        raise NotImplementedError()

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """
        Query and retrieve all attempts associated with a specific rollout ID.
        Returns an empty list if no attempts are found.
        """
        raise NotImplementedError()

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """
        Safely retrieves the latest attempt for a given rollout ID.
        """
        raise NotImplementedError()

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves a specific version of named resources by its ID.
        """
        raise NotImplementedError()

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves the latest version of named resources.
        """
        raise NotImplementedError()

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """
        Get the next span sequence ID for a given rollout and attempt.
        This should be used to assign a unique sequence ID to each span within an attempt.

        Recommend getting the ID before the operation even begins to avoid racing conditions.
        """
        raise NotImplementedError()

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        """
        Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.
        """
        raise NotImplementedError()

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """
        Query and retrieve all spans associated with a specific rollout ID.
        Returns an empty list if no spans are found.
        """
        raise NotImplementedError()

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        raise NotImplementedError()

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        metadata: Dict[str, Any] | Unset = UNSET,
    ) -> RolloutV2:
        """
        Update the rollout status and related metadata.

        Not-listed fields here either cannot be updated, or should be auto-updated (e.g., end_time).

        When status is updated to a finished / problematic state, other states like task
        queues will be updated accordingly.

        Args:
            rollout_id: Unique identifier for the rollout to update
            input: New input data for the rollout. If set, will be updated. Can be updated to None
            mode: New mode for the rollout. If set, will be updated. Can be updated to None
            resources_id: New resources ID for the rollout. If set, will be updated. Can be updated to None
            status: New status for the rollout. If set, will be updated
            metadata: Dictionary of additional metadata to update. If set, will be merged with existing metadata
        """
        raise NotImplementedError()

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Dict[str, Any] | Unset = UNSET,
    ) -> Attempt:
        """
        Update a specific or latest attempt for a given rollout.

        Update the latest attempt will NOT affect the corresponding rollout status.


        Args:
            rollout_id: Unique identifier for the rollout
            attempt_id: Unique identifier for the attempt
            status: Status to set for the attempt, update if provided
            worker_id: Worker identifier, update if provided
            last_heartbeat_time: Timestamp of the last heartbeat from the worker
            metadata: Dictionary of additional metadata to update, will be merged with existing metadata
        """
        raise NotImplementedError()


class LightningStoreWatchDog:
    """
    Watchdog service that monitors rollout health and handles timeouts.

    Monitors running rollouts for:
    - Unresponsive workers (attempted but no response, or no spans for a while)
    - Timeout conditions (total execution time exceeds timeout_seconds)
    """

    def __init__(
        self,
        timeout_seconds: float,
        unresponsive_seconds: float,
        max_attempts: int = 3,
        retry_condition: Optional[Sequence[AttemptStatus]] = None,
    ):
        """
        Initialize the watchdog with timeout configurations.

        Args:
            timeout_seconds: Maximum total time allowed for a rollout execution
            unresponsive_seconds: Maximum time without activity before marking as unresponsive
            max_attempts: Maximum number of retry attempts for a rollout
            retry_condition: List of statuses that should trigger a retry (default: ["unresponsive", "timeout"])
        """
        self.timeout_seconds = timeout_seconds
        self.unresponsive_seconds = unresponsive_seconds
        self.max_attempts = max_attempts
        self.retry_condition = retry_condition or ["unresponsive", "timeout"]

    async def healthcheck(self, store: LightningStore) -> None:
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
            # Get the latest attempt for this rollout
            latest_attempt = await store.get_latest_attempt(rollout.rollout_id)
            if not latest_attempt:
                continue

            # Check if the attempt has already failed or succeeded
            if latest_attempt.status == "failed":
                await self._handle_failed_rollout(store, latest_attempt, latest_attempt.status)
                continue

            if latest_attempt.status == "succeeded":
                await self._handle_succeeded_rollout(store, latest_attempt)
                continue

            # Check for timeout condition (based on attempt start_time, instead of rollout start_time)
            if current_time - latest_attempt.start_time > self.timeout_seconds:
                await self._handle_failed_rollout(store, latest_attempt, "timeout")
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
                if rollout.status == "preparing":
                    rollout = await store.update_rollout(
                        rollout_id=latest_attempt.rollout_id,
                        status="running",
                    )

                # Haven't received heartbeat for a while
                if current_time - cast(float, latest_attempt.last_heartbeat_time) > self.unresponsive_seconds:
                    await self._handle_failed_rollout(store, latest_attempt, "unresponsive")
                    continue

            # Check if there's no last heartbeat (no spans) at all
            if (
                latest_attempt.last_heartbeat_time is None
                and current_time - latest_attempt.start_time > self.unresponsive_seconds
            ):
                await self._handle_failed_rollout(store, latest_attempt, "unresponsive")

    async def _handle_succeeded_rollout(self, store: LightningStore, attempt: Attempt) -> None:
        """
        Handle a succeeded rollout by marking the rollout as succeeded.

        Args:
            store: The LightningStore instance
            attempt: The Attempt instance that has succeeded
        """
        await store.update_rollout(
            rollout_id=attempt.rollout_id,
            status="succeeded",
        )

    async def _handle_failed_rollout(self, store: LightningStore, attempt: Attempt, status: AttemptStatus) -> None:
        """
        Handle a failed rollout by either retrying or marking as failed.

        Args:
            store: The LightningStore instance
            attempt: The Attempt instance that has failed
            status: The failure status ("timeout" or "unresponsive")
        """
        # Update the current attempt with the failure status
        if status != attempt.status:
            await store.update_attempt(
                rollout_id=attempt.rollout_id,
                attempt_id=attempt.attempt_id,
                status=status,
            )

        # Check if this status should trigger a retry
        if status in self.retry_condition:
            # Get current attempt count
            attempts = await store.query_attempts(attempt.rollout_id)
            current_attempts = len(attempts)

            # If we haven't exceeded max attempts, retry
            if current_attempts < self.max_attempts:
                await store.update_rollout(
                    rollout_id=attempt.rollout_id,
                    status="requeuing",
                )
                return

        # If we can't retry or shouldn't retry, mark as failed
        await store.update_rollout(
            rollout_id=attempt.rollout_id,
            status="failed",
        )

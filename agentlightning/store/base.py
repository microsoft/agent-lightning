# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import NamedResources, ResourcesUpdate, RolloutStatus, RolloutV2


def is_queuing(rollout: RolloutV2) -> bool:
    return rollout.status == "queuing" or rollout.status == "requeuing"


def is_running(rollout: RolloutV2) -> bool:
    return rollout.status == "preparing" or rollout.status == "running"


def is_finished(rollout: RolloutV2) -> bool:
    return rollout.status == "error" or rollout.status == "success"


T_callable = TypeVar("T_callable", bound=Callable[..., Any])


def healthcheck(func: T_callable) -> T_callable:
    """
    Decorator to run the watchdog healthcheck before executing the decorated method.
    Only runs if the store has a watchdog configured.
    """

    @functools.wraps(func)
    async def wrapper(self: LightningStore, *args: Any, **kwargs: Any) -> Any:
        # Run watchdog healthcheck if available
        if self.watchdog is not None:
            await self.watchdog.healthcheck(self)

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

    async def add_rollout(self, rollout: RolloutV2) -> None:
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

    async def add_span(self, span: Span) -> None:
        """
        Add a span to the store.
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
        status: RolloutStatus,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        attempt_start_time: Optional[float] = None,
        last_attempt_status: Optional[RolloutStatus] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update the rollout status and related metadata.

        Args:
            rollout_id: Unique identifier for the rollout to update
            status: New status to set for the rollout
            worker_id: Optional worker identifier
            attempt_sequence_id: Optional sequence ID for the attempt
            attempt_id: Optional unique attempt identifier
            attempt_start_time: Optional timestamp when current attempt started
            last_attempt_status: Optional status of the last attempt
            **kwargs: Additional rollout information to update
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
        retry_condition: Optional[Sequence[RolloutStatus]] = None,
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
        1. Check for unresponsive rollouts (attempted but no response, or no spans for a while)
        2. Check for timed-out rollouts (running too long since start_time)
        3. Update rollout status accordingly via store._update_rollout

        Args:
            store: The LightningStore instance to check rollouts from
        """
        current_time = time.time()

        # Get all running rollouts from the store
        # Note: This assumes store has a method to get running rollouts
        # In a real implementation, you'd need to add this method to LightningStore
        running_rollouts = await store.query_rollouts(status=["preparing", "running"])

        for rollout in running_rollouts:
            # Check for timeout condition (based on start_time)
            if rollout.start_time and current_time - rollout.start_time > self.timeout_seconds:
                await self._handle_failed_rollout(store, rollout, "timeout")
                continue

            # Query the recent spans for this rollout
            recent_spans = await store.query_spans(rollout.rollout_id)

            # Check for unresponsive condition (based on attempt_start_time)
            # This checks if worker attempted but no response, or no spans for a while
            if (
                rollout.attempt_start_time
                and current_time - rollout.attempt_start_time > self.unresponsive_seconds
                and not recent_spans
            ):

                await self._handle_failed_rollout(store, rollout, "unresponsive")
                continue

            # Additionally, check if there's no recent span activity
            if self._no_recent_span_activity(recent_spans, current_time):
                await self._handle_failed_rollout(store, rollout, "unresponsive")

    def _no_recent_span_activity(self, spans: List[Span], current_time: float) -> bool:
        """
        Check if there's no recent span activity within the unresponsive threshold.

        Args:
            spans: List of spans for the rollout
            current_time: Current timestamp

        Returns:
            True if no recent activity, False otherwise
        """
        if not spans:
            return True

        # Find the most recent span activity
        latest_span_time = 0.0
        for span in spans:
            if span.end_time:
                latest_span_time = max(latest_span_time, span.end_time)
            elif span.start_time:
                latest_span_time = max(latest_span_time, span.start_time)

        # If no span times found or latest activity is too old
        return latest_span_time == 0.0 or current_time - latest_span_time > self.unresponsive_seconds

    async def _handle_failed_rollout(self, store: LightningStore, rollout: RolloutV2, status: RolloutStatus) -> None:
        """
        Handle a failed rollout by either retrying or marking as failed.

        Args:
            store: The LightningStore instance
            rollout: The rollout that failed
            status: The failure status ("timeout" or "unresponsive")
        """
        # Check if this status should trigger a retry
        if status in self.retry_condition:
            # Get current attempt count (default to 1 if not set)
            current_attempts = rollout.attempt_sequence_id or 1

            # If we haven't exceeded max attempts, retry
            if current_attempts < self.max_attempts:
                await store.update_rollout(
                    rollout_id=rollout.rollout_id,
                    status="requeuing",
                    last_attempt_status=status,
                    attempt_sequence_id=current_attempts + 1,
                )
                return

        # If we can't retry or shouldn't retry, mark with the failure status
        await store.update_rollout(
            rollout_id=rollout.rollout_id,
            status=status,
        )

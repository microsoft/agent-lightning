# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    RolloutConfig,
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


class LightningStore:
    """
    A centralized, thread-safe, async, data store for the lightning's state.
    This holds the task queue, versioned resources, and completed rollouts.

    The store has a built-in clock and it should be responsible for tracking the times.
    All the time-based operations like retry, timeout, etc. should be handled by the store.
    """

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """
        Add one incomplete rollout to the store, and get an attempt created for it.
        This will immediately sets the rollout to a preparing state, and should be
        used by whoever is going to execute the rollout.

        Return a special rollout with attempt object. Do not update it directly.

        But if the rollout fails or timeouts, it's still possible that the watchdog
        sends it back to the queue for retry.

        To enqueue a rollout to the task queue, use `enqueue_rollout` instead.
        """
        raise NotImplementedError()

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        """
        Adds a new task to the queue with specific metadata and
        returns the rollout object with its unique ID.
        """
        raise NotImplementedError()

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.

        Will set the rollout status to preparing.
        """
        raise NotImplementedError()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """
        Create a new attempt for a given rollout ID and return the attempt details.
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

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[RolloutV2]:
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

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[RolloutV2]:
        """
        Safely retrieves a specific rollout by its ID.
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

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        """
        Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.

        TODO: Add support for waiting for 20 new rollouts, or wait until 80% of the pending ids are completed.
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
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
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
            config: New config for the rollout. If set, will be updated
            metadata: Dictionary of additional metadata to update. If set, will replace the existing metadata
        """
        raise NotImplementedError()

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
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
            metadata: Dictionary of additional metadata to update, will replace the existing metadata
        """
        raise NotImplementedError()

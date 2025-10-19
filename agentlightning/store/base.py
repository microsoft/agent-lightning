# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TaskInput,
)


def is_queuing(rollout: Rollout) -> bool:
    return rollout.status == "queuing" or rollout.status == "requeuing"


def is_running(rollout: Rollout) -> bool:
    return rollout.status == "preparing" or rollout.status == "running"


def is_finished(rollout: Rollout) -> bool:
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
    """Asynchronous persistence layer coordinating rollouts and resources.

    Implementations back the shared task queue, store rollout attempts,
    version named resources, and maintain time-based bookkeeping such as retry
    delays. The interface is intentionally high level so strategies and runners
    can swap storage backends without changing orchestration logic.
    """

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """Create a rollout in ``preparing`` state and allocate its first attempt.

        Args:
            input: Task payload supplied by the requester.
            mode: Optional semantic mode, such as ``"train"`` or ``"eval"``.
            resources_id: Version identifier of the resources snapshot to use.
            config: Optional rollout configuration metadata.
            metadata: User-supplied metadata stored alongside the rollout.

        Returns:
            Attempt wrapper containing the rollout and the first attempt record.

        Raises:
            NotImplementedError: Subclasses must persist the rollout.

        !!! note
            Use [enqueue_rollout()][agentlightning.LightningStore.enqueue_rollout]
            to queue work without taking an immediate attempt.
        """
        raise NotImplementedError()

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """Enqueue a rollout without starting an attempt immediately.

        Args:
            input: Task payload supplied by the requester.
            mode: Optional semantic mode, such as ``"train"`` or ``"eval"``.
            resources_id: Version identifier of the resources snapshot to use.
            config: Optional rollout configuration metadata.
            metadata: User-supplied metadata stored alongside the rollout.

        Returns:
            Rollout record queued for future execution.

        Raises:
            NotImplementedError: Subclasses must persist the rollout.
        """
        raise NotImplementedError()

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """Pop the next rollout from the queue and mark it as ``preparing``.

        Returns:
            Attempt wrapper ready for execution, or ``None`` when the queue is
            empty.

        Raises:
            NotImplementedError: Subclasses must implement queue retrieval.
        """
        raise NotImplementedError()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Create a new attempt for ``rollout_id``.

        Args:
            rollout_id: Identifier of the rollout being retried.

        Returns:
            Attempt wrapper containing the rollout and the newly created attempt.

        Raises:
            NotImplementedError: Subclasses must implement attempt creation.
        """
        raise NotImplementedError()

    async def add_span(self, span: Span) -> Span:
        """Persist a span emitted during rollout execution.

        Implementations should update rollout and attempt state to ``running``
        when the stored span represents execution progress.

        Args:
            span: Span metadata to persist.

        Returns:
            Stored span record.

        Raises:
            NotImplementedError: Subclasses must implement span persistence.
        """
        raise NotImplementedError()

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        """Persist an OpenTelemetry span captured for a rollout attempt.

        Args:
            rollout_id: Identifier of the rollout owning the span.
            attempt_id: Attempt identifier for the span.
            readable_span: Span exported by ``opentelemetry``.
            sequence_id: Explicit sequence number. When omitted, the next value
                from [get_next_span_sequence_id()][agentlightning.LightningStore.get_next_span_sequence_id]
                is used.

        Returns:
            Stored span record.

        Raises:
            NotImplementedError: Subclasses must implement span persistence.
        """
        raise NotImplementedError()

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        """Return rollouts filtered by status or identifier.

        Args:
            status: Optional sequence of [`RolloutStatus`][agentlightning.RolloutStatus]
                values to include.
            rollout_ids: Optional explicit identifiers to retrieve.

        Returns:
            Matching rollouts ordered by backend-specific semantics.

        Raises:
            NotImplementedError: Subclasses must implement the query.
        """
        raise NotImplementedError()

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """Return every attempt created for ``rollout_id``.

        Args:
            rollout_id: Identifier of the rollout being inspected.

        Returns:
            All attempts ordered by creation time.

        Raises:
            NotImplementedError: Subclasses must implement the query.
        """
        raise NotImplementedError()

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        """Fetch a rollout by identifier.

        Args:
            rollout_id: Identifier to retrieve.

        Returns:
            Rollout when found, otherwise ``None``.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Fetch the most recent attempt for ``rollout_id``.

        Args:
            rollout_id: Identifier to inspect.

        Returns:
            Latest attempt or ``None`` when no attempts exist.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Return a specific resources snapshot by identifier.

        Args:
            resources_id: Identifier of the snapshot.

        Returns:
            Resources update or ``None`` when missing.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Fetch the most recent resources snapshot.

        Returns:
            Resources update or ``None`` when no resources have been stored.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Allocate a monotonically increasing sequence id for span ordering.

        Args:
            rollout_id: Identifier of the rollout emitting spans.
            attempt_id: Attempt identifier for the span.

        Returns:
            Sequence number unique within the attempt.

        Raises:
            NotImplementedError: Subclasses must provide the allocator.
        """
        raise NotImplementedError()

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Block until each rollout completes or ``timeout`` expires.

        Args:
            rollout_ids: Identifiers to wait for.
            timeout: Maximum number of seconds to wait. ``None`` waits
                indefinitely.

        Returns:
            Completed rollouts. Entries may remain incomplete if the timeout is
            reached.

        Raises:
            NotImplementedError: Subclasses must implement waiting semantics.
        """
        raise NotImplementedError()

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """Return spans for ``rollout_id`` and optionally a specific attempt.

        Args:
            rollout_id: Identifier of the rollout being inspected.
            attempt_id: Attempt identifier, ``"latest"`` for the newest attempt,
                or ``None`` to include all attempts.

        Returns:
            Ordered list of spans.

        Raises:
            NotImplementedError: Subclasses must implement the query.
        """
        raise NotImplementedError()

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.
        Not implemented by many stores yet.
        """
        raise NotImplementedError()

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version or updates an existing version of named resources and sets it as the latest.
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
    ) -> Rollout:
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

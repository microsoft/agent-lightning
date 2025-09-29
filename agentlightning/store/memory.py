# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Sequence

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

from .base import UNSET, LightningStore, LightningStoreWatchDog, Unset, healthcheck, is_finished, is_queuing


class InMemoryLightningStore(LightningStore):
    """
    In-memory implementation of LightningStore using Python data structures.
    Thread-safe and async-compatible but data is not persistent.
    """

    def __init__(self, watchdog: LightningStoreWatchDog | None = None):
        super().__init__(watchdog)
        self._lock = asyncio.Lock()

        # Task queue and rollouts storage
        self._task_queue: deque[RolloutV2] = deque()
        self._rollouts: Dict[str, RolloutV2] = {}

        # Resources storage (similar to legacy server.py)
        self._resources: Dict[str, ResourcesUpdate] = {}
        self._latest_resources_id: Optional[str] = None

        # Spans storage
        self._spans: Dict[str, List[Span]] = {}  # rollout_id -> list of spans

        # Attempt tracking
        self._attempts: Dict[str, List[Attempt]] = {}  # rollout_id -> list of attempts

        # Completion tracking for wait_for_rollouts
        self._completion_events: Dict[str, asyncio.Event] = {}

    def _register_rollout_unlocked(self, rollout: RolloutV2) -> None:
        """Register rollout assuming caller holds ``self._lock``."""
        self._rollouts[rollout.rollout_id] = rollout
        self._task_queue.append(rollout)
        self._completion_events.setdefault(rollout.rollout_id, asyncio.Event())

    @healthcheck
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
        async with self._lock:
            rollout_id = f"rollout-{uuid.uuid4()}"
            current_time = time.time()

            rollout = RolloutV2(
                rollout_id=rollout_id,
                input=sample,
                mode=mode,
                resources_id=resources_id or self._latest_resources_id,
                start_time=current_time,
                status="queuing",
                metadata=metadata or {},
            )

            self._register_rollout_unlocked(rollout)

            return rollout

    @healthcheck
    async def add_rollout(self, rollout: RolloutV2) -> RolloutV2:
        """Add an existing rollout to the store."""
        async with self._lock:
            self._register_rollout_unlocked(rollout)
            return rollout

    @healthcheck
    async def pop_rollout(self) -> Optional[RolloutV2]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.

        Will set the rollout status to preparing and create a new attempt.
        """
        async with self._lock:
            # Keep looking until we find a rollout that's still in queuing status
            # or the queue is empty
            while self._task_queue:
                rollout = self._task_queue.popleft()

                # Check if rollout is still in a queuing state
                # (it might have been updated to a different status while in queue)
                if is_queuing(rollout):
                    # Update status to preparing
                    rollout.status = "preparing"

                    # Create a new attempt (could be first attempt or retry)
                    attempt_id = f"attempt-{uuid.uuid4()}"
                    current_time = time.time()

                    # Get existing attempts to determine sequence number
                    existing_attempts = self._attempts.get(rollout.rollout_id, [])
                    sequence_id = len(existing_attempts) + 1

                    attempt = Attempt(
                        rollout_id=rollout.rollout_id,
                        attempt_id=attempt_id,
                        sequence_id=sequence_id,
                        start_time=current_time,
                        status="preparing",
                    )

                    if rollout.rollout_id not in self._attempts:
                        self._attempts[rollout.rollout_id] = []
                    self._attempts[rollout.rollout_id].append(attempt)

                    return rollout

                # If not in queuing state, skip this rollout and continue
                # (it was updated externally and should not be processed)

            # No valid rollouts found
            return None

    @healthcheck
    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        """
        Query and retrieve rollouts filtered by their status.
        If no status is provided, returns all rollouts.
        """
        async with self._lock:
            if status is None:
                return list(self._rollouts.values())

            status_set = set(status)
            return [rollout for rollout in self._rollouts.values() if rollout.status in status_set]

    @healthcheck
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """
        Query and retrieve all attempts associated with a specific rollout ID.
        Returns an empty list if no attempts are found.
        """
        async with self._lock:
            return self._attempts.get(rollout_id, [])

    @healthcheck
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """
        Safely retrieves the latest attempt for a given rollout ID.
        """
        async with self._lock:
            attempts = self._attempts.get(rollout_id, [])
            if not attempts:
                return None
            return max(attempts, key=lambda a: a.sequence_id)

    @healthcheck
    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        async with self._lock:
            update = ResourcesUpdate(resources_id=resources_id, resources=resources)
            self._resources[resources_id] = update
            self._latest_resources_id = resources_id
            return update

    @healthcheck
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves a specific version of named resources by its ID.
        """
        async with self._lock:
            return self._resources.get(resources_id)

    @healthcheck
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves the latest version of named resources.
        """
        async with self._lock:
            if self._latest_resources_id:
                return self._resources.get(self._latest_resources_id)
            return None

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """
        Get the next span sequence ID for a given rollout and attempt.
        """
        async with self._lock:
            if rollout_id not in self._spans:
                return 1
            existing_spans = [span for span in self._spans[rollout_id] if span.attempt_id == attempt_id]
            return len(existing_spans) + 1

    async def add_span(self, span: Span) -> Span:
        """Persist a pre-converted span."""
        async with self._lock:
            rollout = self._rollouts.get(span.rollout_id)
            if not rollout:
                raise ValueError(f"Rollout {span.rollout_id} not found")
            attempts = self._attempts.get(span.rollout_id, [])
            current_attempt = next((a for a in attempts if a.attempt_id == span.attempt_id), None)
            latest_attempt = max(attempts, key=lambda a: a.sequence_id) if attempts else None
            if not current_attempt:
                raise ValueError(f"Attempt {span.attempt_id} not found for rollout {span.rollout_id}")
            if not latest_attempt:
                raise ValueError(f"No attempts found for rollout {span.rollout_id}")

            if span.rollout_id not in self._spans:
                self._spans[span.rollout_id] = []
            self._spans[span.rollout_id].append(span)

            # Update attempt heartbeat
            current_attempt.last_heartbeat_time = time.time()
            if current_attempt.status == "preparing":
                current_attempt.status = "running"

            # If the status has already timed out or failed, do not change it

            # Update rollout status if it's the latest attempt
            if rollout.status == "preparing" and current_attempt == latest_attempt:
                rollout.status = "running"

            return span

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        """Add an opentelemetry span to the store."""
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)
        span = Span.from_opentelemetry(
            readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
        )
        await self.add_span(span)
        return span

    @healthcheck
    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        """
        Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.
        """
        completed_rollouts: List[RolloutV2] = []

        async def wait_for_rollout(rollout_id: str):
            # First check if already completed
            async with self._lock:
                rollout = self._rollouts.get(rollout_id)
                if rollout and is_finished(rollout):
                    completed_rollouts.append(rollout)
                    return

            # If not completed and we have an event, wait for completion
            if rollout_id in self._completion_events:
                try:
                    await asyncio.wait_for(self._completion_events[rollout_id].wait(), timeout=timeout)
                    async with self._lock:
                        rollout = self._rollouts.get(rollout_id)
                        if rollout and is_finished(rollout):
                            completed_rollouts.append(rollout)
                except asyncio.TimeoutError:
                    pass

        # Wait for all rollouts concurrently
        await asyncio.gather(*[wait_for_rollout(rid) for rid in rollout_ids], return_exceptions=True)

        return completed_rollouts

    @healthcheck
    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """
        Query and retrieve all spans associated with a specific rollout ID.
        Returns an empty list if no spans are found.
        """
        async with self._lock:
            spans = self._spans.get(rollout_id, [])
            if attempt_id is None:
                return spans
            elif attempt_id == "latest":
                # Find the latest attempt_id
                if not spans:
                    return []
                latest_attempt = max(spans, key=lambda s: s.attempt_id if s.attempt_id else "").attempt_id
                return [s for s in spans if s.attempt_id == latest_attempt]
            else:
                return [s for s in spans if s.attempt_id == attempt_id]

    @healthcheck
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
        """
        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")

            # Update fields if they are not UNSET
            if not isinstance(input, Unset):
                rollout.input = input
            if not isinstance(mode, Unset):
                rollout.mode = mode
            if not isinstance(resources_id, Unset):
                rollout.resources_id = resources_id
            if not isinstance(metadata, Unset):
                # Merge metadata
                if rollout.metadata:
                    rollout.metadata = {**rollout.metadata, **metadata}
                else:
                    rollout.metadata = metadata
            if not isinstance(status, Unset):
                rollout.status = status

            # Set end time for finished rollouts
            if status is not UNSET and is_finished(rollout):
                rollout.end_time = time.time()
                # Signal completion
                if rollout_id in self._completion_events:
                    self._completion_events[rollout_id].set()

            # If requeuing, add back to queue
            elif is_queuing(rollout) and rollout not in self._task_queue:
                self._task_queue.append(rollout)

            return rollout

    @healthcheck
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
        """
        async with self._lock:
            attempts = self._attempts.get(rollout_id, [])
            if not attempts:
                raise ValueError(f"No attempts found for rollout {rollout_id}")

            # Find the attempt to update
            if attempt_id == "latest":
                attempt = max(attempts, key=lambda a: a.sequence_id)
            else:
                attempt = next((a for a in attempts if a.attempt_id == attempt_id), None)
                if not attempt:
                    raise ValueError(f"Attempt {attempt_id} not found for rollout {rollout_id}")

            # Update fields if they are not UNSET
            if not isinstance(status, Unset):
                attempt.status = status
                # Also update end_time if the status indicates completion
                if status in ["failed", "succeeded"]:
                    attempt.end_time = time.time()
            if not isinstance(worker_id, Unset):
                attempt.worker_id = worker_id
            if not isinstance(last_heartbeat_time, Unset):
                attempt.last_heartbeat_time = last_heartbeat_time
            if not isinstance(metadata, Unset):
                # Merge metadata
                if attempt.metadata:
                    attempt.metadata = {**attempt.metadata, **metadata}
                else:
                    attempt.metadata = metadata

            return attempt

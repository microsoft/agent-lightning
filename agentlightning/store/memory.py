# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import ResourcesUpdate, RolloutStatus, RolloutV2

from .base import LightningStore, LightningStoreWatchDog, healthcheck, is_finished


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
                attempt_sequence_id=1,
                metadata=metadata or {},
            )

            self._register_rollout_unlocked(rollout)

            return rollout

    @healthcheck
    async def add_rollout(self, rollout: RolloutV2) -> None:
        """Add an existing rollout to the store."""
        async with self._lock:
            self._register_rollout_unlocked(rollout)

    @healthcheck
    async def pop_rollout(self) -> Optional[RolloutV2]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.

        Will set the rollout status to preparing.
        """
        async with self._lock:
            if not self._task_queue:
                return None

            rollout = self._task_queue.popleft()
            # Update status to preparing (similar to legacy implementation)
            rollout.status = "preparing"
            rollout.attempt_start_time = time.time()

            return rollout

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
    async def update_resources(self, update: ResourcesUpdate):
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        async with self._lock:
            self._resources[update.resources_id] = update
            self._latest_resources_id = update.resources_id

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

    async def add_span(self, span: Span) -> None:
        """Persist a pre-converted span."""
        async with self._lock:
            rollout = self._rollouts.get(span.rollout_id)
            if not rollout:
                raise ValueError(f"Rollout {span.rollout_id} not found")

            if span.rollout_id not in self._spans:
                self._spans[span.rollout_id] = []
            self._spans[span.rollout_id].append(span)

            if rollout.status == "preparing":
                rollout.status = "running"

    async def add_otel_span(self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan) -> Span:
        span = Span.from_opentelemetry(readable_span, rollout_id=rollout_id, attempt_id=attempt_id)
        await self.add_span(span)
        return span

    @healthcheck
    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        """
        Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.
        """
        completed_rollouts: List[RolloutV2] = []

        @healthcheck
        async def wait_for_rollout(rollout_id: str):
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

    async def query_spans(self, rollout_id: str) -> List[Span]:
        """
        Query and retrieve all spans associated with a specific rollout ID.
        Returns an empty list if no spans are found.
        """
        async with self._lock:
            return self._spans.get(rollout_id, [])

    async def _update_rollout(
        self,
        rollout_id: str,
        status: RolloutStatus,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        attempt_start_time: Optional[float] = None,
        last_attempt_status: Optional[RolloutStatus] = None,
        **kwargs: Any,
    ):
        """
        Update the rollout status and related metadata.
        This should only be used internally or by watchdog.
        """
        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")

            # Update fields
            rollout.status = status
            if worker_id is not None:
                rollout.worker_id = worker_id
            if attempt_sequence_id is not None:
                rollout.attempt_sequence_id = attempt_sequence_id
            if attempt_id is not None:
                rollout.attempt_id = attempt_id
            if attempt_start_time is not None:
                rollout.attempt_start_time = attempt_start_time
            if last_attempt_status is not None:
                rollout.last_attempt_status = last_attempt_status

            # Update any additional fields
            for key, value in kwargs.items():
                if hasattr(rollout, key):
                    setattr(rollout, key, value)
                else:
                    rollout.metadata[key] = value

            # Set end time for finished rollouts using utility function
            if is_finished(rollout):
                rollout.end_time = time.time()
                # Signal completion
                if rollout_id in self._completion_events:
                    self._completion_events[rollout_id].set()

            # If requeuing, add back to queue (similar to legacy requeue_task)
            elif status == "requeuing":
                rollout.status = "queuing"  # Reset to queuing for the queue
                self._task_queue.append(rollout)

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Counter, Dict, List, Literal, Optional, Sequence, TypeVar, cast

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

from .base import UNSET, LightningStore, Unset, is_finished, is_queuing
from .utils import healthcheck, propagate_status

T_callable = TypeVar("T_callable", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _healthcheck_wrapper(func: T_callable) -> T_callable:
    """
    Decorator to run the watchdog healthcheck **before** executing the decorated method.
    Only runs if the store has a watchdog configured.
    Prevents recursive healthcheck execution using a flag on the store instance.
    """

    @functools.wraps(func)
    async def wrapper(self: InMemoryLightningStore, *args: Any, **kwargs: Any) -> Any:
        # Check if healthcheck is already running to prevent recursion
        if getattr(self, "_healthcheck_running", False):
            # Skip healthcheck if already running
            return await func(self, *args, **kwargs)

        # Set flag to prevent recursive healthcheck calls
        # This flag is not asyncio/thread-safe, but it doesn't matter
        self._healthcheck_running = True  # type: ignore
        try:
            # The following methods should live inside one lock.
            await self._healthcheck()  # pyright: ignore[reportPrivateUsage]
        finally:
            # Always clear the flag, even if healthcheck fails
            self._healthcheck_running = False  # type: ignore

        # Execute the original method
        # This should be outside the lock.
        return await func(self, *args, **kwargs)

    return cast(T_callable, wrapper)


def _generate_resources_id() -> str:
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]
    return "rs-" + short_id


def _generate_rollout_id() -> str:
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]
    return "ro-" + short_id


def _generate_attempt_id() -> str:
    """We don't need that long because attempts are limited to rollouts."""
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:8]
    return "at-" + short_id


class InMemoryLightningStore(LightningStore):
    """
    In-memory implementation of LightningStore using Python data structures.
    Thread-safe and async-compatible but data is not persistent.

    The methods in this class should generally not call each other,
    especially those that are locked.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Task queue and rollouts storage
        self._task_queue: deque[Rollout] = deque()
        self._rollouts: Dict[str, Rollout] = {}

        # Resources storage (similar to legacy server.py)
        self._resources: Dict[str, ResourcesUpdate] = {}
        self._latest_resources_id: Optional[str] = None

        # Spans storage
        self._spans: Dict[str, List[Span]] = {}  # rollout_id -> list of spans
        self._span_sequence_ids: Dict[str, int] = Counter()  # rollout_id -> sequence_id

        # Attempt tracking
        self._attempts: Dict[str, List[Attempt]] = {}  # rollout_id -> list of attempts

        # Completion tracking for wait_for_rollouts (cross-loop safe)
        self._completion_events: Dict[str, threading.Event] = {}

    @_healthcheck_wrapper
    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """
        Notify the store that I'm about to run a rollout.
        """
        async with self._lock:
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id or self._latest_resources_id,
                start_time=current_time,
                status="preparing",
                metadata=metadata or {},
            )

            # Create the initial attempt
            attempt_id = _generate_attempt_id()
            attempt = Attempt(
                rollout_id=rollout.rollout_id,
                attempt_id=attempt_id,
                sequence_id=1,
                start_time=current_time,
                status="preparing",
            )

            self._attempts[rollout.rollout_id] = [attempt]
            self._rollouts[rollout.rollout_id] = rollout

            # Manully added rollout is not added to task queue. It's already preparing
            self._completion_events.setdefault(rollout.rollout_id, threading.Event())

            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @_healthcheck_wrapper
    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        async with self._lock:
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id or self._latest_resources_id,
                start_time=current_time,
                status="queuing",  # should be queuing
                metadata=metadata or {},
            )

            self._rollouts[rollout.rollout_id] = rollout
            self._task_queue.append(rollout)  # add it to the end of the queue
            self._completion_events.setdefault(rollout.rollout_id, threading.Event())

            return rollout

    @_healthcheck_wrapper
    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
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
                    attempt_id = _generate_attempt_id()
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

                    return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

                # If not in queuing state, skip this rollout and continue
                # (it was updated externally and should not be processed)

            # No valid rollouts found
            return None

    @_healthcheck_wrapper
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """
        Create a new attempt for a given rollout ID and return the attempt details.
        """
        async with self._lock:
            # Get the rollout
            rollout = self._rollouts.get(rollout_id)
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")

            # Get existing attempts to determine sequence number
            existing_attempts = self._attempts.get(rollout_id, [])
            sequence_id = len(existing_attempts) + 1

            # We don't care whether the max attempts have reached or not
            # This attempt is from user trigger

            # Create new attempt
            attempt_id = _generate_attempt_id()
            current_time = time.time()

            attempt = Attempt(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                sequence_id=sequence_id,
                start_time=current_time,
                status="preparing",
            )

            # Add attempt to storage
            if rollout_id not in self._attempts:
                self._attempts[rollout_id] = []
            self._attempts[rollout_id].append(attempt)

            self._completion_events.setdefault(rollout.rollout_id, threading.Event())

            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @_healthcheck_wrapper
    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        """
        Query and retrieve rollouts filtered by their status and rollout ids.
        If no status is provided, returns all rollouts.
        """
        async with self._lock:
            rollouts = list(self._rollouts.values())

            # Filter by rollout_ids if provided
            if rollout_ids is not None:
                rollout_ids_set = set(rollout_ids)
                rollouts = [rollout for rollout in rollouts if rollout.rollout_id in rollout_ids_set]

            # Filter by status if provided
            if status is not None:
                status_set = set(status)
                rollouts = [rollout for rollout in rollouts if rollout.status in status_set]

            return rollouts

    @_healthcheck_wrapper
    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        """
        Safely retrieves a specific rollout by its ID.
        """
        async with self._lock:
            return self._rollouts.get(rollout_id)

    @_healthcheck_wrapper
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """
        Query and retrieve all attempts associated with a specific rollout ID.
        Returns an empty list if no attempts are found.
        """
        async with self._lock:
            return self._attempts.get(rollout_id, [])

    @_healthcheck_wrapper
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """
        Safely retrieves the latest attempt for a given rollout ID.
        """
        async with self._lock:
            attempts = self._attempts.get(rollout_id, [])
            if not attempts:
                return None
            return max(attempts, key=lambda a: a.sequence_id)

    @_healthcheck_wrapper
    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        resources_id = _generate_resources_id()
        async with self._lock:
            update = ResourcesUpdate(resources_id=resources_id, resources=resources)
            self._resources[resources_id] = update
            self._latest_resources_id = resources_id
            return update

    @_healthcheck_wrapper
    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        async with self._lock:
            update = ResourcesUpdate(resources_id=resources_id, resources=resources)
            self._resources[resources_id] = update
            self._latest_resources_id = resources_id
            return update

    @_healthcheck_wrapper
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves a specific version of named resources by its ID.
        """
        async with self._lock:
            return self._resources.get(resources_id)

    @_healthcheck_wrapper
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
        The number is strictly increasing for each rollout.
        The store will not issue the same sequence ID twice.
        """
        async with self._lock:
            self._span_sequence_ids[rollout_id] += 1
            return self._span_sequence_ids[rollout_id]

    async def add_span(self, span: Span) -> Span:
        """Persist a pre-converted span."""
        async with self._lock:
            self._span_sequence_ids[span.rollout_id] = max(self._span_sequence_ids[span.rollout_id], span.sequence_id)
            return await self._add_span_unlocked(span)

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        """Add an opentelemetry span to the store."""
        async with self._lock:
            if sequence_id is None:
                # Issue a new sequence ID for the rollout
                self._span_sequence_ids[rollout_id] += 1
                sequence_id = self._span_sequence_ids[rollout_id]
            else:
                # Comes from a provided sequence ID
                # Make sure our counter is strictly increasing
                self._span_sequence_ids[rollout_id] = max(self._span_sequence_ids[rollout_id], sequence_id)

            span = Span.from_opentelemetry(
                readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
            )
            await self._add_span_unlocked(span)
            return span

    async def _add_span_unlocked(self, span: Span) -> Span:
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
        if current_attempt.status in ["preparing", "unresponsive"]:
            current_attempt.status = "running"

        # If the status has already timed out or failed, do not change it

        # Update rollout status if it's the latest attempt
        if current_attempt == latest_attempt:
            if rollout.status == "preparing":
                rollout.status = "running"
            elif rollout.status in ["queuing", "requeuing"]:
                try:
                    self._task_queue.remove(rollout)
                except ValueError:
                    logger.warning(
                        f"Trying to remove rollout {rollout.rollout_id} from the queue but it's not in the queue."
                    )
                rollout.status = "running"

        return span

    @_healthcheck_wrapper
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """
        Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.

        This method does not change the state of the store.
        """
        completed_rollouts: List[Rollout] = []

        async def wait_for_rollout(rollout_id: str):
            # First check if already completed
            async with self._lock:
                rollout = self._rollouts.get(rollout_id)
                if rollout and is_finished(rollout):
                    completed_rollouts.append(rollout)
                    return

            # No timeout, return immediately
            if timeout is not None and timeout <= 0:
                return

            # If not completed and we have an event, wait for completion
            if rollout_id in self._completion_events:
                evt = self._completion_events[rollout_id]

                # Wait for the event with proper timeout handling
                # evt.wait() returns True if event was set, False if timeout occurred
                if timeout is None:
                    # Wait indefinitely by polling with finite timeouts
                    # This allows threads to exit cleanly on shutdown
                    while True:
                        result = await asyncio.to_thread(evt.wait, 10.0)  # Poll every 10 seconds
                        if result:  # Event was set
                            break
                        # Loop and check again (continues indefinitely since timeout=None)
                else:
                    # Wait with the specified timeout
                    result = await asyncio.to_thread(evt.wait, timeout)

                # If event was set (not timeout), check if rollout is finished
                if result:
                    async with self._lock:
                        rollout = self._rollouts.get(rollout_id)
                        if rollout and is_finished(rollout):
                            completed_rollouts.append(rollout)

            # Rollout not found, return

        # Wait for all rollouts concurrently
        await asyncio.gather(*[wait_for_rollout(rid) for rid in rollout_ids], return_exceptions=True)

        return completed_rollouts

    @_healthcheck_wrapper
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
                latest_attempt = max(spans, key=lambda s: s.sequence_id if s.attempt_id else "").attempt_id
                return [s for s in spans if s.attempt_id == latest_attempt]
            else:
                return [s for s in spans if s.attempt_id == attempt_id]

    @_healthcheck_wrapper
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
        """
        async with self._lock:
            return await self._update_rollout_unlocked(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id,
                status=status,
                config=config,
                metadata=metadata,
            )

    @_healthcheck_wrapper
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
        """
        async with self._lock:
            attempt = await self._update_attempt_unlocked(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                status=status,
                worker_id=worker_id,
                last_heartbeat_time=last_heartbeat_time,
                metadata=metadata,
            )

        return attempt

    async def _update_rollout_unlocked(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        # No lock inside this one.
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
        if not isinstance(status, Unset):
            rollout.status = status
        if not isinstance(config, Unset):
            rollout.config = config
        if not isinstance(metadata, Unset):
            rollout.metadata = metadata

        # Set end time for finished rollouts
        # Rollout is only finished when it succeeded or fail with no more retries.
        if not isinstance(status, Unset) and is_finished(rollout):
            rollout.end_time = time.time()
            # Signal completion
            if rollout_id in self._completion_events:
                self._completion_events[rollout_id].set()

        # If requeuing, add back to queue
        elif is_queuing(rollout) and rollout not in self._task_queue:
            self._task_queue.append(rollout)

        # If the rollout is no longer in a queueing state, remove it from the queue.
        if not isinstance(status, Unset) and not is_queuing(rollout) and rollout in self._task_queue:
            try:
                self._task_queue.remove(rollout)
            except ValueError:
                # Another coroutine may have already removed the rollout from the queue.
                logger.warning(
                    f"Trying to remove rollout {rollout.rollout_id} from the queue but it's not in the queue."
                )

        # Re-validate the rollout to ensure legality
        Rollout.model_validate(rollout.model_dump())

        return rollout

    async def _update_attempt_unlocked(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        # No lock, but with status propagation.
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            raise ValueError(f"Rollout {rollout_id} not found")

        attempts = self._attempts.get(rollout_id, [])
        if not attempts:
            raise ValueError(f"No attempts found for rollout {rollout_id}")

        latest_attempt = max(attempts, key=lambda a: a.sequence_id)

        # Find the attempt to update
        if attempt_id == "latest":
            attempt = latest_attempt
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
            attempt.metadata = metadata

        # Re-validate the attempt to ensure legality
        Attempt.model_validate(attempt.model_dump())

        if attempt == latest_attempt:

            async def _update_status(rollout_id: str, status: RolloutStatus) -> Rollout:
                return await self._update_rollout_unlocked(rollout_id, status=status)

            # Propagate the status to the rollout
            await propagate_status(
                _update_status,
                attempt,
                rollout.config,
            )

        return attempt

    async def _healthcheck(self) -> None:
        """Perform healthcheck against all running rollouts in the store."""
        async with self._lock:
            running_rollouts: List[AttemptedRollout] = []
            for rollout in self._rollouts.values():
                if rollout.status in ["preparing", "running"]:
                    all_attempts = self._attempts.get(rollout.rollout_id, [])
                    if not all_attempts:
                        # The rollout is running but has no attempts, this should not happen
                        logger.error(f"Rollout {rollout.rollout_id} is running but has no attempts")
                        continue
                    latest_attempt = max(all_attempts, key=lambda a: a.sequence_id)
                    running_rollouts.append(AttemptedRollout(**rollout.model_dump(), attempt=latest_attempt))

            async def _update_attempt_status(rollout_id: str, attempt_id: str, status: AttemptStatus) -> Attempt:
                return await self._update_attempt_unlocked(rollout_id, attempt_id, status=status)

            async def _update_rollout_status(rollout_id: str, status: RolloutStatus) -> Rollout:
                return await self._update_rollout_unlocked(rollout_id, status=status)

            await healthcheck(
                running_rollouts,
                _update_rollout_status,
                _update_attempt_status,
            )

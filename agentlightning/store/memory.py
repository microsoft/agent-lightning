# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import sys
import threading
import time
import uuid
import weakref
from collections import deque
from collections.abc import Iterable
from collections.abc import Mapping as MappingABC
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    cast,
)

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

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


class _LoopAwareAsyncLock:
    """Async lock that transparently rebinds to the current event loop.

    The lock intentionally remains *thread-unsafe*: callers must only use it from
    one thread at a time. If multiple threads interact with the store, each
    thread gets its own event loop specific lock.
    """

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()

    # When serializing and deserializing, we don't need to serialize the locks.
    # Because another process will have its own set of event loops and its own lock.
    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._locks = weakref.WeakKeyDictionary()

    def _get_lock_for_current_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    async def __aenter__(self) -> asyncio.Lock:
        lock = self._get_lock_for_current_loop()
        await lock.acquire()
        return lock

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None or not lock.locked():
            raise RuntimeError("Lock released without being acquired")
        lock.release()


def estimate_model_size(obj: Any) -> int:
    """Rough recursive size estimate for Pydantic BaseModel instances."""

    if isinstance(obj, BaseModel):
        values = cast(Iterable[Any], obj.__dict__.values())
        return sum(estimate_model_size(value) for value in values) + sys.getsizeof(cast(object, obj))
    if isinstance(obj, MappingABC):
        mapping = cast(Mapping[Any, Any], obj)
        return sum(estimate_model_size(value) for value in mapping.values()) + sys.getsizeof(cast(object, obj))
    if isinstance(obj, (list, tuple, set)):
        iterable = cast(Iterable[Any], obj)
        return sum(estimate_model_size(value) for value in iterable) + sys.getsizeof(cast(object, obj))
    return sys.getsizeof(cast(object, obj))


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


def _detect_total_memory_bytes() -> int:
    """Best-effort detection of the total available system memory in bytes."""

    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        # Fallback to 8GB if memory cannot be detected.
        logger.error("psutil is not installed. Falling back to 8GB of memory in total.")
        return 8 * 1024**3


class InMemoryLightningStore(LightningStore):
    """
    In-memory implementation of LightningStore using Python data structures.
    Thread-safe and async-compatible but data is not persistent.

    The methods in this class should generally not call each other,
    especially those that are locked.

    Args:
        eviction_memory_threshold: The threshold for evicting spans in bytes.
            By default, it's 70% of the total VRAM available.
        safe_memory_threshold: The threshold for safe memory usage in bytes.
            By default, it's 80% of the eviction threshold.
        span_size_estimator: A function to estimate the size of a span in bytes.
            By default, it's a simple size estimator that uses sys.getsizeof.
    """

    def __init__(
        self,
        *,
        eviction_memory_threshold: float | int | None = None,
        safe_memory_threshold: float | int | None = None,
        span_size_estimator: Callable[[Span], int] | None = None,
    ):
        self._lock = _LoopAwareAsyncLock()

        # Task queue and rollouts storage
        self._task_queue: deque[Rollout] = deque()
        self._rollouts: Dict[str, Rollout] = {}

        # Resources storage (similar to legacy server.py)
        self._resources: Dict[str, ResourcesUpdate] = {}
        self._latest_resources_id: Optional[str] = None

        # Spans storage
        self._spans: Dict[str, List[Span]] = {}  # rollout_id -> list of spans
        self._span_sequence_ids: Dict[str, int] = Counter()  # rollout_id -> sequence_id
        self._span_bytes_by_rollout: Dict[str, int] = Counter()
        self._total_span_bytes: int = 0
        self._evicted_rollout_span_sets: Set[str] = set()

        self._memory_capacity_bytes = _detect_total_memory_bytes()
        if self._memory_capacity_bytes <= 0:
            raise ValueError("Detected memory capacity must be positive")

        self._eviction_threshold_bytes = self._resolve_memory_threshold(
            eviction_memory_threshold,
            default_ratio=0.7,
            capacity_bytes=self._memory_capacity_bytes,
            name="eviction_memory_threshold",
            minimum=1,
        )

        if safe_memory_threshold is None:
            safe_memory_threshold = max(int(self._eviction_threshold_bytes * 0.8), 0)

        self._safe_threshold_bytes = self._resolve_memory_threshold(
            safe_memory_threshold,
            default_ratio=self._eviction_threshold_bytes / self._memory_capacity_bytes,
            capacity_bytes=self._memory_capacity_bytes,
            name="safe_memory_threshold",
            minimum=0,
        )

        if not (0 <= self._safe_threshold_bytes < self._eviction_threshold_bytes):
            raise ValueError("safe_memory_threshold must be smaller than eviction_memory_threshold")
        self._custom_span_size_estimator = span_size_estimator

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
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """Notify the store that I'm about to run a rollout.

        See [`LightningStore.start_rollout()`][agentlightning.LightningStore.start_rollout] for semantics.
        """
        async with self._lock:
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
            rollout_metadata = dict(metadata) if metadata is not None else {}

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id or self._latest_resources_id,
                start_time=current_time,
                status="preparing",
                config=rollout_config,
                metadata=rollout_metadata,
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
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """Adds a new task to the queue with specific metadata and returns the rollout.

        See [`LightningStore.enqueue_rollout()`][agentlightning.LightningStore.enqueue_rollout] for semantics.
        """
        async with self._lock:
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
            rollout_metadata = dict(metadata) if metadata is not None else {}

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id or self._latest_resources_id,
                start_time=current_time,
                status="queuing",  # should be queuing
                config=rollout_config,
                metadata=rollout_metadata,
            )

            self._rollouts[rollout.rollout_id] = rollout
            self._task_queue.append(rollout)  # add it to the end of the queue
            self._completion_events.setdefault(rollout.rollout_id, threading.Event())

            return rollout

    @_healthcheck_wrapper
    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """Retrieves the next task from the queue without blocking.
        Returns `None` if the queue is empty.

        Will set the rollout status to preparing and create a new attempt.

        See [`LightningStore.dequeue_rollout()`][agentlightning.LightningStore.dequeue_rollout] for semantics.
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

                    # Sync attempt status to rollout
                    await self._update_rollout_unlocked(rollout.rollout_id, status="preparing")

                    return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

                # If not in queuing state, skip this rollout and continue
                # (it was updated externally and should not be processed)

            # No valid rollouts found
            return None

    @_healthcheck_wrapper
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Creates a new attempt for a given rollout ID and return the attempt details.

        See [`LightningStore.start_attempt()`][agentlightning.LightningStore.start_attempt] for semantics.
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

            # Sync attempt status to rollout
            await self._update_rollout_unlocked(rollout_id, status="preparing")

            self._completion_events.setdefault(rollout.rollout_id, threading.Event())

            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @_healthcheck_wrapper
    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        """Retrieves rollouts filtered by their status and rollout ids.
        If no status is provided, returns all rollouts.

        See [`LightningStore.query_rollouts()`][agentlightning.LightningStore.query_rollouts] for semantics.
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
        """Retrieves a specific rollout by its ID.

        See [`LightningStore.get_rollout_by_id()`][agentlightning.LightningStore.get_rollout_by_id] for semantics.
        """
        async with self._lock:
            return self._rollouts.get(rollout_id)

    @_healthcheck_wrapper
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """Retrieves all attempts associated with a specific rollout ID.
        Returns an empty list if no attempts are found.

        See [`LightningStore.query_attempts()`][agentlightning.LightningStore.query_attempts] for semantics.
        """
        async with self._lock:
            return self._attempts.get(rollout_id, [])

    @_healthcheck_wrapper
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Retrieves the latest attempt for a given rollout ID.

        See [`LightningStore.get_latest_attempt()`][agentlightning.LightningStore.get_latest_attempt] for semantics.
        """
        async with self._lock:
            attempts = self._attempts.get(rollout_id, [])
            if not attempts:
                return None
            return max(attempts, key=lambda a: a.sequence_id)

    @_healthcheck_wrapper
    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """Stores a new version of named resources and sets it as the latest.

        See [`LightningStore.add_resources()`][agentlightning.LightningStore.add_resources] for semantics.
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

        See [`LightningStore.update_resources()`][agentlightning.LightningStore.update_resources] for semantics.
        """
        async with self._lock:
            update = ResourcesUpdate(resources_id=resources_id, resources=resources)
            self._resources[resources_id] = update
            self._latest_resources_id = resources_id
            return update

    @_healthcheck_wrapper
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Retrieves a specific version of named resources by its ID.

        See [`LightningStore.get_resources_by_id()`][agentlightning.LightningStore.get_resources_by_id] for semantics.
        """
        async with self._lock:
            return self._resources.get(resources_id)

    @_healthcheck_wrapper
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Retrieves the latest version of named resources.

        See [`LightningStore.get_latest_resources()`][agentlightning.LightningStore.get_latest_resources] for semantics.
        """
        async with self._lock:
            if self._latest_resources_id:
                return self._resources.get(self._latest_resources_id)
            return None

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Get the next span sequence ID for a given rollout and attempt.
        The number is strictly increasing for each rollout.
        The store will not issue the same sequence ID twice.

        See [`LightningStore.get_next_span_sequence_id()`][agentlightning.LightningStore.get_next_span_sequence_id] for semantics.
        """
        async with self._lock:
            self._span_sequence_ids[rollout_id] += 1
            return self._span_sequence_ids[rollout_id]

    async def add_span(self, span: Span) -> Span:
        """Persist a pre-converted span.

        See [`LightningStore.add_span()`][agentlightning.LightningStore.add_span] for semantics.
        """
        async with self._lock:
            self._span_sequence_ids[span.rollout_id] = max(self._span_sequence_ids[span.rollout_id], span.sequence_id)
            return await self._add_span_unlocked(span)

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        """Add an opentelemetry span to the store.

        See [`LightningStore.add_otel_span()`][agentlightning.LightningStore.add_otel_span] for semantics.
        """
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
        self._account_span_size(span)
        self._maybe_evict_spans()

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

    @staticmethod
    def _resolve_memory_threshold(
        value: float | int | None,
        *,
        default_ratio: float,
        capacity_bytes: int,
        name: str,
        minimum: int,
    ) -> int:
        if value is None:
            resolved = int(capacity_bytes * default_ratio)
        elif isinstance(value, float):
            if minimum == 0:
                if not (0 <= value <= 1):
                    raise ValueError(f"{name} ratio must be between 0 and 1 inclusive")
            else:
                if not (0 < value <= 1):
                    raise ValueError(f"{name} ratio must be greater than 0 and at most 1")
            resolved = int(capacity_bytes * value)
        else:
            value_int = value
            if value_int < 0:
                raise ValueError(f"{name} must be non-negative")
            resolved = value_int

        if resolved < minimum:
            raise ValueError(f"{name} must be at least {minimum} bytes")

        return resolved

    def _account_span_size(self, span: Span) -> int:
        if self._custom_span_size_estimator is not None:
            size = max(int(self._custom_span_size_estimator(span)), 0)
        else:
            size = estimate_model_size(span)

        self._span_bytes_by_rollout[span.rollout_id] += size
        self._total_span_bytes += size
        return size

    def _maybe_evict_spans(self) -> None:
        if self._total_span_bytes <= self._eviction_threshold_bytes:
            return

        candidates: List[tuple[float, str]] = []
        for rollout_id, spans in self._spans.items():
            if not spans:
                continue
            rollout = self._rollouts.get(rollout_id)
            start_time = rollout.start_time if rollout is not None else (spans[0].start_time or 0.0)
            candidates.append((start_time, rollout_id))

        candidates.sort(key=lambda item: item[0])

        logger.info(f"Evicting spans for {len(candidates)} rollouts to free up memory...")
        memory_consumed_before = self._total_span_bytes
        for _, rollout_id in candidates:
            if self._total_span_bytes <= self._safe_threshold_bytes:
                break
            logger.debug(f"Evicting spans for rollout {rollout_id} to free up memory...")
            self._evict_spans_for_rollout(rollout_id)
        logger.info(f"Freed up {memory_consumed_before - self._total_span_bytes} bytes of memory")

    def _evict_spans_for_rollout(self, rollout_id: str) -> None:
        spans = self._spans.pop(rollout_id, [])
        if not spans:
            return
        removed_bytes = self._span_bytes_by_rollout.pop(rollout_id, 0)
        self._total_span_bytes = max(self._total_span_bytes - removed_bytes, 0)
        self._evicted_rollout_span_sets.add(rollout_id)

    @_healthcheck_wrapper
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.

        This method does not change the state of the store.

        See [`LightningStore.wait_for_rollouts()`][agentlightning.LightningStore.wait_for_rollouts] for semantics.
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

        See [`LightningStore.query_spans()`][agentlightning.LightningStore.query_spans] for semantics.
        """
        async with self._lock:
            if rollout_id in self._evicted_rollout_span_sets:
                raise RuntimeError(f"Spans for rollout {rollout_id} have been evicted")
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
        """Update the rollout status and related metadata.

        See [`LightningStore.update_rollout()`][agentlightning.LightningStore.update_rollout] for semantics.
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
        """Update a specific or latest attempt for a given rollout.

        See [`LightningStore.update_attempt()`][agentlightning.LightningStore.update_attempt] for semantics.
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

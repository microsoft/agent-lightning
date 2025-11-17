# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
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
    Worker,
)

from .base import UNSET, LightningStore, LightningStoreCapabilities, Unset, is_finished, is_queuing
from .collection import Filter, LightningCollections
from .utils import healthcheck, propagate_status

T_callable = TypeVar("T_callable", bound=Callable[..., Any])
T_model = TypeVar("T_model", bound=BaseModel)
T_collections = TypeVar("T_collections", bound=LightningCollections)


logger = logging.getLogger(__name__)


def _healthcheck_wrapper(func: T_callable) -> T_callable:
    """
    Decorator to run the watchdog healthcheck **before** executing the decorated method.
    Only runs if the store has a watchdog configured.
    Prevents recursive healthcheck execution using a flag on the store instance.
    """

    @functools.wraps(func)
    async def wrapper(self: CollectionBasedLightningStore[T_collections], *args: Any, **kwargs: Any) -> Any:
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


class CollectionBasedLightningStore(LightningStore, Generic[T_collections]):
    """It's the standard implementation of LightningStore that uses collections to store data.

    If the store implementation is to use the store's default behavior, it's recommended to
    inherit from this class and override the methods if needed.
    Bring your own collection implementation by using a different `collections` argument.

    Args:
        collections: The collections to use for storage.
    """

    def __init__(
        self,
        *,
        collections: T_collections,
    ):
        # rollouts and spans' storage
        self.collections = collections

        # Caches the latest resources ID.
        self._latest_resources_id: Union[str, None, Unset] = UNSET

    async def _ensure_latest_resources_id(self) -> Optional[str]:
        if isinstance(self._latest_resources_id, Unset):
            latest_resources = await self.collections.resources.get(sort_by="update_time", sort_order="desc")
            if latest_resources:
                self._latest_resources_id = latest_resources.resources_id
            else:
                self._latest_resources_id = None
        return self._latest_resources_id

    async def _get_or_create_worker(self, worker_id: str) -> Worker:
        """Create a worker if it doesn't exist.

        This is different from upsert because we don't want to update the worker if it already exists.
        """
        worker = await self.collections.workers.get({"worker_id": {"exact": worker_id}})
        if worker is None:
            worker = Worker(worker_id=worker_id)
            await self.collections.workers.insert([worker])
        return worker

    async def _sync_worker_with_attempt(self, attempt: Attempt) -> None:
        worker_id = attempt.worker_id
        if not worker_id:
            return

        worker = await self._get_or_create_worker(worker_id)
        now = time.time()

        if attempt.status in ("succeeded", "failed"):
            if worker.status != "idle":
                worker.last_idle_time = now
            worker.status = "idle"
            worker.current_rollout_id = None
            worker.current_attempt_id = None
        elif attempt.status in ("timeout", "unresponsive"):
            if worker.status != "unknown":
                worker.last_idle_time = now
            worker.status = "unknown"
            worker.current_rollout_id = None
            worker.current_attempt_id = None
        else:
            transitioned = worker.status != "busy" or worker.current_attempt_id != attempt.attempt_id
            if transitioned:
                worker.last_busy_time = now
            worker.status = "busy"
            worker.current_rollout_id = attempt.rollout_id
            worker.current_attempt_id = attempt.attempt_id

        # Validate the schema to make sure it's valid.
        Worker.model_validate(worker.model_dump())
        await self.collections.workers.update([worker])

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store.

        This store supports no capability. The capability depends on the underlying collections.
        """
        return LightningStoreCapabilities()

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
        async with self.collections.atomic():
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
            rollout_metadata = dict(metadata) if metadata is not None else {}

            resources_id = resources_id if resources_id is not None else await self._ensure_latest_resources_id()

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id,
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

            await self.collections.attempts.insert([attempt])
            await self.collections.rollouts.insert([rollout])

            # Notify the subclass that the rollout status has changed.
            await self.on_rollout_update(rollout)

            # Return a rollout with attempt attached.
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
        async with self.collections.atomic():
            rollout_id = _generate_rollout_id()
            current_time = time.time()

            rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
            rollout_metadata = dict(metadata) if metadata is not None else {}

            resources_id = resources_id if resources_id is not None else await self._ensure_latest_resources_id()

            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id,
                start_time=current_time,
                status="queuing",  # should be queuing
                config=rollout_config,
                metadata=rollout_metadata,
            )

            await self.collections.rollouts.insert([rollout])
            await self.collections.rollout_queue.enqueue([rollout])  # add it to the end of the queue

            # Notify the subclass that the rollout status has changed.
            await self.on_rollout_update(rollout)

            # Return the rollout with no attempt attached.
            return rollout

    @_healthcheck_wrapper
    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        """Retrieves the next task from the queue without blocking.
        Returns `None` if the queue is empty.

        Will set the rollout status to preparing and create a new attempt.

        See [`LightningStore.dequeue_rollout()`][agentlightning.LightningStore.dequeue_rollout] for semantics.
        """
        async with self.collections.atomic():
            if worker_id is not None:
                worker = await self._get_or_create_worker(worker_id)
                worker.last_dequeue_time = time.time()
                worker.status = "idle"
                await self.collections.workers.update([worker])

            # Keep looking until we find a rollout that's still in queuing status
            # or the queue is empty
            while self.collections.rollout_queue.size() > 0:
                dequeued = await self.collections.rollout_queue.dequeue(1)
                if not dequeued:
                    break
                rollout = dequeued[0]

                # Check if rollout is still in a queuing state
                # (it might have been updated to a different status while in queue)
                if is_queuing(rollout):
                    # Create a new attempt (could be first attempt or retry)
                    attempt_id = _generate_attempt_id()
                    current_time = time.time()

                    # Get existing attempts to determine sequence number
                    existing_attempts = await self._query_attempts_for_rollout_unlocked(rollout.rollout_id)
                    sequence_id = len(existing_attempts) + 1

                    attempt = Attempt(
                        rollout_id=rollout.rollout_id,
                        attempt_id=attempt_id,
                        sequence_id=sequence_id,
                        start_time=current_time,
                        status="preparing",
                    )

                    await self.collections.attempts.insert([attempt])

                    # Sync attempt status to rollout
                    await self._update_rollout_unlocked(rollout.rollout_id, status="preparing")

                    return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

                # If not in queuing state, skip this rollout and continue
                # (it was updated externally and should not be processed)
                logger.warning(
                    f"Rollout {rollout.rollout_id} is not in queuing state: {rollout.status}, skipping dequeuing"
                )

            # No valid rollouts found
            return None

    @_healthcheck_wrapper
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Creates a new attempt for a given rollout ID and return the attempt details.

        See [`LightningStore.start_attempt()`][agentlightning.LightningStore.start_attempt] for semantics.
        """
        async with self.collections.atomic():
            # Get the rollout
            rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")

            # Get existing attempts to determine sequence number
            existing_attempts = await self._query_attempts_for_rollout_unlocked(rollout_id)
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
            await self.collections.attempts.insert([attempt])

            # Sync attempt status to rollout
            await self._update_rollout_unlocked(rollout_id, status="preparing")

            # Return the rollout with the new attempt attached.
            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @_healthcheck_wrapper
    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        """Retrieves rollouts filtered by their status and rollout ids.
        If no status is provided, returns all rollouts.

        See [`LightningStore.query_rollouts()`][agentlightning.LightningStore.query_rollouts] for semantics.
        """
        # Construct filters condition
        filters: Filter = {}
        if rollout_ids is not None:
            filters["rollout_id"] = {"within": list(rollout_ids)}
        if status is not None:
            filters["status"] = {"within": list(status)}

        async with self.collections.atomic():
            rollouts = await self.collections.rollouts.query(filters=filters)

            # Attach the latest attempt to the rollout objects
            # TODO: Maybe we can use asyncio.gather here to speed up the process?
            attempted_rollouts = [
                await self._rollout_to_attempted_rollout_unlocked(rollout) for rollout in rollouts.items
            ]

            return attempted_rollouts

    async def _query_attempts_for_rollout_unlocked(self, rollout_id: str) -> List[Attempt]:
        """The unlocked version of `query_attempts_for_rollout`."""
        result = await self.collections.attempts.query(
            filters={"rollout_id": {"exact": rollout_id}}, sort_by="sequence_id", sort_order="asc"
        )
        return list(result.items)

    @_healthcheck_wrapper
    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Union[Rollout, AttemptedRollout]]:
        """Retrieves a specific rollout by its ID.

        See [`LightningStore.get_rollout_by_id()`][agentlightning.LightningStore.get_rollout_by_id] for semantics.

        If the rollout has been attempted, the latest attempt will also be returned.
        """
        async with self.collections.atomic():
            rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if rollout is None:
                return None
            return await self._rollout_to_attempted_rollout_unlocked(rollout)

    async def _rollout_to_attempted_rollout_unlocked(self, rollout: Rollout) -> Union[Rollout, AttemptedRollout]:
        """Query the latest attempt for the rollout, and attach it to the rollout object.

        If the rollout has no attempts, return the rollout object itself.
        """
        latest_attempt = await self._get_latest_attempt_unlocked(rollout.rollout_id)
        if latest_attempt is None:
            return rollout
        else:
            return AttemptedRollout(**rollout.model_dump(), attempt=latest_attempt)

    async def _get_latest_attempt_unlocked(self, rollout_id: str) -> Optional[Attempt]:
        """The unlocked version of `get_latest_attempt`."""
        return await self.collections.attempts.get(
            filters={"rollout_id": {"exact": rollout_id}}, sort_by="sequence_id", sort_order="desc"
        )

    @_healthcheck_wrapper
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """Retrieves all attempts associated with a specific rollout ID.
        Returns an empty list if no attempts are found.

        See [`LightningStore.query_attempts()`][agentlightning.LightningStore.query_attempts] for semantics.
        """
        async with self.collections.atomic():
            result = await self.collections.attempts.query(
                filters={"rollout_id": {"exact": rollout_id}}, sort_by="sequence_id", sort_order="asc"
            )
            return list(result.items)

    @_healthcheck_wrapper
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Retrieves the latest attempt for a given rollout ID.

        See [`LightningStore.get_latest_attempt()`][agentlightning.LightningStore.get_latest_attempt] for semantics.
        """
        async with self.collections.atomic():
            return await self._get_latest_attempt_unlocked(rollout_id)

    @_healthcheck_wrapper
    async def query_resources(self) -> List[ResourcesUpdate]:
        """Return every stored resource snapshot in insertion order."""
        async with self.collections.atomic():
            # No sorting and no pagination by default
            result = await self.collections.resources.query()
            return list(result.items)

    @_healthcheck_wrapper
    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """Stores a new version of named resources and sets it as the latest.

        See [`LightningStore.add_resources()`][agentlightning.LightningStore.add_resources] for semantics.
        """
        resources_id = _generate_resources_id()
        async with self.collections.atomic():
            current_time = time.time()
            update = ResourcesUpdate(
                resources_id=resources_id,
                resources=resources,
                create_time=current_time,
                update_time=current_time,
                version=1,
            )
            await self.collections.resources.insert([update])
            self._latest_resources_id = resources_id
            return update

    @_healthcheck_wrapper
    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.

        See [`LightningStore.update_resources()`][agentlightning.LightningStore.update_resources] for semantics.
        """
        async with self.collections.atomic():
            current_time = time.time()
            existing = await self.collections.resources.get({"resources_id": {"exact": resources_id}})
            if existing is None:
                update = ResourcesUpdate(
                    resources_id=resources_id,
                    resources=resources,
                    create_time=current_time,
                    update_time=current_time,
                    version=1,
                )
                await self.collections.resources.insert([update])
            else:
                update = existing.model_copy(
                    update={
                        "resources": resources,
                        "update_time": current_time,
                        "version": existing.version + 1,
                    }
                )
                await self.collections.resources.update([update])
            self._latest_resources_id = resources_id
            return update

    @_healthcheck_wrapper
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Retrieves a specific version of named resources by its ID.

        See [`LightningStore.get_resources_by_id()`][agentlightning.LightningStore.get_resources_by_id] for semantics.
        """
        async with self.collections.atomic():
            return await self.collections.resources.get({"resources_id": {"exact": resources_id}})

    @_healthcheck_wrapper
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Retrieves the latest version of named resources.

        See [`LightningStore.get_latest_resources()`][agentlightning.LightningStore.get_latest_resources] for semantics.
        """
        async with self.collections.atomic():
            if self._latest_resources_id:
                return await self.collections.resources.get({"resources_id": {"exact": self._latest_resources_id}})
            return None

    async def _issue_span_sequence_id_unlocked(self, rollout_id: str) -> int:
        """Issue a new span sequence ID for a given rollout."""
        sequence_id = await self.collections.span_sequence_ids.get(rollout_id)
        if sequence_id is None:
            sequence_id = 1
        else:
            sequence_id += 1
        await self.collections.span_sequence_ids.set(rollout_id, sequence_id)
        return sequence_id

    async def _sync_span_sequence_id_unlocked(self, rollout_id: str, sequence_id: int) -> None:
        """Sync the span sequence ID for a given rollout from the input span sequence ID."""
        existing_sequence_id = await self.collections.span_sequence_ids.get(rollout_id)
        if existing_sequence_id is None:
            existing_sequence_id = 0
        await self.collections.span_sequence_ids.set(rollout_id, max(existing_sequence_id, sequence_id))

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Get the next span sequence ID for a given rollout and attempt.
        The number is strictly increasing for each rollout.
        The store will not issue the same sequence ID twice.

        See [`LightningStore.get_next_span_sequence_id()`][agentlightning.LightningStore.get_next_span_sequence_id] for semantics.
        """
        async with self.collections.atomic():
            return await self._issue_span_sequence_id_unlocked(rollout_id)

    async def add_span(self, span: Span) -> Span:
        """Persist a pre-converted span.

        See [`LightningStore.add_span()`][agentlightning.LightningStore.add_span] for semantics.
        """
        async with self.collections.atomic():
            # Update the sequence ID to be synced with latest input span
            await self._sync_span_sequence_id_unlocked(span.rollout_id, span.sequence_id)
            return await self._add_span_unlocked(span)

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        """Add an opentelemetry span to the store.

        See [`LightningStore.add_otel_span()`][agentlightning.LightningStore.add_otel_span] for semantics.
        """
        async with self.collections.atomic():
            if sequence_id is None:
                # Issue a new sequence ID for the rollout
                sequence_id = await self._issue_span_sequence_id_unlocked(rollout_id)
            else:
                # Comes from a provided sequence ID
                # Make sure our counter is strictly increasing
                await self._sync_span_sequence_id_unlocked(rollout_id, sequence_id)

            span = Span.from_opentelemetry(
                readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
            )
            await self._add_span_unlocked(span)
            return span

    async def _add_span_unlocked(self, span: Span) -> Span:
        rollout = await self.collections.rollouts.get({"rollout_id": {"exact": span.rollout_id}})
        if not rollout:
            raise ValueError(f"Rollout {span.rollout_id} not found")
        current_attempt = await self.collections.attempts.get(
            filters={"rollout_id": {"exact": span.rollout_id}, "attempt_id": {"exact": span.attempt_id}},
        )
        latest_attempt = await self.collections.attempts.get(
            filters={"rollout_id": {"exact": span.rollout_id}}, sort_by="sequence_id", sort_order="desc"
        )
        if not current_attempt:
            raise ValueError(f"Attempt {span.attempt_id} not found for rollout {span.rollout_id}")
        if not latest_attempt:
            raise ValueError(f"No attempts found for rollout {span.rollout_id}")

        await self.collections.spans.insert([span])

        # Update attempt heartbeat
        current_attempt.last_heartbeat_time = time.time()
        if current_attempt.status in ["preparing", "unresponsive"]:
            current_attempt.status = "running"

        # If the status has already timed out or failed, do not change it

        # Update rollout status if it's the latest attempt
        if current_attempt.attempt_id == latest_attempt.attempt_id:
            if rollout.status == "preparing":
                rollout.status = "running"
                await self.collections.rollouts.update([rollout])
                await self.on_rollout_update(rollout)
            elif rollout.status in ["queuing", "requeuing"]:
                rollout.status = "running"
                await self.collections.rollouts.update([rollout])
                await self.on_rollout_update(rollout)

        return span

    @_healthcheck_wrapper
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Wait for specified rollouts to complete with a timeout.
        Returns the completed rollouts, potentially incomplete if timeout is reached.

        This method does not change the state of the store.

        See [`LightningStore.wait_for_rollouts()`][agentlightning.LightningStore.wait_for_rollouts] for semantics.
        """
        # Wait for all rollouts concurrently
        rollouts = await asyncio.gather(
            *[self.wait_for_rollout(rid, timeout) for rid in rollout_ids], return_exceptions=True
        )

        for rollout_id, rollout in zip(rollout_ids, rollouts):
            if isinstance(rollout, Exception):
                logger.error(f"Error waiting for rollout {rollout_id}: {rollout}")

        # Filter out the exceptions
        return [rollout for rollout in rollouts if isinstance(rollout, Rollout)]

    async def wait_for_rollout(self, rollout_id: str, timeout: Optional[float] = None) -> Optional[Rollout]:
        """Wait for a specific rollout to complete with a timeout.

        Subclass may use advanced mechanisms like events to accelerate this.

        Returns the completed rollout, or None if timeout is reached.
        """
        # First check if already completed
        async with self.collections.atomic():
            rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if rollout and is_finished(rollout):
                return rollout

        # No timeout, return immediately
        if timeout is not None and timeout <= 0:
            return None

        start_time = time.time()
        deadline = start_time + timeout if timeout is not None else None

        # If not completed, wait for completion
        while deadline is None or time.time() < deadline:
            # Poll every 10 seconds by default
            rest_time = max(0.01, deadline - time.time()) if deadline is not None else 10.0
            await asyncio.sleep(rest_time)
            rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            # check if rollout is finished
            if rollout and is_finished(rollout):
                return rollout

    @_healthcheck_wrapper
    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """
        Query and retrieve all spans associated with a specific rollout ID.
        Returns an empty list if no spans are found.

        See [`LightningStore.query_spans()`][agentlightning.LightningStore.query_spans] for semantics.
        """
        async with self.collections.atomic():
            if attempt_id is None:
                spans = await self.collections.spans.query(filters={"rollout_id": {"exact": rollout_id}})
            elif attempt_id == "latest":
                latest_attempt = await self.collections.attempts.get(
                    filters={"rollout_id": {"exact": rollout_id}}, sort_by="sequence_id", sort_order="desc"
                )
                if not latest_attempt:
                    logger.debug(f"No attempts found for rollout {rollout_id} when querying latest spans")
                    return []
                spans = await self.collections.spans.query(
                    filters={"rollout_id": {"exact": rollout_id}, "attempt_id": {"exact": latest_attempt.attempt_id}}
                )
            else:
                spans = await self.collections.spans.query(
                    filters={"rollout_id": {"exact": rollout_id}, "attempt_id": {"exact": attempt_id}}
                )
            return list(spans.items)

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
        async with self.collections.atomic():
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
        async with self.collections.atomic():
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
        rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
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

        # If requeuing, add back to queue
        # NOTE: We don't need to check if the rollout is already in the queue,
        # because when dequeuing, we will check if the rollout is still in queuing state.
        # If the rollout has been dequeued twice, its status will be changed to preparing in the first dequeue.
        elif is_queuing(rollout):
            await self.collections.rollout_queue.enqueue([rollout])

        # We also don't need to remove non-queuing rollouts from the queue, for similar reasons.

        # Re-validate the rollout to ensure legality
        Rollout.model_validate(rollout.model_dump())

        await self.collections.rollouts.update([rollout])
        await self.on_rollout_update(rollout)

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
        rollout = await self.collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
        if not rollout:
            raise ValueError(f"Rollout {rollout_id} not found")

        latest_attempt = await self.collections.attempts.get(
            {"rollout_id": {"exact": rollout_id}}, sort_by="sequence_id", sort_order="desc"
        )
        if not latest_attempt:
            raise ValueError(f"No attempts found for rollout {rollout_id}")

        # Find the attempt to update
        if attempt_id == "latest":
            attempt = latest_attempt
        else:
            attempt = await self.collections.attempts.get(
                {"rollout_id": {"exact": rollout_id}, "attempt_id": {"exact": attempt_id}}
            )
            if not attempt:
                raise ValueError(f"Attempt {attempt_id} not found for rollout {rollout_id}")

        worker_sync_required = False

        # Update fields if they are not UNSET
        if not isinstance(worker_id, Unset):
            attempt.worker_id = worker_id
            worker_sync_required = worker_sync_required or bool(worker_id)
        if not isinstance(status, Unset):
            attempt.status = status
            # Also update end_time if the status indicates completion
            if status in ["failed", "succeeded"]:
                attempt.end_time = time.time()
            worker_sync_required = worker_sync_required or bool(attempt.worker_id)
        if not isinstance(last_heartbeat_time, Unset):
            attempt.last_heartbeat_time = last_heartbeat_time
        if not isinstance(metadata, Unset):
            attempt.metadata = metadata

        if worker_sync_required and attempt.worker_id:
            await self._sync_worker_with_attempt(attempt)

        # Re-validate the attempt to ensure legality
        Attempt.model_validate(attempt.model_dump())
        # Update the attempt in storage
        await self.collections.attempts.update([attempt])

        if attempt.attempt_id == latest_attempt.attempt_id:

            async def _update_status(rollout_id: str, status: RolloutStatus) -> Rollout:
                return await self._update_rollout_unlocked(rollout_id, status=status)

            # Propagate the status to the rollout
            await propagate_status(
                _update_status,
                attempt,
                rollout.config,
            )

        return attempt

    @_healthcheck_wrapper
    async def query_workers(self) -> List[Worker]:
        """Return the current snapshot of all workers."""
        async with self.collections.atomic():
            result = await self.collections.workers.query()
            return list(result.items)

    @_healthcheck_wrapper
    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        async with self.collections.atomic():
            return await self.collections.workers.get({"worker_id": {"exact": worker_id}})

    @_healthcheck_wrapper
    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        """Create or update a worker entry."""
        async with self.collections.atomic():
            worker = await self._get_or_create_worker(worker_id)
            if not isinstance(heartbeat_stats, Unset):
                worker.heartbeat_stats = dict(heartbeat_stats)
            worker.last_heartbeat_time = time.time()

            Worker.model_validate(worker.model_dump())
            await self.collections.workers.update([worker])
            return worker

    async def on_rollout_update(self, rollout: Rollout) -> None:
        """Callback for subclasses to implement specific logic when a rollout changes.

        Subclass should not lock this method with `collections.atomic()` because the caller will already hold the lock.
        """
        pass

    async def get_running_rollouts(self) -> List[AttemptedRollout]:
        """Get all running rollouts.

        As this is invoked very frequently (probably at every requests),
        subclass can implement hacks to make it more efficient.
        It should also be unlocked and let the caller hold the lock.
        """
        running_rollouts: List[AttemptedRollout] = []
        rollouts = await self.collections.rollouts.query(filters={"status": {"within": ["preparing", "running"]}})

        for rollout in rollouts.items:
            latest_attempt = await self.collections.attempts.get(
                filters={"rollout_id": {"exact": rollout.rollout_id}}, sort_by="sequence_id", sort_order="desc"
            )
            if not latest_attempt:
                # The rollout is running but has no attempts, this should not happen
                logger.error(f"Rollout {rollout.rollout_id} is running but has no attempts")
                continue
            running_rollouts.append(AttemptedRollout(**rollout.model_dump(), attempt=latest_attempt))
        return running_rollouts

    async def _healthcheck(self) -> None:
        """Perform healthcheck against all running rollouts in the store."""
        async with self.collections.atomic():
            running_rollouts = await self.get_running_rollouts()

            async def _update_attempt_status(rollout_id: str, attempt_id: str, status: AttemptStatus) -> Attempt:
                return await self._update_attempt_unlocked(rollout_id, attempt_id, status=status)

            async def _update_rollout_status(rollout_id: str, status: RolloutStatus) -> Rollout:
                return await self._update_rollout_unlocked(rollout_id, status=status)

            await healthcheck(
                running_rollouts,
                _update_rollout_status,
                _update_attempt_status,
            )

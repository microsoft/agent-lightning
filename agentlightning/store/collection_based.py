# Copyright (c) Microsoft. All rights reserved.

"""Collection-based LightningStore implementation.

To developers, please check whether the implementation is correct by checking the following:

1. Whether all `_unlocked_*` methods are guarded by some `atomic()` or `execute()` context.
2. Whether all `atomic()` or `execute()` contexts are labeled (labels="...") correctly.
3. `_unlocked_update_attempt_and_rollout` should be accompanied by `_post_update_rollout`, `_unlocked_sync_worker_with_attempt`.
4. `_post_add_spans` should be called after the spans are inserted into the store.
5. `_unlocked_update_rollout_only` should be accompanied by `_post_update_rollout`.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
import uuid
import warnings
from collections import defaultdict
from types import CoroutineType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel
from typing_extensions import Concatenate

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    FilterField,
    NamedResources,
    PaginatedResult,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    SortOptions,
    Span,
    TaskInput,
    Worker,
    WorkerStatus,
)

from .base import (
    UNSET,
    LightningStore,
    LightningStoreCapabilities,
    LightningStoreStatistics,
    Unset,
    is_finished,
    is_queuing,
)
from .collection import FilterOptions, LightningCollections
from .collection.base import AtomicLabels
from .utils import LATENCY_BUCKETS, rollout_status_from_attempt, scan_unhealthy_rollouts

T_callable = TypeVar("T_callable", bound=Callable[..., Any])
T_model = TypeVar("T_model", bound=BaseModel)
T_collections = TypeVar("T_collections", bound=LightningCollections)

P = ParamSpec("P")
R = TypeVar("R")
C = TypeVar("C")  # The collections type

SelfT = TypeVar("SelfT", bound="CollectionBasedLightningStore[Any]")

logger = logging.getLogger(__name__)


def _with_collections_execute(labels: Sequence[AtomicLabels]):
    """Hands over the function execution to the collections.execute method.
    Used to enable committing and automatic retries.

    The wrapped function should accept an extra locked collection as its first argument.
    """

    def decorator(
        func: Callable[Concatenate[SelfT, T_collections, P], CoroutineType[Any, Any, R]],
    ) -> Callable[Concatenate[SelfT, P], CoroutineType[Any, Any, R]]:

        @functools.wraps(func)
        async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
            async def callback(collections: T_collections) -> R:
                return await func(self, collections, *args, **kwargs)

            return await self.collections.execute(
                callback,
                mode="rw",  # Read-write all enabled.
                snapshot=self._read_snapshot,  # pyright: ignore[reportPrivateUsage]
                commit=True,  # Enable committing.
                labels=labels,
            )

        return wrapper

    return decorator


def tracked(name: str):
    """Decorator to track the execution of the decorated method with Prometheus."""

    _public_methods = frozenset([name for name in LightningStore.__dict__ if not name.startswith("_")])

    def decorator(func: T_callable) -> T_callable:

        @functools.wraps(func)
        async def wrapper(self: CollectionBasedLightningStore[T_collections], *args: Any, **kwargs: Any) -> Any:
            # For backtracking in Mongo-collection methods.
            # Only track the public methods (+healthcheck)
            if name in _public_methods or name == "_healthcheck":
                method_name = name  # pyright: ignore[reportUnusedVariable]
            else:
                method_name = None  # pyright: ignore[reportUnusedVariable]

            if not self._prometheus:  # pyright: ignore[reportPrivateUsage]
                # Skip the tracking because tracking is not configured
                return await func(self, *args, **kwargs)

            start_time = time.perf_counter()
            try:
                ret = await func(self, *args, **kwargs)
                self._total_metric.labels(name, "OK").inc()  # pyright: ignore[reportPrivateUsage]
                return ret
            except Exception as exc:
                self._total_metric.labels(name, exc.__class__.__name__).inc()  # pyright: ignore[reportPrivateUsage]
                raise
            finally:
                elapsed = time.perf_counter() - start_time
                self._latency_metric.labels(name).observe(elapsed)  # pyright: ignore[reportPrivateUsage]

        return cast(T_callable, wrapper)

    return decorator


def healthcheck_before(func: T_callable) -> T_callable:
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
            await self._scan_for_unhealthy_rollouts()  # pyright: ignore[reportPrivateUsage]
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

    The methods in this class should generally not call each other,
    especially those that are locked.

    Args:
        collections: The collections to use for storage.
        read_snapshot: Make sure read operations are atomic. If set to true,
            all read operations like `query_rollouts` will have better consistency.
            It may use an isolated snapshot that supports repeatable reads.
        prometheus: Enable Prometheus tracking.
    """

    def __init__(self, collections: T_collections, *, read_snapshot: bool = False, prometheus: bool = False):
        # rollouts and spans' storage
        self.collections = collections
        self._read_snapshot = read_snapshot
        self._prometheus = prometheus
        self._launch_time = time.time()

        if prometheus:
            from prometheus_client import Counter, Histogram

            self._latency_metric = Histogram(
                "collection_store_latency_seconds",
                "Latency of CollectionBasedLightningStore methods",
                ["method"],
                buckets=LATENCY_BUCKETS,
            )
            self._total_metric = Counter(
                "collection_store_total",
                "Total MongoDB operations",
                ["method", "error_type"],
            )
            self._rollout_counter = Counter(
                "collection_store_rollout_total",
                "Total rollouts",
                ["status", "mode"],
            )
            self._rollout_duration_metric = Histogram(
                "collection_store_rollout_duration_seconds",
                "Duration of rollouts",
                ["status", "mode"],
                buckets=LATENCY_BUCKETS,
            )

    async def statistics(self) -> LightningStoreStatistics:
        """Return the statistics of the store."""
        current_time = time.time()
        return {
            "name": self.__class__.__name__,
            "total_rollouts": await self.collections.rollouts.size(),
            "total_attempts": await self.collections.attempts.size(),
            "total_spans": await self.collections.spans.size(),
            "total_resources": await self.collections.resources.size(),
            "total_workers": await self.collections.workers.size(),
            "uptime": current_time - self._launch_time,
        }

    @tracked("_get_latest_resources")
    async def _get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Get the latest resources from the collections. Returns `None` if no resources are found."""
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["resources"]) as collections:
            return await collections.resources.get(sort={"name": "update_time", "order": "desc"})

    @tracked("_update_or_insert_worker")
    async def _update_or_insert_worker(self, worker: Worker, update_fields: Sequence[str] | None = None) -> Worker:
        """Create a worker if it doesn't exist. Update its `update_fields` if it already exists."""
        async with self.collections.atomic(mode="rw", snapshot=self._read_snapshot, labels=["workers"]) as collections:
            updated_workers = await collections.workers.upsert([worker], update_fields=update_fields)
            return updated_workers[0]

    @tracked("_unlocked_sync_worker_with_attempt")
    async def _unlocked_sync_worker_with_attempt(self, collections: T_collections, attempt: Attempt) -> None:
        """Update the worker's status. This can be done in a separate session."""
        worker_id = attempt.worker_id
        if not worker_id:
            return

        worker = await collections.workers.get({"worker_id": {"exact": worker_id}})
        if not worker:
            worker = Worker(worker_id=worker_id)
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
        await collections.workers.upsert([worker])

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store.

        This store supports no capability. The capability depends on the underlying collections.
        """
        return LightningStoreCapabilities()

    @tracked("_sync_workers_with_attempts")
    @_with_collections_execute(labels=["workers", "attempts"])
    async def _sync_workers_with_attempts(self, collections: T_collections, attempts: Sequence[Attempt]) -> None:
        """Update the worker's status. Locked bulk version of `_unlocked_sync_workers_with_attempts`."""
        for attempt in attempts:
            await self._unlocked_sync_worker_with_attempt(collections, attempt)

    @tracked("start_rollout")
    @healthcheck_before
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
        rollout_id = _generate_rollout_id()
        current_time = time.time()

        rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
        rollout_metadata = dict(metadata) if metadata is not None else {}

        if resources_id is None:
            latest_resources = await self._get_latest_resources()
            resources_id = latest_resources.resources_id if latest_resources is not None else None

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

        async def _insert_rollout_and_attempt(collections: T_collections) -> None:
            await collections.attempts.insert([attempt])
            await collections.rollouts.insert([rollout])

        await self.collections.execute(
            _insert_rollout_and_attempt,
            mode="rw",
            snapshot=self._read_snapshot,
            commit=True,
            labels=["rollouts", "attempts"],
        )
        # Notify the subclass that the rollout status has changed.
        all_fields = list(rollout.__class__.model_fields.keys())
        await self._post_update_rollout([(rollout, all_fields)])

        # Return a rollout with attempt attached.
        return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @tracked("enqueue_rollout")
    @healthcheck_before
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
        rollout_id = _generate_rollout_id()
        current_time = time.time()

        rollout_config = config.model_copy(deep=True) if config is not None else RolloutConfig()
        rollout_metadata = dict(metadata) if metadata is not None else {}

        if resources_id is None:
            latest_resources = await self._get_latest_resources()
            resources_id = latest_resources.resources_id if latest_resources is not None else None

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

        async def _insert_rollout_and_enqueue(collections: T_collections) -> None:
            await collections.rollouts.insert([rollout])
            await collections.rollout_queue.enqueue([rollout.rollout_id])  # add it to the end of the queue

        await self.collections.execute(
            _insert_rollout_and_enqueue,
            mode="rw",
            snapshot=self._read_snapshot,
            commit=True,
            labels=["rollouts", "rollout_queue"],
        )
        # Notify the subclass that the rollout status has changed.
        all_fields = list(rollout.__class__.model_fields.keys())
        await self._post_update_rollout([(rollout, all_fields)])

        # Return the rollout with no attempt attached.
        return rollout

    @tracked("_post_dequeue_rollout")
    @_with_collections_execute(labels=["rollouts", "attempts"])
    async def _post_dequeue_rollout(
        self, collections: T_collections, rollout_id: str
    ) -> Optional[Tuple[AttemptedRollout, Sequence[str]]]:
        """Post-dequeue logic for the rollout. Returns the rollout and the update fields (for post-update logic)."""
        rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
        if not rollout:
            logger.warning(f"Rollout {rollout_id} not found, skipping dequeuing")
            return None

        # Check if rollout is still in a queuing state
        # (it might have been updated to a different status while in queue)
        if is_queuing(rollout):
            # Create a new attempt (could be first attempt or retry)
            attempt_id = _generate_attempt_id()
            current_time = time.time()

            # Get existing attempts to determine sequence number
            existing_attempts = await self._unlocked_query_attempts_for_rollout(collections, rollout.rollout_id)
            sequence_id = len(existing_attempts) + 1

            attempt = Attempt(
                rollout_id=rollout.rollout_id,
                attempt_id=attempt_id,
                sequence_id=sequence_id,
                start_time=current_time,
                status="preparing",
            )

            await collections.attempts.insert([attempt])

            # Sync attempt status to rollout
            rollout, update_fields = await self._unlocked_update_rollout_only(
                collections, rollout.rollout_id, status="preparing"
            )
            return AttemptedRollout(**rollout.model_dump(), attempt=attempt), update_fields

        else:
            # If not in queuing state, skip this rollout and continue
            # (it was updated externally and should not be processed)
            logger.warning(
                f"Rollout {rollout.rollout_id} is not in queuing state: {rollout.status}, skipping dequeuing"
            )
            return None

    @tracked("dequeue_rollout")
    @healthcheck_before
    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        """Retrieves the next task from the queue without blocking.
        Returns `None` if the queue is empty.

        Will set the rollout status to preparing and create a new attempt.

        See [`LightningStore.dequeue_rollout()`][agentlightning.LightningStore.dequeue_rollout] for semantics.
        """
        if worker_id is not None:
            new_worker = Worker(worker_id=worker_id, status="idle", last_dequeue_time=time.time())
            await self._update_or_insert_worker(new_worker, update_fields=["status", "last_dequeue_time"])

        # Keep looking until we find a rollout that's still in queuing status
        # or the queue is empty
        while True:
            async with self.collections.atomic(
                mode="rw", snapshot=self._read_snapshot, labels=["rollout_queue"]
            ) as collections:
                dequeued = await collections.rollout_queue.dequeue(1)
            if not dequeued:
                break
            rollout_id = dequeued[0]

            post_dequeue_result = await self._post_dequeue_rollout(rollout_id)
            if post_dequeue_result is not None:
                attempted_rollout, update_fields = post_dequeue_result
                await self._post_update_rollout([(attempted_rollout, update_fields)])
                return attempted_rollout

            # else continue the loop

        # No valid rollouts found
        return None

    @tracked("start_attempt")
    @healthcheck_before
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Creates a new attempt for a given rollout ID and return the attempt details.

        See [`LightningStore.start_attempt()`][agentlightning.LightningStore.start_attempt] for semantics.
        """

        async def _create_attempt(collections: T_collections):
            # Get the rollout
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")

            # Get existing attempts to determine sequence number
            existing_attempts = await self._unlocked_query_attempts_for_rollout(collections, rollout_id)
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
            await collections.attempts.insert([attempt])

            # Sync attempt status to rollout
            rollout, update_fields = await self._unlocked_update_rollout_only(
                collections, rollout_id, status="preparing"
            )
            return attempt, rollout, update_fields

        attempt, rollout, update_fields = await self.collections.execute(
            _create_attempt, mode="rw", snapshot=self._read_snapshot, commit=True, labels=["rollouts", "attempts"]
        )
        await self._post_update_rollout([(rollout, update_fields)])

        # Return the rollout with the new attempt attached.
        return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @tracked("query_rollouts")
    @healthcheck_before
    async def query_rollouts(
        self,
        *,
        status_in: Optional[Sequence[RolloutStatus]] = None,
        rollout_id_in: Optional[Sequence[str]] = None,
        rollout_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
        status: Optional[Sequence[RolloutStatus]] = None,
        rollout_ids: Optional[Sequence[str]] = None,
    ) -> PaginatedResult[Union[Rollout, AttemptedRollout]]:
        """Retrieve rollouts with filtering and pagination.

        See [`LightningStore.query_rollouts()`][agentlightning.LightningStore.query_rollouts] for semantics.
        """
        # Construct filters condition
        if status_in is not None:
            resolved_status = status_in
        elif status is not None:
            warnings.warn("status is deprecated, use status_in instead", DeprecationWarning, stacklevel=3)
            resolved_status = status
        else:
            resolved_status = None

        if rollout_id_in is not None:
            resolved_rollout_ids = rollout_id_in
        elif rollout_ids is not None:
            warnings.warn("rollout_ids is deprecated, use rollout_id_in instead", DeprecationWarning, stacklevel=3)
            resolved_rollout_ids = rollout_ids
        else:
            resolved_rollout_ids = None

        filters: FilterOptions = {}
        filters["_aggregate"] = filter_logic
        if resolved_status is not None:
            filters["status"] = {"within": list(resolved_status)}
        if resolved_rollout_ids is not None:
            rollout_id_field = cast(FilterField, filters.setdefault("rollout_id", {}))
            rollout_id_field["within"] = list(resolved_rollout_ids)
        if rollout_id_contains is not None:
            rollout_id_field = cast(FilterField, filters.setdefault("rollout_id", {}))
            rollout_id_field["contains"] = rollout_id_contains

        async with self.collections.atomic(
            mode="r", snapshot=self._read_snapshot, labels=["rollouts", "attempts"]
        ) as collections:
            rollouts = await collections.rollouts.query(
                filter=filters if list(filters.keys()) != ["_aggregate"] else None,
                sort=SortOptions(name=sort_by, order=sort_order) if sort_by else None,
                limit=limit,
                offset=offset,
            )

            # Attach the latest attempt to the rollout objects
            attempted_rollouts = await self._unlocked_many_rollouts_to_attempted_rollouts(collections, rollouts.items)

        return PaginatedResult(
            items=attempted_rollouts, limit=rollouts.limit, offset=rollouts.offset, total=rollouts.total
        )

    @tracked("_unlocked_query_attempts_for_rollout")
    async def _unlocked_query_attempts_for_rollout(self, collections: T_collections, rollout_id: str) -> List[Attempt]:
        """The unlocked version of `query_attempts_for_rollout`."""
        result = await collections.attempts.query(
            filter={"rollout_id": {"exact": rollout_id}},
            sort={"name": "sequence_id", "order": "asc"},
        )
        return list(result.items)

    @tracked("get_rollout_by_id")
    @healthcheck_before
    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Union[Rollout, AttemptedRollout]]:
        """Retrieves a specific rollout by its ID.

        See [`LightningStore.get_rollout_by_id()`][agentlightning.LightningStore.get_rollout_by_id] for semantics.

        If the rollout has been attempted, the latest attempt will also be returned.
        """
        async with self.collections.atomic(
            mode="r", snapshot=self._read_snapshot, labels=["rollouts", "attempts"]
        ) as collections:
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if rollout is None:
                return None
            return await self._unlocked_rollout_to_attempted_rollout(collections, rollout)

    @tracked("_unlocked_rollout_to_attempted_rollout")
    async def _unlocked_rollout_to_attempted_rollout(
        self, collections: T_collections, rollout: Rollout
    ) -> Union[Rollout, AttemptedRollout]:
        """Query the latest attempt for the rollout, and attach it to the rollout object.

        If the rollout has no attempts, return the rollout object itself.
        """
        latest_attempt = await self._unlocked_get_latest_attempt(collections, rollout.rollout_id)
        if latest_attempt is None:
            return rollout
        else:
            return AttemptedRollout(**rollout.model_dump(), attempt=latest_attempt)

    @tracked("_unlocked_many_rollouts_to_attempted_rollouts")
    async def _unlocked_many_rollouts_to_attempted_rollouts(
        self, collections: T_collections, rollouts: Sequence[Rollout]
    ) -> List[Union[Rollout, AttemptedRollout]]:
        """Query the latest attempts for the rollouts, and attach them to the rollout objects."""
        # TODO: Maybe we can use asyncio.gather here to speed up the process?
        return [await self._unlocked_rollout_to_attempted_rollout(collections, rollout) for rollout in rollouts]

    @tracked("_unlocked_get_latest_attempt")
    async def _unlocked_get_latest_attempt(self, collections: T_collections, rollout_id: str) -> Optional[Attempt]:
        """The unlocked version of `get_latest_attempt`."""
        return await collections.attempts.get(
            filter={"rollout_id": {"exact": rollout_id}},
            sort={"name": "sequence_id", "order": "desc"},
        )

    @tracked("query_attempts")
    @healthcheck_before
    async def query_attempts(
        self,
        rollout_id: str,
        *,
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[Attempt]:
        """Retrieve attempts for a rollout with optional ordering/pagination."""
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["attempts"]) as collections:
            return await collections.attempts.query(
                filter={"rollout_id": {"exact": rollout_id}},
                sort={"name": sort_by, "order": sort_order} if sort_by else None,
                limit=limit,
                offset=offset,
            )

    @tracked("get_latest_attempt")
    @healthcheck_before
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Retrieves the latest attempt for a given rollout ID.

        See [`LightningStore.get_latest_attempt()`][agentlightning.LightningStore.get_latest_attempt] for semantics.
        """
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["attempts"]) as collections:
            return await self._unlocked_get_latest_attempt(collections, rollout_id)

    @tracked("query_resources")
    async def query_resources(
        self,
        *,
        resources_id: Optional[str] = None,
        resources_id_contains: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[ResourcesUpdate]:
        """Return every stored resource snapshot in insertion order."""
        filters: FilterOptions = {}
        if resources_id is not None:
            resources_id_field = cast(FilterField, filters.setdefault("resources_id", {}))
            resources_id_field["exact"] = resources_id
        if resources_id_contains is not None:
            resources_id_field = cast(FilterField, filters.setdefault("resources_id", {}))
            resources_id_field["contains"] = resources_id_contains

        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["resources"]) as collections:
            return await collections.resources.query(
                filter=filters or None,
                sort={"name": sort_by, "order": sort_order} if sort_by else None,
                limit=limit,
                offset=offset,
            )

    @tracked("add_resources")
    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """Stores a new version of named resources and sets it as the latest.

        See [`LightningStore.add_resources()`][agentlightning.LightningStore.add_resources] for semantics.
        """
        resources_id = _generate_resources_id()
        current_time = time.time()
        update = ResourcesUpdate(
            resources_id=resources_id,
            resources=resources,
            create_time=current_time,
            update_time=current_time,
            version=1,
        )
        async with self.collections.atomic(mode="w", snapshot=self._read_snapshot, labels=["resources"]) as collections:
            await collections.resources.insert([update])
        return update

    @tracked("update_resources")
    @healthcheck_before
    @_with_collections_execute(labels=["resources"])
    async def update_resources(
        self, collections: T_collections, resources_id: str, resources: NamedResources
    ) -> ResourcesUpdate:
        """
        Safely stores a new version of named resources and sets it as the latest.

        See [`LightningStore.update_resources()`][agentlightning.LightningStore.update_resources] for semantics.
        """
        current_time = time.time()
        existing = await collections.resources.get({"resources_id": {"exact": resources_id}})
        if existing is None:
            update = ResourcesUpdate(
                resources_id=resources_id,
                resources=resources,
                create_time=current_time,
                update_time=current_time,
                version=1,
            )
            await collections.resources.insert([update])
        else:
            update = existing.model_copy(
                update={
                    "resources": resources,
                    "update_time": current_time,
                    "version": existing.version + 1,
                }
            )
            await collections.resources.update([update])
        return update

    @tracked("get_resources_by_id")
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Retrieves a specific version of named resources by its ID.

        See [`LightningStore.get_resources_by_id()`][agentlightning.LightningStore.get_resources_by_id] for semantics.
        """
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["resources"]) as collections:
            return await collections.resources.get({"resources_id": {"exact": resources_id}})

    @tracked("get_latest_resources")
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Retrieves the latest version of named resources.

        See [`LightningStore.get_latest_resources()`][agentlightning.LightningStore.get_latest_resources] for semantics.
        """
        return await self._get_latest_resources()

    @tracked("_issue_many_span_sequence_ids")
    @_with_collections_execute(labels=["span_sequence_ids"])
    async def _issue_many_span_sequence_ids(self, collections: T_collections, rollout_ids: List[str]) -> List[int]:
        """Issue a new span sequence ID for a given rollout."""
        # Cache the next sequence IDs for the rollouts (for both RW)
        next_sequence_ids_cache: Dict[str, int] = {}
        result: List[int] = []
        for rollout_id in rollout_ids:
            if rollout_id not in next_sequence_ids_cache:
                retrieved_id = await collections.span_sequence_ids.get(rollout_id)
                if retrieved_id is None:
                    retrieved_id = 0
                next_sequence_ids_cache[rollout_id] = retrieved_id

            # Increment the sequence ID for the rollout
            next_sequence_ids_cache[rollout_id] += 1
            result.append(next_sequence_ids_cache[rollout_id])

        # Propagate the cache to storage
        for rollout_id, sequence_id in next_sequence_ids_cache.items():
            await collections.span_sequence_ids.set(rollout_id, sequence_id)

        return result

    @tracked("_sync_span_sequence_id")
    @_with_collections_execute(labels=["span_sequence_ids"])
    async def _sync_span_sequence_id(self, collections: T_collections, rollout_id: str, sequence_id: int) -> None:
        """Sync the span sequence ID for a given rollout from the input span sequence ID."""
        existing_sequence_id = await collections.span_sequence_ids.get(rollout_id)
        if existing_sequence_id is None:
            existing_sequence_id = 0
        await collections.span_sequence_ids.set(rollout_id, max(existing_sequence_id, sequence_id))

    @tracked("get_next_span_sequence_id")
    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Get the next span sequence ID for a given rollout and attempt.
        The number is strictly increasing for each rollout.
        The store will not issue the same sequence ID twice.

        See [`LightningStore.get_next_span_sequence_id()`][agentlightning.LightningStore.get_next_span_sequence_id] for semantics.
        """
        ret = await self._issue_many_span_sequence_ids([rollout_id])
        return ret[0]

    @tracked("get_many_span_sequence_ids")
    async def get_many_span_sequence_ids(self, rollout_attempt_ids: Sequence[Tuple[str, str]]) -> Sequence[int]:
        """Get the next span sequence IDs for a given list of rollout and attempt identifiers."""
        return await self._issue_many_span_sequence_ids([rollout_id for rollout_id, _ in rollout_attempt_ids])

    @tracked("add_span")
    async def add_span(self, span: Span) -> Optional[Span]:
        """Persist a pre-converted span.

        See [`LightningStore.add_span()`][agentlightning.LightningStore.add_span] for semantics.
        """
        # Update the sequence ID to be synced with latest input span
        await self._sync_span_sequence_id(span.rollout_id, span.sequence_id)
        successful_spans = await self._add_many_spans_helper(span.rollout_id, span.attempt_id, [span])
        return successful_spans[0] if len(successful_spans) > 0 else None

    @tracked("add_many_spans")
    async def add_many_spans(self, spans: Sequence[Span]) -> Sequence[Span]:
        """Persist a sequence of pre-converted spans.

        See [`LightningStore.add_many_spans()`][agentlightning.LightningStore.add_many_spans] for semantics.
        """
        # Group spans by rollout and attempt
        spans_by_rollout_attempt: Dict[Tuple[str, str], List[Span]] = defaultdict(list)
        for span in spans:
            spans_by_rollout_attempt[(span.rollout_id, span.attempt_id)].append(span)

        # Bulk add spans for each rollout and attempt
        successful_spans: List[Span] = []
        for (rollout_id, attempt_id), spans in spans_by_rollout_attempt.items():
            await self._sync_span_sequence_id(rollout_id, max(span.sequence_id for span in spans))
            ret = await self._add_many_spans_helper(rollout_id, attempt_id, spans)
            successful_spans.extend(ret)
        return successful_spans

    @tracked("add_otel_span")
    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Optional[Span]:
        """Add an opentelemetry span to the store.

        See [`LightningStore.add_otel_span()`][agentlightning.LightningStore.add_otel_span] for semantics.
        """
        if sequence_id is None:
            # Issue a new sequence ID for the rollout
            sequence_id = (await self._issue_many_span_sequence_ids([rollout_id]))[0]
        else:
            # Comes from a provided sequence ID
            # Make sure our counter is strictly increasing
            await self._sync_span_sequence_id(rollout_id, sequence_id)

        span = Span.from_opentelemetry(
            readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
        )
        ret = await self._add_many_spans_helper(rollout_id, attempt_id, [span])
        return ret[0] if len(ret) > 0 else None

    @tracked("_add_many_spans_helper")
    async def _add_many_spans_helper(self, rollout_id: str, attempt_id: str, spans: Sequence[Span]) -> Sequence[Span]:
        """Add many spans to the store. All spans must be for the same rollout and attempt.

        This method is divided into three parts:

        1. Verify the rollout and attempt exist;
        2. Insert the spans in bulk; if insert fails, fallback to inserting one by one;
        3. Update rollout and attempt status if necessary.
        """

        async with self.collections.atomic(
            mode="r", snapshot=self._read_snapshot, labels=["rollouts", "attempts"]
        ) as collections:
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if not rollout:
                raise ValueError(f"Rollout {rollout_id} not found")
            current_attempt = await collections.attempts.get(
                filter={"rollout_id": {"exact": rollout_id}, "attempt_id": {"exact": attempt_id}},
            )

            if not current_attempt:
                raise ValueError(f"Attempt {attempt_id} not found for rollout {rollout_id}")

        async def _add_span_fallback(span: Span) -> bool:
            try:
                await collections.spans.insert([span])
                return True
            except ValueError as e:
                if "already exists" in str(e) or "contains duplicate" in str(e):
                    logger.error(
                        f"Duplicated span added for rollout={span.rollout_id}, attempt={span.attempt_id}, span={span.span_id}. Skipping."
                    )
                    return False
                raise

        successful_spans: List[Span] = []
        try:
            # This is not guarded by commit=True.
            async with self.collections.atomic(
                mode="w", snapshot=self._read_snapshot, commit=False, labels=["spans"]
            ) as collections:
                await collections.spans.insert(spans)
            successful_spans.extend(spans)
        except ValueError as e:
            if "already exists" in str(e) or "contains duplicate" in str(e):
                # There is a duplicate span, we warn it
                # We fallback to adding the spans one by one
                async def _add_many_spans_fallback(collections: T_collections):
                    for span in spans:
                        if await _add_span_fallback(span):
                            successful_spans.append(span)

                await self.collections.execute(
                    _add_many_spans_fallback, mode="w", snapshot=self._read_snapshot, commit=True, labels=["spans"]
                )
            else:
                raise

        if not successful_spans:
            # No spans were added, skip the rest.
            return []

        await self._post_add_spans(successful_spans, rollout_id, attempt_id)

        return successful_spans

    @tracked("_post_add_spans")
    async def _post_add_spans(self, spans: Sequence[Span], rollout_id: str, attempt_id: str) -> None:
        """Update attempt heartbeat and rollout status after spans are inserted.

        Args:
            spans: Newly inserted spans.
            rollout_id: Identifier for the rollout receiving the spans.
            attempt_id: Identifier for the attempt receiving the spans.

        Note:
            The method refetches the attempt/rollout inside the transactional callback to
            avoid clobbering fields that might have changed after the spans were queued.
        """
        if not spans:
            return

        async def _update_rollout_attempt(collections: T_collections) -> Optional[Tuple[Rollout, Sequence[str]]]:
            attempt = await collections.attempts.get(
                {"rollout_id": {"exact": rollout_id}, "attempt_id": {"exact": attempt_id}}
            )
            if attempt is None:
                return None
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if rollout is None:
                return None

            # Update attempt heartbeat and ensure persistence
            attempt.last_heartbeat_time = time.time()
            if attempt.status in ["preparing", "unresponsive"]:
                attempt.status = "running"
            await collections.attempts.update([attempt], update_fields=["last_heartbeat_time", "status"])

            # If the status has already timed out or failed, do not change it (but heartbeat is still recorded)

            # Update rollout status if it's the latest attempt
            rollout_updated: bool = False
            updated_fields: List[str] = []
            latest_attempt = await self._unlocked_get_latest_attempt(collections, rollout.rollout_id)
            if latest_attempt is not None and attempt.attempt_id == latest_attempt.attempt_id:
                if rollout.status == "preparing":
                    rollout.status = "running"
                    await collections.rollouts.update([rollout], update_fields=["status"])
                    rollout_updated = True
                    updated_fields = ["status"]
                elif rollout.status in ["queuing", "requeuing"]:
                    rollout.status = "running"
                    await collections.rollouts.update([rollout], update_fields=["status"])
                    rollout_updated = True
                    updated_fields = ["status"]
            return (rollout, updated_fields) if rollout_updated else None

        rollout_update = await self.collections.execute(
            _update_rollout_attempt,
            mode="rw",
            snapshot=self._read_snapshot,
            commit=True,
            labels=["rollouts", "attempts"],
        )
        if rollout_update is not None:
            await self._post_update_rollout([rollout_update])

    @tracked("wait_for_rollouts")
    @healthcheck_before
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

    @tracked("wait_for_rollout")
    async def wait_for_rollout(self, rollout_id: str, timeout: Optional[float] = None) -> Optional[Rollout]:
        """Wait for a specific rollout to complete with a timeout.

        Subclass may use advanced mechanisms like events to accelerate this.

        Returns the completed rollout, or None if timeout is reached.
        """
        # First check if already completed
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["rollouts"]) as collections:
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})

        if rollout is None:
            # Rollout does not exist, return immediately
            return None

        if is_finished(rollout):
            # Rollout is already finished, return immediately
            return rollout

        # No timeout, return immediately
        if timeout is not None and timeout <= 0:
            return None

        start_time = time.time()
        deadline = start_time + timeout if timeout is not None else None

        # If not completed, wait for completion
        while deadline is None or time.time() < deadline:
            # Poll every 10 seconds by default
            rest_time = max(0.01, min(deadline - time.time(), 10.0)) if deadline is not None else 10.0
            await asyncio.sleep(rest_time)
            async with self.collections.atomic(
                mode="r", snapshot=self._read_snapshot, labels=["rollouts"]
            ) as collections:
                rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
                # check if rollout is finished
                if rollout and is_finished(rollout):
                    return rollout

        return None

    @tracked("query_spans")
    @healthcheck_before  # latest can point to a different attempt
    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
        *,
        trace_id: Optional[str] = None,
        trace_id_contains: Optional[str] = None,
        span_id: Optional[str] = None,
        span_id_contains: Optional[str] = None,
        parent_id: Optional[str] = None,
        parent_id_contains: Optional[str] = None,
        name: Optional[str] = None,
        name_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        limit: int = -1,
        offset: int = 0,
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> PaginatedResult[Span]:
        """
        Query and retrieve spans associated with a specific rollout ID.
        Returns an empty list if no spans are found.

        See [`LightningStore.query_spans()`][agentlightning.LightningStore.query_spans] for semantics.
        """

        resolved_attempt_id: Optional[str]
        if attempt_id is None:
            resolved_attempt_id = None
        elif attempt_id == "latest":
            async with self.collections.atomic(
                mode="r", snapshot=self._read_snapshot, labels=["attempts"]
            ) as collections:
                latest_attempt = await self._unlocked_get_latest_attempt(collections, rollout_id)
            if not latest_attempt:
                logger.debug(f"No attempts found for rollout {rollout_id} when querying latest spans")
                return PaginatedResult(items=[], limit=limit, offset=offset, total=0)
            resolved_attempt_id = latest_attempt.attempt_id
        else:
            resolved_attempt_id = attempt_id

        must_filter: Dict[str, FilterField] = {"rollout_id": {"exact": rollout_id}}
        if resolved_attempt_id is not None:
            must_filter["attempt_id"] = {"exact": resolved_attempt_id}
        filter_options: FilterOptions = {
            "_aggregate": filter_logic,  # this can be and/or
            "_must": must_filter,  # Must satisfy all the filters in the must list
        }

        def _resolve_filter_field(
            field_name: str, filter_exact: Optional[str] | None, filter_contains: Optional[str] | None
        ) -> None:
            field = cast(FilterField, filter_options.setdefault(field_name, {}))
            if filter_exact is not None:
                field["exact"] = filter_exact
            if filter_contains is not None:
                field["contains"] = filter_contains

        _resolve_filter_field("trace_id", trace_id, trace_id_contains)
        _resolve_filter_field("span_id", span_id, span_id_contains)
        _resolve_filter_field("parent_id", parent_id, parent_id_contains)
        _resolve_filter_field("name", name, name_contains)

        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["spans"]) as collections:
            return await collections.spans.query(
                filter=filter_options,
                sort={"name": sort_by, "order": sort_order} if sort_by else None,
                limit=limit,
                offset=offset,
            )

    @tracked("update_rollout")
    @healthcheck_before
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
        async with self.collections.atomic(mode="w", snapshot=self._read_snapshot, labels=["rollouts"]) as collections:
            rollout, update_fields = await self._unlocked_update_rollout_only(
                collections=collections,
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id,
                status=status,
                config=config,
                metadata=metadata,
            )

        await self._post_update_rollout([(rollout, update_fields)])
        return rollout

    @tracked("update_attempt")
    @healthcheck_before
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
        attempt, rollout_update, worker_sync_required = await self.collections.execute(
            lambda collections: self._unlocked_update_attempt_and_rollout(
                collections=collections,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                status=status,
                worker_id=worker_id,
                last_heartbeat_time=last_heartbeat_time,
                metadata=metadata,
            ),
            mode="rw",
            snapshot=self._read_snapshot,
            commit=True,
            labels=["rollouts", "attempts"],
        )
        if rollout_update:
            await self._post_update_rollout([rollout_update])
        if worker_sync_required:
            await self._sync_workers_with_attempts([attempt])
        return attempt

    @tracked("_unlocked_update_rollout_only")
    async def _unlocked_update_rollout_only(
        self,
        collections: T_collections,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Tuple[Rollout, Sequence[str]]:
        """Update the rollout status and related metadata only.

        Not updating related attempts or workers.

        There is only one update operation call inside; so commit is not strictly required.
        """
        rollout_construct_params: Dict[str, Any] = {"rollout_id": rollout_id}
        # Update fields if they are not UNSET
        if not isinstance(input, Unset):
            rollout_construct_params["input"] = input
        if not isinstance(mode, Unset):
            rollout_construct_params["mode"] = mode
        if not isinstance(resources_id, Unset):
            rollout_construct_params["resources_id"] = resources_id
        if not isinstance(status, Unset):
            rollout_construct_params["status"] = status
        if not isinstance(config, Unset):
            rollout_construct_params["config"] = config
        if not isinstance(metadata, Unset):
            rollout_construct_params["metadata"] = metadata

        # Set end time for finished rollouts
        # Rollout is only finished when it succeeded or fail with no more retries.
        if not isinstance(status, Unset) and status in ["failed", "succeeded", "cancelled"]:
            rollout_construct_params["end_time"] = time.time()

        update_fields = list(rollout_construct_params.keys())

        # Set required fields for validation purposes.
        rollout_construct_params.setdefault("input", None)
        rollout_construct_params.setdefault("start_time", 0.0)
        rollout_obj = Rollout.model_validate(rollout_construct_params)

        rollouts_updated = await collections.rollouts.update([rollout_obj], update_fields=update_fields)
        return rollouts_updated[0], update_fields

    @tracked("_post_update_rollout")
    async def _post_update_rollout(self, rollouts: Sequence[Tuple[Rollout, Sequence[str]]]) -> None:
        """Post-update logic for the rollout.

        This method has locks inside, so it should be called with the lock held.

        Args:
            rollouts: A sequence of tuples, each containing a rollout and the fields that were updated.
        """
        for rollout, updated_fields in rollouts:
            # Sometimes "end_time" is set but it's not really updated.
            if "end_time" in updated_fields and is_finished(rollout):
                if self._prometheus:
                    self._rollout_counter.labels(rollout.status, rollout.mode).inc()
                    self._rollout_duration_metric.labels(rollout.status, rollout.mode).observe(
                        cast(float, rollout.end_time) - rollout.start_time
                    )

        # If requeuing, add back to queue.
        # Check whether the rollout is already in queue.
        candidate_requeue_rollouts: List[str] = []
        async with self.collections.atomic(
            mode="r", snapshot=self._read_snapshot, labels=["rollout_queue"]
        ) as collections:
            for rollout, updated_fields in rollouts:
                # I think it's okay that the rollout_queue reports stale data;
                # the queue can have duplicated rollouts; the dequeue operation will handle it.
                if (
                    "status" in updated_fields
                    and is_queuing(rollout)
                    and not await collections.rollout_queue.has(rollout.rollout_id)
                ):
                    candidate_requeue_rollouts.append(rollout.rollout_id)

        async with self.collections.atomic(
            mode="w", snapshot=self._read_snapshot, labels=["rollout_queue"]
        ) as collections:
            await collections.rollout_queue.enqueue(candidate_requeue_rollouts)

        # NOTE: We also don't need to remove non-queuing rollouts from the queue.

    @tracked("_unlocked_update_attempt_and_rollout")
    async def _unlocked_update_attempt_and_rollout(
        self,
        collections: T_collections,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Tuple[Attempt, Optional[Tuple[Rollout, Sequence[str]]], bool]:
        """Update an attempt.

        The attempt status is propagated to the rollout if the attempt is the latest attempt.

        Returns:
            - The updated attempt
            - The updated rollout (or none if unchanged); post-rollout-update is not invoked yet
            - Whether the worker needs to be synced
        """
        # No lock, but with status propagation.
        rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
        if not rollout:
            raise ValueError(f"Rollout {rollout_id} not found")

        latest_attempt = await self._unlocked_get_latest_attempt(collections, rollout_id)
        if not latest_attempt:
            raise ValueError(f"No attempts found for rollout {rollout_id}")

        # Find the attempt to update
        if attempt_id == "latest":
            attempt = latest_attempt
        else:
            attempt = await collections.attempts.get(
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

        # Re-validate the attempt to ensure legality
        Attempt.model_validate(attempt.model_dump())
        # Update the attempt in storage
        await collections.attempts.update([attempt])

        rollout_update: Optional[Tuple[Rollout, Sequence[str]]] = None
        if attempt.attempt_id == latest_attempt.attempt_id:
            # Propagate the status to the rollout
            rollout_status = await rollout_status_from_attempt(attempt, rollout.config)
            if rollout_status != rollout.status:
                updated_rollout, update_fields = await self._unlocked_update_rollout_only(
                    collections, rollout_id, status=rollout_status
                )
                rollout_update = (updated_rollout, update_fields)

        return attempt, rollout_update, worker_sync_required

    @tracked("query_workers")
    @healthcheck_before
    async def query_workers(
        self,
        *,
        status_in: Optional[Sequence[WorkerStatus]] = None,
        worker_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[Worker]:
        """Return the current snapshot of all workers."""
        filters: FilterOptions = {}
        if status_in is not None:
            filters["status"] = {"within": list(status_in)}
        if worker_id_contains is not None:
            filters["worker_id"] = {"contains": worker_id_contains}
        filters["_aggregate"] = filter_logic

        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["workers"]) as collections:
            return await collections.workers.query(
                filter=filters if list(filters.keys()) != ["_aggregate"] else None,
                sort={"name": sort_by, "order": sort_order} if sort_by else None,
                limit=limit,
                offset=offset,
            )

    @tracked("get_worker_by_id")
    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        async with self.collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["workers"]) as collections:
            return await collections.workers.get({"worker_id": {"exact": worker_id}})

    @tracked("update_worker")
    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        """Create or update a worker entry."""
        update_fields = ["last_heartbeat_time"]
        new_worker = Worker(worker_id=worker_id, last_heartbeat_time=time.time())
        if not isinstance(heartbeat_stats, Unset):
            update_fields.append("heartbeat_stats")
            new_worker.heartbeat_stats = dict(heartbeat_stats)
        return await self._update_or_insert_worker(new_worker, update_fields=update_fields)

    @tracked("_unlocked_get_running_rollouts")
    async def _unlocked_get_running_rollouts(self, collections: T_collections) -> List[AttemptedRollout]:
        """Get all running rollouts.

        As this is invoked very frequently (probably at every requests),
        subclass can implement hacks to make it more efficient.
        It should also be unlocked and let the caller hold the lock.
        """
        filtered_rollouts = await collections.rollouts.query(filter={"status": {"within": ["preparing", "running"]}})
        running_rollouts = await self._unlocked_many_rollouts_to_attempted_rollouts(collections, filtered_rollouts)

        running_attempted_rollouts: List[AttemptedRollout] = []
        for rollout in running_rollouts:
            if not isinstance(rollout, AttemptedRollout):
                logger.error(f"Rollout {rollout.rollout_id} is running but has no attempts")
                continue
            running_attempted_rollouts.append(rollout)

        return running_attempted_rollouts

    @tracked("_scan_for_unhealthy_rollouts")
    async def _scan_for_unhealthy_rollouts(self) -> None:
        """Perform healthcheck against all running rollouts in the store."""
        rollouts, attempts_sync_required = await self._find_and_update_unhealthy_rollouts()

        if rollouts:
            await self._post_update_rollout(rollouts)

        # Sync worker status
        if attempts_sync_required:
            await self._sync_workers_with_attempts(attempts_sync_required)

    @tracked("_find_and_update_unhealthy_rollouts")
    @_with_collections_execute(labels=["rollouts", "attempts"])
    async def _find_and_update_unhealthy_rollouts(
        self, collections: T_collections
    ) -> Tuple[List[Tuple[Rollout, Sequence[str]]], List[Attempt]]:
        """Batch update the status of unhealthy attempts.

        Returns:
            - The list of rollouts that have been updated
            - The list of attempts that need worker-sync
        """
        running_rollouts = await self._unlocked_get_running_rollouts(collections)

        candidate_updates = await scan_unhealthy_rollouts(running_rollouts)
        if not candidate_updates:
            return [], []

        rollouts: List[Tuple[Rollout, Sequence[str]]] = []
        attempts: List[Attempt] = []
        for (rollout_id, attempt_id), status in candidate_updates.items():
            attempt, rollout_update, worker_sync_required = await self._unlocked_update_attempt_and_rollout(
                collections, rollout_id, attempt_id, status=status
            )
            if rollout_update:
                rollouts.append(rollout_update)
            if worker_sync_required:
                attempts.append(attempt)
        return rollouts, attempts

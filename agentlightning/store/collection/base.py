# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
import time
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from agentlightning.store.utils import LATENCY_BUCKETS
from agentlightning.utils.metrics import MetricsBackend

if TYPE_CHECKING:
    from typing import Self

from agentlightning.store.base import LightningStore
from agentlightning.types import (
    Attempt,
    FilterField,
    FilterOptions,
    PaginatedResult,
    ResourcesUpdate,
    Rollout,
    SortOptions,
    Span,
    Worker,
)

T = TypeVar("T")  # Recommended to be a BaseModel
K = TypeVar("K")
V = TypeVar("V")
T_callable = TypeVar("T_callable", bound=Callable[..., Any])

AtomicMode = Literal["r", "w", "rw"]
"""What is expected within the atomic context. Can be "read", "write", or "read-write"."""

AtomicLabels = Literal["rollouts", "attempts", "spans", "resources", "workers", "rollout_queue", "span_sequence_ids"]
"""Labels for atomic operations.

These labels are used to identify the collections that are affected by the atomic operation.
"""


COLLECTION_TRACKING_STORE_METHODS = frozenset(
    [name for name in LightningStore.__dict__ if not name.startswith("_")] + ["_healthcheck"]
)

_UNKNOWN_STORE_METHOD = "unknown"


def _nearest_lightning_store_method_from_stack() -> str:
    """Stack introspection so that we capture the nearest public API method from the
    call stack whenever metrics are recorded."""
    frame = inspect.currentframe()
    try:
        if frame is None:
            return _UNKNOWN_STORE_METHOD
        frame = frame.f_back
        while frame is not None:
            self_obj = frame.f_locals.get("self")
            method_name = frame.f_locals.get("method_name")
            if method_name in COLLECTION_TRACKING_STORE_METHODS and isinstance(self_obj, LightningStore):
                return method_name
            frame = frame.f_back
        return _UNKNOWN_STORE_METHOD
    except Exception:
        return _UNKNOWN_STORE_METHOD
    finally:
        del frame


def resolve_error_type(exc: BaseException | None) -> str:
    if exc is None:
        return "N/A"

    try:
        from .mongo import resolve_mongo_error_type

        error_type = resolve_mongo_error_type(exc)
        if error_type is not None:
            return error_type
    except ImportError:
        # If the mongo backend is not available, fall back to using the exception's class name.
        pass

    return exc.__class__.__name__


def tracked(operation: str):
    """Decorator to track the execution of the decorated method."""

    def decorator(func: T_callable) -> T_callable:

        @functools.wraps(func)
        async def wrapper(self: TrackedCollection, *args: Any, **kwargs: Any) -> Any:
            async with self.tracking_context(operation, self.collection_name):
                return await func(self, *args, **kwargs)

        return cast(T_callable, wrapper)

    return decorator


class TrackedCollection:
    """An object that can be tracked by the metrics backend."""

    def __init__(self, tracker: MetricsBackend | None = None):
        self._tracker = tracker

    @property
    def tracker(self) -> MetricsBackend | None:
        return self._tracker

    @property
    def collection_name(self) -> str:
        """The identifier of the collection."""
        raise NotImplementedError()

    @property
    def extra_tracking_labels(self) -> Mapping[str, Any]:
        """Extra labels to add to the tracking context."""
        return {}

    @asynccontextmanager
    async def tracking_context(self, operation: str, collection: str):
        """Context manager to track the execution of the decorated method.

        Args:
            operation: The operation to track.
            collection: The collection to track.
        """
        if self._tracker is None:
            # no-op context manager
            yield

        else:
            # Enable tracking
            start_time = time.perf_counter()
            status: str = "OK"
            store_method = _nearest_lightning_store_method_from_stack()
            try:
                yield
            except BaseException as exc:
                status = resolve_error_type(exc)
                raise
            finally:
                elapsed = time.perf_counter() - start_time
                await self._tracker.inc_counter(  # pyright: ignore[reportPrivateUsage]
                    "agl.collections.total",
                    labels={
                        "store_method": store_method,
                        "operation": operation,
                        "collection": collection,
                        "status": status,
                        **self.extra_tracking_labels,
                    },
                )
                await self._tracker.observe_histogram(  # pyright: ignore[reportPrivateUsage]
                    "agl.collections.latency",
                    value=elapsed,
                    labels={
                        "store_method": store_method,
                        "operation": operation,
                        "collection": collection,
                        "status": status,
                        **self.extra_tracking_labels,
                    },
                )


class Collection(TrackedCollection, Generic[T]):
    """Standard collection interface. Behaves like a list of items. Supporting addition, updating, and deletion of items."""

    def primary_keys(self) -> Sequence[str]:
        """Get the primary keys of the collection."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}]>"

    def item_type(self) -> Type[T]:
        """Get the type of the items in the collection."""
        raise NotImplementedError()

    async def size(self) -> int:
        """Get the number of items in the collection."""
        raise NotImplementedError()

    async def query(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with the given filters, sort order, and pagination.

        Args:
            filter:
                The filters to apply to the collection. See [`FilterOptions`][agentlightning.FilterOptions].

            sort:
                The options for sorting the collection. See [`SortOptions`][agentlightning.SortOptions].
                The field must exist in the model. If field might contain null values, in which case the behavior is undefined
                (i.e., depending on the implementation).

            limit:
                Max number of items to return. Use -1 for "no limit".

            offset:
                Number of items to skip from the start of the *matching* items.

        Returns:
            PaginatedResult with items, limit, offset, and total matched items.
        """
        raise NotImplementedError()

    async def get(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
    ) -> Optional[T]:
        """Get the first item that matches the given filters.

        Args:
            filter: The filters to apply to the collection.
                See [`FilterOptions`][agentlightning.store.collection.FilterOptions].
            sort: Sort options. See [`SortOptions`][agentlightning.store.collection.SortOptions].

        Returns:
            The first item that matches the given filters, or None if no item matches.
        """
        raise NotImplementedError()

    async def insert(self, items: Sequence[T]) -> None:
        """Add the given items to the collection.

        Raises:
            ValueError: If an item with the same primary key already exists.
        """
        raise NotImplementedError()

    async def update(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        """Update the given items in the collection.

        Args:
            items: The items to update in the collection.
            update_fields: The fields to update. If not provided, all fields in the type will be updated.
                Only applicable if the item type is a Pydantic BaseModel.

        Raises:
            ValueError: If an item with the primary keys does not exist.

        Returns:
            The items that were updated.
        """
        raise NotImplementedError()

    async def upsert(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        """Upsert the given items into the collection.

        If the items with the same primary keys already exist, they will be updated.
        Otherwise, they will be inserted.

        The operation has three semantics configurable via `update_fields`:

        - `update_or_insert` via `collection.upsert(items, update_fields=["status", "updated_at"])`.
          If the item with the same primary keys already exists, only the specified fields will be updated.
          Otherwise, the item will be inserted.
        - `get_or_insert` via `collection.upsert(items, update_fields=[])`.
          If the item with the same primary keys already exists, the item will be left unchanged.
          Otherwise, the item will be inserted.
        - `replace_ish` via `collection.upsert(items)`.
          If the item with the same primary keys already exists, all fields from the item will be set.
          Otherwise, the item will be inserted.

        Returns:
            The items that were upserted.
        """
        raise NotImplementedError()

    async def delete(self, items: Sequence[T]) -> None:
        """Delete the given items from the collection.

        Args:
            items: The items to delete from the collection.

        Raises:
            ValueError: If the items with the primary keys to be deleted do not exist.
        """
        raise NotImplementedError()


class Queue(TrackedCollection, Generic[T]):
    """Behaves like a deque. Supporting appending items to the end and popping items from the front."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}]>"

    def item_type(self) -> Type[T]:
        """Get the type of the items in the queue."""
        raise NotImplementedError()

    async def has(self, item: T) -> bool:
        """Check if the given item is in the queue."""
        raise NotImplementedError()

    async def enqueue(self, items: Sequence[T]) -> Sequence[T]:
        """Append the given items to the end of the queue.

        Args:
            items: The items to append to the end of the queue.

        Returns:
            The items that were appended to the end of the queue.
        """
        raise NotImplementedError()

    async def dequeue(self, limit: int = 1) -> Sequence[T]:
        """Pop the given number of items from the front of the queue.

        Args:
            limit: The number of items to pop from the front of the queue.

        Returns:
            The items that were popped from the front of the queue.
            If there are less than `limit` items in the queue, the remaining items will be returned.
        """
        raise NotImplementedError()

    async def peek(self, limit: int = 1) -> Sequence[T]:
        """Peek the given number of items from the front of the queue.

        Args:
            limit: The number of items to peek from the front of the queue.

        Returns:
            The items that were peeked from the front of the queue.
            If there are less than `limit` items in the queue, the remaining items will be returned.
        """
        raise NotImplementedError()

    async def size(self) -> int:
        """Get the number of items in the queue."""
        raise NotImplementedError()


class KeyValue(TrackedCollection, Generic[K, V]):
    """Behaves like a dictionary. Supporting addition, updating, and deletion of items."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    async def has(self, key: K) -> bool:
        """Check if the given key is in the dictionary."""
        raise NotImplementedError()

    async def get(self, key: K, default: V | None = None) -> V | None:
        """Get the value for the given key, or the default value if the key is not found."""
        raise NotImplementedError()

    async def set(self, key: K, value: V) -> None:
        """Set the value for the given key."""
        raise NotImplementedError()

    async def pop(self, key: K, default: V | None = None) -> V | None:
        """Pop the value for the given key, or the default value if the key is not found."""
        raise NotImplementedError()

    async def size(self) -> int:
        """Get the number of items in the dictionary."""
        raise NotImplementedError()


class LightningCollections(TrackedCollection):
    """Collections of rollouts, attempts, spans, resources, and workers.

    [LightningStore][agentlightning.LightningStore] implementations can use this as a storage base
    to implement the store API.
    """

    def __init__(self, tracker: MetricsBackend | None = None, extra_labels: Optional[Sequence[str]] = None):
        super().__init__(tracker=tracker)
        self.register_collection_metrics(extra_labels)

    def register_collection_metrics(self, extra_labels: Optional[Sequence[str]] = None) -> None:
        if self._tracker is None:
            return
        labels = ["store_method", "operation", "collection", "status"]
        if extra_labels is not None:
            labels.extend(extra_labels)
        self._tracker.register_histogram(
            "agl.collections.latency",
            labels,
            buckets=LATENCY_BUCKETS,
            group_level=2,
        )
        self._tracker.register_counter("agl.collections.total", labels, group_level=2)

    @property
    def tracker(self) -> MetricsBackend | None:
        return self._tracker

    @property
    def rollouts(self) -> Collection[Rollout]:
        """Collections of rollouts."""
        raise NotImplementedError()

    @property
    def attempts(self) -> Collection[Attempt]:
        """Collections of attempts."""
        raise NotImplementedError()

    @property
    def spans(self) -> Collection[Span]:
        """Collections of spans."""
        raise NotImplementedError()

    @property
    def resources(self) -> Collection[ResourcesUpdate]:
        """Collections of resources."""
        raise NotImplementedError()

    @property
    def workers(self) -> Collection[Worker]:
        """Collections of workers."""
        raise NotImplementedError()

    @property
    def rollout_queue(self) -> Queue[str]:
        """Queue of rollouts (tasks)."""
        raise NotImplementedError()

    @property
    def span_sequence_ids(self) -> KeyValue[str, int]:
        """Dictionary (counter) of span sequence IDs."""
        raise NotImplementedError()

    def atomic(
        self,
        *,
        mode: AtomicMode = "rw",
        snapshot: bool = False,
        commit: bool = False,
        labels: Optional[Sequence[AtomicLabels]] = None,
        **kwargs: Any,
    ) -> AsyncContextManager[Self]:
        """Perform a atomic operation on the collections.

        Subclass may use args and kwargs to support multiple levels of atomicity.
        The arguments can be seen as tags. They only imply the behavior of the operation, not the implementation.

        Args:
            mode: The mode of atomicity. See [`AtomicMode`][agentlightning.store.collection.AtomicMode].
            snapshot: Enable read snapshot for repeatable reads. Data consistency is guaranteed. The real behavior is implementation-dependent.
            commit: Enable commitment for write operations. Unsuccessful operations will be rolled back depending on the implementation.
                Recommend to use [`execute()`][agentlightning.store.collection.LightningCollections.execute] for this level to enable automatic retries.
                Remember that the real behavior is implementation-dependent.
            labels: Labels to add to the atomic operation (commonly used as lock names or collection names).
            **kwargs: Keyword arguments to pass to the operation.
        """
        raise NotImplementedError()

    async def execute(
        self,
        callback: Callable[[Self], Awaitable[T]],
        *,
        mode: AtomicMode = "rw",
        snapshot: bool = False,
        commit: bool = False,
        labels: Optional[Sequence[AtomicLabels]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute the given callback within an atomic operation. Retry on transient errors is implied.

        See [`atomic()`][agentlightning.store.collection.LightningCollections.atomic] for more details.
        """
        async with self.atomic(mode=mode, snapshot=snapshot, commit=commit, labels=labels, **kwargs) as collections:
            return await callback(collections)


FilterMap = Mapping[str, FilterField]


def merge_must_filters(target: MutableMapping[str, FilterField], definition: Any) -> None:
    """Normalize a `_must` filter group into the provided mapping.

    Mainly for validation purposes.
    """
    if definition is None:
        return

    entries: List[Mapping[str, FilterField]] = []
    if isinstance(definition, Mapping):
        entries.append(cast(Mapping[str, FilterField], definition))
    elif isinstance(definition, Sequence) and not isinstance(definition, (str, bytes)):
        for entry in definition:  # type: ignore
            if not isinstance(entry, Mapping):
                raise TypeError("Each `_must` entry must be a mapping of field names to operators")
            entries.append(cast(Mapping[str, FilterField], entry))
    else:
        raise TypeError("`_must` filters must be provided as a mapping or sequence of mappings")

    for entry in entries:
        for field_name, ops in entry.items():
            existing = target.get(field_name, {})
            merged_ops: Dict[str, Any] = dict(existing)
            for op_name, expected in ops.items():
                if op_name in merged_ops:
                    raise ValueError(f"Duplicate operator '{op_name}' for field '{field_name}' in must filters")
                merged_ops[op_name] = expected
            target[field_name] = cast(FilterField, merged_ops)


def normalize_filter_options(
    filter_options: Optional[FilterOptions],
) -> Tuple[Optional[FilterMap], Optional[FilterMap], Literal["and", "or"]]:
    """Convert FilterOptions to the internal structure and resolve aggregate logic."""
    if not filter_options:
        return None, None, "and"

    aggregate = cast(Literal["and", "or"], filter_options.get("_aggregate", "and"))
    if aggregate not in ("and", "or"):
        raise ValueError(f"Unsupported filter aggregate '{aggregate}'")

    # Extract normalized filters and must filters from the filter options.
    normalized: Dict[str, FilterField] = {}
    must_filters: Dict[str, FilterField] = {}
    for field_name, ops in filter_options.items():
        if field_name == "_aggregate":
            continue
        if field_name == "_must":
            merge_must_filters(must_filters, ops)
            continue
        normalized[field_name] = cast(FilterField, dict(ops))  # type: ignore

    return (normalized or None, must_filters or None, aggregate)


def resolve_sort_options(sort: Optional[SortOptions]) -> Tuple[Optional[str], Literal["asc", "desc"]]:
    """Extract sort field/order from the caller-provided SortOptions."""
    if not sort:
        return None, "asc"

    sort_name = sort.get("name")
    if not sort_name:
        raise ValueError("Sort options must include a 'name' field")

    sort_order = sort.get("order", "asc")
    if sort_order not in ("asc", "desc"):
        raise ValueError(f"Unsupported sort order '{sort_order}'")

    return sort_name, sort_order

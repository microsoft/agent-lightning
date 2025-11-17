# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import weakref
from collections import deque
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Deque,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from agentlightning.types import (
    Attempt,
    ResourcesUpdate,
    Rollout,
    Span,
    Worker,
)

from .base import Collection, Filter, KeyValue, LightningCollections, PaginatedResult, Queue

T = TypeVar("T", bound=BaseModel)
K = TypeVar("K")
V = TypeVar("V")

# Nested structure type:
# dict[pk1] -> dict[pk2] -> ... -> item
ListBasedCollectionItemType = Union[
    Dict[Any, "ListBasedCollectionItemType[T]"],  # intermediate node
    Dict[Any, T],  # leaf node dictionary
]


MutationMode = Literal["insert", "update", "upsert", "delete"]


class ListBasedCollection(Collection[T]):
    """In-memory implementation of Collection using a nested dict for O(1) primary-key lookup.

    The internal structure is:

        {
            pk1_value: {
                pk2_value: {
                    ...
                        pkN_value: item
                }
            }
        }

    where the nesting depth equals the number of primary keys.
    """

    def __init__(self, items: List[T], item_type: Type[T], primary_keys: Sequence[str]):
        if not primary_keys:
            raise ValueError("primary_keys must be non-empty")

        self._items: Dict[Any, Any] = {}
        self._size: int = 0
        self._item_type: Type[T] = item_type
        self._primary_keys: Tuple[str, ...] = tuple(primary_keys)

        # Pre-populate the collection with the given items.
        for item in items or []:
            self._mutate_single(item, mode="insert")

    def primary_keys(self) -> Sequence[str]:
        """Return the primary key field names for this collection."""
        return self._primary_keys

    def item_type(self) -> Type[T]:
        """Return the Pydantic model type of items stored in this collection."""
        return self._item_type

    def size(self) -> int:
        """Return the number of items stored in the collection."""
        return self._size

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({self.size()})>"

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _ensure_item_type(self, item: T) -> None:
        """Validate that the item matches the declared item_type."""
        if not isinstance(item, self._item_type):
            raise TypeError(f"Expected item of type {self._item_type.__name__}, " f"got {type(item).__name__}")

    def _extract_primary_key_values(self, item: T) -> Tuple[Any, ...]:
        """Extract the primary key values from an item.

        Raises:
            ValueError: If any primary key is missing on the item.
        """
        values: List[Any] = []
        for key in self._primary_keys:
            if not hasattr(item, key):
                raise ValueError(f"Item {item} does not have primary key field '{key}'")
            values.append(getattr(item, key))
        return tuple(values)

    def _render_key_values(self, key_values: Sequence[Any]) -> str:
        return ", ".join(f"{name}={value!r}" for name, value in zip(self._primary_keys, key_values))

    def _locate_node(
        self,
        key_values: Sequence[Any],
        create_missing: bool,
    ) -> Tuple[MutableMapping[Any, Any], Any]:
        """Locate the parent mapping and final key for an item path.

        Args:
            key_values: The sequence of primary key values.
            create_missing: Whether to create intermediate dictionaries as needed.

        Returns:
            (parent_mapping, final_key)

        Raises:
            KeyError: If the path does not exist and create_missing is False.
            ValueError: If the internal structure is corrupted (non-dict where dict is expected).
        """
        if not key_values:
            raise ValueError("key_values must be non-empty")

        current: MutableMapping[Any, Any] = self._items
        for idx, value in enumerate(key_values):
            is_last = idx == len(key_values) - 1
            if is_last:
                # At the final level, current[value] is the item (or will be).
                return current, value  # type: ignore

            # Intermediate level: current[value] must be a dict.
            if value not in current:
                if not create_missing:
                    raise KeyError(f"Path does not exist for given primary keys: {self._render_key_values(key_values)}")
                current[value] = {}
            next_node = current[value]  # type: ignore
            if not isinstance(next_node, dict):
                raise ValueError(f"Internal structure corrupted: expected dict, got {type(next_node)!r}")  # type: ignore
            current = next_node  # type: ignore

        # We should always return inside the loop.
        raise RuntimeError("Unreachable")

    def _mutate_single(self, item: T, mode: MutationMode) -> None:
        """Core mutation logic shared by insert, update, upsert, and delete."""
        self._ensure_item_type(item)
        key_values = self._extract_primary_key_values(item)

        if mode in ("insert", "upsert"):
            parent, final_key = self._locate_node(key_values, create_missing=True)
            exists = final_key in parent

            if mode == "insert":
                if exists:
                    raise ValueError(f"Item already exists with primary key(s): {self._render_key_values(key_values)}")
                parent[final_key] = item
                self._size += 1
            else:  # upsert
                if not exists:
                    self._size += 1
                parent[final_key] = item

        elif mode in ("update", "delete"):
            # For update/delete we must not create missing paths.
            try:
                parent, final_key = self._locate_node(key_values, create_missing=False)
            except KeyError:
                raise ValueError(
                    f"Item does not exist with primary key(s): {self._render_key_values(key_values)}"
                ) from None

            if final_key not in parent:
                raise ValueError(f"Item does not exist with primary key(s): {self._render_key_values(key_values)}")

            if mode == "update":
                parent[final_key] = item
            else:  # delete
                del parent[final_key]
                self._size -= 1
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")

    def _iter_items(self) -> Iterable[T]:
        """Iterate over all items in the nested dictionary structure."""
        if not self._items:
            return
        stack: List[Mapping[Any, Any]] = [self._items]
        while stack:
            node = stack.pop()
            for value in node.values():
                # Leaf nodes contain items; intermediate nodes are dicts.
                if isinstance(value, self._item_type):
                    yield value
                elif isinstance(value, dict):
                    stack.append(value)  # type: ignore
                else:
                    raise ValueError(
                        f"Internal structure corrupted: expected dict or {self._item_type.__name__}, "
                        f"got {type(value)!r}"
                    )

    @staticmethod
    def _item_matches_filters(
        item: T,
        filters: Optional[Filter],
        filter_logic: Literal["and", "or"],
    ) -> bool:
        """Check whether an item matches the provided filter definition.

        Filter format:

        ```json
        {
            "field_name": {
                "exact": <value>,
                "within": <iterable_of_allowed_values>,
                "contains": <substring_or_element>,
            },
            ...
        }
        ```

        Operators within the same field are stored in a unified pool and combined using
        a universal logical operator.
        """
        if not filters:
            return True

        all_conditions_match: List[bool] = []

        for field_name, ops in filters.items():
            item_value = getattr(item, field_name, None)

            for op_name, expected in ops.items():
                # Ignore no-op filters
                if expected is None:
                    continue

                if op_name == "exact":
                    all_conditions_match.append(item_value == expected)

                elif op_name == "within":
                    try:
                        all_conditions_match.append(item_value in expected)
                    except TypeError:
                        all_conditions_match.append(False)

                elif op_name == "contains":
                    if item_value is None:
                        all_conditions_match.append(False)
                    elif isinstance(item_value, str) and isinstance(expected, str):
                        all_conditions_match.append(expected in item_value)
                    else:
                        # Fallback: treat as generic iterable containment.
                        try:
                            all_conditions_match.append(expected in item_value)  # type: ignore[arg-type]
                        except TypeError:
                            all_conditions_match.append(False)
                else:
                    raise ValueError(f"Unsupported filter operator '{op_name}' for field '{field_name}'")

        return all(all_conditions_match) if filter_logic == "and" else any(all_conditions_match)

    @staticmethod
    def _get_sort_value(item: T, sort_by: str) -> Any:
        """Get a sort key for the given item/field.

        - If the field name ends with '_time', values are treated as comparable timestamps.
        - For other fields we try to infer a safe default from the Pydantic model annotation.
        """
        value = getattr(item, sort_by, None)

        if sort_by.endswith("_time"):
            # For *_time fields, push missing values to the end.
            return float("inf") if value is None else value

        if value is None:
            # Introspect model field type to choose a reasonable default for None.
            model_fields = getattr(item.__class__, "model_fields", {})
            if sort_by not in model_fields:
                raise ValueError(
                    f"Failed to sort items by '{sort_by}': field does not exist " f"on {item.__class__.__name__}"
                )

            field_type_str = str(model_fields[sort_by].annotation)
            if "str" in field_type_str or "Literal" in field_type_str:
                return ""
            if "int" in field_type_str:
                return 0
            if "float" in field_type_str:
                return 0.0
            raise ValueError(f"Failed to sort items by '{sort_by}': unsupported field type {field_type_str!r}")

        return value

    async def query(
        self,
        filters: Optional[Filter] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with filters, sort order, and pagination.

        Args:
            filters:
                Mapping of field name to operator dict. For each field:

                    {
                        "exact":   value_for_equality,
                        "within":  iterable_of_allowed_values,
                        "contains": substring_or_element_to_match,
                    }

            filter_logic:
                How to combine per-field results:
                - "and": all fields must match.
                - "or":  at least one field must match.

            sort_by:
                Optional field name to sort by. Must exist on the model.

            sort_order:
                "asc" or "desc" for ascending / descending sort.

            limit:
                Max number of items to return. Use -1 for "no limit".

            offset:
                Number of items to skip from the start of the *matching* items.

        Returns:
            PaginatedResult with:
                - items: the page of results
                - limit: the requested limit
                - offset: the requested offset
                - total: total number of items that matched the filters
        """
        # No sorting: stream through items and apply filters on the fly.
        if not sort_by:
            matched_items: List[T] = []
            total_matched = 0

            for item in self._iter_items():
                if not self._item_matches_filters(item, filters, filter_logic):
                    continue

                # Count every match for 'total'
                total_matched += 1

                # Apply offset/limit window
                if total_matched <= offset:
                    continue
                if limit != -1 and len(matched_items) >= limit:
                    # Still need to finish iteration to get accurate total_matched.
                    continue

                matched_items.append(item)

            return PaginatedResult(
                items=matched_items,
                limit=limit,
                offset=offset,
                total=total_matched,
            )

        # With sorting: we must materialize all matching items to sort them.
        all_matches: List[T] = []
        for item in self._iter_items():
            if self._item_matches_filters(item, filters, filter_logic):
                all_matches.append(item)

        total_matched = len(all_matches)
        reverse = sort_order == "desc"
        all_matches.sort(key=lambda x: self._get_sort_value(x, sort_by), reverse=reverse)

        if limit == -1:
            paginated_items = all_matches[offset:]
        else:
            paginated_items = all_matches[offset : offset + limit]

        return PaginatedResult(
            items=paginated_items,
            limit=limit,
            offset=offset,
            total=total_matched,
        )

    async def get(
        self,
        filters: Optional[Filter] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> Optional[T]:
        """Return the first item that matches the given filters, or None."""
        result = await self.query(
            filters=filters,
            filter_logic=filter_logic,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=1,
            offset=0,
        )
        return result.items[0] if result.items else None

    async def insert(self, items: Sequence[T]) -> None:
        """Insert the given items.

        Raises:
            ValueError: If any item with the same primary keys already exists.
        """
        for item in items:
            self._mutate_single(item, mode="insert")

    async def update(self, items: Sequence[T]) -> None:
        """Update the given items.

        Raises:
            ValueError: If any item with the given primary keys does not exist.
        """
        for item in items:
            self._mutate_single(item, mode="update")

    async def upsert(self, items: Sequence[T]) -> None:
        """Upsert the given items (insert if missing, otherwise update)."""
        for item in items:
            self._mutate_single(item, mode="upsert")

    async def delete(self, items: Sequence[T]) -> None:
        """Delete the given items.

        Raises:
            ValueError: If any item with the given primary keys does not exist.
        """
        # We use a two-phase approach to avoid partial deletion if one fails:
        # first compute key_values to validate, then perform deletions.
        for item in items:
            # _mutate_single will validate existence and update size.
            self._mutate_single(item, mode="delete")


class DequeBasedQueue(Queue[T]):
    """Queue implementation backed by collections.deque.

    Provides O(1) amortized enqueue (append) and dequeue (popleft).
    """

    def __init__(self, item_type: Type[T], items: Optional[Sequence[T]] = None):
        self._items: Deque[T] = deque()
        self._item_type: Type[T] = item_type
        if items:
            self._items.extend(items)

    def item_type(self) -> Type[T]:
        return self._item_type

    async def enqueue(self, items: Sequence[T]) -> Sequence[T]:
        for item in items:
            if not isinstance(item, self._item_type):
                raise TypeError(f"Expected item of type {self._item_type.__name__}, got {type(item).__name__}")
            self._items.append(item)
        return items

    async def dequeue(self, limit: int = 1) -> Sequence[T]:
        if limit <= 0:
            return []
        out: List[T] = []
        for _ in range(min(limit, len(self._items))):
            out.append(self._items.popleft())
        return out

    async def peek(self, limit: int = 1) -> Sequence[T]:
        if limit <= 0:
            return []
        result: List[T] = []
        count = min(limit, len(self._items))
        for idx, item in enumerate(self._items):
            if idx >= count:
                break
            result.append(item)
        return result

    def size(self) -> int:
        return len(self._items)


class DictBasedKeyValue(KeyValue[K, V]):
    """KeyValue implementation backed by a plain dictionary."""

    def __init__(self, data: Optional[Mapping[K, V]] = None):
        self._values: Dict[K, V] = dict(data) if data else {}

    async def get(self, key: K, default: V | None = None) -> V | None:
        return self._values.get(key, default)

    async def set(self, key: K, value: V) -> None:
        self._values[key] = value

    async def pop(self, key: K, default: V | None = None) -> V | None:
        return self._values.pop(key, default)

    def size(self) -> int:
        return len(self._values)


class InMemoryLightningCollections(LightningCollections):
    """In-memory implementation of LightningCollections using Python data structures.

    Serves as the storage base for [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore].
    """

    def __init__(self):
        self._lock = _LoopAwareAsyncLock()
        self._rollouts = ListBasedCollection(items=[], item_type=Rollout, primary_keys=["rollout_id"])
        self._attempts = ListBasedCollection(items=[], item_type=Attempt, primary_keys=["rollout_id", "attempt_id"])
        self._spans = ListBasedCollection(
            items=[], item_type=Span, primary_keys=["rollout_id", "attempt_id", "span_id"]
        )
        self._resources = ListBasedCollection(items=[], item_type=ResourcesUpdate, primary_keys=["resources_id"])
        self._workers = ListBasedCollection(items=[], item_type=Worker, primary_keys=["worker_id"])
        self._rollout_queue = DequeBasedQueue(items=[], item_type=Rollout)
        self._span_sequence_ids = DictBasedKeyValue[str, int](data={})  # rollout_id -> sequence_id

    @property
    def rollouts(self) -> ListBasedCollection[Rollout]:
        return self._rollouts

    @property
    def attempts(self) -> ListBasedCollection[Attempt]:
        return self._attempts

    @property
    def spans(self) -> ListBasedCollection[Span]:
        return self._spans

    @property
    def resources(self) -> ListBasedCollection[ResourcesUpdate]:
        return self._resources

    @property
    def workers(self) -> ListBasedCollection[Worker]:
        return self._workers

    @property
    def rollout_queue(self) -> DequeBasedQueue[Rollout]:
        return self._rollout_queue

    @property
    def span_sequence_ids(self) -> DictBasedKeyValue[str, int]:
        return self._span_sequence_ids

    @asynccontextmanager
    async def atomic(self, *args: Any, **kwargs: Any) -> AsyncGenerator[None, None]:
        async with self._lock:
            yield

    async def evict_spans_for_rollout(self, rollout_id: str) -> None:
        """Evict all spans for a given rollout ID.

        Uses private API for efficiency.
        """
        self._spans._items.pop(rollout_id, [])  # pyright: ignore[reportPrivateUsage]


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

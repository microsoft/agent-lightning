# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from typing import (
    Any,
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
    FilterField,
    FilterOptions,
    PaginatedResult,
    ResourcesUpdate,
    Rollout,
    SortOptions,
    Span,
    Worker,
)

from .base import (
    AtomicMode,
    Collection,
    FilterMap,
    KeyValue,
    LightningCollections,
    Queue,
    normalize_filter_options,
    resolve_sort_options,
)

T = TypeVar("T")  # Recommended to be a BaseModel, not a dict
K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)

# Nested structure type:
# dict[pk1] -> dict[pk2] -> ... -> item
ListBasedCollectionItemType = Union[
    Dict[Any, "ListBasedCollectionItemType[T]"],  # intermediate node
    Dict[Any, T],  # leaf node dictionary
]

MutationMode = Literal["insert", "update", "upsert", "delete"]


def _item_matches_filters(
    item: object,
    filters: Optional[FilterMap],
    filter_logic: Literal["and", "or"],
    must_filters: Optional[FilterMap] = None,
) -> bool:
    """Check whether an item matches the provided filter definition.

    Filter format:

    ```json
    {
        "_aggregate": "or",
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
    if must_filters and not _item_matches_filters(item, must_filters, "and"):
        return False

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
                    all_conditions_match.append(item_value in expected)  # type: ignore[arg-type]
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


def _get_sort_value(item: object, sort_by: str) -> Any:
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

    Sorting behavior:

    1. If no sort_by is provided, the items are returned in the order of insertion.
    2. If sort_by is provided, the items are sorted by the value of the sort_by field.
    3. If the sort_by field is a timestamp, the null values are treated as infinity.
    4. If the sort_by field is not a timestamp, the null values are treated as empty string
       if the field is str-like, 0 if the field is int-like, 0.0 if the field is float-like.
    """

    def __init__(self, items: List[T], item_type: Type[T], primary_keys: Sequence[str]):
        if not primary_keys:
            raise ValueError("primary_keys must be non-empty")

        self._items: Dict[Any, Any] = {}
        self._size: int = 0
        if issubclass(item_type, dict):
            raise TypeError(f"Expect item to be not a dict, got {item_type.__name__}")
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

    async def size(self) -> int:
        """Return the number of items stored in the collection."""
        return self._size

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({self._size})>"

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

    def _mutate_single(self, item: T, mode: MutationMode, update_fields: Sequence[str] | None = None) -> Optional[T]:
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

                elif update_fields is None:
                    # update_or_insert: update all fields
                    parent[final_key] = item

                else:
                    if not issubclass(self._item_type, BaseModel):
                        raise TypeError(
                            f"When using update_fields, the item type must be a Pydantic BaseModel, got {self._item_type.__name__}"
                        )

                    # Try to fetch the existing item
                    existing = parent[final_key]
                    if not isinstance(existing, self._item_type):
                        raise ValueError(
                            f"Internal structure corrupted: expected {self._item_type.__name__}, got {type(existing)!r}"
                        )

                    if not isinstance(item, self._item_type):
                        raise TypeError(
                            f"When using update_fields, the item type must be a Pydantic BaseModel, got {type(item).__name__}"
                        )

                    parent[final_key] = parent[final_key].model_copy(
                        update={field: getattr(item, field) for field in update_fields}
                    )

            return parent[final_key]

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
                if update_fields is None:
                    # replace the entire item
                    parent[final_key] = item
                else:
                    if not issubclass(self._item_type, BaseModel):
                        raise TypeError(
                            f"When using update_fields, the item type must be a Pydantic BaseModel, got {self._item_type.__name__}"
                        )
                    if not isinstance(item, self._item_type):
                        raise TypeError(
                            f"When using update_fields, the item type must be a Pydantic BaseModel, got {type(item).__name__}"
                        )
                    parent[final_key] = parent[final_key].model_copy(
                        update={field: getattr(item, field) for field in update_fields}
                    )
                return parent[final_key]
            else:  # delete
                del parent[final_key]
                self._size -= 1
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")

    def _iter_items(
        self,
        root: Optional[Mapping[Any, Any]] = None,
        filters: Optional[FilterMap] = None,
        must_filters: Optional[FilterMap] = None,
        filter_logic: Literal["and", "or"] = "and",
    ) -> Iterable[T]:
        """Iterate over all items in the nested dictionary structure, optionally applying filters."""
        if root is None:
            root = self._items
        if not root:
            return
        stack: List[Mapping[Any, Any]] = [root]
        while stack:
            node = stack.pop()
            for value in node.values():
                # Leaf nodes contain items; intermediate nodes are dicts.
                if isinstance(value, self._item_type):
                    if _item_matches_filters(value, filters, filter_logic, must_filters):
                        yield value
                elif isinstance(value, dict):
                    stack.append(value)  # type: ignore
                else:
                    raise ValueError(
                        f"Internal structure corrupted: expected dict or {self._item_type.__name__}, "
                        f"got {type(value)!r}"
                    )

    def _iter_matching_items(
        self,
        filters: Optional[FilterMap],
        must_filters: Optional[FilterMap],
        filter_logic: Literal["and", "or"],
    ) -> Iterable[T]:
        """Efficiently iterate over items matching filters, using primary-key prefix when possible."""
        # Fast path: when optional filters can't form a prefix, fall back to scanning.
        if filter_logic != "and" and must_filters is None:
            return self._iter_items(filters=filters, must_filters=must_filters, filter_logic=filter_logic)

        # Try to derive a primary-key prefix from exact filters.
        pk_values_prefix: List[Any] = []
        prefix_sources: List[FilterMap] = []
        if must_filters:
            prefix_sources.append(must_filters)
        if filter_logic == "and" and filters:
            prefix_sources.append(filters)

        for pk in self._primary_keys:
            # combined_ops are: [{"exact": value}, {"within": [...]}, ...]
            combined_ops: List[FilterField] = []
            for source in prefix_sources:
                field_ops = source.get(pk)  # type: ignore[union-attr]
                if field_ops:
                    combined_ops.append(field_ops)
            if not combined_ops:
                break
            # Only allow a pure {"exact": value} constraint.
            exact_value: Any | None = None
            allow_prefix = True
            for ops in combined_ops:
                if set(ops.keys()) != {"exact"}:
                    allow_prefix = False
                    break
                candidate = ops.get("exact")
                if candidate is None:
                    allow_prefix = False
                    break
                if exact_value is not None and candidate != exact_value:
                    # Contradictory exact filters mean no items can match.
                    logger.warning(f"Contradictory exact filters for field '{pk}': {exact_value} != {candidate}")
                    return ()
                exact_value = candidate

            if not allow_prefix:
                break

            value = exact_value
            if value is None:
                break
            pk_values_prefix.append(value)

        if not pk_values_prefix:
            return self._iter_items(filters=filters, must_filters=must_filters, filter_logic=filter_logic)

        try:
            if len(pk_values_prefix) == len(self._primary_keys):
                # All primary keys specified -> at most a single item.
                parent, final_key = self._locate_node(pk_values_prefix, create_missing=False)
                single_item = parent.get(final_key)
                if isinstance(single_item, self._item_type) and _item_matches_filters(
                    single_item,
                    filters,
                    filter_logic,
                    must_filters,
                ):
                    return (single_item,)
                return ()
            else:
                # Prefix of primary keys specified -> iterate only the subtree below that prefix.
                parent, final_key = self._locate_node(pk_values_prefix, create_missing=False)
                subtree = parent.get(final_key)
                if isinstance(subtree, dict):
                    return self._iter_items(
                        subtree,  # type: ignore
                        filters=filters,
                        must_filters=must_filters,
                        filter_logic=filter_logic,
                    )
                return ()
        except KeyError:
            # No items exist for this primary-key prefix.
            return ()

    async def query(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with filters, sort order, and pagination.

        Args:
            filter: Mapping of field name to operator dict along with the optional `_aggregate` logic.
            sort: Options describing which field to sort by and in which order.
            limit: Max number of items to return. Use -1 for "no limit".
            offset: Number of items to skip from the start of the *matching* items.
        """
        filters, must_filters, filter_logic = normalize_filter_options(filter)
        sort_by, sort_order = resolve_sort_options(sort)
        items_iter: Iterable[T] = self._iter_matching_items(filters, must_filters, filter_logic)

        # No sorting: stream through items and apply pagination on the fly.
        if not sort_by:
            matched_items: List[T] = []
            total_matched = 0

            for item in items_iter:
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
        all_matches: List[T] = list(items_iter)

        total_matched = len(all_matches)
        reverse = sort_order == "desc"
        all_matches.sort(key=lambda x: _get_sort_value(x, sort_by), reverse=reverse)

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
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
    ) -> Optional[T]:
        """Return the first (or best-sorted) item that matches the given filters, or None."""
        filters, must_filters, filter_logic = normalize_filter_options(filter)
        sort_by, sort_order = resolve_sort_options(sort)
        items_iter: Iterable[T] = self._iter_matching_items(filters, must_filters, filter_logic)

        if not sort_by:
            # Just return the first matching item, if any.
            for item in items_iter:
                return item
            return None

        # Single-pass min/max according to sort_order.
        best_item: Optional[T] = None
        best_key: Any = None

        for item in items_iter:
            key = _get_sort_value(item, sort_by)
            if best_item is None:
                best_item = item
                best_key = key
                continue

            if sort_order == "asc":
                if key < best_key:
                    best_item, best_key = item, key
            else:
                if key > best_key:
                    best_item, best_key = item, key

        return best_item

    async def insert(self, items: Sequence[T]) -> None:
        """Insert the given items.

        Raises:
            ValueError: If any item with the same primary keys already exists.
        """
        seen_keys: set[Tuple[Any, ...]] = set()
        prepared: List[T] = []
        for item in items:
            self._ensure_item_type(item)
            key_values = self._extract_primary_key_values(item)
            if key_values in seen_keys:
                raise ValueError(
                    f"Insert payload contains duplicate primary key(s): {self._render_key_values(key_values)}"
                )
            seen_keys.add(key_values)
            prepared.append(item)

        for item in prepared:
            self._mutate_single(item, mode="insert")

    async def update(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        """Update the given items.

        Raises:
            ValueError: If any item with the given primary keys does not exist.
        """
        updated_items: List[T] = []
        for item in items:
            updated = self._mutate_single(item, mode="update", update_fields=update_fields)
            if updated is None:
                raise RuntimeError(f"_mutate_single returned None for item {item}. This should never happen.")
            updated_items.append(updated)
        return updated_items

    async def upsert(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        """Upsert the given items (insert if missing, otherwise update)."""
        upserted_items: List[T] = []
        for item in items:
            upserted = self._mutate_single(item, mode="upsert", update_fields=update_fields)
            if upserted is None:
                raise RuntimeError(f"_mutate_single returned None for item {item}. This should never happen.")
            upserted_items.append(upserted)
        return upserted_items

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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({len(self._items)})>"

    async def has(self, item: T) -> bool:
        if not isinstance(item, self._item_type):
            raise TypeError(f"Expected item of type {self._item_type.__name__}, got {type(item).__name__}")
        return item in self._items

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

    async def size(self) -> int:
        return len(self._items)


class DictBasedKeyValue(KeyValue[K, V]):
    """KeyValue implementation backed by a plain dictionary."""

    def __init__(self, data: Optional[Mapping[K, V]] = None):
        self._values: Dict[K, V] = dict(data) if data else {}

    async def has(self, key: K) -> bool:
        return key in self._values

    async def get(self, key: K, default: V | None = None) -> V | None:
        return self._values.get(key, default)

    async def set(self, key: K, value: V) -> None:
        self._values[key] = value

    async def pop(self, key: K, default: V | None = None) -> V | None:
        return self._values.pop(key, default)

    async def size(self) -> int:
        return len(self._values)


class InMemoryLightningCollections(LightningCollections):
    """In-memory implementation of LightningCollections using Python data structures.

    Serves as the storage base for [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore].
    """

    def __init__(self, lock_type: Literal["thread", "asyncio"]):
        self._lock = {
            "rollouts": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "attempts": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "spans": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "resources": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "workers": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "rollout_queue": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
            "span_sequence_ids": _LoopAwareAsyncLock() if lock_type == "asyncio" else _ThreadSafeAsyncLock(),
        }
        self._rollouts = ListBasedCollection(items=[], item_type=Rollout, primary_keys=["rollout_id"])
        self._attempts = ListBasedCollection(items=[], item_type=Attempt, primary_keys=["rollout_id", "attempt_id"])
        self._spans = ListBasedCollection(
            items=[], item_type=Span, primary_keys=["rollout_id", "attempt_id", "span_id"]
        )
        self._resources = ListBasedCollection(items=[], item_type=ResourcesUpdate, primary_keys=["resources_id"])
        self._workers = ListBasedCollection(items=[], item_type=Worker, primary_keys=["worker_id"])
        self._rollout_queue = DequeBasedQueue(items=[], item_type=str)
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
    def rollout_queue(self) -> DequeBasedQueue[str]:
        return self._rollout_queue

    @property
    def span_sequence_ids(self) -> DictBasedKeyValue[str, int]:
        return self._span_sequence_ids

    @asynccontextmanager
    async def atomic(
        self, *, mode: AtomicMode = "rw", snapshot: bool = False, labels: Optional[Sequence[str]] = None, **kwargs: Any
    ):
        """In-memory collections apply a lock outside. It doesn't need to manipulate the collections inside.

        Skip the locking if mode is "r" and snapshot is False.

        This collection implementation does NOT support rollback / commit.
        """
        if mode == "r" and not snapshot:
            yield self
            return
        if not labels:
            # If no labels are provided, use all locks.
            labels = list(self._lock.keys())
        managers = [self._lock[label] for label in labels]
        async with AsyncExitStack() as stack:
            for manager in managers:
                await stack.enter_async_context(manager)
            yield self

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


class _ThreadSafeAsyncLock:
    """A threading.Lock that can be used in both async and sync contexts."""

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any, **kwargs: Any):
        self._lock.release()

    async def __aenter__(self):
        # We run the blocking .acquire() in a thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._lock.acquire)
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any):
        # .release() is non-blocking, so we can call it directly
        self._lock.release()

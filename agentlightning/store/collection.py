from __future__ import annotations

from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel

from agentlightning.types import (
    Attempt,
    ResourcesUpdate,
    Rollout,
    Span,
    Worker,
)

T = TypeVar("T", bound=BaseModel)

Filter = Mapping[str, Mapping[Literal["exact", "within", "contains"], Any]]
"""Mapping of field name to filter conditions."""


class PaginatedResult(BaseModel, Generic[T]):
    """Result of a paginated query."""

    items: Sequence[T]
    """Items in the result."""
    limit: int
    """Limit of the result."""
    offset: int
    """Offset of the result."""
    total: int
    """Total number of items in the collection."""


class Collection(Generic[T]):
    """Behaves like a list of items. Supporting addition, updating, and deletion of items."""

    def primary_keys(self) -> Sequence[str]:
        """Get the primary keys of the collection."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({self.size()})>"

    def item_type(self) -> Type[T]:
        """Get the type of the items in the collection."""
        raise NotImplementedError()

    def size(self) -> int:
        """Get the number of items in the collection."""
        raise NotImplementedError()

    async def query(
        self,
        filters: Optional[Filter] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with the given filters, sort order, and pagination.

        Args:
            filters:
                A mapping of field name -> operator dict. Each operator dict can contain:
                - "exact": value for exact equality.
                - "in": iterable of allowed values.
                - "contains": substring to search for in string fields.

                Example:
                    {
                        "status": {"exact": "active"},
                        "id": {"in": [1, 2, 3]},
                        "name": {"contains": "foo"},
                    }

            filter_logic:
                How to combine per-field results:
                - "and": all fields must match.
                - "or": at least one field must match.

            sort_by:
                Optional field to sort by. Field must exist in the model.

            sort_order:
                "asc" or "desc" for ascending / descending sort.

            limit:
                Max number of items to return. Use -1 for "no limit".

            offset:
                Number of items to skip from the start of the *matching* items.

        Returns:
            PaginatedResult with items, limit, offset, and total matched items.
        """
        raise NotImplementedError()

    async def get(self, filters: Filter, filter_logic: Literal["and", "or"] = "and") -> Optional[T]:
        """Get the first item that matches the given filters.

        Args:
            filters: The filters to apply to the collection.

        Returns:
            The first item that matches the given filters, or None if no item matches.
        """
        raise NotImplementedError()

    async def insert(self, items: Sequence[T]) -> Sequence[T]:
        """Add the given items to the collection.

        Raises:
            ValueError: If an item with the same primary key already exists.
        """
        raise NotImplementedError()

    async def update(self, items: Sequence[T]) -> Sequence[T]:
        """Update the given items in the collection.

        Raises:
            ValueError: If an item with the primary keys does not exist.
        """
        raise NotImplementedError()

    async def upsert(self, items: Sequence[T]) -> Sequence[T]:
        """Upsert the given items into the collection.

        If the items with the same primary keys already exist, they will be updated.
        Otherwise, they will be inserted.
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


class Queue(Generic[T]):
    """Behaves like a deque. Supporting appending items to the end and popping items from the front."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({self.size()})>"

    def item_type(self) -> Type[T]:
        """Get the type of the items in the queue."""
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

    def size(self) -> int:
        """Get the number of items in the queue."""
        raise NotImplementedError()


# Nested structure type:
# dict[pk1] -> dict[pk2] -> ... -> item
ListCollectionItemType = Union[
    Dict[Any, "ListCollectionItemType[T]"],  # intermediate node
    Dict[Any, T],  # leaf node dictionary
]


MutationMode = Literal["insert", "update", "upsert", "delete"]


class ListCollection(Collection[T]):
    """List-based implementation of Collection using a nested dict for efficient lookup.

    The internal structure is:

    {
        pk1_value: {
            pk2_value: {
                ...: item
            }
        }
    }

    where the nesting depth equals the number of primary keys.
    """

    def __init__(self, items: List[T], item_type: Type[T], primary_keys: Sequence[str]):
        self._items: ListCollectionItemType[T] = dict()
        self._size = 0
        self._item_type = item_type
        if not primary_keys:
            raise ValueError("primary_keys must be non-empty")
        self._primary_keys = tuple(primary_keys)

    def primary_keys(self) -> Sequence[str]:
        """Get the primary keys of the collection."""
        return self._primary_keys

    def item_type(self) -> Type[T]:
        """Get the type of the items in the collection."""
        return self._item_type

    def size(self) -> int:
        """Get the number of items in the collection."""
        return len(self._items)

    def _flatten_items(self, items: ListCollectionItemType[T]) -> Iterable[T]:
        """Flatten the nested dictionary of items into a flat list of items."""
        if not items:
            return

        # Peek the first value
        first_value = next(iter(items.values()))
        if isinstance(first_value, dict):
            yield from self._flatten_items(first_value)
        else:
            yield from cast(Iterable[T], items.values())

    async def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with the given filters, sort order, and pagination."""
        # Apply filters
        filtered_items: List[T] = []
        if not filters:
            filtered_items = list(self._flatten_items(self._items))
        else:
            for item in self._flatten_items(self._items):
                matches: List[bool] = []
                for key, value in filters.items():
                    if value is None:
                        continue

                    # Handle _in suffix (list membership)
                    if key.endswith("_in"):
                        field = key[:-3]
                        item_value = getattr(item, field, None)
                        matches.append(item_value in value if isinstance(value, list) else False)
                    # Handle _contains suffix (substring match)
                    elif key.endswith("_contains"):
                        field = key[:-9]
                        item_value = getattr(item, field, None)
                        if item_value is not None and isinstance(item_value, str) and isinstance(value, str):
                            matches.append(value in item_value)
                        else:
                            matches.append(False)
                    # Exact match
                    else:
                        item_value = getattr(item, key, None)
                        matches.append(item_value == value)

                if matches:
                    if filter_logic == "and":
                        if all(matches):
                            filtered_items.append(item)
                    else:  # "or"
                        if any(matches):
                            filtered_items.append(item)

        # Apply sorting

        def _get_sort_value(item: T, sort_by: str) -> Any:
            if sort_by.endswith("_time"):
                value = getattr(item, sort_by, None)
                if value is None:
                    value = float("inf")
                return value
            else:
                # Other than _time, we assume the value must be a string
                value = getattr(item, sort_by, None)
                if value is None:
                    if sort_by not in item.__class__.model_fields:  # type: ignore
                        raise ValueError(
                            f"Failed to sort items by {sort_by}: {sort_by} is not a field of {item.__class__.__name__}"
                        )
                    field_type = str(item.__class__.model_fields[sort_by].annotation)  # type: ignore
                    if "str" in field_type or "Literal" in field_type:
                        return ""
                    if "int" in field_type:
                        return 0
                    if "float" in field_type:
                        return 0.0
                    raise ValueError(f"Failed to sort items by {sort_by}: {value} is not a string or number")
                return value

        if sort_by:
            reverse = sort_order == "desc"
            filtered_items.sort(key=lambda x: _get_sort_value(x, sort_by), reverse=reverse)

        # Get total count before pagination
        total = len(filtered_items)

        # Apply pagination
        if limit == -1:
            paginated_items = filtered_items[offset:]
        else:
            paginated_items = filtered_items[offset : offset + limit]

        return PaginatedResult(items=paginated_items, limit=limit, offset=offset, total=total)

    async def insert(self, items: Sequence[T]) -> Sequence[T]:
        """Add the given items to the collection."""
        primary_keys = self.primary_keys()
        for item in items:
            primary_key_values: List[Any] = []
            for primary_key in primary_keys:
                if not hasattr(item, primary_key):
                    raise ValueError(f"Item {item} does not have primary key {primary_key}")
                value = getattr(item, primary_key)
                primary_key_values.append(value)

            # Find the target dictionary to insert the item into
            target: ListCollectionItemType[T] = self._items
            for i, value in enumerate(primary_key_values):
                if i + 1 == len(primary_keys):
                    # For the final layer, directly assign the item
                    if value in target:
                        key_values_str = ", ".join(
                            [f"{key}={value}" for key, value in zip(primary_keys, primary_key_values)]
                        )
                        raise ValueError(f"Item {item} already exists with primary key {key_values_str}")
                    target[value] = item  # type: ignore
                else:
                    if value not in target:
                        target[value] = {}  # type: ignore
                    target = target[value]  # type: ignore
                    if not isinstance(target, dict):  # type: ignore
                        raise ValueError(f"Insert target {target} is not a dictionary")
        return items


class DequeQueue(Queue[T]):
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


class LightningCollections:
    """Collections of rollouts, attempts, spans, resources, and workers.

    [LightningStore][agentlightning.LightningStore] implementations can use this as a storage base
    to implement the store API.
    """

    rollouts: Collection[Rollout]
    """Collections of rollouts."""
    attempts: Collection[Attempt]
    """Collections of attempts."""
    spans: Collection[Span]
    """Collections of spans."""
    resources: Collection[ResourcesUpdate]
    """Collections of resources."""
    workers: Collection[Worker]
    """Collections of workers."""
    rollout_queue: Queue[Rollout]
    """Queue of rollouts (tasks)."""

    async def atomic(self, *args: Any, **kwargs: Any) -> None:
        """Perform a atomic operation on the collections.

        Subclass may use args and kwargs to support multiple levels of atomicity.

        Args:
            *args: Arguments to pass to the operation.
            **kwargs: Keyword arguments to pass to the operation.
        """
        raise NotImplementedError()

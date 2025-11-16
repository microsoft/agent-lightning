from __future__ import annotations

from typing import Any, Dict, Generic, Iterable, List, Literal, Mapping, Optional, Sequence, Type, TypeVar, Union, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PaginatedResult(BaseModel, Generic[T]):
    items: Sequence[T]
    limit: int
    offset: int
    total: int


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

    def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        """Query the collection with the given filters, sort order, and pagination.

        Args:
            filters: Optional filters to apply to the query.
                Use `_in` suffix to filter by a list of values.
                Use `_contains` suffix to filter by a substring.
                Use no suffix to filter by an exact match.
            filter_logic: The logic to apply to the filters.
                Use `and` to require all filters to be true.
                Use `or` to require any filter to be true.
            sort_by: Optional field to sort by.
                The field must be a string field of the item.
            sort_order: The order to sort by.
                Use `asc` to sort in ascending order.
                Use `desc` to sort in descending order.
            limit: The maximum number of items to return.
                Use `-1` to return all items.
            offset: The offset to start from.

        Returns:
            A paginated result containing the items, limit, offset, and total number of items.
        """
        raise NotImplementedError()

    def insert(self, items: Sequence[T]) -> Sequence[T]:
        """Add the given items to the collection.

        Args:
            items: The items to add to the collection.

        Returns:
            The items that were added to the collection.

        Raises:
            ValueError: If the items with the primary keys have already been added.
        """
        raise NotImplementedError()

    def update(self, items: Sequence[T]) -> Sequence[T]:
        """Update the given items in the collection.

        If the items with the primary keys do not exist, an error will be raised.

        Args:
            items: The items to update in the collection.

        Returns:
            The items that were updated in the collection.

        Raises:
            ValueError: If the items with the primary keys do not exist.
        """
        raise NotImplementedError()

    def upsert(self, items: Sequence[T]) -> Sequence[T]:
        """Upsert the given items into the collection.

        If the items with the same primary keys already exist, they will be updated.
        Otherwise, they will be inserted.
        """
        raise NotImplementedError()

    def delete(self, items: Sequence[T]) -> None:
        """Delete the given items from the collection.

        Args:
            items: The items to delete from the collection.

        Returns:
            The items that were deleted from the collection.

        Raises:
            ValueError: If the items with the primary keys do not exist.
        """
        raise NotImplementedError()


class Queue(Generic[T]):
    """Behaves like a deque. Supporting appending items to the end and popping items from the front."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.item_type().__name__}] ({self.size()})>"

    def item_type(self) -> Type[T]:
        """Get the type of the items in the queue."""
        raise NotImplementedError()

    def enqueue(self, items: Sequence[T]) -> Sequence[T]:
        """Append the given items to the end of the queue.

        Args:
            items: The items to append to the end of the queue.

        Returns:
            The items that were appended to the end of the queue.
        """
        raise NotImplementedError()

    def dequeue(self, limit: int = 1) -> Sequence[T]:
        """Pop the given number of items from the front of the queue.

        Args:
            limit: The number of items to pop from the front of the queue.

        Returns:
            The items that were popped from the front of the queue.
            If there are less than `limit` items in the queue, the remaining items will be returned.
        """
        raise NotImplementedError()

    def peek(self, limit: int = 1) -> Sequence[T]:
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


ListCollectionItemType = Union[Dict[str, "ListCollectionItemType[T]"], Dict[str, T]]


class ListCollection(Collection[T]):
    """List-based implementation of Collection.

    To make the items lookup efficient, the items will be stored in a nested dictionary:

    ```python
    {
        "<primary_key_1>": {
            "<primary_key_2>": {
                "<primary_key_3>": T,
            },
            "<primary_key_2>": {
                ...
            },
        }
        "<primary_key_1>": {
            ...
        },
    }
    ```

    Args:
        item_type: The type of the items in the collection.
        primary_keys: The primary keys of the items in the collection.
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

    def query(
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

    def insert(self, items: Sequence[T]) -> Sequence[T]:
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

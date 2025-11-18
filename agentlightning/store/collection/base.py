# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import (
    Any,
    AsyncContextManager,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from agentlightning.types import (
    Attempt,
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

    async def update(self, items: Sequence[T]) -> None:
        """Update the given items in the collection.

        Raises:
            ValueError: If an item with the primary keys does not exist.
        """
        raise NotImplementedError()

    async def upsert(self, items: Sequence[T]) -> None:
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

    def size(self) -> int:
        """Get the number of items in the queue."""
        raise NotImplementedError()


class KeyValue(Generic[K, V]):
    """Behaves like a dictionary. Supporting addition, updating, and deletion of items."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self.size()})>"

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

    def size(self) -> int:
        """Get the number of items in the dictionary."""
        raise NotImplementedError()


class LightningCollections:
    """Collections of rollouts, attempts, spans, resources, and workers.

    [LightningStore][agentlightning.LightningStore] implementations can use this as a storage base
    to implement the store API.
    """

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

    def atomic(self, *args: Any, **kwargs: Any) -> AsyncContextManager[None]:
        """Perform a atomic operation on the collections.

        Subclass may use args and kwargs to support multiple levels of atomicity.

        Args:
            *args: Arguments to pass to the operation.
            **kwargs: Keyword arguments to pass to the operation.
        """
        raise NotImplementedError()

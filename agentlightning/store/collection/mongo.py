# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Type, TypeVar, cast

from pydantic import BaseModel, TypeAdapter
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, DuplicateKeyError

from agentlightning.types import FilterOptions, PaginatedResult, SortOptions

from .base import Collection, KeyValue, Queue, normalize_filter_options, resolve_sort_options

T_model = TypeVar("T_model", bound=BaseModel)

T_generic = TypeVar("T_generic")
K = TypeVar("K")
V = TypeVar("V")


def _field_ops_to_conditions(field: str, ops: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert a FilterField (ops) into one or more Mongo conditions."""
    conditions: List[Dict[str, Any]] = []

    if "exact" in ops:
        conditions.append({field: ops["exact"]})

    if "within" in ops and ops["within"] is not None:
        conditions.append({field: {"$in": list(ops["within"])}})

    if "contains" in ops and ops["contains"] is not None:
        # Use case-insensitive substring match via regex.
        value = str(ops["contains"])
        pattern = f".*{re.escape(value)}.*"
        conditions.append({field: {"$regex": pattern, "$options": "i"}})

    return conditions


def _build_mongo_filter(filter_options: Optional[FilterOptions]) -> Dict[str, Any]:
    """Translate FilterOptions into a MongoDB filter dict."""
    normalized, must_filters, aggregate = normalize_filter_options(filter_options)

    regular_conditions: List[Dict[str, Any]] = []
    must_conditions: List[Dict[str, Any]] = []

    # Normal filters
    if normalized:
        for field_name, ops in normalized.items():
            regular_conditions.extend(_field_ops_to_conditions(field_name, ops))

    # Must filters
    if must_filters:
        for field_name, ops in must_filters.items():
            must_conditions.extend(_field_ops_to_conditions(field_name, ops))

    # No filters at all
    if not regular_conditions and not must_conditions:
        return {}

    # Aggregate logic for regular conditions; _must always ANDs in.
    if aggregate == "and":
        all_conds = regular_conditions + must_conditions
        if len(all_conds) == 1:
            return all_conds[0]
        return {"$and": all_conds}

    # aggregate == "or"
    if regular_conditions and must_conditions:
        # (OR of regular) AND (all must)
        if len(regular_conditions) == 1:
            or_part: Dict[str, Any] = regular_conditions[0]
        else:
            or_part = {"$or": regular_conditions}

        and_parts: List[Dict[str, Any]] = [or_part] + must_conditions
        if len(and_parts) == 1:
            return and_parts[0]
        return {"$and": and_parts}

    if regular_conditions:
        if len(regular_conditions) == 1:
            return regular_conditions[0]
        return {"$or": regular_conditions}

    # Only must conditions
    if len(must_conditions) == 1:
        return must_conditions[0]
    return {"$and": must_conditions}


class MongoBasedCollection(Collection[T_model]):
    """Mongo-based implementation of Collection.

    Args:
        db: The MongoDB database.
        collection_name: The name of the collection.
        partition_id: The partition ID. Used to partition the collection into multiple collections.
        primary_keys: The primary keys of the collection.
        item_type: The type of the items in the collection.
    """

    def __init__(
        self,
        db: AsyncDatabase[Mapping[str, Any]],
        collection_name: str,
        partition_id: str,
        primary_keys: Sequence[str],
        item_type: Type[T_model],
    ):
        self._db = db
        self._collection = db[collection_name]
        self._partition_id = partition_id

        if not primary_keys:
            raise ValueError("primary_keys must be non-empty")
        self._primary_keys = list(primary_keys)

        if not issubclass(item_type, BaseModel):  # type: ignore
            raise ValueError(f"item_type must be a subclass of BaseModel, got {item_type.__name__}")
        self._item_type = item_type

    async def ensure_collection(self, *, create_indexes: bool = True) -> None:
        """Ensure the backing MongoDB collection exists (and optionally its indexes).

        This method is idempotent and safe to call multiple times.

        Args:
            create_indexes:
                If True, create a unique index across the configured primary
                key fields. If such an index already exists with the same
                definition, MongoDB will treat this as a no-op.
        """
        # Create collection if it doesn't exist yet
        existing = await self._db.list_collection_names()
        if self._collection.name not in existing:
            await self._db.create_collection(self._collection.name)

        # Optionally create a unique index on primary keys (scoped by partition_id)
        if create_indexes and self._primary_keys:
            # Always include the partition field in the unique index.
            keys = [("partition_id", 1)] + [(pk, 1) for pk in self._primary_keys]
            await self._collection.create_index(
                keys,
                unique=True,
                name=f"uniq_partition_{'_'.join(self._primary_keys)}",
            )

    def primary_keys(self) -> Sequence[str]:
        """Return the primary key field names for this collection."""
        return self._primary_keys

    def item_type(self) -> Type[T_model]:
        return self._item_type

    async def size(self) -> int:
        return await self._collection.count_documents({"partition_id": self._partition_id})

    def _pk_filter(self, item: T_model) -> Dict[str, Any]:
        """Build a Mongo filter for the primary key(s) of a model instance."""
        data = item.model_dump()
        missing = [pk for pk in self._primary_keys if pk not in data]
        if missing:
            raise ValueError(f"Missing primary key fields {missing} on item {item!r}")
        pk_filter: Dict[str, Any] = {"partition_id": self._partition_id}
        pk_filter.update({pk: data[pk] for pk in self._primary_keys})
        return pk_filter

    async def query(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T_model]:
        # Always require partition_id via _must in FilterOptions
        if filter is None:
            combined: Dict[str, Any] = {
                "_must": {
                    "partition_id": {"exact": self._partition_id},
                }
            }
        else:
            combined = dict(filter)
            existing_must = combined.get("_must")
            partition_must = {"partition_id": {"exact": self._partition_id}}
            if existing_must is None:
                combined["_must"] = partition_must
            else:
                # Merge it with the existing must filters.
                combined["_must"] = {**existing_must, **partition_must}

        mongo_filter = _build_mongo_filter(cast(FilterOptions, combined))

        total = await self._collection.count_documents(mongo_filter)

        cursor = self._collection.find(mongo_filter)

        sort_name, sort_order = resolve_sort_options(sort)
        if sort_name is not None:
            direction = 1 if sort_order == "asc" else -1
            cursor = cursor.sort(sort_name, direction)

        if offset > 0:
            cursor = cursor.skip(offset)

        if limit >= 0:
            cursor = cursor.limit(limit)

        items: List[T_model] = []
        async for raw in cursor:
            # Convert Mongo document to Pydantic model
            items.append(self._item_type.model_validate(raw))  # type: ignore[arg-type]

        return PaginatedResult[T_model](items=items, limit=limit, offset=offset, total=total)

    async def get(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
    ) -> Optional[T_model]:
        result = await self.query(filter=filter, sort=sort, limit=1, offset=0)
        return result.items[0] if result.items else None

    async def insert(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        docs: List[Mapping[str, Any]] = []
        for item in items:
            # Pre-check for existence to provide a clearer ValueError
            pk_filter = self._pk_filter(item)
            existing = await self._collection.find_one(pk_filter)
            if existing is not None:
                raise ValueError(f"Item with primary key(s) {pk_filter} already exists")

            doc = item.model_dump()
            doc["partition_id"] = self._partition_id
            docs.append(doc)

        if not docs:
            return

        try:
            await self._collection.insert_many(docs)
        except DuplicateKeyError as exc:
            # In case the DB enforces uniqueness via index, normalize to ValueError
            raise ValueError("Duplicate key error while inserting items") from exc

    async def update(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        for item in items:
            pk_filter = self._pk_filter(item)
            doc = item.model_dump()
            doc["partition_id"] = self._partition_id
            result = await self._collection.replace_one(pk_filter, doc)
            if result.matched_count == 0:
                raise ValueError(f"Item with primary key(s) {pk_filter} does not exist")

    async def upsert(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        for item in items:
            pk_filter = self._pk_filter(item)
            doc = item.model_dump()
            doc["partition_id"] = self._partition_id
            await self._collection.replace_one(pk_filter, doc, upsert=True)

    async def delete(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        for item in items:
            pk_filter = self._pk_filter(item)
            result = await self._collection.delete_one(pk_filter)
            if result.deleted_count == 0:
                raise ValueError(f"Item with primary key(s) {pk_filter} does not exist")


class MongoBasedQueue(Queue[T_generic], Generic[T_generic]):
    """Mongo-based implementation of Queue backed by a MongoDB collection.

    Items are stored append-only; dequeue marks items as consumed instead of deleting them.
    """

    def __init__(
        self,
        db: AsyncDatabase[Mapping[str, Any]],
        collection_name: str,
        partition_id: str,
        item_type: Type[T_generic],
    ) -> None:
        """
        Args:
            db: The MongoDB database.
            collection_name: The name of the collection backing the queue.
            partition_id: Partition identifier; allows multiple logical queues in one collection.
            item_type: The Python type of queue items (primitive or BaseModel subclass).
        """
        self._db = db
        self._collection = db[collection_name]
        self._partition_id = partition_id
        self._item_type = item_type
        self._adapter: TypeAdapter[T_generic] = TypeAdapter(item_type)

    def item_type(self) -> Type[T_generic]:
        return self._item_type

    async def ensure_collection(self) -> None:
        """Ensure the backing collection exists.

        If it already exists, this is a no-op.
        """
        existing = await self._db.list_collection_names()
        if self._collection.name in existing:
            # Assume it's already correctly configured.
            return

        try:
            # Create a normal (non-capped) collection.
            await self._db.create_collection(self._collection.name)
        except CollectionInvalid as exc:
            raise CollectionInvalid(f"Failed to create collection '{self._collection.name}'") from exc

        # Index to speed up partition + consumed lookups.
        await self._collection.create_index(
            [("partition_id", 1), ("consumed", 1), ("_id", 1)],
            name="queue_partition_consumed_id",
        )

    async def has(self, item: T_generic) -> bool:
        encoded = self._adapter.dump_python(item, mode="python")
        doc = await self._collection.find_one(
            {
                "partition_id": self._partition_id,
                "consumed": False,
                "value": encoded,
            }
        )
        return doc is not None

    async def enqueue(self, items: Sequence[T_generic]) -> Sequence[T_generic]:
        if not items:
            return []

        docs: Sequence[Mapping[str, Any]] = [
            {
                "partition_id": self._partition_id,
                "value": self._adapter.dump_python(item, mode="python"),
                "consumed": False,
                "created_at": datetime.now(),
            }
            for item in items
        ]

        await self._collection.insert_many(list(docs))
        return list(items)

    async def dequeue(self, limit: int = 1) -> Sequence[T_generic]:
        if limit <= 0:
            return []

        results: list[T_generic] = []

        # Atomic claim loop using find_one_and_update
        for _ in range(limit):
            doc = await self._collection.find_one_and_update(
                {
                    "partition_id": self._partition_id,
                    "consumed": False,
                },
                {"$set": {"consumed": True}},
                sort=[("_id", 1)],  # FIFO using insertion order
                return_document=True,
            )
            if doc is None:  # type: ignore
                # No more items to dequeue
                break

            raw_value = doc["value"]
            item = self._adapter.validate_python(raw_value)
            results.append(item)

        return results

    async def peek(self, limit: int = 1) -> Sequence[T_generic]:
        if limit <= 0:
            return []

        cursor = (
            self._collection.find(
                {
                    "partition_id": self._partition_id,
                    "consumed": False,
                }
            )
            .sort("_id", 1)
            .limit(limit)
        )

        items: list[T_generic] = []
        async for doc in cursor:
            raw_value = doc["value"]
            items.append(self._adapter.validate_python(raw_value))

        return items

    async def size(self) -> int:
        return await self._collection.count_documents(
            {
                "partition_id": self._partition_id,
                "consumed": False,
            }
        )


class MongoBasedKeyValue(KeyValue[K, V], Generic[K, V]):
    """Mongo-based implementation of KeyValue."""

    def __init__(
        self,
        db: AsyncDatabase[Mapping[str, Any]],
        collection_name: str,
        partition_id: str,
        key_type: Type[K],
        value_type: Type[V],
    ) -> None:
        """
        Args:
            db: The MongoDB database.
            collection_name: The name of the collection backing the key-value store.
            partition_id: Partition identifier; allows multiple logical maps in one collection.
            key_type: The Python type of keys (primitive or BaseModel).
            value_type: The Python type of values (primitive or BaseModel).
        """
        self._db = db
        self._collection = db[collection_name]
        self._partition_id = partition_id
        self._key_type = key_type
        self._value_type = value_type
        self._key_adapter: TypeAdapter[K] = TypeAdapter(key_type)
        self._value_adapter: TypeAdapter[V] = TypeAdapter(value_type)

    async def ensure_collection(self, *, create_indexes: bool = True) -> None:
        """Ensure the backing collection exists (and optionally its indexes)."""
        existing = await self._db.list_collection_names()
        if self._collection.name not in existing:
            await self._db.create_collection(self._collection.name)

        if create_indexes:
            await self._collection.create_index(
                [("partition_id", 1), ("key", 1)],
                unique=True,
                name="uniq_partition_key",
            )

    async def has(self, key: K) -> bool:
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await self._collection.find_one(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            }
        )
        return doc is not None

    async def get(self, key: K, default: V | None = None) -> V | None:
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await self._collection.find_one(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            }
        )
        if doc is None:
            return default

        raw_value = doc["value"]
        return self._value_adapter.validate_python(raw_value)

    async def set(self, key: K, value: V) -> None:
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        encoded_value = self._value_adapter.dump_python(value, mode="python")
        try:
            await self._collection.replace_one(
                {
                    "partition_id": self._partition_id,
                    "key": encoded_key,
                },
                {
                    "partition_id": self._partition_id,
                    "key": encoded_key,
                    "value": encoded_value,
                },
                upsert=True,
            )
        except DuplicateKeyError as exc:
            # Very unlikely with replace_one+upsert, but normalize anyway.
            raise ValueError("Duplicate key error while setting key-value item") from exc

    async def pop(self, key: K, default: V | None = None) -> V | None:
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await self._collection.find_one_and_delete(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            }
        )
        if doc is None:  # type: ignore
            return default

        raw_value = doc["value"]
        return self._value_adapter.validate_python(raw_value)

    async def size(self) -> int:
        return await self._collection.count_documents(
            {
                "partition_id": self._partition_id,
            }
        )

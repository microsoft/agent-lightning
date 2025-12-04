# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import random
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from typing import Self

from pydantic import BaseModel, TypeAdapter
from pymongo import AsyncMongoClient, ReadPreference, ReturnDocument, WriteConcern
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, ConnectionFailure, DuplicateKeyError, OperationFailure, PyMongoError
from pymongo.read_concern import ReadConcern

from agentlightning.store.base import LightningStore
from agentlightning.store.utils import LATENCY_BUCKETS
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

from .base import (
    AtomicMode,
    Collection,
    KeyValue,
    LightningCollections,
    Queue,
    normalize_filter_options,
    resolve_sort_options,
)

T_model = TypeVar("T_model", bound=BaseModel)

T_generic = TypeVar("T_generic")

T_mapping = TypeVar("T_mapping", bound=Mapping[str, Any])

T_callable = TypeVar("T_callable", bound=Callable[..., Any])

K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)

_OPERATION_CONTEXT: contextvars.ContextVar["_MongoOperationContext | None"] = contextvars.ContextVar(
    "_mongo_operation_context", default=None
)

_LIGHTNING_STORE_PUBLIC_METHODS = frozenset(
    [name for name, value in LightningStore.__dict__.items() if not name.startswith("_") and callable(value)]
    + ["_healthcheck"]
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
            if method_name in _LIGHTNING_STORE_PUBLIC_METHODS and isinstance(self_obj, LightningStore):
                return method_name
            frame = frame.f_back
        return _UNKNOWN_STORE_METHOD
    except Exception:
        return _UNKNOWN_STORE_METHOD
    finally:
        del frame


class MongoOperationPrometheusTracker:
    """A tracker for MongoDB operations metrics.

    All classes should share one single instance of this tracker.
    """

    def __init__(self, enabled: bool):
        self._enabled = enabled

        if enabled:
            from prometheus_client import Counter, Histogram

            base_labels = ["operation", "database", "collection", "store_method"]
            self._latency_metric = Histogram(
                "mongo_operation_duration_seconds",
                "Latency of MongoDB operations",
                base_labels,
                buckets=LATENCY_BUCKETS,
            )
            self._total_metric = Counter(
                "mongo_operation_total",
                "Total MongoDB operations",
                base_labels + ["status"],
            )
            self._error_metric = Counter(
                "mongo_operation_errors_total",
                "Total MongoDB operations that failed",
                base_labels + ["error_type"],
            )
            self._num_attempts_metric = Histogram(
                "mongo_operation_num_attempts",
                "Number of attempts for MongoDB operations",
                base_labels,
                buckets=list(range(10)) + list(range(10, 100, 5)),
            )

    def track(self, operation: str, database: str, collection: str) -> _MongoOperationContext | _DummyOperationContext:
        if not self._enabled:
            return _DummyOperationContext()
        return _MongoOperationContext(self, operation, database, collection)

    @staticmethod
    def classify_error(exc: BaseException | None) -> str:
        if exc is None:
            return "Other"
        is_transient = isinstance(exc, PyMongoError) and exc.has_error_label("TransientTransactionError")
        if isinstance(exc, OperationFailure):
            if is_transient:
                return f"OperationFailure-{exc.code}-Transient"
            else:
                return f"OperationFailure-{exc.code}"
        if isinstance(exc, DuplicateKeyError):
            return "DuplicateKeyError-Transient" if is_transient else "DuplicateKeyError"
        if isinstance(exc, PyMongoError):
            if is_transient:
                return f"{exc.__class__.__name__}-Transient"
            else:
                return exc.__class__.__name__
        if isinstance(exc, ConnectionFailure):
            return "ConnectionFailure-Transient" if is_transient else "ConnectionFailure"
        return "Other-Transient" if is_transient else "Other"

    def observe(
        self,
        *,
        operation: str,
        database: str,
        collection: str,
        elapsed: float,
        status: str,
        error_type: str | None,
        num_attempts: int | None = None,
    ) -> None:
        if not self._enabled:
            return

        store_method = _nearest_lightning_store_method_from_stack()
        self._total_metric.labels(operation, database, collection, store_method, status).inc()
        self._latency_metric.labels(operation, database, collection, store_method).observe(elapsed)
        if status == "error" and error_type:
            self._error_metric.labels(operation, database, collection, store_method, error_type).inc()
        if num_attempts is not None:
            self._num_attempts_metric.labels(operation, database, collection, store_method).observe(num_attempts)


class _MongoOperationContext:
    """A context manager for tracking MongoDB operations.

    Used via:

    ```python
    with self.tracker.track("insert", "database", "collection") as track_context:
        try:
            await collection.insert_one({})
        except Exception as exc:
            # For errors that can be ignored, report the error to the tracker.
            track_context.report_error(exc)
    ```
    """

    def __init__(self, tracker: MongoOperationPrometheusTracker, operation: str, database: str, collection: str):
        self._tracker = tracker
        self._operation = operation
        self._database = database
        self._collection = collection
        self._start: float = 0.0
        self._active: bool = False
        self._error_type: str | None = None
        self._num_attempts: int | None = None

    def __enter__(self) -> "_MongoOperationContext":
        self._active = True
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if not self._active:
            return False

        elapsed = time.perf_counter() - self._start

        # Try to classify the error
        if self._error_type is not None:
            error_type = self._error_type
        elif exc is not None:
            error_type = self._tracker.classify_error(exc)
        else:
            error_type = None

        status = "ok" if error_type is None else "error"
        self._tracker.observe(
            operation=self._operation,
            database=self._database,
            collection=self._collection,
            elapsed=elapsed,
            status=status,
            error_type=error_type,
            num_attempts=self._num_attempts,
        )
        return False

    def report_error(self, exc: BaseException) -> None:
        """Used to report errors that occurred in the middle of an operation."""
        self._error_type = self._tracker.classify_error(exc)

    def report_num_attempts(self, num_attempts: int) -> None:
        """Used to report the number of attempts that occurred in the middle of an operation."""
        self._num_attempts = num_attempts


class _DummyOperationContext:
    """A dummy context manager that does nothing, but compatible with _MongoOperationContext."""

    def __init__(self):
        pass

    def __enter__(self) -> "_DummyOperationContext":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        return False

    def report_error(self, exc: BaseException) -> None:
        pass

    def report_num_attempts(self, num_attempts: int) -> None:
        pass


def _mongo_operation(operation: str) -> Callable[[T_callable], T_callable]:
    def decorator(func: T_callable) -> T_callable:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"_mongo_operation decorator requires coroutine functions, got {func.__name__}")

        @functools.wraps(func)
        async def wrapper(
            self: (
                MongoBasedCollection[T_model]
                | MongoBasedQueue[T_generic]
                | MongoBasedKeyValue[K, V]
                | MongoLightningCollections
            ),
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            tracker = self._prometheus_tracker  # pyright: ignore[reportPrivateUsage]
            if not tracker._enabled:  # pyright: ignore[reportPrivateUsage]
                # Skip the tracking because tracking is not configured
                return await func(self, *args, **kwargs)
            with tracker.track(
                operation, self._database_name, self._collection_name  # pyright: ignore[reportPrivateUsage]
            ):
                return await func(self, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def _field_ops_to_conditions(field: str, ops: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert a FilterField (ops) into one or more Mongo conditions."""
    conditions: List[Dict[str, Any]] = []

    for op_name, raw_value in ops.items():
        if op_name == "exact":
            if raw_value is None:
                logger.debug(f"Skipping exact filter for field '{field}' with None value")
                continue
            conditions.append({field: raw_value})
        elif op_name == "within":
            if raw_value is None:
                logger.debug(f"Skipping within filter for field '{field}' with None value")
                continue
            try:
                iterable = list(raw_value)
            except TypeError as exc:
                raise ValueError(f"Invalid iterable for within filter for field '{field}': {raw_value!r}") from exc
            conditions.append({field: {"$in": iterable}})
        elif op_name == "contains":
            if raw_value is None:
                logger.debug(f"Skipping contains filter for field '{field}' with None value")
                continue
            value = str(raw_value)
            pattern = f".*{re.escape(value)}.*"
            conditions.append({field: {"$regex": pattern, "$options": "i"}})
        else:
            raise ValueError(f"Unsupported filter operator '{op_name}' for field '{field}'")

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


async def _ensure_collection(
    db: AsyncDatabase[Mapping[str, Any]],
    collection_name: str,
    primary_keys: Optional[Sequence[str]] = None,
    extra_indexes: Optional[Sequence[Sequence[str]]] = None,
) -> bool:
    """Ensure the backing MongoDB collection exists.

    This method is idempotent and safe to call multiple times.
    """
    # Create collection if it doesn't exist yet
    try:
        await db.create_collection(collection_name)
    except CollectionInvalid as exc:
        # Thrown if collection already exists
        logger.debug(f"Collection '{collection_name}' may have already existed. No need to create it: {exc!r}")
    except OperationFailure as exc:
        logger.debug(f"Failed to create collection '{collection_name}'. Probably already exists: {exc!r}")
        # Some servers use OperationFailure w/ specific codes for "NamespaceExists"
        if exc.code in (48, 68):  # 48: NamespaceExists, 68: already exists on older versions
            pass
        else:
            raise

    # Optionally create a unique index on primary keys (scoped by partition_id)
    if primary_keys:
        # Always include the partition field in the unique index.
        keys = [("partition_id", 1)] + [(pk, 1) for pk in primary_keys]
        try:
            await db[collection_name].create_index(keys, name=f"uniq_partition_{'_'.join(primary_keys)}", unique=True)
        except OperationFailure as exc:
            logger.debug(f"Index for collection '{collection_name}' already exists. No need to create it: {exc!r}")
            # Ignore "index already exists" type errors
            if exc.code in (68, 85):  # IndexOptionsConflict, etc.
                pass
            else:
                raise

    # Optionally create extra indexes
    if extra_indexes:
        for index in extra_indexes:
            try:
                await db[collection_name].create_index(index, name=f"idx_{'_'.join(index)}")
            except OperationFailure as exc:
                logger.debug(f"Index for collection '{collection_name}' already exists. No need to create it: {exc!r}")
                # Ignore "index already exists" type errors
                if exc.code in (68, 85):  # IndexOptionsConflict, etc.
                    pass
                else:
                    raise

    return True


class MongoClientPool(Generic[T_mapping]):
    """A pool of MongoDB clients, each binded to a specific event loop.

    This class is to resolve the issue of MongoDB client cannot be shared across event loops:

    ```
    Cannot use AsyncMongoClient in different event loop. AsyncMongoClient uses low-level asyncio APIs that bind it to the event loop it was created on.
    ```

    Use the client pool with a context manager to ensure all clients are closed when the context is exited.
    """

    def __init__(self, client: AsyncMongoClient[T_mapping]):
        self._lock = threading.Lock()

        self._client_base = client
        self._client_pool: Dict[int, AsyncMongoClient[T_mapping]] = {}

        self._collection_pool: Dict[Tuple[int, str, str], AsyncCollection[T_mapping]] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all clients in the pool (except the base client)."""

        with self._lock:
            clients = list(self._client_pool.values())
            self._client_pool.clear()

        for client in clients:
            try:
                if client is not self._client_base:
                    await client.close()
                else:
                    logger.debug("Skipping closing base client: %s", client)
            except Exception:
                logger.exception("Error closing MongoDB client: %s", client)

    async def get_client(self) -> AsyncMongoClient[T_mapping]:
        loop = asyncio.get_running_loop()
        key = id(loop)

        # If there is already a client specifically for this loop, return it.
        if key in self._client_pool:
            # Verify that the client still works.
            await self._client_pool[key].aconnect()
            return self._client_pool[key]

        try:
            # Try whether the base client will work.
            await self._client_base.aconnect()

            with self._lock:
                # If it works, add it to the pool and return it.
                self._client_pool.setdefault(key, self._client_base)
            return self._client_base
        except RuntimeError as exc:
            if "Cannot use AsyncMongoClient in different event loop" in str(exc):
                with self._lock:
                    # Create a new client for this loop.
                    client = self._client_base._duplicate()  # type: ignore
                    # Try whether the new client will work.
                    await client.aconnect()
                    # Add it to the pool and return it.
                    self._client_pool.setdefault(key, client)  # type: ignore
                return client  # type: ignore

            raise

    async def get_collection(self, database_name: str, collection_name: str) -> AsyncCollection[T_mapping]:
        loop = asyncio.get_running_loop()
        key = (id(loop), database_name, collection_name)
        if key in self._collection_pool:
            return self._collection_pool[key]

        # Create a new collection for this loop.
        client = await self.get_client()
        collection = client[database_name][collection_name]
        with self._lock:
            self._collection_pool.setdefault(key, collection)
        return collection


class MongoBasedCollection(Collection[T_model]):
    """Mongo-based implementation of Collection.

    Args:
        client_pool: The pool of MongoDB clients.
        database_name: The name of the database.
        collection_name: The name of the collection.
        partition_id: The partition ID. Used to partition the collection into multiple collections.
        primary_keys: The primary keys of the collection.
        item_type: The type of the items in the collection.
        extra_indexes: The extra indexes to create on the collection.
    """

    def __init__(
        self,
        client_pool: MongoClientPool[Mapping[str, Any]] | AsyncMongoClient[Mapping[str, Any]],
        database_name: str,
        collection_name: str,
        partition_id: str,
        primary_keys: Sequence[str],
        item_type: Type[T_model],
        extra_indexes: Sequence[Sequence[str]] = [],
        prometheus_tracker: MongoOperationPrometheusTracker | None = None,
    ):
        if isinstance(client_pool, AsyncMongoClient):
            self._client_pool = MongoClientPool(client_pool)
        else:
            self._client_pool = client_pool
        self._database_name = database_name
        self._collection_name = collection_name
        self._partition_id = partition_id
        self._collection_created = False
        self._extra_indexes = [list(index) for index in extra_indexes]
        self._session: Optional[AsyncClientSession] = None
        self._prometheus_tracker = (
            prometheus_tracker if prometheus_tracker is not None else MongoOperationPrometheusTracker(enabled=False)
        )

        if not primary_keys:
            raise ValueError("primary_keys must be non-empty")
        self._primary_keys = list(primary_keys)

        if not issubclass(item_type, BaseModel):  # type: ignore
            raise ValueError(f"item_type must be a subclass of BaseModel, got {item_type.__name__}")
        self._item_type = item_type

    @_mongo_operation("ensure_collection")
    async def ensure_collection(self) -> AsyncCollection[Mapping[str, Any]]:
        """Ensure the backing MongoDB collection exists (and optionally its indexes).

        This method is idempotent and safe to call multiple times.

        It will also create a unique index across the configured primary key fields.
        """
        if not self._collection_created:
            client = await self._client_pool.get_client()
            self._collection_created = await _ensure_collection(
                client[self._database_name], self._collection_name, self._primary_keys, self._extra_indexes
            )

        return await self._client_pool.get_collection(self._database_name, self._collection_name)

    def with_session(self, session: AsyncClientSession) -> MongoBasedCollection[T_model]:
        """Create a new collection with the same configuration but a new session."""
        collection = MongoBasedCollection(
            client_pool=self._client_pool,
            database_name=self._database_name,
            collection_name=self._collection_name,
            partition_id=self._partition_id,
            primary_keys=self._primary_keys,
            item_type=self._item_type,
            extra_indexes=self._extra_indexes,
            prometheus_tracker=self._prometheus_tracker,
        )
        collection._collection_created = self._collection_created
        collection._session = session
        return collection

    def primary_keys(self) -> Sequence[str]:
        """Return the primary key field names for this collection."""
        return self._primary_keys

    def item_type(self) -> Type[T_model]:
        return self._item_type

    @_mongo_operation("size")
    async def size(self) -> int:
        collection = await self.ensure_collection()
        return await collection.count_documents({"partition_id": self._partition_id}, session=self._session)

    def _pk_filter(self, item: T_model) -> Dict[str, Any]:
        """Build a Mongo filter for the primary key(s) of a model instance."""
        data = item.model_dump()
        missing = [pk for pk in self._primary_keys if pk not in data]
        if missing:
            raise ValueError(f"Missing primary key fields {missing} on item {item!r}")
        pk_filter: Dict[str, Any] = {"partition_id": self._partition_id}
        pk_filter.update({pk: data[pk] for pk in self._primary_keys})
        return pk_filter

    def _render_pk_values(self, values: Sequence[Any]) -> str:
        return ", ".join(f"{pk}={value!r}" for pk, value in zip(self._primary_keys, values))

    def _ensure_item_type(self, item: T_model) -> None:
        if not isinstance(item, self._item_type):
            raise TypeError(f"Expected item of type {self._item_type.__name__}, got {type(item).__name__}")

    def _inject_partition_filter(self, filter: Optional[FilterOptions]) -> Dict[str, Any]:
        """Ensure every query is scoped to this collection's partition."""
        combined: Dict[str, Any]
        if filter is None:
            combined = {}
        else:
            combined = dict(filter)

        partition_must = {"partition_id": {"exact": self._partition_id}}
        existing_must = combined.get("_must")
        if existing_must is None:
            combined["_must"] = partition_must
            return combined

        if isinstance(existing_must, Mapping):
            combined["_must"] = [existing_must, partition_must]
        elif isinstance(existing_must, Sequence) and not isinstance(existing_must, (str, bytes)):
            combined["_must"] = [*existing_must, partition_must]
        else:
            raise TypeError("`_must` filters must be a mapping or sequence of mappings")

        return combined

    @_mongo_operation("query")
    async def query(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T_model]:
        """Mongo-based implementation of Collection.query.

        The handling of null-values in sorting is different from memory-based implementation.
        In MongoDB, null values are treated as less than non-null values.
        """
        await self.ensure_collection()

        combined = self._inject_partition_filter(filter)
        mongo_filter = _build_mongo_filter(cast(FilterOptions, combined))

        collection = await self.ensure_collection()
        total = await collection.count_documents(mongo_filter, session=self._session)

        if limit == 0:
            return PaginatedResult[T_model](items=[], limit=0, offset=offset, total=total)

        cursor = collection.find(mongo_filter, session=self._session)

        sort_name, sort_order = resolve_sort_options(sort)
        if sort_name is not None:
            model_fields = getattr(self._item_type, "model_fields", {})
            if sort_name not in model_fields:
                raise ValueError(
                    f"Failed to sort items by '{sort_name}': field does not exist on {self._item_type.__name__}"
                )
            direction = 1 if sort_order == "asc" else -1
            cursor = cursor.sort(sort_name, direction)

        if offset > 0:
            cursor = cursor.skip(offset)
        if limit >= 0:
            cursor = cursor.limit(limit)

        items: List[T_model] = []
        item_type_has_id = "_id" in self._item_type.model_fields
        async for raw in cursor:
            # Remove _id from the raw document if the item type does not have it.
            if not item_type_has_id:
                raw.pop("_id", None)  # type: ignore
            # Convert Mongo document to Pydantic model
            items.append(self._item_type.model_validate(raw))  # type: ignore[arg-type]

        return PaginatedResult[T_model](items=items, limit=limit, offset=offset, total=total)

    @_mongo_operation("get")
    async def get(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
    ) -> Optional[T_model]:
        result = await self.query(filter=filter, sort=sort, limit=1, offset=0)
        return result.items[0] if result.items else None

    @_mongo_operation("insert")
    async def insert(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        collection = await self.ensure_collection()
        docs: List[Mapping[str, Any]] = []
        pk_conditions: List[Dict[str, Any]] = []
        seen_primary_keys: set[Tuple[Any, ...]] = set()
        for item in items:
            self._ensure_item_type(item)
            pk_filter = self._pk_filter(item)
            pk_values = tuple(pk_filter[pk] for pk in self._primary_keys)
            if pk_values in seen_primary_keys:
                raise ValueError(
                    f"Insert payload contains duplicate primary key(s): {self._render_pk_values(pk_values)}"
                )
            seen_primary_keys.add(pk_values)
            pk_conditions.append({pk: pk_filter[pk] for pk in self._primary_keys})

            doc = item.model_dump()
            doc["partition_id"] = self._partition_id
            docs.append(doc)

        if not docs:
            return

        if len(pk_conditions) == 1:
            existing_filter: Dict[str, Any] = {"partition_id": self._partition_id, **pk_conditions[0]}
        else:
            existing_filter = {"partition_id": self._partition_id, "$or": pk_conditions}

        with self._prometheus_tracker.track("insert__find_existing", self._database_name, self._collection_name):
            existing = await collection.find_one(existing_filter, session=self._session)
            if existing is not None:
                existing_values = tuple(existing.get(pk) for pk in self._primary_keys)
                raise ValueError(f"Item with primary key(s) {self._render_pk_values(existing_values)} already exists")

        with self._prometheus_tracker.track(
            "insert__insert_many", self._database_name, self._collection_name
        ) as tracker:
            try:
                await collection.insert_many(docs, session=self._session)
            except DuplicateKeyError as exc:
                # In case the DB enforces uniqueness via index, normalize to ValueError
                tracker.report_error(exc)
                raise ValueError("Duplicate key error while inserting items") from exc

    @_mongo_operation("update")
    async def update(self, items: Sequence[T_model], update_fields: Sequence[str] | None = None) -> List[T_model]:
        if not items:
            return []

        updated_items: List[T_model] = []
        collection = await self.ensure_collection()

        for item in items:
            self._ensure_item_type(item)
            pk_filter = self._pk_filter(item)
            doc = item.model_dump()
            doc["partition_id"] = self._partition_id

            updated_doc = None

            # Branch 1: Full Replace
            if update_fields is None:
                with self._prometheus_tracker.track(
                    "update__find_one_and_replace", self._database_name, self._collection_name
                ):
                    updated_doc = await collection.find_one_and_replace(
                        filter=pk_filter,
                        replacement=doc,
                        session=self._session,
                        return_document=ReturnDocument.AFTER,  # Returns the new version
                    )

            # Branch 2: Partial Update
            else:
                update_doc = {field: doc[field] for field in update_fields if field in doc}
                with self._prometheus_tracker.track(
                    "update__find_one_and_update", self._database_name, self._collection_name
                ):
                    updated_doc = await collection.find_one_and_update(
                        filter=pk_filter,
                        update={"$set": update_doc},
                        session=self._session,
                        return_document=ReturnDocument.AFTER,  # Returns the new version
                    )

            # Validation and Reconstruction
            if updated_doc is None:  # type: ignore
                raise ValueError(f"Item with primary key(s) {pk_filter} does not exist")

            # Re-instantiate the model from the raw MongoDB dictionary.
            new_item = self._item_type.model_validate(updated_doc)  # type: ignore[arg-type]
            updated_items.append(new_item)

        return updated_items

    @_mongo_operation("upsert")
    async def upsert(self, items: Sequence[T_model], update_fields: Sequence[str] | None = None) -> List[T_model]:
        if not items:
            return []

        upserted_items: List[T_model] = []
        collection = await self.ensure_collection()

        for item in items:
            self._ensure_item_type(item)
            pk_filter = self._pk_filter(item)

            insert_doc = item.model_dump()
            insert_doc["partition_id"] = self._partition_id

            # If update_fields is None, we update ALL fields (standard upsert behavior).
            # Otherwise, we only update specific fields, but insert the full doc if it's new.
            target_fields = update_fields if update_fields is not None else list(insert_doc.keys())

            # 1. $set: Fields that should be overwritten if the document exists
            update_subset = {field: insert_doc[field] for field in target_fields if field in insert_doc}

            # 2. $setOnInsert: Fields that are only set if we are creating a NEW document
            # (Everything in the model that isn't in the update_subset)
            set_on_insert = {k: v for k, v in insert_doc.items() if k not in update_subset}

            update_spec: Dict[str, Dict[str, Any]] = {}
            if set_on_insert:
                update_spec["$setOnInsert"] = set_on_insert
            if update_subset:
                update_spec["$set"] = update_subset

            with self._prometheus_tracker.track(
                "upsert__find_one_and_update", self._database_name, self._collection_name
            ):
                result_doc = await collection.find_one_and_update(
                    filter=pk_filter,
                    update=update_spec,
                    upsert=True,
                    session=self._session,
                    return_document=ReturnDocument.AFTER,
                )

            # Because upsert=True, result_doc is guaranteed to be not None
            new_item = self._item_type.model_validate(result_doc)  # type: ignore[arg-type]
            upserted_items.append(new_item)

        return upserted_items

    @_mongo_operation("delete")
    async def delete(self, items: Sequence[T_model]) -> None:
        if not items:
            return

        collection = await self.ensure_collection()
        for item in items:
            self._ensure_item_type(item)
            pk_filter = self._pk_filter(item)
            with self._prometheus_tracker.track("delete__delete_one", self._database_name, self._collection_name):
                result = await collection.delete_one(pk_filter, session=self._session)
            if result.deleted_count == 0:
                raise ValueError(f"Item with primary key(s) {pk_filter} does not exist")


class MongoBasedQueue(Queue[T_generic], Generic[T_generic]):
    """Mongo-based implementation of Queue backed by a MongoDB collection.

    Items are stored append-only; dequeue marks items as consumed instead of deleting them.
    """

    def __init__(
        self,
        client_pool: MongoClientPool[Mapping[str, Any]] | AsyncMongoClient[Mapping[str, Any]],
        database_name: str,
        collection_name: str,
        partition_id: str,
        item_type: Type[T_generic],
        prometheus_tracker: MongoOperationPrometheusTracker | None = None,
    ) -> None:
        """
        Args:
            client_pool: The pool of MongoDB clients.
            database_name: The name of the database.
            collection_name: The name of the collection backing the queue.
            partition_id: Partition identifier; allows multiple logical queues in one collection.
            item_type: The Python type of queue items (primitive or BaseModel subclass).
        """
        if isinstance(client_pool, AsyncMongoClient):
            self._client_pool = MongoClientPool(client_pool)
        else:
            self._client_pool = client_pool
        self._database_name = database_name
        self._collection_name = collection_name
        self._partition_id = partition_id
        self._item_type = item_type
        self._adapter: TypeAdapter[T_generic] = TypeAdapter(item_type)
        self._collection_created = False

        self._session: Optional[AsyncClientSession] = None
        self._prometheus_tracker = (
            prometheus_tracker if prometheus_tracker is not None else MongoOperationPrometheusTracker(enabled=False)
        )

    def item_type(self) -> Type[T_generic]:
        return self._item_type

    @_mongo_operation("ensure_collection")
    async def ensure_collection(self) -> AsyncCollection[Mapping[str, Any]]:
        """Ensure the backing collection exists.

        If it already exists, it returns the existing collection.
        """
        if not self._collection_created:
            client = await self._client_pool.get_client()
            self._collection_created = await _ensure_collection(
                client[self._database_name], self._collection_name, primary_keys=["consumed", "_id"]
            )
        return await self._client_pool.get_collection(self._database_name, self._collection_name)

    def with_session(self, session: AsyncClientSession) -> MongoBasedQueue[T_generic]:
        queue = MongoBasedQueue(
            client_pool=self._client_pool,
            database_name=self._database_name,
            collection_name=self._collection_name,
            partition_id=self._partition_id,
            item_type=self._item_type,
            prometheus_tracker=self._prometheus_tracker,
        )
        queue._collection_created = self._collection_created
        queue._session = session
        return queue

    @_mongo_operation("has")
    async def has(self, item: T_generic) -> bool:
        collection = await self.ensure_collection()
        encoded = self._adapter.dump_python(item, mode="python")
        doc = await collection.find_one(
            {
                "partition_id": self._partition_id,
                "consumed": False,
                "value": encoded,
            },
            session=self._session,
        )
        return doc is not None

    @_mongo_operation("enqueue")
    async def enqueue(self, items: Sequence[T_generic]) -> Sequence[T_generic]:
        if not items:
            return []

        collection = await self.ensure_collection()
        docs: List[Mapping[str, Any]] = []
        for item in items:
            if not isinstance(item, self._item_type):
                raise TypeError(f"Expected item of type {self._item_type.__name__}, got {type(item).__name__}")
            docs.append(
                {
                    "partition_id": self._partition_id,
                    "value": self._adapter.dump_python(item, mode="python"),
                    "consumed": False,
                    "created_at": datetime.now(),
                }
            )

        with self._prometheus_tracker.track("enqueue__insert_many", self._database_name, self._collection_name):
            await collection.insert_many(docs, session=self._session)
        return list(items)

    @_mongo_operation("dequeue")
    async def dequeue(self, limit: int = 1) -> Sequence[T_generic]:
        if limit <= 0:
            return []

        collection = await self.ensure_collection()
        results: list[T_generic] = []

        # Atomic claim loop using find_one_and_update
        for _ in range(limit):
            with self._prometheus_tracker.track(
                "dequeue__find_one_and_update", self._database_name, self._collection_name
            ):
                doc = await collection.find_one_and_update(
                    {
                        "partition_id": self._partition_id,
                        "consumed": False,
                    },
                    {"$set": {"consumed": True}},
                    sort=[("_id", 1)],  # FIFO using insertion order
                    return_document=True,
                    session=self._session,
                )
            if doc is None:  # type: ignore
                # No more items to dequeue
                break

            raw_value = doc["value"]
            item = self._adapter.validate_python(raw_value)
            results.append(item)

        return results

    @_mongo_operation("peek")
    async def peek(self, limit: int = 1) -> Sequence[T_generic]:
        if limit <= 0:
            return []

        collection = await self.ensure_collection()
        with self._prometheus_tracker.track("peek__find", self._database_name, self._collection_name):
            cursor = (
                collection.find(
                    {
                        "partition_id": self._partition_id,
                        "consumed": False,
                    },
                    session=self._session,
                )
                .sort("_id", 1)
                .limit(limit)
            )

        items: list[T_generic] = []
        async for doc in cursor:
            raw_value = doc["value"]
            items.append(self._adapter.validate_python(raw_value))

        return items

    @_mongo_operation("size")
    async def size(self) -> int:
        collection = await self.ensure_collection()
        return await collection.count_documents(
            {
                "partition_id": self._partition_id,
                "consumed": False,
            },
            session=self._session,
        )


class MongoBasedKeyValue(KeyValue[K, V], Generic[K, V]):
    """Mongo-based implementation of KeyValue."""

    def __init__(
        self,
        client_pool: MongoClientPool[Mapping[str, Any]] | AsyncMongoClient[Mapping[str, Any]],
        database_name: str,
        collection_name: str,
        partition_id: str,
        key_type: Type[K],
        value_type: Type[V],
        prometheus_tracker: MongoOperationPrometheusTracker | None = None,
    ) -> None:
        """
        Args:
            client_pool: The pool of MongoDB clients.
            database_name: The name of the database.
            collection_name: The name of the collection backing the key-value store.
            partition_id: Partition identifier; allows multiple logical maps in one collection.
            key_type: The Python type of keys (primitive or BaseModel).
            value_type: The Python type of values (primitive or BaseModel).
        """
        if isinstance(client_pool, AsyncMongoClient):
            self._client_pool = MongoClientPool(client_pool)
        else:
            self._client_pool = client_pool
        self._database_name = database_name
        self._collection_name = collection_name
        self._partition_id = partition_id
        self._key_type = key_type
        self._value_type = value_type
        self._key_adapter: TypeAdapter[K] = TypeAdapter(key_type)
        self._value_adapter: TypeAdapter[V] = TypeAdapter(value_type)
        self._collection_created = False

        self._session: Optional[AsyncClientSession] = None
        self._prometheus_tracker = (
            prometheus_tracker if prometheus_tracker is not None else MongoOperationPrometheusTracker(enabled=False)
        )

    @_mongo_operation("ensure_collection")
    async def ensure_collection(self, *, create_indexes: bool = True) -> AsyncCollection[Mapping[str, Any]]:
        """Ensure the backing collection exists (and optionally its indexes)."""
        if not self._collection_created:
            client = await self._client_pool.get_client()
            self._collection_created = await _ensure_collection(
                client[self._database_name], self._collection_name, primary_keys=["key"]
            )
        return await self._client_pool.get_collection(self._database_name, self._collection_name)

    def with_session(self, session: AsyncClientSession) -> MongoBasedKeyValue[K, V]:
        key_value = MongoBasedKeyValue(
            client_pool=self._client_pool,
            database_name=self._database_name,
            collection_name=self._collection_name,
            partition_id=self._partition_id,
            key_type=self._key_type,
            value_type=self._value_type,
            prometheus_tracker=self._prometheus_tracker,
        )
        key_value._collection_created = self._collection_created
        key_value._session = session

        return key_value

    @_mongo_operation("has")
    async def has(self, key: K) -> bool:
        collection = await self.ensure_collection()
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await collection.find_one(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            },
            session=self._session,
        )
        return doc is not None

    @_mongo_operation("get")
    async def get(self, key: K, default: V | None = None) -> V | None:
        collection = await self.ensure_collection()
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await collection.find_one(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            },
            session=self._session,
        )
        if doc is None:
            return default

        raw_value = doc["value"]
        return self._value_adapter.validate_python(raw_value)

    @_mongo_operation("set")
    async def set(self, key: K, value: V) -> None:
        collection = await self.ensure_collection()
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        encoded_value = self._value_adapter.dump_python(value, mode="python")
        with self._prometheus_tracker.track(
            "upsert__replace_one", self._database_name, self._collection_name
        ) as tracer:
            try:
                await collection.replace_one(
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
                    session=self._session,
                )
            except DuplicateKeyError as exc:
                # Very unlikely with replace_one+upsert, but normalize anyway.
                tracer.report_error(exc)
                raise ValueError("Duplicate key error while setting key-value item") from exc

    @_mongo_operation("pop")
    async def pop(self, key: K, default: V | None = None) -> V | None:
        collection = await self.ensure_collection()
        encoded_key = self._key_adapter.dump_python(key, mode="python")
        doc = await collection.find_one_and_delete(
            {
                "partition_id": self._partition_id,
                "key": encoded_key,
            },
            session=self._session,
        )
        if doc is None:  # type: ignore
            return default

        raw_value = doc["value"]
        return self._value_adapter.validate_python(raw_value)

    @_mongo_operation("size")
    async def size(self) -> int:
        collection = await self.ensure_collection()
        return await collection.count_documents(
            {
                "partition_id": self._partition_id,
            },
            session=self._session,
        )


class MongoLightningCollections(LightningCollections):
    """Mongo implementation of LightningCollections using MongoDB collections.

    Serves as the storage base for [`MongoLightningStore`][agentlightning.store.MongoLightningStore].
    """

    def __init__(
        self,
        client_pool: MongoClientPool[Mapping[str, Any]],
        database_name: str,
        partition_id: str,
        rollouts: Optional[MongoBasedCollection[Rollout]] = None,
        attempts: Optional[MongoBasedCollection[Attempt]] = None,
        spans: Optional[MongoBasedCollection[Span]] = None,
        resources: Optional[MongoBasedCollection[ResourcesUpdate]] = None,
        workers: Optional[MongoBasedCollection[Worker]] = None,
        rollout_queue: Optional[MongoBasedQueue[str]] = None,
        span_sequence_ids: Optional[MongoBasedKeyValue[str, int]] = None,
        prometheus_tracker: MongoOperationPrometheusTracker | None = None,
    ):
        self._client_pool = client_pool
        self._database_name = database_name
        self._collection_name = "collections"  # Special collection name for tracking transactions
        self._partition_id = partition_id
        self._prometheus_tracker = (
            prometheus_tracker if prometheus_tracker is not None else MongoOperationPrometheusTracker(enabled=False)
        )
        self._collection_ensured = False
        self._rollouts = (
            rollouts
            if rollouts is not None
            else MongoBasedCollection(
                self._client_pool,
                self._database_name,
                "rollouts",
                self._partition_id,
                ["rollout_id"],
                Rollout,
                [["status"]],
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._attempts = (
            attempts
            if attempts is not None
            else MongoBasedCollection(
                self._client_pool,
                self._database_name,
                "attempts",
                self._partition_id,
                ["rollout_id", "attempt_id"],
                Attempt,
                [["status"], ["sequence_id"]],
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._spans = (
            spans
            if spans is not None
            else MongoBasedCollection(
                self._client_pool,
                self._database_name,
                "spans",
                self._partition_id,
                ["rollout_id", "attempt_id", "span_id"],
                Span,
                [["sequence_id"]],
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._resources = (
            resources
            if resources is not None
            else MongoBasedCollection(
                self._client_pool,
                self._database_name,
                "resources",
                self._partition_id,
                ["resources_id"],
                ResourcesUpdate,
                ["update_time"],
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._workers = (
            workers
            if workers is not None
            else MongoBasedCollection(
                self._client_pool,
                self._database_name,
                "workers",
                self._partition_id,
                ["worker_id"],
                Worker,
                ["status"],
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._rollout_queue = (
            rollout_queue
            if rollout_queue is not None
            else MongoBasedQueue(
                self._client_pool,
                self._database_name,
                "rollout_queue",
                self._partition_id,
                str,
                prometheus_tracker=self._prometheus_tracker,
            )
        )
        self._span_sequence_ids = (
            span_sequence_ids
            if span_sequence_ids is not None
            else MongoBasedKeyValue(
                self._client_pool,
                self._database_name,
                "span_sequence_ids",
                self._partition_id,
                str,
                int,
                prometheus_tracker=self._prometheus_tracker,
            )
        )

    def with_session(self, session: AsyncClientSession) -> Self:
        instance = self.__class__(
            client_pool=self._client_pool,
            database_name=self._database_name,
            partition_id=self._partition_id,
            rollouts=self._rollouts.with_session(session),
            attempts=self._attempts.with_session(session),
            spans=self._spans.with_session(session),
            resources=self._resources.with_session(session),
            workers=self._workers.with_session(session),
            rollout_queue=self._rollout_queue.with_session(session),
            span_sequence_ids=self._span_sequence_ids.with_session(session),
            prometheus_tracker=self._prometheus_tracker,
        )
        instance._collection_ensured = self._collection_ensured
        return instance

    @property
    def rollouts(self) -> MongoBasedCollection[Rollout]:
        return self._rollouts

    @property
    def attempts(self) -> MongoBasedCollection[Attempt]:
        return self._attempts

    @property
    def spans(self) -> MongoBasedCollection[Span]:
        return self._spans

    @property
    def resources(self) -> MongoBasedCollection[ResourcesUpdate]:
        return self._resources

    @property
    def workers(self) -> MongoBasedCollection[Worker]:
        return self._workers

    @property
    def rollout_queue(self) -> MongoBasedQueue[str]:
        return self._rollout_queue

    @property
    def span_sequence_ids(self) -> MongoBasedKeyValue[str, int]:
        return self._span_sequence_ids

    @_mongo_operation("ensure_collections")
    async def _ensure_collections(self) -> None:
        """Ensure all collections exist."""
        if self._collection_ensured:
            return
        await self._rollouts.ensure_collection()
        await self._attempts.ensure_collection()
        await self._spans.ensure_collection()
        await self._resources.ensure_collection()
        await self._workers.ensure_collection()
        await self._rollout_queue.ensure_collection()
        await self._span_sequence_ids.ensure_collection()
        self._collection_ensured = True

    @asynccontextmanager
    async def atomic(
        self, mode: AtomicMode = "rw", snapshot: bool = False, commit: bool = False, *args: Any, **kwargs: Any
    ):
        """Perform a atomic operation on the collections."""
        if commit:
            raise ValueError("Commit should be used with execute() instead.")
        with self._prometheus_tracker.track("atomic", self._database_name, self._collection_name):
            # First step: ensure all collections exist before going into the atomic block
            if not self._collection_ensured:
                await self._ensure_collections()
            # Execute directly without commit
            yield self

    @_mongo_operation("execute")
    async def execute(
        self,
        callback: Callable[[Self], Awaitable[T_generic]],
        *,
        mode: AtomicMode = "rw",
        snapshot: bool = False,
        commit: bool = False,
        **kwargs: Any,
    ) -> T_generic:
        """Execute the given callback within an atomic operation, and with retries on transient errors."""
        if not self._collection_ensured:
            await self._ensure_collections()
        client = await self._client_pool.get_client()

        # If commit is not turned on, just execute the callback directly.
        if not commit:
            return await callback(self)

        # If snapshot is enabled, use snapshot read concern.
        read_concern = ReadConcern("snapshot") if snapshot else ReadConcern("local")
        # If mode is "r", write_concern is not needed.
        write_concern = WriteConcern("majority") if mode != "r" else None

        async with client.start_session() as session:
            collections = self.with_session(session)
            with self._prometheus_tracker.track(
                "execute__transaction", self._database_name, self._collection_name
            ) as tracker:
                try:
                    return await self._with_transaction(
                        session, collections, callback, read_concern, write_concern, tracker
                    )
                except (ConnectionFailure, OperationFailure) as exc:
                    # Un-retryable errors.
                    tracker.report_error(exc)
                    raise RuntimeError("Transaction failed with connection or operation error") from exc

    async def _with_transaction(
        self,
        session: AsyncClientSession,
        collections: Self,
        callback: Callable[[Self], Awaitable[T_generic]],
        read_concern: ReadConcern,
        write_concern: Optional[WriteConcern],
        transaction_tracker: _MongoOperationContext | _DummyOperationContext,
    ) -> T_generic:
        # This will start a transaction, run transaction callback, and commit.
        # It will also transparently retry on some transient errors.
        # Expanded implementation of with_transaction from client_session
        num_attempts = 0
        read_preference = ReadPreference.PRIMARY
        transaction_retry_time_limit = 120
        start_time = time.monotonic()

        def _within_time_limit() -> bool:
            return time.monotonic() - start_time < transaction_retry_time_limit

        async def _jitter_before_retry() -> None:
            with self._prometheus_tracker.track("execute__jitter", self._database_name, self._collection_name):
                await asyncio.sleep(random.uniform(0, 0.05))

        while True:
            await session.start_transaction(read_concern, write_concern, read_preference)

            with self._prometheus_tracker.track(
                "execute__callback", self._database_name, self._collection_name
            ) as callback_tracker:
                try:
                    num_attempts += 1
                    transaction_tracker.report_num_attempts(num_attempts)
                    # The _session is always the same within one transaction,
                    # so we can use the same collections object.
                    ret = await callback(collections)
                # Catch KeyboardInterrupt, CancelledError, etc. and cleanup.
                except BaseException as exc:
                    callback_tracker.report_error(exc)
                    if session.in_transaction:
                        await session.abort_transaction()
                    if (
                        isinstance(exc, PyMongoError)
                        and exc.has_error_label("TransientTransactionError")
                        and _within_time_limit()
                    ):
                        # Retry the entire transaction.
                        await _jitter_before_retry()
                        continue
                    raise

            if not session.in_transaction:
                # Assume callback intentionally ended the transaction.
                return ret

            commit_num_attempts = 0

            # Tracks the commit operation.
            with self._prometheus_tracker.track(
                "execute__commit", self._database_name, self._collection_name
            ) as commit_tracker:
                # Loop until the commit succeeds or we hit the time limit.
                while True:
                    # Tracks the commit attempt.
                    with self._prometheus_tracker.track(
                        "execute__commit_attempt", self._database_name, self._collection_name
                    ) as commit_attempt_tracker:
                        try:
                            commit_num_attempts += 1
                            commit_tracker.report_num_attempts(commit_num_attempts)
                            await session.commit_transaction()
                        except PyMongoError as exc:
                            commit_attempt_tracker.report_error(exc)
                            if (
                                exc.has_error_label("UnknownTransactionCommitResult")
                                and _within_time_limit()
                                and not (isinstance(exc, OperationFailure) and exc.code == 50)  # max_time_expired_error
                            ):
                                # Retry the commit.
                                await _jitter_before_retry()
                                continue

                            if exc.has_error_label("TransientTransactionError") and _within_time_limit():
                                # Retry the entire transaction.
                                await _jitter_before_retry()
                                break
                            raise

                        # Commit succeeded.
                        return ret

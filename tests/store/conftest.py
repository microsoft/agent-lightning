# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
import time
from itertools import count
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Sequence
from unittest.mock import Mock
from uuid import uuid4

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field
from pytest import FixtureRequest

from agentlightning.store.base import LightningStore
from agentlightning.store.collection import DequeBasedQueue, DictBasedKeyValue, KeyValue, ListBasedCollection, Queue
from agentlightning.store.collection.base import Collection
from agentlightning.store.memory import InMemoryLightningStore

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient
    from pymongo.asynchronous.database import AsyncDatabase

__all__ = [
    "inmemory_store",
    "mock_readable_span",
    "sample_items",
    "sample_collection",
    "SampleItem",
    "QueueItem",
    "deque_queue",
    "dict_key_value",
    "dict_key_value_data",
    "temporary_mongo_database",
]


mongo_uri = os.getenv("AGL_TEST_MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")


@pytest.fixture
def inmemory_store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


@pytest_asyncio.fixture
async def mongo_store(temporary_mongo_database: AsyncDatabase[Any]):
    """Fixture for MongoDB store implementation."""
    from agentlightning.store.mongo import MongoLightningStore

    db = MongoLightningStore(client=temporary_mongo_database.client, database_name=temporary_mongo_database.name)
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture(
    params=[
        "inmemory_store",
        pytest.param("mongo_store", marks=pytest.mark.mongo),
    ]
)
def store_fixture(request: FixtureRequest) -> AsyncGenerator[LightningStore, None]:
    """Parameterized fixture that provides different store implementations for testing."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_readable_span() -> ReadableSpan:
    """Create a mock ReadableSpan for testing."""
    span = Mock()
    span.name = "test_span"
    context_counter = count(1)

    def _make_context() -> Mock:
        """Generate a distinct span context each time it is requested."""
        index = next(context_counter)
        context = Mock()
        context.trace_id = 111111
        context.span_id = 222222 + index
        context.is_remote = False
        context.trace_state = {}
        return context

    # Mock context
    span.get_span_context = Mock(side_effect=_make_context)

    # Mock other attributes
    span.parent = None
    # Fix mock status to return proper string values
    status_code_mock = Mock()
    status_code_mock.name = "OK"
    span.status = Mock(status_code=status_code_mock, description=None)
    span.attributes = {"test": "value"}
    span.events = []
    span.links = []
    span.start_time = time.time_ns()
    span.end_time = time.time_ns() + 1000000
    span.resource = Mock(attributes={}, schema_url="")

    return span


class SampleItem(BaseModel):
    partition: str
    index: int
    name: str
    status: str
    tags: List[str] = Field(default_factory=list)
    score: float | None = None
    rank: int | None = None
    updated_time: float | None = None
    payload: Dict[str, int] = Field(default_factory=dict)
    metadata: str | None = None


class QueueItem(BaseModel):
    idx: int


@pytest_asyncio.fixture
async def mongo_client():
    from pymongo import AsyncMongoClient

    client = AsyncMongoClient[Any](mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
    except Exception as exc:  # depends on external service
        await client.close()
        raise RuntimeError(f"MongoDB not available: {exc}")

    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def temporary_mongo_database(mongo_client: AsyncMongoClient[Any]):
    """Yield a temporary MongoDB database for integration tests."""
    db_name = f"agentlightning-test-{uuid4().hex}"
    db = mongo_client[db_name]  # type: ignore
    try:
        yield db
    finally:
        await mongo_client.drop_database(db_name)


### Collection fixtures ###


@pytest.fixture()
def sample_items() -> List[SampleItem]:
    return [
        SampleItem(
            partition="alpha",
            index=1,
            name="urgent-phase-one",
            status="new",
            tags=["core", "urgent"],
            score=10.5,
            rank=3,
            updated_time=12.0,
            payload={"priority": 10},
            metadata="alpha-start",
        ),
        SampleItem(
            partition="alpha",
            index=2,
            name="phase-two",
            status="running",
            tags=["core"],
            score=5.0,
            rank=2,
            updated_time=None,
            payload={"priority": 5},
            metadata=None,
        ),
        SampleItem(
            partition="alpha",
            index=3,
            name="delayed-phase",
            status="blocked",
            tags=["delayed"],
            score=None,
            rank=5,
            updated_time=15.1,
            payload={"priority": 8},
            metadata="delayed-phase",
        ),
        SampleItem(
            partition="beta",
            index=1,
            name="beta-critical",
            status="new",
            tags=["beta", "urgent"],
            score=8.0,
            rank=1,
            updated_time=7.0,
            payload={"priority": 7},
            metadata="beta critical",
        ),
        SampleItem(
            partition="beta",
            index=2,
            name="beta optional",
            status="done",
            tags=["beta"],
            score=3.0,
            rank=None,
            updated_time=2.0,
            payload={"priority": 1},
            metadata="optional path",
        ),
        SampleItem(
            partition="gamma",
            index=1,
            name="gamma-phase",
            status="running",
            tags=[],
            score=9.5,
            rank=4,
            updated_time=None,
            payload={"priority": 9},
            metadata="gamma-phase data",
        ),
        SampleItem(
            partition="gamma",
            index=2,
            name="gamma-late",
            status="done",
            tags=["late", "core"],
            score=1.0,
            rank=6,
            updated_time=20.0,
            payload={"priority": 2},
            metadata="gamma late entry",
        ),
        SampleItem(
            partition="delta",
            index=1,
            name="delta misc",
            status="archived",
            tags=["misc"],
            score=4.2,
            rank=7,
            updated_time=11.0,
            payload={"priority": 3},
            metadata="delta misc block",
        ),
    ]


### Generic collection fixtures ###


@pytest.fixture()
def sample_collection_memory(sample_items: Sequence[SampleItem]) -> ListBasedCollection[SampleItem]:
    collection: Collection[SampleItem] = ListBasedCollection(list(sample_items), SampleItem, ("partition", "index"))
    setattr(collection, "_test_backend", "memory")
    return collection


@pytest_asyncio.fixture
async def sample_collection_mongo(temporary_mongo_database: AsyncDatabase[Any], sample_items: Sequence[SampleItem]):
    from agentlightning.store.collection.mongo import MongoBasedCollection, MongoClientPool

    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        collection = MongoBasedCollection(
            client_pool,
            temporary_mongo_database.name,
            "sample-items",
            "partition-123",
            ["partition", "index"],
            SampleItem,
        )
        await collection.insert(sample_items)
        setattr(collection, "_test_backend", "mongo")
        yield collection


@pytest.fixture(
    params=[
        "memory",
        pytest.param("mongo", marks=pytest.mark.mongo),
    ]
)
def sample_collection(request: pytest.FixtureRequest):
    backend = request.param
    return request.getfixturevalue("sample_collection_" + backend)


### Generic queue fixtures ###


@pytest.fixture
def deque_queue_memory() -> DequeBasedQueue[QueueItem]:
    return DequeBasedQueue(QueueItem, [QueueItem(idx=i) for i in range(3)])


@pytest_asyncio.fixture
async def deque_queue_mongo(temporary_mongo_database: AsyncDatabase[Any]):
    from agentlightning.store.collection.mongo import MongoBasedQueue, MongoClientPool

    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        queue = MongoBasedQueue[QueueItem](
            client_pool,
            temporary_mongo_database.name,
            "queue-items",
            "partition-1",
            QueueItem,
        )
        await queue.enqueue([QueueItem(idx=i) for i in range(3)])
        yield queue


@pytest.fixture(
    params=[
        "memory",
        pytest.param("mongo", marks=pytest.mark.mongo),
    ]
)
def deque_queue(request: pytest.FixtureRequest) -> AsyncGenerator[Queue[QueueItem], None]:
    backend = request.param
    return request.getfixturevalue("deque_queue_" + backend)


### Generic key-value fixtures ###


@pytest.fixture()
def dict_key_value_data() -> Dict[str, int]:
    return {"alpha": 1, "beta": 2}


@pytest.fixture()
def dict_key_value_memory(dict_key_value_data: Dict[str, int]) -> DictBasedKeyValue[str, int]:
    return DictBasedKeyValue(dict_key_value_data)


@pytest_asyncio.fixture
async def dict_key_value_mongo(temporary_mongo_database: AsyncDatabase[Any], dict_key_value_data: Dict[str, int]):
    from agentlightning.store.collection.mongo import MongoBasedKeyValue, MongoClientPool

    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        key_value = MongoBasedKeyValue[str, int](
            client_pool,
            temporary_mongo_database.name,
            "key-value-items",
            "partition-1",
            str,
            int,
        )
        for key, value in dict_key_value_data.items():
            await key_value.set(key, value)
        yield key_value


@pytest.fixture(
    params=[
        "memory",
        pytest.param("mongo", marks=pytest.mark.mongo),
    ]
)
def dict_key_value(request: pytest.FixtureRequest) -> AsyncGenerator[KeyValue[str, int], None]:
    backend = request.param
    return request.getfixturevalue("dict_key_value_" + backend)

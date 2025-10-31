# Copyright (c) Microsoft. All rights reserved.

import time
from unittest.mock import Mock

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store import InMemoryLightningStore, DatabaseLightningStore

__all__ = [
    "inmemory_store",
    "db_store",
    "mock_readable_span",
]


@pytest.fixture
def inmemory_store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


import os
import uuid
import typing

@pytest_asyncio.fixture
async def db_store() -> typing.AsyncGenerator[DatabaseLightningStore, None]:
    """Create a DatabaseLightningStore using a SQLite file for testing."""
    tmp_path = ".pytest_cache"
    # Ensure the directory exists and create a random file in it
    os.makedirs(tmp_path, exist_ok=True)
    db_path = os.path.join(tmp_path, f"test_db_{uuid.uuid4().hex}.sqlite3")
    database_url = f"sqlite+aiosqlite:///{db_path}"
    store = DatabaseLightningStore(database_url=database_url)
    await store.start()
    try:
        yield store
    finally:
        await store.stop()
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.fixture
def mock_readable_span() -> ReadableSpan:
    """Create a mock ReadableSpan for testing."""
    span = Mock()
    span.name = "test_span"

    # Mock context
    context = Mock()
    context.trace_id = 111111
    context.span_id = 222222
    context.is_remote = False
    context.trace_state = {}  # Make it an empty dict instead of Mock
    span.get_span_context = Mock(return_value=context)

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

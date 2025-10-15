# Copyright (c) Microsoft. All rights reserved.

import time
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.store.sqlite import SqliteLightningStore

__all__ = [
    "inmemory_store",
    "sqlite_store",
    "mock_readable_span",
]


@pytest.fixture
def inmemory_store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


@pytest.fixture
def sqlite_store(tmp_path: Path) -> Generator[SqliteLightningStore, None, None]:
    """Create a SqliteLightningStore backed by a temporary database file."""
    store = SqliteLightningStore(str(tmp_path / "lightning.db"))
    yield store
    store.close()


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
    context.trace_state = {}
    span.get_span_context = Mock(return_value=context)

    # Mock other attributes
    span.parent = None
    status_code_mock = Mock()
    status_code_mock.name = "OK"
    span.status = Mock(status_code=status_code_mock, description=None)
    span.attributes = {"test": "value"}
    span.events = []
    span.links = []
    span.start_time = time.time_ns()
    span.end_time = time.time_ns() + 1_000_000
    span.resource = Mock(attributes={}, schema_url="")

    return span

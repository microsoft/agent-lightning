# Copyright (c) Microsoft. All rights reserved.

import time
from itertools import count
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from pytest import FixtureRequest

from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore

__all__ = [
    "inmemory_store",
    "mock_readable_span",
]


@pytest.fixture
def inmemory_store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


@pytest.fixture
def sql_store():
    """Placeholder fixture for SQL store implementation. Returns None until SQL store is ready."""
    return None


# Uncomment this when sql store is ready
# @pytest.fixture(params=["inmemory_store", "sql_store"])
@pytest.fixture(params=["inmemory_store"])
def store_fixture(request: FixtureRequest) -> LightningStore:
    """Parameterized fixture that provides different store implementations for testing.
    Currently supports InMemoryLightningStore, with SQL store support planned.
    """
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

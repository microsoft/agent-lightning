# Copyright (c) Microsoft. All rights reserved.

import time
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan

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

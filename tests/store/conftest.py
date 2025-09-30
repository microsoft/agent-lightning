# Copyright (c) Microsoft. All rights reserved.

"""
Comprehensive tests for InMemoryLightningStore.

Test categories:
- Core CRUD operations
- Queue operations (FIFO behavior)
- Resource versioning
- Span tracking and sequencing
- Rollout lifecycle and status transitions
- Concurrent access patterns
- Error handling and edge cases
"""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest

# from agentlightning.store.base import LightningStoreWatchDog  # TODO: Re-enable when watchdog is implemented
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import Span
from agentlightning.types import LLM, Attempt, AttemptStatus, PromptTemplate, ResourcesUpdate, RolloutStatus, RolloutV2

__all__ = [
    "store",
    # "store_with_watchdog",  # TODO: Re-enable when watchdog is implemented
    "mock_readable_span",
]


@pytest.fixture
def store() -> InMemoryLightningStore:
    """Create a fresh InMemoryLightningStore instance."""
    return InMemoryLightningStore()


# TODO: Re-enable when watchdog is implemented
# @pytest.fixture
# def store_with_watchdog() -> InMemoryLightningStore:
#     """Create a store with watchdog configured."""
#     watchdog = LightningStoreWatchDog(
#         timeout_seconds=5.0,
#         unresponsive_seconds=2.0,
#         max_attempts=3,
#         retry_condition=["unresponsive", "timeout"],
#     )
#     return InMemoryLightningStore(watchdog=watchdog)


@pytest.fixture
def mock_readable_span() -> Mock:
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

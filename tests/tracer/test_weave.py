# Copyright (c) Microsoft. All rights reserved.

import asyncio
from unittest.mock import patch

import pytest
import weave
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.base import LightningStore
from agentlightning.tracer.weave import WeaveTracer
from agentlightning.types import Span


class MockLightningStore(LightningStore):
    """A minimal stub-only LightningStore, only implements methods likely used in tests."""

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        if sequence_id is None:
            sequence_id = 0

        span = Span.from_opentelemetry(
            readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
        )

        return span


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


def _mock_weave_init():
    with patch("weave.init"):
        weave.init(project_name="agentlightning.tracer.weave_test")


@pytest.mark.skip(reason="Skipping this test temporarily")
def test_weave_trace_workable_store_valid():
    asyncio.run(_test_weave_trace_workable_store_valid_async())


async def _test_weave_trace_workable_store_valid_async():
    _mock_weave_init()

    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    store = MockLightningStore()
    calls = tracer.get_last_trace()
    print(len(calls))
    try:
        # Case where store, rollout_id, and attempt_id are all non-none.
        tracer.trace_run(_func_without_exception)
        calls = tracer.get_last_trace()
        assert len(calls) > 0

        # Case where store, rollout_id, and attempt_id are all non-none.
        async with tracer.trace_context(
            name="weave_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        calls = tracer.get_last_trace()
        assert len(calls) > 0
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()

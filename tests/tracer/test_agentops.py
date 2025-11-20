# Copyright (c) Microsoft. All rights reserved.

import multiprocessing
import sys
from typing import Any, Callable, Coroutine, Optional, Union

import agentops
import pytest
from agentops.sdk.core import TraceContext
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace.status import StatusCode

from agentlightning.store.base import LightningStore, LightningStoreCapabilities
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Span


class MockLightningStore(LightningStore):
    """A minimal stub-only LightningStore, only implements methods likely used in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.otlp_traces = False

    def enable_otlp_traces(self) -> None:
        self.otlp_traces = True

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

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            async_safe=False,
            thread_safe=False,
            zero_copy=False,
            otlp_traces=self.otlp_traces,
        )

    def otlp_traces_endpoint(self) -> str:
        return "dump://"


def _func_with_exception():
    """Function that always raises an exception to test error tracing."""
    raise ValueError("This is a test exception")


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


@pytest.mark.parametrize("with_exception", [True, False])
def test_trace_error_status_from_instance(with_exception: bool):
    """
    Test that AgentOpsTracer correctly sets trace end state based on execution result.

    This test replaces `agentops.end_trace` with a custom function to capture
    the `end_state` passed in. It verifies that traces ending after a raised
    exception have `StatusCode.ERROR`, while normal runs have `StatusCode.OK`.
    """

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_test_trace_error_status_from_instance_imp, args=(with_exception,))
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."

    if with_exception:
        assert proc.exitcode != 0, (
            f"Child process for test_trace_error_status_from_instance with exception exited with exit code {proc.exitcode}. "
            "Check child traceback in test output."
        )
    else:
        assert proc.exitcode == 0, (
            f"Child process for test_trace_error_status_from_instance without exception failed with exit code {proc.exitcode}. "
            "Check child traceback in test output."
        )


def _test_trace_error_status_from_instance_imp(with_exception: bool):
    captured_state = {}
    old_end_trace = agentops.end_trace

    def custom_end_trace(
        trace_context: Optional[TraceContext] = None, end_state: Union[Any, StatusCode, str] = None
    ) -> None:
        captured_state["state"] = end_state
        return old_end_trace(trace_context, end_state=end_state)

    agentops.end_trace = custom_end_trace

    tracer = AgentOpsTracer()
    tracer.init()
    tracer.init_worker(0)

    try:
        if with_exception:
            tracer.trace_run(_func_with_exception)
            if captured_state["state"] != StatusCode.ERROR:
                sys.exit(-1)
        else:
            tracer.trace_run(_func_without_exception)
            if captured_state["state"] != StatusCode.OK:
                sys.exit(-1)
    finally:
        agentops.end_trace = old_end_trace
        tracer.teardown_worker(0)
        tracer.teardown()


def test_agentops_trace_with_store_or_not():
    """
    The purpose of this test is to verify whether the following two scenarios both work correctly:

    1. Using AgentOpsTracer to trace a function without providing a store, rollout_id, or attempt_id.
    2. Using AgentOpsTracer to trace a function with providing a store which disabled native otlp exporter, rollout_id, and attempt_id.
    3. Using AgentOpsTracer to trace a function with providing a store which enabled native otlp exporter, rollout_id, and attempt_id.
    """

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_async, args=(_test_agentops_trace_with_store_or_not_imp,))
    proc.start()
    proc.join(30.0)  # On GPU server, the time is around 10 seconds.

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()

        assert False, "Child process hung. Check test output for details."

    assert proc.exitcode == 0, (
        f"Child process for test_trace_error_status_from_instance failed with exit code {proc.exitcode}. "
        "Check child traceback in test output."
    )


def _run_async(coro: Callable[[], Coroutine[Any, Any, Any]]) -> None:
    """Small wrapper: run async function inside multiprocessing target."""
    import asyncio

    asyncio.run(coro())


async def _test_agentops_trace_with_store_or_not_imp():
    tracer = AgentOpsTracer()
    tracer.init()
    tracer.init_worker(0)

    try:
        # Using AgentOpsTracer to trace a function without providing a store, rollout_id, or attempt_id.
        tracer.trace_run(_func_without_exception)
        spans = tracer.get_last_trace()
        assert len(spans) > 0

        # Using AgentOpsTracer to trace a function with providing a store which disabled native otlp exporter, rollout_id, and attempt_id.
        store = MockLightningStore()
        async with tracer.trace_context(
            name="agentops_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        spans = tracer.get_last_trace()
        assert len(spans) > 0

        # Using AgentOpsTracer to trace a function with providing a store which enabled native otlp exporter, rollout_id, and attempt_id.
        store.enable_otlp_traces()
        async with tracer.trace_context(
            name="agentops_test", store=store, rollout_id="test_rollout_id", attempt_id="test_attempt_id"
        ):
            _func_without_exception()
        spans = tracer.get_last_trace()
        assert len(spans) > 0
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()

# Copyright (c) Microsoft. All rights reserved.

import multiprocessing
import sys
from typing import Any, Optional, Union

import agentops
import pytest
from agentops.sdk.core import TraceContext
from opentelemetry.trace.status import StatusCode

from agentlightning.tracer.agentops import AgentOpsTracer


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

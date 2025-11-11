# Copyright (c) Microsoft. All rights reserved.

import multiprocessing
from unittest.mock import patch

import weave

from agentlightning.tracer.weave import WeaveTracer


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


def test_weave_trace_workable():
    """
    Test that WeaveTracer can record and retrieve a simple function trace.
    """

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_test_weave_trace_workable_imp)
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


def _mock_weave_init():
    with patch("weave.init"):
        weave.init(project_name="agentlightning.tracer.weave_test")


def _test_weave_trace_workable_imp():
    _mock_weave_init()

    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    try:
        tracer.trace_run(_func_without_exception)
        calls = tracer.get_last_trace()

        assert len(calls) > 0
    finally:
        tracer.teardown_worker(0)
        tracer.teardown()

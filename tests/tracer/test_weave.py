# Copyright (c) Microsoft. All rights reserved.

import multiprocessing
from typing import Any, Optional, Union

from dotenv import load_dotenv

from agentlightning.tracer.weave import WeaveTracer

load_dotenv()


def _func_with_exception():
    """Function that always raises an exception to test error tracing."""
    raise ValueError("This is a test exception")


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


# def test_trace_error_status_from_instance():
#     """
#     Test that AgentOpsTracer correctly sets trace end state based on execution result.

#     This test replaces `agentops.end_trace` with a custom function to capture
#     the `end_state` passed in. It verifies that traces ending after a raised
#     exception have `StatusCode.ERROR`, while normal runs have `StatusCode.OK`.
#     """

#     ctx = multiprocessing.get_context("spawn")
#     proc = ctx.Process(target=_test_trace_error_status_from_instance_imp)
#     proc.start()
#     proc.join(30.0)  # On GPU server, the time is around 10 seconds.

#     if proc.is_alive():
#         proc.terminate()
#         proc.join(5)
#         if proc.is_alive():
#             proc.kill()

#         assert False, "Child process hung. Check test output for details."

#     assert proc.exitcode == 0, (
#         f"Child process for test_trace_error_status_from_instance failed with exit code {proc.exitcode}. "
#         "Check child traceback in test output."
#     )

from openai import OpenAI


def _create_completion(message: str) -> str:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="mock-key")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": message}],
    )
    return response.choices[0].message.content


def test_weave_trace_workable():
    tracer = WeaveTracer()
    tracer.init()
    tracer.init_worker(0)

    try:
        tracer.trace_run(_create_completion, "Hello there, how are you?")
        # tracer.trace_run(_func_with_exception)
        calls = tracer.get_last_trace()
        assert len(calls) > 0

    finally:
        tracer.teardown_worker(0)
        tracer.teardown()

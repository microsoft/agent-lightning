# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import patch

import pytest
import weave

from agentlightning.tracer.weave import WeaveTracer


def _func_without_exception():
    """Function that always executed successfully to test success tracing."""
    pass


def _mock_weave_init():
    with patch("weave.init"):
        weave.init(project_name="agentlightning.tracer.weave_test")


@pytest.mark.skip(reason="Skipping this test temporarily")
def test_weave_trace_workable():
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

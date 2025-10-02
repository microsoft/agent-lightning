# Copyright (c) Microsoft. All rights reserved.

import agentops
import agentops.sdk.core as agentops_core
import opentelemetry.trace as trace_api


# pyright: reportPrivateUsage=false
def clear_tracer_provider() -> None:
    """OpenTelemetry tracer provider does not allow set twice.

    Reset the tracer provider to allow setting it again in tests.
    This is a hack to the internal state of OpenTelemetry.
    Not a good idea to use in production.
    Always remember to setup two processes if you need two tracers.
    """

    if hasattr(trace_api, "_TRACER_PROVIDER"):
        if trace_api._TRACER_PROVIDER is not None:
            trace_api._TRACER_PROVIDER = None

    if hasattr(trace_api, "_TRACER_PROVIDER_SET_ONCE"):
        if hasattr(trace_api._TRACER_PROVIDER_SET_ONCE, "_done"):
            if trace_api._TRACER_PROVIDER_SET_ONCE._done:
                trace_api._TRACER_PROVIDER_SET_ONCE._done = False

        if hasattr(trace_api._TRACER_PROVIDER_SET_ONCE, "_flag"):
            if trace_api._TRACER_PROVIDER_SET_ONCE._flag:  # type: ignore
                trace_api._TRACER_PROVIDER_SET_ONCE._flag = False  # type: ignore


def clear_agentops_init() -> None:
    """Make agentops.init() runnable again."""
    agentops.get_client().initialized = False
    agentops_core.tracer._initialized = False

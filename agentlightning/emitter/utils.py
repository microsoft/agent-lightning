# Copyright (c) Microsoft. All rights reserved.

"""Common utilities for the emitter module."""

import opentelemetry.trace as trace_api
from opentelemetry.trace import get_tracer_provider


def get_tracer() -> trace_api.Tracer:
    """Return the tracer used for AgentLightning spans.

    Raises:
        RuntimeError: If the tracer is not initialized.

    Returns:
        The AgentLightning tracer instance.
    """
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()
    return tracer_provider.get_tracer("agentlightning")

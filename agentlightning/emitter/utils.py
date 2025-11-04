# Copyright (c) Microsoft. All rights reserved.

"""Utilities shared across emitter implementations."""

import opentelemetry.trace as trace_api
from opentelemetry.trace import get_tracer_provider


def get_tracer() -> trace_api.Tracer:
    """Resolve the OpenTelemetry tracer configured for Agent Lightning.

    Returns:
        OpenTelemetry tracer tagged with the `agentlightning` instrumentation name.

    Raises:
        RuntimeError: If OpenTelemetry was not initialized before calling this helper.
    """
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()
    return tracer_provider.get_tracer("agentlightning")

# Copyright (c) Microsoft. All rights reserved.

import json
import logging
from typing import Any

import opentelemetry.trace as trace_api
from opentelemetry.trace import get_tracer_provider

from agentlightning.types.tracer import SpanNames

logger = logging.getLogger(__name__)


def _get_tracer() -> trace_api.Tracer:
    """Return the tracer used for AgentLightning spans."""
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()
    return tracer_provider.get_tracer("agentlightning")


def emit_object(object: Any) -> None:
    """Emit any object as a span. Make sure the object is JSON serializable."""
    try:
        serialized = json.dumps(object)
    except (TypeError, ValueError):
        logger.error(f"Object must be JSON serializable, got: {type(object)}. Skip emit_object.")
        return

    tracer = _get_tracer()
    span = tracer.start_span(
        SpanNames.OBJECT.value,
        attributes={"object": serialized},
    )
    logger.debug("Emitting object span with payload size %d characters", len(serialized))
    with span:
        pass

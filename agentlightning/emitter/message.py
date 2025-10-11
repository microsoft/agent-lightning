# Copyright (c) Microsoft. All rights reserved.

import logging

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


def emit_message(message: str) -> None:
    """Emit a string message as a span.

    OpenTelemetry has a dedicated design of logs by design, but we can also use spans to emit messages.
    So that it can all be unified in the data store and analyzed together.
    """
    if not isinstance(message, str):  # type: ignore
        logger.error(f"Message must be a string, got: {type(message)}. Skip emit_message.")
        return

    tracer = _get_tracer()
    span = tracer.start_span(
        SpanNames.MESSAGE.value,
        attributes={"message": message},
    )
    logger.debug("Emitting message span with message: %s", message)
    with span:
        pass

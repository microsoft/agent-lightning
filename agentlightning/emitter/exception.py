# Copyright (c) Microsoft. All rights reserved.

import logging
import traceback

import opentelemetry.trace as trace_api
from opentelemetry.semconv.attributes import exception_attributes
from opentelemetry.trace import get_tracer_provider

from agentlightning.types.tracer import SpanNames

logger = logging.getLogger(__name__)


def _get_tracer() -> trace_api.Tracer:
    """Return the tracer used for AgentLightning spans."""
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()
    return tracer_provider.get_tracer("agentlightning")


def emit_exception(exception: BaseException) -> None:
    """Emit an exception as a span."""
    if not isinstance(exception, BaseException):  # type: ignore
        logger.error(f"Expected an BaseException instance, got: {type(exception)}. Skip emit_exception.")
        return

    tracer = _get_tracer()
    stacktrace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    attributes = {
        exception_attributes.EXCEPTION_TYPE: type(exception).__name__,
        exception_attributes.EXCEPTION_MESSAGE: str(exception),
        exception_attributes.EXCEPTION_ESCAPED: True,
    }
    if stacktrace.strip():
        attributes[exception_attributes.EXCEPTION_STACKTRACE] = stacktrace

    span = tracer.start_span(
        SpanNames.EXCEPTION.value,
        attributes=attributes,
    )
    logger.debug("Emitting exception span for %s", type(exception).__name__)
    with span:
        span.record_exception(exception)
        # We don't set the status of the span here. They have other semantics.

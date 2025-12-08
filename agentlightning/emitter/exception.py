# Copyright (c) Microsoft. All rights reserved.

import logging
import traceback

from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.types import SpanNames

from .utils import get_tracer

logger = logging.getLogger(__name__)


def emit_exception(exception: BaseException) -> None:
    """Record an exception with OpenTelemetry metadata.

    Args:
        exception: Raised exception instance to serialize into telemetry attributes.

    !!! note
        The helper validates its input. Non-exception values are ignored to prevent
        noisy telemetry and indicate programming mistakes via the logger.
    """
    if not isinstance(exception, BaseException):  # type: ignore
        logger.error(f"Expected an BaseException instance, got: {type(exception)}. Skip emit_exception.")
        return

    tracer = get_tracer()
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

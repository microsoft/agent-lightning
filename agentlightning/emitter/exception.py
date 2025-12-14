# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Any, Dict, Optional

from agentlightning.semconv import AGL_EXCEPTION
from agentlightning.utils.otel import format_exception_attributes, get_tracer

logger = logging.getLogger(__name__)


def emit_exception(
    exception: BaseException, attributes: Optional[Dict[str, Any]] = None, propagate: bool = True
) -> None:
    """Record an exception with OpenTelemetry metadata.

    Classic OpenTelemetry records exceptions in a dedicated logging service.
    We simplify the model and use trace spans to record exceptions as well.

    Args:
        exception: Raised exception instance to serialize into telemetry attributes.
        attributes: Additional attributes to attach to the exception span.
        propagate: Whether to propagate the span to exporters automatically.

    !!! note

        The helper validates its input. If a non-exception value is provided,
        a TypeError is raised to indicate a programming mistake.
    """
    if not isinstance(exception, BaseException):  # type: ignore
        raise TypeError(f"Expected a BaseException instance, got: {type(exception)}.")

    tracer = get_tracer(use_active_span_processor=propagate)
    span_attributes = format_exception_attributes(exception)

    if attributes:
        span_attributes.update(attributes)

    span = tracer.start_span(
        AGL_EXCEPTION,
        attributes=span_attributes,
    )
    logger.debug("Emitting exception span for %s", type(exception).__name__)
    with span:
        span.record_exception(exception)
        # We don't set the status of the span here. They have other semantics.

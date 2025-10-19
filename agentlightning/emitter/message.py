# Copyright (c) Microsoft. All rights reserved.

import logging

from agentlightning.types import SpanAttributeNames, SpanNames

from .utils import get_tracer

logger = logging.getLogger(__name__)


def emit_message(message: str) -> None:
    """Emit a textual message as an OpenTelemetry span.

    Args:
        message: Human readable message to attach as a span attribute.

    !!! note
        OpenTelemetry distinguishes between logs and spans. Emitting the message as a
        span keeps all Agent Lightning telemetry in a single data store for analysis.
    """
    if not isinstance(message, str):  # type: ignore
        logger.error(f"Message must be a string, got: {type(message)}. Skip emit_message.")
        return

    tracer = get_tracer()
    span = tracer.start_span(
        SpanNames.MESSAGE.value,
        attributes={SpanAttributeNames.MESSAGE.value: message},
    )
    logger.debug("Emitting message span with message: %s", message)
    with span:
        pass

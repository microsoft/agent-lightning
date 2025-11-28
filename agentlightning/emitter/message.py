# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Any, Dict, Optional

from agentlightning.semconv import AGL_MESSAGE, LightningSpanAttributes
from agentlightning.types import SpanLike
from agentlightning.utils.otel import get_tracer

logger = logging.getLogger(__name__)


def emit_message(message: str, attributes: Optional[Dict[str, Any]] = None, propagate: bool = True) -> None:
    """Emit a textual message as an OpenTelemetry span.

    Commonly used for sending debugging and logging messages.

    Args:
        message: Human readable message to attach as a span attribute.
        attributes: Additional attributes to attach to the message span.
        propagate: Whether to propagate the span to exporters automatically.

    !!! note
        OpenTelemetry distinguishes between logs and spans. Emitting the message as a
        span keeps all Agent Lightning telemetry in a single data store for analysis.
    """
    if not isinstance(message, str):  # type: ignore
        raise TypeError(f"Message must be a string or list of strings, got: {type(message)}.")

    tracer = get_tracer(use_active_span_processor=propagate)
    span_attributes = {LightningSpanAttributes.MESSAGE_BODY.value: message}
    if attributes:
        span_attributes.update(attributes)
    span = tracer.start_span(
        AGL_MESSAGE,
        attributes=span_attributes,
    )
    logger.debug("Emitting message span with message: %s", message)
    with span:
        pass


def get_message_value(span: SpanLike) -> Optional[str]:
    """Extract the message string from a message span.

    Args:
        span: Span-like object to extract the message from.
    """
    span_attributes = span.attributes or {}
    if LightningSpanAttributes.MESSAGE_BODY.value not in span_attributes:
        return None
    message = span_attributes[LightningSpanAttributes.MESSAGE_BODY.value]
    if isinstance(message, str):
        return message
    raise TypeError(f"Message must be a string, got: {type(message)}.")

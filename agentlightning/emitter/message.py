# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Any, Dict, Optional

from agentlightning.semconv import AGL_MESSAGE, LightningSpanAttributes
from agentlightning.tracer.base import get_active_tracer
from agentlightning.tracer.dummy import DummyTracer
from agentlightning.types import Attributes, SpanLike
from agentlightning.utils.otel import flatten_attributes, sanitize_attributes

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

    if propagate:
        tracer = get_active_tracer()
        if tracer is None:
            raise RuntimeError("No active tracer found. Cannot emit message span.")
    else:
        tracer = DummyTracer()
    span_attributes: Attributes = {LightningSpanAttributes.MESSAGE_BODY.value: message}
    if attributes:
        flattened = flatten_attributes(attributes, expand_leaf_lists=False)
        span_attributes.update(sanitize_attributes(flattened))
    logger.debug("Emitting message span with message: %s", message)
    tracer.create_span(
        AGL_MESSAGE,
        attributes=span_attributes,
    )


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

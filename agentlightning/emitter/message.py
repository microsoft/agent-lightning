# Copyright (c) Microsoft. All rights reserved.

import logging

from agentlightning.types import SpanAttributeNames, SpanNames

from .utils import get_tracer

logger = logging.getLogger(__name__)


def emit_message(message: str) -> None:
    """Emit a string message as a span.

    OpenTelemetry has a dedicated design of logs by design, but we can also use spans to emit messages.
    So that it can all be unified in the data store and analyzed together.
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

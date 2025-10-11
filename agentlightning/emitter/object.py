# Copyright (c) Microsoft. All rights reserved.

import json
import logging
from typing import Any

from agentlightning.types import SpanAttributeNames, SpanNames

from .utils import get_tracer

logger = logging.getLogger(__name__)


def emit_object(object: Any) -> None:
    """Emit any object as a span. Make sure the object is JSON serializable."""
    try:
        serialized = json.dumps(object)
    except (TypeError, ValueError):
        logger.error(f"Object must be JSON serializable, got: {type(object)}. Skip emit_object.")
        return

    tracer = get_tracer()
    span = tracer.start_span(
        SpanNames.OBJECT.value,
        attributes={SpanAttributeNames.OBJECT.value: serialized},
    )
    logger.debug("Emitting object span with payload size %d characters", len(serialized))
    with span:
        pass

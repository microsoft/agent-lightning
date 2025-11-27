# Copyright (c) Microsoft. All rights reserved.

import base64
import json
import logging
from typing import Any, Dict, Optional

from agentlightning.semconv import AGL_OBJECT, LightningSpanAttributes
from agentlightning.types import SpanLike
from agentlightning.utils.otel import full_qualified_name, get_tracer

logger = logging.getLogger(__name__)


def emit_object(object: Any, attributes: Optional[Dict[str, Any]] = None, propagate: bool = True) -> None:
    """Emit an object's serialized representation as an OpenTelemetry span.

    Args:
        object: Data structure to encode as JSON and attach to the span payload.
        attributes: Additional attributes to attach to the object span.
        propagate: Whether to propagate the span to exporters automatically.

    !!! note
        The payload must be JSON serializable. Non-serializable objects will lead to a RuntimeError.
    """
    span_attributes = encode_object(object)
    if attributes:
        span_attributes.update(attributes)
    tracer = get_tracer(use_active_span_processor=propagate)
    span = tracer.start_span(
        AGL_OBJECT,
        attributes=span_attributes,
    )
    attr_length = 0
    if LightningSpanAttributes.OBJECT_JSON.value in span_attributes:
        attr_length = len(span_attributes[LightningSpanAttributes.OBJECT_JSON.value])
    elif LightningSpanAttributes.OBJECT_LITERAL.value in span_attributes:
        attr_length = len(span_attributes[LightningSpanAttributes.OBJECT_LITERAL.value])
    logger.debug("Emitting object span with payload size %d characters", attr_length)
    with span:
        pass


def encode_object(object: Any) -> Dict[str, Any]:
    """Encode an object as span attributes.

    Args:
        object: Data structure to encode as JSON.
    """
    span_attributes = {}
    if isinstance(object, (str, int, float, bool)):
        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: type(object).__name__,
            LightningSpanAttributes.OBJECT_LITERAL.value: str(object),
        }
    elif isinstance(object, bytes):
        b64_encoded = base64.b64encode(object).decode("utf-8")
        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: "bytes",
            LightningSpanAttributes.OBJECT_LITERAL.value: b64_encoded,
        }
    else:
        try:
            serialized = json.dumps(object)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Object must be JSON serializable, got: {type(object)}.") from exc

        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: full_qualified_name(type(object)),  # type: ignore
            LightningSpanAttributes.OBJECT_JSON.value: serialized,
        }

    return span_attributes


def get_object_value(span: SpanLike) -> Any:
    """Extract the object payload from an object span.

    Args:
        span: Span object produced by Agent Lightning emitters.
    """
    attributes = span.attributes or {}
    if LightningSpanAttributes.OBJECT_JSON.value in attributes:
        serialized = attributes[LightningSpanAttributes.OBJECT_JSON.value]
        try:
            return json.loads(serialized)  # type: ignore
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Failed to deserialize object JSON from span.") from exc
    elif LightningSpanAttributes.OBJECT_LITERAL.value in attributes:
        literal = attributes[LightningSpanAttributes.OBJECT_LITERAL.value]
        obj_type = attributes.get(LightningSpanAttributes.OBJECT_TYPE.value, "str")
        if obj_type == "str":
            return literal
        elif obj_type == "int":
            # Let it raise errors if there are any
            return int(literal)  # type: ignore
        elif obj_type == "float":
            return float(literal)  # type: ignore
        elif obj_type == "bool":
            return literal.lower() == "true"  # type: ignore
        elif obj_type == "bytes":
            return base64.b64decode(literal.encode("utf-8"))  # type: ignore
        else:
            raise RuntimeError(f"Unsupported object type for literal deserialization: {obj_type}")
    else:
        return None

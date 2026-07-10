# Copyright (c) Microsoft. All rights reserved.

import base64
import json
import logging
from typing import Any, Dict, Optional, cast

from agentlightning.semconv import AGL_OBJECT, LightningSpanAttributes
from agentlightning.tracer.base import get_active_tracer
from agentlightning.tracer.dummy import DummyTracer
from agentlightning.types import SpanCoreFields, SpanLike, TraceStatus
from agentlightning.utils.otel import flatten_attributes, full_qualified_name, sanitize_attributes

logger = logging.getLogger(__name__)


def emit_object(value: Any, attributes: Optional[Dict[str, Any]] = None, propagate: bool = True) -> SpanCoreFields:
    """Emit an object's serialized representation as an OpenTelemetry span.

    Args:
        value: Data structure to encode as JSON and attach to the span payload.
        attributes: Additional attributes to attach to the object span.
        propagate: Whether to propagate the span to exporters automatically.

    !!! note
        The payload must be JSON serializable. Non-serializable objects will lead to a RuntimeError.
    """
    span_attributes = encode_object(value)
    if attributes:
        flattened = flatten_attributes(attributes, expand_leaf_lists=False)
        span_attributes.update(sanitize_attributes(flattened))

    attr_length = 0
    if LightningSpanAttributes.OBJECT_JSON.value in span_attributes:
        attr_length = len(span_attributes[LightningSpanAttributes.OBJECT_JSON.value])
    elif LightningSpanAttributes.OBJECT_LITERAL.value in span_attributes:
        attr_length = len(span_attributes[LightningSpanAttributes.OBJECT_LITERAL.value])
    logger.debug("Emitting object span with payload size %d characters", attr_length)

    if propagate:
        tracer = get_active_tracer()
        if tracer is None:
            raise RuntimeError("No active tracer found. Cannot emit object span.")
    else:
        # Do not actually propagate to any store or tracer backend.
        tracer = DummyTracer()

    return tracer.create_span(
        name=AGL_OBJECT,
        attributes=span_attributes,
        status=TraceStatus(status_code="OK"),
    )


def encode_object(value: Any) -> Dict[str, Any]:
    """Encode an object as span attributes.

    Args:
        value: Data structure to encode as JSON.
    """
    span_attributes = {}
    if isinstance(value, (str, int, float, bool)):
        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: type(value).__name__,
            LightningSpanAttributes.OBJECT_LITERAL.value: str(value),
        }
    elif isinstance(value, bytes):
        b64_encoded = base64.b64encode(value).decode("utf-8")
        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: "bytes",
            LightningSpanAttributes.OBJECT_LITERAL.value: b64_encoded,
        }
    else:
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Object must be JSON serializable, got: {type(value)}.") from exc

        value_type = cast(type[Any], type(value))
        span_attributes = {
            LightningSpanAttributes.OBJECT_TYPE.value: full_qualified_name(value_type),
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
        if not isinstance(serialized, str):
            raise RuntimeError("Object JSON span attribute must be a string.")
        try:
            return json.loads(serialized)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Failed to deserialize object JSON from span.") from exc
    elif LightningSpanAttributes.OBJECT_LITERAL.value in attributes:
        literal = attributes[LightningSpanAttributes.OBJECT_LITERAL.value]
        obj_type = attributes.get(LightningSpanAttributes.OBJECT_TYPE.value, "str")
        if not isinstance(literal, str):
            raise RuntimeError("Object literal span attribute must be a string.")
        if obj_type == "str":
            return literal
        elif obj_type == "int":
            # Let it raise errors if there are any
            return int(literal)
        elif obj_type == "float":
            return float(literal)
        elif obj_type == "bool":
            return literal.lower() == "true"
        elif obj_type == "bytes":
            return base64.b64decode(literal.encode("utf-8"))
        else:
            raise RuntimeError(f"Unsupported object type for literal deserialization: {obj_type}")
    else:
        return None

# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting annotation spans."""

import logging
from typing import Any, Dict

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.semconv import AGL_ANNOTATION
from agentlightning.utils.otel import flatten_attributes, get_tracer

logger = logging.getLogger(__name__)


def emit_annotation(annotation: Dict[str, Any], propagate: bool = True) -> ReadableSpan:
    """Emit a new annotation span.

    This is the underlying implementation of [`emit_reward`][agentlightning.emit_reward].

    Annotation spans are used to annotate a specific event or a part of rollout.
    See [semconv][agentlightning.semconv] for conventional annotation keys in Agent-lightning.

    If annotations contain nested dicts, they will be flattened before emitting.
    Complex objects will lead to emitting failures.

    Args:
        annotation: Dictionary containing annotation key-value pairs.
            Representatives are rewards, tags, and metadata.
        propagate: Whether to propagate the span to exporters automatically.
    """
    annotation_attributes = flatten_attributes(annotation)
    if any(not isinstance(v, (str, int, float, bool, bytes)) for v in annotation_attributes.values()):
        raise TypeError("All annotation attributes must be primitive types (str, int, float, bool, bytes)")

    # TODO: this should use a tracer from current context rather than the singleton
    tracer = get_tracer(use_active_span_processor=propagate)
    span = tracer.start_span(
        AGL_ANNOTATION,
        attributes=annotation_attributes,
    )
    logger.debug("Emitting annotation span with keys %s", annotation_attributes)
    with span:
        pass
    if not isinstance(span, ReadableSpan):
        raise ValueError(f"Span is not a ReadableSpan: {span}")

    return span

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

"""Data models that mirror OpenTelemetry spans for Agent Lightning."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event as OtelEvent
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace.status import Status as OtelStatus
from pydantic import BaseModel, ConfigDict

__all__ = [
    "AttributeValue",
    "Attributes",
    "TraceState",
    "SpanContext",
    "TraceStatus",
    "Event",
    "Link",
    "OtelResource",
    "Span",
    "SpanNames",
    "SpanAttributeNames",
    "SpanLike",
]


def convert_timestamp(timestamp: Optional[int]) -> Optional[float]:
    """Normalize OpenTelemetry timestamps to seconds.

    Args:
        timestamp: Timestamp expressed either in seconds or nanoseconds.

    Returns:
        Timestamp in seconds when `timestamp` is provided; otherwise `None`.
    """
    if not timestamp:
        return None
    return timestamp / 1_000_000_000 if timestamp > 1e12 else timestamp


def extract_extra_fields(src: Any, excluded_fields: List[str]) -> Dict[str, Any]:
    """Capture custom attributes from an OpenTelemetry object.

    Args:
        src: Object that exposes a `__dict__` of potential attributes.
        excluded_fields: Attribute names that should be removed from the output.

    Returns:
        Dictionary containing JSON-serializable representations of the remaining fields.
    """
    excluded_fields_set = set(excluded_fields) | set(["_" + k for k in excluded_fields])
    # Exclude the function fields
    excluded_fields_set |= set(src.__class__.__dict__.keys())
    stripped_dict = {k.lstrip("_"): v for k, v in src.__dict__.items()}
    candidates = {k: v for k, v in stripped_dict.items() if k not in excluded_fields_set and not k.startswith("_")}
    # This should strip or flatten the unserializable fields
    candidates_serialized = json.dumps(candidates, default=str)
    return json.loads(candidates_serialized)


AttributeValue = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]
"""Possible values for OpenTelemetry attributes."""
Attributes = Dict[str, AttributeValue]
"""Mapping from attribute names to their values. Same as OpenTelemetry `Attributes` type."""
TraceState = Dict[str, str]
"""Mapping from trace state key to its value. Same as OpenTelemetry `TraceState` type."""


class SpanContext(BaseModel):
    """Pydantic representation of `opentelemetry.trace.SpanContext` values."""

    trace_id: str
    """The trace ID of the span."""
    span_id: str
    """The span ID of the span."""
    is_remote: bool
    """Whether the span is remote."""
    trace_state: TraceState
    """Mapping from trace state key to its value."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_opentelemetry(cls, src: trace_api.SpanContext) -> "SpanContext":
        """Construct a [`SpanContext`][agentlightning.SpanContext] from OpenTelemetry data."""

        return cls(
            trace_id=trace_api.format_trace_id(src.trace_id),
            span_id=trace_api.format_span_id(src.span_id),
            is_remote=src.is_remote,
            trace_state={k: v for k, v in src.trace_state.items()} if src.trace_state else {},
            **extract_extra_fields(src, ["trace_id", "span_id", "is_remote", "trace_state"]),
        )


class TraceStatus(BaseModel):
    """Serializable variant of `opentelemetry.trace.Status`."""

    status_code: str
    """The status code of the span. Same as OpenTelemetry `Status.status_code` type."""
    description: Optional[str] = None
    """The description of the span. Same as OpenTelemetry `Status.description` type."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_opentelemetry(cls, src: OtelStatus) -> "TraceStatus":
        """Create a [`TraceStatus`][agentlightning.TraceStatus] from OpenTelemetry metadata."""

        return cls(
            status_code=src.status_code.name,
            description=src.description,
            **extract_extra_fields(src, ["status_code", "description"]),
        )


class Event(BaseModel):
    """Serializable representation of OpenTelemetry `Event` values."""

    name: str
    """The name of the event."""
    attributes: Attributes
    """Mapping from attribute names to their values. Same as OpenTelemetry `Attributes` type."""
    timestamp: Optional[float] = None
    """The timestamp of the event. Same as OpenTelemetry `Event.timestamp` type."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_opentelemetry(cls, src: OtelEvent) -> "Event":
        """Create an [`Event`][agentlightning.Event] from an OpenTelemetry event."""

        return cls(
            name=src.name,
            attributes=dict(src.attributes) if src.attributes else {},
            timestamp=convert_timestamp(src.timestamp),
            **extract_extra_fields(src, ["name", "attributes", "timestamp"]),
        )


class Link(BaseModel):
    """Serializable representation of OpenTelemetry `Link` values."""

    context: SpanContext
    """The context of the link."""
    attributes: Optional[Attributes] = None
    """Optional attributes."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_opentelemetry(cls, src: trace_api.Link) -> "Link":
        """Create a [`Link`][agentlightning.Link] from an OpenTelemetry link."""

        return cls(
            context=SpanContext.from_opentelemetry(src.context),
            attributes=dict(src.attributes) if src.attributes else None,
            **extract_extra_fields(src, ["context", "attributes"]),
        )


class OtelResource(BaseModel):
    """Serializable representation of OpenTelemetry `Resource` values.

    Named as `OtelResource` to avoid confusion with the [`Resource`][agentlightning.Resource] class.
    Users will very rarely need to construct this class directly. Most of the times,
    they deal with the [`Resource`][agentlightning.Resource] class instead, which describes
    a very different concept.
    """

    attributes: Attributes
    """Mapping from attribute names to their values. Same as OpenTelemetry `Attributes` type."""
    schema_url: str
    """The schema URL of the resource."""

    @classmethod
    def from_opentelemetry(cls, src: Resource) -> "OtelResource":
        """Create a [`Resource`][agentlightning.Resource] from an OpenTelemetry resource."""

        return cls(
            attributes=dict(src.attributes) if src.attributes else {},
            schema_url=src.schema_url if src.schema_url else "",
            **extract_extra_fields(src, ["attributes", "schema_url"]),
        )


class Span(BaseModel):
    """Agent Lightning's canonical span model used for persistence and analytics.

    The model captures the most relevant fields from
    `opentelemetry.sdk.trace.ReadableSpan` instances while preserving unmodeled
    attributes in Pydantic `BaseModel`'s extra storage. This keeps the serialized format
    stable even as upstream OpenTelemetry types evolve.
    """

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    """The rollout which this span belongs to."""
    attempt_id: str
    """The attempt which this span belongs to."""
    sequence_id: int
    """The ID to make spans ordered within a single attempt."""

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: str  # one rollout can have traces coming from multiple places
    """The trace ID of the span. One rollout/attempt can have multiple traces.
    This ID comes from the OpenTelemetry trace ID generator.
    """
    span_id: str
    """The span ID of the span. This ID comes from the OpenTelemetry span ID generator."""
    parent_id: Optional[str]
    """The parent span ID of the span."""

    # Core ReadableSpan fields
    name: str
    """The name of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    status: TraceStatus
    """The status of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    attributes: Attributes
    """The attributes of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    events: List[Event]
    """The events of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    links: List[Link]
    """The links of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""

    # Timestamps
    start_time: Optional[float]
    """The start time of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    end_time: Optional[float]
    """The end time of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""

    # Other parsable fields
    context: Optional[SpanContext]
    """The context of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    parent: Optional[SpanContext]
    """The parent context of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    resource: OtelResource
    """The resource of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""

    # Preserve other fields in the readable span as extra fields
    # Make sure that are json serializable (so no bytes, complex objects, ...)

    @classmethod
    def from_opentelemetry(
        cls,
        src: ReadableSpan,
        rollout_id: str,
        attempt_id: str,
        sequence_id: int,
    ) -> "Span":
        """Convert an OpenTelemetry span into the Agent Lightning data model.

        Args:
            src: Span captured by OpenTelemetry.
            rollout_id: Identifier for the rollout that produced the span.
            attempt_id: Identifier of the attempt within the rollout.
            sequence_id: Monotonically increasing identifier assigned to the span.

        Returns:
            Parsed [`Span`][agentlightning.Span] instance suitable for persistence.
        """
        context = src.get_span_context()
        if context is None:
            trace_id = span_id = 0
        else:
            trace_id = context.trace_id
            span_id = context.span_id
        return cls(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
            trace_id=trace_api.format_trace_id(trace_id),
            span_id=trace_api.format_span_id(span_id),
            parent_id=(trace_api.format_span_id(src.parent.span_id) if src.parent else None),
            name=src.name,
            status=TraceStatus.from_opentelemetry(src.status),
            attributes=dict(src.attributes) if src.attributes else {},
            events=[Event.from_opentelemetry(event) for event in src.events] if src.events else [],
            links=[Link.from_opentelemetry(link) for link in src.links] if src.links else [],
            start_time=convert_timestamp(src.start_time),
            end_time=convert_timestamp(src.end_time),
            context=SpanContext.from_opentelemetry(context) if context else None,
            parent=(SpanContext.from_opentelemetry(src.parent) if src.parent else None),
            resource=OtelResource.from_opentelemetry(src.resource),
            **extract_extra_fields(
                src,
                [
                    "name",
                    "context",
                    "parent",
                    "resource",
                    "attributes",
                    "events",
                    "links",
                    "start_time",
                    "end_time",
                    "status",
                    "span_processor",
                    "rollout_id",
                    "attempt_id",
                    "trace_id",
                    "span_id",
                    "parent_id",
                ],
            ),
        )

    @classmethod
    def from_attributes(
        cls,
        *,
        attributes: Attributes,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        sequence_id: Optional[int] = None,
        name: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        resource: Optional[OtelResource] = None,
    ) -> "Span":
        """Build a synthetic span from raw attributes.
        Different from the [`from_opentelemetry`][agentlightning.Span.from_opentelemetry] method,
        all parameters other than `attributes` are optional and will be generated if not provided.

        Args:
            attributes: Span attributes to persist.
            rollout_id: Optional rollout identifier associated with the span.
            attempt_id: Optional attempt identifier associated with the span.
            sequence_id: Optional sequence number to preserve ordering.
            name: Optional human-readable span name.
            trace_id: Custom trace identifier. When omitted, a random identifier is generated.
            span_id: Custom span identifier. When omitted, a random identifier is generated.
            parent_id: Optional parent span identifier.
            start_time: Span start timestamp in seconds.
            end_time: Span end timestamp in seconds.
            resource: Explicit resource information to attach to the span.

        Returns:
            [`Span`][agentlightning.Span] populated with the provided attributes.
        """

        id_generator = RandomIdGenerator()
        trace_id = trace_id or trace_api.format_trace_id(id_generator.generate_trace_id())
        span_id = span_id or trace_api.format_span_id(id_generator.generate_span_id())

        return cls(
            rollout_id=rollout_id or "",
            attempt_id=attempt_id or "",
            sequence_id=sequence_id or 0,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            start_time=start_time,
            end_time=end_time,
            context=SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_state={},
            ),
            name=name or SpanNames.VIRTUAL.value,
            resource=resource or OtelResource(attributes={}, schema_url=""),
            attributes=attributes,
            status=TraceStatus(status_code="OK"),
            events=[],
            links=[],
            parent=(
                SpanContext(
                    trace_id=trace_id,
                    span_id=parent_id,
                    is_remote=False,
                    trace_state={},
                )
                if parent_id
                else None
            ),
        )


class SpanNames(str, Enum):
    """Enumerated span names recognised by Agent-lightning."""

    REWARD = "agentlightning.reward"
    """The name of the reward span."""
    MESSAGE = "agentlightning.message"
    """The name of the message span."""
    OBJECT = "agentlightning.object"
    """The name of the object span."""
    EXCEPTION = "agentlightning.exception"
    """The name of the exception span."""
    VIRTUAL = "agentlightning.virtual"
    """The name of the virtual span. It represents derived spans without concrete operations."""
    ROLLOUT_ID = "agentlightning.rollout_id"
    """The name of the rollout ID."""
    ATTEMPT_ID = "agentlightning.attempt_id"
    """The name of the attempt ID."""
    SPAN_SEQUENCE_ID = "agentlightning.span_sequence_id"
    """The name of the span sequence ID."""


class SpanAttributeNames(str, Enum):
    """Canonical attribute names written by Agent Lightning emitters."""

    MESSAGE = "message"
    """The name of the message attribute."""
    OBJECT = "object"
    """The name of the object attribute."""


SpanLike = Union[ReadableSpan, Span]
"""Union type of OpenTelemetry `ReadableSpan` and Agent-lightning [`Span`][agentlightning.Span]."""

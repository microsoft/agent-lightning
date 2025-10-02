# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as OtelResource
from opentelemetry.sdk.trace import Event as OtelEvent
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace.status import Status as OtelStatus
from pydantic import BaseModel

__all__ = [
    "AttributeValue",
    "Attributes",
    "TraceState",
    "SpanContext",
    "TraceStatus",
    "Event",
    "Link",
    "Resource",
    "Span",
    "SpanNames",
]


def convert_timestamp(timestamp: Optional[int]) -> Optional[float]:
    """Convert timestamp from nanoseconds to seconds if needed.

    Auto-detects format: if > 1e12, assumes nanoseconds; otherwise seconds.
    """
    if not timestamp:
        return None
    return timestamp / 1_000_000_000 if timestamp > 1e12 else timestamp


def extract_extra_fields(src: Any, excluded_fields: List[str]) -> Dict[str, Any]:
    """Extract extra fields from source object, excluding specified fields and private fields."""
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
Attributes = Dict[str, AttributeValue]
TraceState = Dict[str, str]


class SpanContext(BaseModel):
    """Corresponding to opentelemetry.trace.SpanContext"""

    trace_id: str
    span_id: str
    is_remote: bool
    trace_state: TraceState

    class Config:
        allow_extra = True

    @classmethod
    def from_opentelemetry(cls, src: trace_api.SpanContext) -> "SpanContext":
        return cls(
            trace_id=trace_api.format_trace_id(src.trace_id),
            span_id=trace_api.format_span_id(src.span_id),
            is_remote=src.is_remote,
            trace_state={k: v for k, v in src.trace_state.items()} if src.trace_state else {},
            **extract_extra_fields(src, ["trace_id", "span_id", "is_remote", "trace_state"]),
        )


class TraceStatus(BaseModel):
    """Corresponding to opentelemetry.trace.Status"""

    status_code: str
    description: Optional[str] = None

    class Config:
        allow_extra = True

    @classmethod
    def from_opentelemetry(cls, src: OtelStatus) -> "TraceStatus":
        return cls(
            status_code=src.status_code.name,
            description=src.description,
            **extract_extra_fields(src, ["status_code", "description"]),
        )


class Event(BaseModel):
    """Corresponding to opentelemetry.trace.Event"""

    name: str
    attributes: Attributes
    timestamp: Optional[float] = None

    class Config:
        allow_extra = True

    @classmethod
    def from_opentelemetry(cls, src: OtelEvent) -> "Event":
        return cls(
            name=src.name,
            attributes=dict(src.attributes) if src.attributes else {},
            timestamp=convert_timestamp(src.timestamp),
            **extract_extra_fields(src, ["name", "attributes", "timestamp"]),
        )


class Link(BaseModel):
    """Corresponding to opentelemetry.trace.Link"""

    context: SpanContext
    attributes: Optional[Attributes] = None

    class Config:
        allow_extra = True

    @classmethod
    def from_opentelemetry(cls, src: trace_api.Link) -> "Link":
        return cls(
            context=SpanContext.from_opentelemetry(src.context),
            attributes=dict(src.attributes) if src.attributes else None,
            **extract_extra_fields(src, ["context", "attributes"]),
        )


class Resource(BaseModel):
    """Corresponding to opentelemetry.sdk.resources.Resource"""

    attributes: Attributes
    schema_url: str

    @classmethod
    def from_opentelemetry(cls, src: OtelResource) -> "Resource":
        return cls(
            attributes=dict(src.attributes) if src.attributes else {},
            schema_url=src.schema_url if src.schema_url else "",
            **extract_extra_fields(src, ["attributes", "schema_url"]),
        )


class Span(BaseModel):

    class Config:
        allow_extra = True  # allow extra fields if needed

    rollout_id: str
    attempt_id: str
    # The ID to make spans ordered within a single attempt
    sequence_id: int

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: str  # one rollout can have traces coming from multiple places
    span_id: str
    parent_id: Optional[str]

    # Core ReadableSpan fields
    name: str
    status: TraceStatus
    attributes: Attributes
    events: List[Event]
    links: List[Link]

    # Timestamps
    start_time: Optional[float]
    end_time: Optional[float]

    # Other parsable fields
    context: Optional[SpanContext]
    parent: Optional[SpanContext]
    resource: Resource

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
            resource=Resource.from_opentelemetry(src.resource),
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


class SpanNames(str, Enum):
    """Standard span name values for AgentLightning.

    Currently only reward spans are supported.
    We will add more spans related to error handling in the future.
    """

    REWARD = "agentlightning.reward"

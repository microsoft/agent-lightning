from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Sequence
from enum import Enum
from pydantic import BaseModel
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource as OtelResource
from opentelemetry import trace as trace_api


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


class SpanKind(str, Enum):
    INTERNAL = "SpanKind.INTERNAL"
    SERVER = "SpanKind.SERVER"
    CLIENT = "SpanKind.CLIENT"
    PRODUCER = "SpanKind.PRODUCER"
    CONSUMER = "SpanKind.CONSUMER"


class StatusCode(str, Enum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanContext(BaseModel):
    """Corresponding to opentelemetry.trace.SpanContext"""
    trace_id: str
    span_id: str
    is_remote: bool
    trace_state: TraceState

    @classmethod
    def from_opentelemetry(cls, src: trace_api.SpanContext) -> "SpanContext":
        return cls(
            trace_id=trace_api.format_trace_id(src.trace_id),
            span_id=trace_api.format_span_id(src.span_id),
            is_remote=src.is_remote,
            trace_state={k: v for k, v in src.trace_state.items()} if src.trace_state else {},
        )


class TraceStatus(BaseModel):
    status_code: StatusCode = StatusCode.UNSET
    description: Optional[str] = None


class Event(BaseModel):
    """Corresponding to opentelemetry.trace.Event"""
    name: str
    attributes: Attributes
    timestamp: Optional[float] = None

    class Config:
        allow_extra = True


class Link(BaseModel):
    """Corresponding to opentelemetry.trace.Link"""
    context: SpanContext
    attributes: Optional[Attributes] = None

    class Config:
        allow_extra = True


class Resource(BaseModel):
    """Corresponding to opentelemetry.sdk.resources.Resource"""
    attributes: Attributes
    schema_url: str

    @classmethod
    def from_opentelemetry(cls, src: OtelResource) -> "Resource":
        return cls(
            attributes=dict(src.attributes) if src.attributes else {},
            schema_url=src.schema_url if src.schema_url else "",
        )


class Span(BaseModel):

    class Config:
        allow_extra = True  # allow extra fields if needed

    rollout_id: str
    attempt_id: str

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: str
    span_id: str
    parent_id: str

    # Core ReadableSpan fields
    name: str
    kind: SpanKind
    status: TraceStatus
    attributes: Attributes
    events: List[Event]
    links: List[Link]

    # Timestamps
    start_time: Optional[float]
    end_time: Optional[float]

    # Other parsable fields
    context: SpanContext
    parent: Optional[SpanContext]
    resource: Resource

    # Preserve other fields in the readable span as extra fields
    # Make sure that are json serializable (so no bytes, complex objects, ...)

    def from_opentelemetry(
        self,
        src: ReadableSpan,
        rollout_id: str,
        attempt_id: str,
    ) -> Span:
        self.rollout_id = rollout_id
        self.attempt_id = attempt_id

        self.trace_id = trace

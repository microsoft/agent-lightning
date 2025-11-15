# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

logger = logging.getLogger(__name__)

from agentlightning.types.tracer import (
    Attributes,
    AttributeValue,
    Event,
    Link,
    OtelResource,
    Span,
    SpanContext,
    TraceStatus,
)

from .base import NamedDictBase, PydanticInDB, PydanticListInDB, SqlAlchemyBase


class TraceStatusInDB(PydanticInDB):
    target_type = TraceStatus


class AttributesInDB(NamedDictBase):
    target_alias = None  # type: ignore
    value_type = AttributeValue


class EventListInDB(PydanticListInDB):
    value_type = Event


class LinkListInDB(PydanticListInDB):
    value_type = Link


class SpanContextInDB(PydanticInDB):
    target_type = SpanContext


class OtelResourceInDB(PydanticInDB):
    target_type = OtelResource


class SpanInDB(SqlAlchemyBase):
    __tablename__ = "spans"

    rollout_id: Mapped[str] = mapped_column(String, nullable=False)  # The rollout which this span belongs to.
    attempt_id: Mapped[str] = mapped_column(String, nullable=False)  # The attempt which this span belongs to.
    sequence_id: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # The ID to make spans ordered within a single attempt.

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: Mapped[str] = mapped_column(
        String, nullable=False
    )  # one rollout can have traces coming from multiple places

    # FIXME: span_id may be not unique across different attempts/rollouts, use (rollout_id, attempt_id, sequence_id) as the primary key instead
    span_id: Mapped[str] = mapped_column(
        String, nullable=False
    )  # The span ID of the span. This ID comes from the OpenTelemetry span ID generator.
    parent_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # The parent span ID of the span.

    # Core ReadableSpan fields
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[TraceStatus] = mapped_column(TraceStatusInDB, nullable=False)
    attributes: Mapped[Attributes] = mapped_column(AttributesInDB, nullable=False)
    events: Mapped[List[Event]] = mapped_column(EventListInDB, nullable=False)
    links: Mapped[List[Link]] = mapped_column(LinkListInDB, nullable=False)

    # Timestamps
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Other parsable fields
    context: Mapped[Optional[SpanContext]] = mapped_column(SpanContextInDB, nullable=True)
    parent: Mapped[Optional[SpanContext]] = mapped_column(SpanContextInDB, nullable=True)
    resource: Mapped[OtelResource] = mapped_column(OtelResourceInDB, nullable=False)

    # extra fields can be added here as needed
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True, default=None)

    __mapper_args__ = {
        "primary_key": [rollout_id, attempt_id, sequence_id],
    }

    def as_span(self) -> Span:
        return Span(
            **self.model_dump(
                exclude={"extra"},
                mapper={"*": lambda obj: obj.extra or {}},  # type: ignore
            )
        )

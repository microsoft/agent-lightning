# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from sqlalchemy import Float, Integer, String, JSON
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm.exc import StaleDataError
from typing import Any, Dict, Optional, List

import time
import logging
logger = logging.getLogger(__name__)

from agentlightning.types.tracer import Span, SpanContext, TraceStatus, Attributes, Event, Link, OtelResource, AttributeValue

from .base import SqlAlchemyBase, PydanticInDB, NamedDictBase, PydanticListInDB
from .rollout import RolloutInDB
from .attempt import AttemptInDB


class TraceStatusInDB(PydanticInDB):
    target_type = TraceStatus


class AttributesInDB(NamedDictBase):
    target_alias = None  # type: ignore
    target_type = AttributeValue


class EventListInDB(PydanticListInDB):
    target_type = Event


class LinkListInDB(PydanticListInDB):
    target_type = Link


class SpanContextInDB(PydanticInDB):
    target_type = SpanContext


class OtelResourceInDB(PydanticInDB):
    target_type = OtelResource


class SpanInDB(SqlAlchemyBase):
    __tablename__ = "spans"

    rollout_id: Mapped[str] = mapped_column(String, nullable=False) # The rollout which this span belongs to.
    attempt_id: Mapped[str] = mapped_column(String, nullable=False) # The attempt which this span belongs to.
    sequence_id: Mapped[int] = mapped_column(Integer, nullable=False) # The ID to make spans ordered within a single attempt.

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: Mapped[str] = mapped_column(String, nullable=False) # one rollout can have traces coming from multiple places

    # FIXME: span_id may be not unique across different attempts/rollouts, use (rollout_id, attempt_id, sequence_id) as the primary key instead
    span_id: Mapped[str] = mapped_column(String, nullable=False) # The span ID of the span. This ID comes from the OpenTelemetry span ID generator.
    parent_id: Mapped[Optional[str]] = mapped_column(String, nullable=True) # The parent span ID of the span.

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
        # FIXME extra field is not included yet
        dic = {k: getattr(self, k) for k in self.__table__.columns.keys() if k != "extra"}
        if self.extra is not None:
            dic.update(self.extra)
        return Span(**dic)

    @classmethod
    async def add_span(cls: type[SpanInDB], session_factory: async_sessionmaker[AsyncSession], span: Dict[str, Any], seq_id: Optional[int] = None) -> Span:
        """Add a new span to the database."""
        if seq_id is not None:
            span['sequence_id'] = seq_id
        extra_dic: Dict[str, Any] = {}
        for k in list(span.keys()):
            if k not in cls.__table__.columns.keys():
                extra_dic[k] = span.pop(k)
        span["extra"] = extra_dic if extra_dic else None

        async with session_factory() as session:
            async with session.begin():
                # create SpanInDB object
                span_obj = cls(**span)
                session.add(span_obj)
                # update attempt's last_heartbeat_time and status
                attempt_obj = await session.get(AttemptInDB, span["attempt_id"])
                if attempt_obj is None:
                    raise ValueError(f"AttemptInDB not found for attempt_id={span['attempt_id']}")
                # ensure the attempt and rollout are in running status
                if attempt_obj.status in ["preparing", "requeuing"]:
                    attempt_obj.status = "running"
                attempt_obj.last_heartbeat_time = time.time()
                # update rollout status if needed
                await session.execute(
                    update(RolloutInDB)
                    .where(
                        RolloutInDB.rollout_id == span["rollout_id"],
                        RolloutInDB.latest_attempt_id == span["attempt_id"],
                        RolloutInDB.status.in_(["preparing", "requeuing"]),
                    )
                    .values(status="running")
                )
                await session.flush()  # ensure the object is written to the DB
                return span_obj.as_span()

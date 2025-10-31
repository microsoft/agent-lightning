# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import uuid
import hashlib

from agentlightning.types import Attempt
from .base import SqlAlchemyBase
from sqlalchemy import String, Integer, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


def _generate_attempt_id() -> str:
    """We don't need that long because attempts are limited to rollouts."""
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:8]
    return "at-" + short_id


class AttemptInDB(SqlAlchemyBase):
    __tablename__ = "attempts"

    rollout_id: Mapped[str] = mapped_column(String, nullable=False)
    attempt_id: Mapped[str] = mapped_column(String, primary_key=True, default_factory=_generate_attempt_id)
    sequence_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_time: Mapped[float] = mapped_column(Float, default_factory=time.time, nullable=False)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    status: Mapped[str] = mapped_column(String, default="preparing", nullable=False)
    worker_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, default=None)
    last_heartbeat_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    attempt_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True, default=None)

    def as_attempt(self) -> Attempt:
        return Attempt(
            rollout_id=self.rollout_id,
            attempt_id=self.attempt_id,
            sequence_id=self.sequence_id,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,  # type: ignore
            worker_id=self.worker_id,
            last_heartbeat_time=self.last_heartbeat_time,
            metadata=self.attempt_metadata if self.attempt_metadata is not None else {},
        )

    @classmethod
    async def get_latest_attempt_for_rollout(cls: type[AttemptInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str) -> Optional[Attempt]:
        async with session_factory() as session:
            async with session.begin():
                result = await session.scalars(
                    select(cls)
                    .where(cls.rollout_id == rollout_id)
                    .order_by(cls.sequence_id.desc())
                    .limit(1)
                )
                attempt_obj = result.one_or_none()
                if attempt_obj is None:
                    return None
                return attempt_obj.as_attempt()


    @classmethod
    async def get_attempts_for_rollout(cls: type[AttemptInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str) -> List[Attempt]:
        async with session_factory() as session:
            async with session.begin():
                result = await session.scalars(
                    select(cls)
                    .where(cls.rollout_id == rollout_id)
                    .order_by(cls.sequence_id.asc())
                )
                return [attempt.as_attempt() for attempt in result.all()]




class SpanSeqIdInDB(SqlAlchemyBase):
    __tablename__ = "span_sequence"

    rollout_id: Mapped[str] = mapped_column(nullable=False)

    # FIXME InMemoryLightningStore let all attempts under the same rollout share the same span sequence for sorting
    # attempt_id: Mapped[str] = mapped_column(nullable=False)
    attempt_id: str # not mapped column, just for type hinting

    current_sequence: Mapped[int] = mapped_column(default=0, nullable=False)

    # Versioning for optimistic concurrency control
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {
        "version_id_col": version_id,
        # "primary_key": [rollout_id, attempt_id],
        "primary_key": [rollout_id],
    }

    @classmethod
    async def get_next_sequence_id(cls: type[SpanSeqIdInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str, attempt_id: str) -> int:
        """Get the next sequence ID with retries to handle race conditions.
        """
        async with session_factory() as session:
            async with session.begin():
                seq_obj = await session.get(cls, rollout_id)
                # seq_obj = await session.get(cls, [rollout_id, attempt_id])
                if seq_obj is None:
                    raise ValueError(f"SpanSeqIdInDB not found for rollout_id={rollout_id}, attempt_id={attempt_id}")
                else:
                    seq_obj.current_sequence += 1
                    await session.flush()
                    return seq_obj.current_sequence  # type: int
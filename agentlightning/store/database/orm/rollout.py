# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, cast
import time
import uuid
import hashlib

from sqlalchemy import String, Integer, Float, JSON
from sqlalchemy import update, and_
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentlightning.types import Rollout, RolloutConfig, Attempt, AttemptedRollout
from agentlightning.types.core import StatusDescription
from .base import PydanticInDB, SqlAlchemyBase
from .attempt import AttemptInDB, SpanSeqIdInDB


def _generate_rollout_id() -> str:
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]
    return "ro-" + short_id


class RolloutConfigInDB(PydanticInDB):
    """Custom SQLAlchemy type to store RolloutConfig as JSON in the database."""

    target_type = RolloutConfig


class RolloutInDB(SqlAlchemyBase):
    __tablename__ = "rollouts"

    input: Mapped[Any] = mapped_column(JSON, nullable=False)
    rollout_id: Mapped[str] = mapped_column(String, primary_key=True, default_factory=_generate_rollout_id)
    start_time: Mapped[float] = mapped_column(Float, default_factory=time.time, nullable=False)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    mode: Mapped[Optional[str]] = mapped_column(String, nullable=True, default=None)
    resources_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, default=None)
    status: Mapped[str] = mapped_column(String, default="queuing", nullable=False)
    config: Mapped[Optional[RolloutConfig]] = mapped_column(RolloutConfigInDB, nullable=True, default=None)  # JSON serialized, convert to RolloutConfig when needed
    rollout_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True, default=None) # JSON serialized, convert to Dict when needed

    # Attempt-related helper methods can be added here if needed
    num_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # number of attempts made for this rollout
    latest_attempt_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, default=None)  # the attempt_id of the latest attempt
    enqueue_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default_factory=time.time)  # time when the rollout was enqueued (for FIFO scheduling)

    # use optimistic concurrency control
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {
        "version_id_col": version_id,
    }

    def as_rollout(self) -> Rollout:
        return Rollout(
            rollout_id=self.rollout_id,
            input=self.input,
            start_time=self.start_time,
            end_time=self.end_time,
            mode=self.mode, # type: ignore
            resources_id=self.resources_id,
            status=self.status, # type: ignore
            config=self.config if self.config is not None else RolloutConfig(),
            metadata=self.rollout_metadata if self.rollout_metadata is not None else {},
        )

    @classmethod
    async def get_rollout_by_id(cls: type[RolloutInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str) -> Optional[Rollout]:
        """Query a specific rollout from the database."""
        async with session_factory() as session:
            async with session.begin():
                rollout_obj = await session.get(cls, rollout_id)
                if rollout_obj is None:
                    return None
                return rollout_obj.as_rollout()

    @classmethod
    async def query_rollouts(cls: type[RolloutInDB], session_factory: async_sessionmaker[AsyncSession], *, statuses: Optional[List[str]] = None, ids: Optional[List[str]] = None) -> List[Rollout]:
        """
        Query rollouts from the database with optional filters.
        """
        async with session_factory() as session:
            async with session.begin():
                conditions :list[Any] = []
                if statuses is not None:
                    conditions.append(cls.status.in_(statuses))
                if ids is not None:
                    conditions.append(cls.rollout_id.in_(ids))
                query = select(cls)
                if conditions:
                    query = query.where(and_(*conditions))
                result = await session.scalars(query)
                rollout_objs = result.all()
                return [obj.as_rollout() for obj in rollout_objs]

    @classmethod
    async def fifo_dequeue_rollout(cls: type[RolloutInDB], session_factory: async_sessionmaker[AsyncSession]) -> Optional[AttemptedRollout]:
        """Dequeue the next rollout in FIFO order (the one with the earliest enqueue_time).
        Returns the RolloutInDB object if found, else None.
        Note: This method does not update the status of the rollout. The caller should handle that.
        """
        async with session_factory() as session:
            async with session.begin():
                # use the update...returning to atomically select the next rollout and claim it by updating its status to 'preparing'
                result = await session.scalars(
                    select(cls)
                    .where(cls.status.in_(StatusDescription.queuing_statuses), cls.enqueue_time.isnot(None))
                    .order_by(cls.enqueue_time.asc())
                    .limit(1)
                )
                rollout_obj = result.one_or_none()
                if rollout_obj is None:
                    return None  # no rollout available
                # update the status of the rollout to 'preparing' via Compare-and-Swap to avoid race
                attempted_rollout = cls.start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    @classmethod
    def start_attempt_for_rollout(cls, session: AsyncSession, rollout_obj: RolloutInDB) -> AttemptedRollout:
        """Create a new attempt for the given rollout and update the rollout's fields."""
        # create a new attempt for this rollout
        attempt_obj = AttemptInDB(
            rollout_id=rollout_obj.rollout_id,
            sequence_id=rollout_obj.num_attempts + 1,
            status="preparing",
        )
        session.add(attempt_obj)
        # pre-update the rollout_obj fields for CAS
        rollout_obj.status = "preparing"  # pre-update the status in the object for CAS
        rollout_obj.enqueue_time = None  # pre-update the enqueue_time in the object for CAS
        rollout_obj.num_attempts += 1  # pre-update the num_attempts in the object for CAS
        rollout_obj.latest_attempt_id = attempt_obj.attempt_id  # pre-update the latest_attempt_id in the object for CAS

        # create a sequence id tracker for each attempt
        seq_obj = SpanSeqIdInDB(
            rollout_id=rollout_obj.rollout_id,
            attempt_id=attempt_obj.attempt_id,
            current_sequence=0,
        )
        session.add(seq_obj)

        return AttemptedRollout(**rollout_obj.as_rollout().model_dump(), attempt=attempt_obj.as_attempt())


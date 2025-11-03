# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import uuid
import hashlib

from sqlalchemy import String, Integer, Float, JSON
from sqlalchemy import and_
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentlightning.types import Rollout, RolloutConfig
from .base import PydanticInDB, SqlAlchemyBase
from ...base import is_finished, is_queuing


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

    def _validate_status_message(self, msg: Dict[str, str]) -> None:
        """Validate the status update message.
        Raises:
            ValueError: If the message is invalid.
        """
        if "event" not in msg:
            raise ValueError("Status update message must contain 'event' field.")
        event = msg["event"]
        if event not in [
            "attempt_status_update", # from attempt status update
            "user_update",         # from user-initiated update
        ]:
            raise ValueError(f"Invalid event type in status update message: {event}")
        if event == "user_update":
            if "new_status" not in msg:
                raise ValueError("Status update message for event 'user_update' must contain 'new_status' field.")
        if event == "attempt_status_update":
            for field in ["new_status", "old_status", "attempt_id", "is_failed"]:
                if field not in msg:
                    raise ValueError(f"Status update message for event '{event}' must contain '{field}' field.")

    def update_status(self, msg: Dict[str, Any]) -> None:
        """Update the rollout status based on the provided message.
        Args:
            msg (Dict[str, str]): The status update message. Refer to `_validate_status_message` for the expected format.
            current_time (Optional[float]): The current time to set end_time or enqueue_time if needed.
        """
        self._validate_status_message(msg)
        event, old_status, new_status = msg["event"], self.status, None
        current_time = msg.get("timestamp", time.time())

        # Step 1: Determine the new status based on the event
        if event == "user_update":
            new_status = msg["new_status"]
        elif event == "attempt_status_update":
            if msg["attempt_id"] != self.latest_attempt_id:
                # outdated attempt status update, ignore
                # TODO if latest attempt fails but an older attempt still running or succeed, we may need to handle that
                return
            else:
                attempt_new_status = msg["new_status"]
                if msg["is_failed"]:
                    # attempt failed
                    config = self.config if self.config is not None else RolloutConfig()
                    if attempt_new_status in config.retry_condition and config.max_attempts > self.num_attempts:
                        new_status = "requeuing"
                    else:
                        new_status = "failed"
                elif attempt_new_status == "running":
                    if old_status in ["preparing", "requeuing"]:
                        new_status = "running"
                else:
                    new_status = attempt_new_status

        # Step 2: Update the status if it has changed and handle follow-up actions
        if new_status is None:
            raise RuntimeError("New status could not be determined from the message.")
        if new_status == old_status:
            return
        self.status = new_status

        if is_finished(self): # type: ignore
            self.end_time = current_time
        if is_queuing(self): # type: ignore
            self.enqueue_time = current_time
            # When requeuing, we do not reset latest_attempt_id or num_attempts,
            # as they should persist across requeues.

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


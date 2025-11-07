# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import JSON, Float, Integer, String, and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from agentlightning.types import AttemptedRollout, Rollout, RolloutConfig, RolloutStatus

from ...base import is_finished, is_queuing
from .attempt import AttemptInDB
from .base import AttemptStatusUpdateMessage, PydanticInDB, SqlAlchemyBase

logger = logging.getLogger(__name__)


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
    status: Mapped[RolloutStatus] = mapped_column(String, default="queuing", nullable=False)
    config: Mapped[RolloutConfig] = mapped_column(
        RolloutConfigInDB, nullable=False, default_factory=RolloutConfig
    )  # JSON serialized, convert to RolloutConfig when needed
    rollout_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, default=None
    )  # JSON serialized, convert to Dict when needed

    # Attempt-related helper methods can be added here if needed
    num_attempts: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )  # number of attempts made for this rollout
    enqueue_time: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, default_factory=time.time
    )  # time when the rollout was enqueued (for FIFO scheduling)
    latest_attempt_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, default=None
    )  # the attempt_id of the latest attempt

    # use optimistic concurrency control
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {
        "version_id_col": version_id,
    }

    def __post_init__(self):
        if self.status not in ["queuing", "running", "succeeded", "failed", "requeuing"]:
            raise ValueError(f"Invalid rollout status: {self.status}")

    def as_rollout(self) -> Rollout:
        return Rollout(
            **self.model_dump(
                exclude={"rollout_metadata", "num_attempts", "enqueue_time", "latest_attempt_id", "version_id"},
                mapper={
                    "metadata": lambda obj: obj.rollout_metadata,  # type: ignore
                    "config": lambda obj: obj.config if obj.config is not None else RolloutConfig(),  # type: ignore
                },
            )
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
            "attempt_status_update",  # from attempt status update
            "user_update",  # from user-initiated update
        ]:
            raise ValueError(f"Invalid event type in status update message: {event}")
        if event == "user_update":
            if "new_status" not in msg:
                raise ValueError("Status update message for event 'user_update' must contain 'new_status' field.")
        if event == "attempt_status_update":
            # leverage AttemptStatusUpdateMessage for validation
            pass

    async def update_status(self, msg: Dict[str, Any] | AttemptStatusUpdateMessage) -> None:
        """Update the rollout status based on the provided message.
        Args:
            msg (Dict[str, str]): The status update message. Refer to `_validate_status_message` for the expected format.
            current_time (Optional[float]): The current time to set end_time or enqueue_time if needed.
        """
        if isinstance(msg, dict):
            self._validate_status_message(msg)
            event = msg["event"]
            current_time = msg.get("timestamp", time.time())
        else:
            event = msg.event
            current_time = msg.timestamp

        old_status = self.status
        new_status = self.status  # initialize new_status with old_status

        # Step 1: Determine the new status based on the event
        if event == "user_update":
            assert isinstance(msg, dict)
            new_status = msg["new_status"]
        elif event == "attempt_status_update":
            msg = AttemptStatusUpdateMessage(**msg) if isinstance(msg, dict) else msg
            if msg.attempt_id == self.latest_attempt_id:
                new_status = msg.new_status  # directly take the latest attempt status
                if msg.is_succeeded:
                    new_status = "succeeded"
                elif msg.is_failed:
                    # no other attempts running, decide whether to requeue or fail
                    config = self.config
                    if config.max_attempts > self.num_attempts and msg.new_status in config.retry_condition:
                        new_status = "requeuing"
                    else:
                        new_status = "failed"
                # elif msg.is_running and old_status in ["failed", "requeuing"]:
                # new_status = "running"
            else:
                # ignore attempts from old attempts
                new_status = old_status

        # Step 2: Update the status if it has changed and handle follow-up actions
        if new_status is None:
            raise RuntimeError(
                f"New status of `{old_status}` and `{self.latest_attempt_id}` could not be determined from the message {msg}."
            )
        if new_status == old_status:
            return
        self.status = cast(RolloutStatus, new_status)

        if is_finished(self):  # type: ignore
            self.end_time = current_time
        if is_queuing(self):  # type: ignore
            self.enqueue_time = current_time
            # When requeuing, we do not reset latest_attempt_id or num_attempts,
            # as they should persist across requeues.

    @classmethod
    async def get_rollout_by_id(
        cls: type[RolloutInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str
    ) -> Optional[Rollout | AttemptedRollout]:
        """Query a specific rollout from the database."""
        async with session_factory() as session:
            async with session.begin():
                rollout_obj = await session.get(cls, rollout_id)
                if rollout_obj is None:
                    return None
                if rollout_obj.latest_attempt_id is not None:
                    attempt_obj = await session.get(AttemptInDB, rollout_obj.latest_attempt_id)
                    if attempt_obj is not None:
                        return AttemptedRollout(
                            **rollout_obj.as_rollout().model_dump(), attempt=attempt_obj.as_attempt()
                        )
                return rollout_obj.as_rollout()

    @classmethod
    async def query_rollouts(
        cls: type[RolloutInDB],
        session_factory: async_sessionmaker[AsyncSession],
        *,
        statuses: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[RolloutInDB]:
        """
        Query rollouts from the database with optional filters.
        """
        async with session_factory() as session:
            async with session.begin():
                conditions: list[Any] = []
                if statuses is not None:
                    conditions.append(cls.status.in_(statuses))
                if ids is not None:
                    conditions.append(cls.rollout_id.in_(ids))
                query = select(cls)
                if conditions:
                    query = query.where(and_(*conditions))
                result = await session.scalars(query)
                rollout_objs = result.all()
                return list(rollout_objs)

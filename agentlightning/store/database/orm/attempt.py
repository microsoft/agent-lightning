# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import uuid
import hashlib
import logging
from dataclasses import InitVar
from sqlalchemy import String, Integer, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentlightning.types import Attempt
from .base import SqlAlchemyBase

logger = logging.getLogger(__name__)


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
    last_heartbeat_time: Mapped[Optional[float]] = mapped_column(Float, nullable=False, default_factory=time.time)
    attempt_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True, default=None)

    # addition columns for processing
    max_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)  # maximum duration allowed for this attempt in seconds
    max_heartbeat_interval: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)  # maximum allowed heartbeat interval in seconds

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

    def _validate_status_message(self, msg: Dict[str, Any]) -> None:
        """This function validates the status update message from caller.
        Raises ValueError if the message is invalid.
        """
        if "event" not in msg:
            raise ValueError("Status update message must contain 'event' field.")
        if "timestamp" not in msg:
            msg["timestamp"] = time.time()
        if msg["event"] not in [
            "user_update", # user update attempt status via dbstore.update_attempt()
            "span_received", # new span received
            "single_step_timeout", # single step timeout detected (from last span heartbeat)
            "overall_timeout", # overall timeout detected
        ]:
            raise ValueError(f"Unsupported event type: {msg['event']}")
        if msg["event"] == "user_update" and "new_status" not in msg:
            raise ValueError("User update event must contain 'new_status' field.")

    def get_finished_statuses(self) -> List[str]:
        """This function returns the list of statuses that are considered finished.
        """
        return [
            "succeeded",
            "failed",
            "timeout",
        ]

    def update_status(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """This function updates the status of the attempt based on the event.
        Args:
            msg: A dictionary containing the status update message. It must contain an "event" field, and optionally a "new_status" field.
                More details about the message format can be found in the `_validate_status_message`() method.
            current_time: The current time to use for updating timestamps. If None, uses time.time().
        Returns:
            A dictionary containing the status update message: {"event": "attempt_status_updated", "old_status": old_status, "new_status": new_status}.
                IF no meaningful status update is performed, returns None.
        Raises:
            ValueError: If the event is not recognized or the status transition is invalid.
            NotImplementedError: If the event handling is not implemented for the current status.
            RuntimeError: If the new status is not set after processing the event.
        """
        self._validate_status_message(msg)
        event = msg["event"]
        current_time = msg.get("timestamp", time.time())
        old_status = self.status
        new_status = msg.get("new_status", None)

        # Step 1: Determine the new status based on the event and current status
        if event == "user_update":
            if not new_status:
                raise ValueError("new_status must be provided for user_update event.")
        elif event == "span_received":
            self.last_heartbeat_time = current_time
            if old_status in ["preparing", "unresponsive", "running"]:
                new_status = "running"
            elif old_status in self.get_finished_statuses():
                logger.warning(f"Span received after attempt is already in status {self.status}. No status update performed.")
                return # no further status update needed
            else:
                raise NotImplementedError(f"Event {event} is not implemented for status {old_status}.")
        elif event == "single_step_timeout":
            if old_status in ["preparing", "running", ]:
                new_status = "unresponsive"
            else:
                logger.warning(f"Single step timeout detected but attempt is in status {self.status}. No status update performed.")
                return # no further status update needed
        elif event == "overall_timeout":
            if old_status not in self.get_finished_statuses():
                new_status = "timeout"
            else:
                logger.warning(f"Overall timeout detected but attempt is in status {self.status}. No status update performed.")
                return # no further status update needed
        else:
            raise NotImplementedError(f"Event {event} is not implemented for status update.")

        # Step 2: Update the status
        if not new_status:
            raise RuntimeError(f"new_status should not be {new_status} after processing event for {event} on status {old_status}.")
        if new_status == old_status:
            return # no status change
        if new_status in self.get_finished_statuses():
            # when attempt is finished, set end_time
            self.end_time = current_time
        self.status = new_status

        # Step 3: Return the status update info for further processing
        return {
            "event": "attempt_status_update",
            "timestamp": current_time,
            "old_status": old_status,
            "new_status": new_status,
            "attempt_id": self.attempt_id,
            "is_failed": new_status in ["failed", "timeout", "unresponsive"],
        }

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

    rollout_id: Mapped[str] = mapped_column(nullable=False, primary_key=True)

    # FIXME InMemoryLightningStore let all attempts under the same rollout share the same span sequence for sorting
    # attempt_id: Mapped[str] = mapped_column(nullable=False)
    attempt_id: InitVar[str] # not mapped column, just for type hinting

    current_sequence: Mapped[int] = mapped_column(default=1, nullable=False)

    # Versioning for optimistic concurrency control
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {
        "version_id_col": version_id,
        # "primary_key": [rollout_id, attempt_id],
        # "primary_key": [rollout_id],
    }

    @classmethod
    async def get_next_sequence_id(cls: type[SpanSeqIdInDB], session_factory: async_sessionmaker[AsyncSession], rollout_id: str, attempt_id: str, external_seq_id: Optional[int] = None) -> int:
        """Get the next sequence ID with retries to handle race conditions.
        IF external_seq_id is provided and is greater than current_sequence, set current_sequence to external_seq_id.
        """
        async with session_factory() as session:
            async with session.begin():
                seq_obj = await session.get(cls, rollout_id)
                # seq_obj = await session.get(cls, [rollout_id, attempt_id])
                if seq_obj is None:
                    raise ValueError(f"Rollout {rollout_id} not found")
                else:
                    current_seq = external_seq_id if external_seq_id is not None and external_seq_id > seq_obj.current_sequence else seq_obj.current_sequence
                    seq_obj.current_sequence = current_seq + 1
                    await session.flush()
                    return current_seq
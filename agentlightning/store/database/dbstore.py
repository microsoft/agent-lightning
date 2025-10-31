# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
import time
from opentelemetry.sdk.trace import ReadableSpan
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from tenacity import AsyncRetrying, stop_before_delay, wait_exponential_jitter
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar



from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TaskInput,
)

from agentlightning.types.core import StatusDescription

from ..base import UNSET, LightningStore, Unset
from .sqlite import RolloutInDB, AttemptInDB, ResourcesUpdateInDB, SpanInDB, SpanSeqIdInDB
from .orm import SqlAlchemyBase
from .utils import register_retry_config

logger = logging.getLogger(__name__)

# TODO add periodic heartbeat checker for attempts and timeout watchdog
# TODO add retry decorators to dbstore operations
# TODO add periodic cleanup of old rollouts/attempts/spans


class DatabaseLightningStore(LightningStore):
    """
    A LightningStore implementation that uses a database backend to store and manage rollouts and attempts.
    The database backend is expected to support asynchronous operations.
    The store uses SQLAlchemy ORM models to interact with the database
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        retry_config: Optional[dict[str, Any]] = None,
        watchdog_mode: Literal["thread", "asyncio"] = "asyncio",
    ) -> None:
        super().__init__()
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", None)
        if database_url is None:
            raise ValueError("A database URL must be provided either via the 'database_url' parameter or the 'DATABASE_URL' environment variable.")

        self._engine = create_async_engine(database_url, echo=False)
        self._async_session = async_sessionmaker(self._engine, expire_on_commit=False)
        if retry_config is not None:
            register_retry_config("dbstore", retry_config)
            # FIXME add retry to dbstore operations
        self._latest_resources_id = None

    async def start(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(SqlAlchemyBase.metadata.create_all)

    async def stop(self):
        await self._engine.dispose()

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = RolloutInDB(
                    input=input,
                    mode=mode,
                    resources_id=resources_id or self._latest_resources_id,
                    status="queuing",
                    config=config,
                    rollout_metadata=metadata,
                )
                session.add(rollout_obj)
                attempted_rollout = RolloutInDB.start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = RolloutInDB(
                    input=input,
                    mode=mode,
                    resources_id=resources_id or self._latest_resources_id,
                    status="queuing",
                    config=config,
                    rollout_metadata=metadata,
                )
                session.add(rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return rollout_obj.as_rollout()

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await RolloutInDB.fifo_dequeue_rollout(self._async_session)

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = await session.get(RolloutInDB, rollout_id)
                if rollout_obj is None:
                    raise ValueError(f"Rollout {rollout_id} does not exist. Cannot start new attempt.")
                attempted_rollout = RolloutInDB.start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    async def add_span(self, span: Span) -> Span:
        seq_id = await SpanSeqIdInDB.get_next_sequence_id(self._async_session, span.rollout_id, span.attempt_id)
        return await SpanInDB.add_span(self._async_session, span.model_dump(), seq_id=seq_id)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)
        span = Span.from_opentelemetry(
            src=readable_span,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
        )
        return await SpanInDB.add_span(self._async_session, span.model_dump(), seq_id=sequence_id)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        return await RolloutInDB.query_rollouts(self._async_session, statuses=status, ids=rollout_ids) # type: ignore

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await AttemptInDB.get_attempts_for_rollout(self._async_session, rollout_id) # type: ignore

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        return await RolloutInDB.get_rollout_by_id(self._async_session, rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await AttemptInDB.get_latest_attempt_for_rollout(self._async_session, rollout_id)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await ResourcesUpdateInDB.get_resources_by_id(self._async_session, resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        if self._latest_resources_id is None:
            return None
        return await ResourcesUpdateInDB.get_resources_by_id(self._async_session, self._latest_resources_id)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await SpanSeqIdInDB.get_next_sequence_id(self._async_session, rollout_id, attempt_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        # implementation the timeout via tenacity retry mechanism, by a `with` context
        wait_min = 0.1 if timeout is None else min(0.1, timeout / 10) # at least one tenth of the timeout or 0.1s
        wait_max = 60 if timeout is None else max(60, timeout / 2) # at most half of the timeout or 60s
        retry_config: Dict[str, Any] = {
            "wait": wait_exponential_jitter(initial=wait_min, max=wait_max, jitter=0.1 * wait_min),
            "reraise": True,
        }
        if timeout is not None:
            retry_config["stop"] = stop_before_delay(timeout)
        async for retry_attempt in AsyncRetrying(**retry_config):
            with retry_attempt:
                async with self._async_session() as session:
                    async with session.begin():
                        result = await session.scalars(
                            select(RolloutInDB).where(RolloutInDB.rollout_id.in_(rollout_ids))
                        )
                        rollouts = result.all()
                        if len(rollouts) != len(rollout_ids):
                            existing_ids = {rollout.rollout_id for rollout in rollouts}
                            missing_ids = set(rollout_ids) - existing_ids
                            raise ValueError(f"Some rollouts do not exist: {missing_ids}")
                        if all(
                            rollout.status in StatusDescription.finishing_statuses
                            for rollout in rollouts
                        ):
                            return [rollout.as_rollout() for rollout in rollouts]
                        else:
                            raise Exception("Not all rollouts have reached terminal status yet.")


    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        async with self._async_session() as session:
            async with session.begin():
                conditions: List[Any] = [SpanInDB.rollout_id == rollout_id]
                if attempt_id is not None:
                    if attempt_id == "latest":
                        rollout_obj = await session.get(RolloutInDB, rollout_id)
                        if rollout_obj is None:
                            raise ValueError(f"Rollout {rollout_id} does not exist. Cannot query latest attempt spans.")
                        attempt_id = rollout_obj.latest_attempt_id
                    conditions.append(SpanInDB.attempt_id == attempt_id)
                query = select(SpanInDB).where(and_(*conditions)).order_by(SpanInDB.sequence_id.asc())
                result = await session.scalars(query)
                span_objs = result.all()
                return [obj.as_span() for obj in span_objs]

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        async with self._async_session() as session:
            async with session.begin():
                resource_obj = ResourcesUpdateInDB(
                    resources=resources,
                )
                session.add(resource_obj)
                await session.flush()  # ensure the object is written to the DB
                self._latest_resources_id = resource_obj.resources_id
                return resource_obj.as_resources_update()

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        async with self._async_session() as session:
            async with session.begin():
                obj = await session.get(ResourcesUpdateInDB, resources_id)
                if obj is None:
                    # raise ValueError(f"Failed to update resources {resources_id}. It may not exist.")
                    # FIXME InMemoryLightningStore will create the resources if not exist, but the base method require to raise error
                    # HACK here stick to the behavior of InMemoryLightningStore for compatibility
                    obj = ResourcesUpdateInDB(
                        resources_id=resources_id,
                        resources=resources,
                    )
                    session.add(obj)
                else:
                    obj.resources = resources
                await session.flush()
                self._latest_resources_id = resources_id
                return obj.as_resources_update()

    async def update_rollout(
        self,
        rollout_id: str|None,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        if rollout_id is None:
            raise ValueError("rollout_id must be provided for updating a rollout.")

        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = await session.get(RolloutInDB, rollout_id)
                if rollout_obj is None:
                    raise ValueError(f"Rollout {rollout_id} does not exist and cannot be updated.")
                # udpate fields
                if not isinstance(input, Unset):
                    rollout_obj.input = input
                if not isinstance(mode, Unset):
                    rollout_obj.mode = mode
                if not isinstance(resources_id, Unset):
                    rollout_obj.resources_id = resources_id
                if not isinstance(status, Unset):
                    rollout_obj.status = status
                    descriptor = StatusDescription()
                    if status in descriptor.finishing_statuses:
                        rollout_obj.end_time = time.time()
                    if status in descriptor.queuing_statuses:
                        rollout_obj.enqueue_time = time.time()
                    if status in descriptor.statuses_from_rollout_to_attempt:
                        # propagate to latest attempt
                        latest_attempt = await session.get(AttemptInDB, rollout_obj.latest_attempt_id)
                        if latest_attempt is not None:
                            latest_attempt.status = status
                            if status in descriptor.finishing_statuses:
                                latest_attempt.end_time = rollout_obj.end_time
                if not isinstance(config, Unset):
                    rollout_obj.config = config
                if not isinstance(metadata, Unset):
                    rollout_obj.rollout_metadata = metadata
                await session.flush()  # ensure the object is written to the DB
                return rollout_obj.as_rollout()

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = await session.get(RolloutInDB, rollout_id)
                if rollout_obj is None:
                    raise ValueError(f"Rollout {rollout_id} does not exist.")
                if attempt_id == "latest":
                    if rollout_obj.latest_attempt_id is None:
                        raise ValueError(f"Rollout {rollout_id} has no attempts. Cannot update latest attempt.")
                    attempt_id = rollout_obj.latest_attempt_id
                if attempt_id != rollout_obj.latest_attempt_id:
                    logger.warning(f"Updating attempt {attempt_id} which is not the latest attempt for rollout {rollout_id}. Latest is {rollout_obj.latest_attempt_id}.")
                attempt_obj = await session.get(AttemptInDB, attempt_id)
                if attempt_obj is None:
                    raise ValueError(f"Attempt {attempt_id} for rollout {rollout_id} does not exist.")
                if attempt_obj.rollout_id != rollout_id:
                    raise ValueError(f"Attempt {attempt_id} does not belong to rollout {rollout_id}.")
                # update fields
                if not isinstance(status, Unset):
                    attempt_obj.status = status
                    descriptor = StatusDescription()
                    if status in descriptor.finishing_statuses:
                        attempt_obj.end_time = time.time()
                        # propagate to rollout if this is the latest attempt
                        # FIXME should comply with th propagate_status() of InMemoryLightningStore
                        rollout_obj.status = status
                        if status in descriptor.finishing_statuses:
                            rollout_obj.end_time = attempt_obj.end_time
                if not isinstance(worker_id, Unset):
                    attempt_obj.worker_id = worker_id
                if not isinstance(last_heartbeat_time, Unset):
                    attempt_obj.last_heartbeat_time = last_heartbeat_time
                if not isinstance(metadata, Unset):
                    attempt_obj.attempt_metadata = metadata
                await session.flush()  # ensure the object is written to the DB
                return attempt_obj.as_attempt()

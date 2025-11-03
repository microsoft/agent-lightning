# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
import time
from opentelemetry.sdk.trace import ReadableSpan
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from tenacity import (
    AsyncRetrying, RetryError, stop_before_delay, wait_exponential_jitter,
)
from typing import Any, Dict, List, Literal, Optional, Sequence

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

from ..base import UNSET, LightningStore, Unset, is_finished
from .orm import SqlAlchemyBase
from .sqlite import RolloutInDB, AttemptInDB, ResourcesUpdateInDB, SpanInDB, SpanSeqIdInDB
from .retry_helper import RetryStrategy, ExceptionRegistry, AsyncTypeBasedRetry

logger = logging.getLogger(__name__)

# TODO add periodic heartbeat checker for attempts and timeout watchdog
# TODO add periodic cleanup of old rollouts/attempts/spans

ExceptionRegistry.register("sqlalchemy.orm.exc.StaleDataError")
ExceptionRegistry.register("sqlalchemy.exc.OperationalError")

db_retry = AsyncTypeBasedRetry({
    "sqlalchemy.exc.OperationalError": RetryStrategy(max_attempts=5, wait_seconds=1, backoff=1.5, jitter=0.3, log=True),
    "sqlalchemy.orm.exc.StaleDataError": RetryStrategy(max_attempts=100, wait_seconds=1e-3, backoff=1.0, jitter=0.1, log=True)
})


class DatabaseLightningStore(LightningStore):
    """
    A LightningStore implementation that uses a database backend to store and manage rollouts and attempts.
    The database backend is expected to support asynchronous operations.
    The store uses SQLAlchemy ORM models to interact with the database
    Args:
        database_url: The database connection URL. If not provided, it will be read from the 'DATABASE_URL' environment variable.
        watchdog_mode: The mode for the watchdog that monitors long-running attempts. Can be 'thread' or 'asyncio'.
        dequeue_strategy: The strategy to dequeue rollouts. Currently only 'fifo' is supported.
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

        self._latest_resources_id = None

    async def start(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(SqlAlchemyBase.metadata.create_all)

    async def stop(self):
        await self._engine.dispose()

    @db_retry
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
                attempted_rollout = await self._start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    @db_retry
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

    # @retry(
    #     retry=retry_if_exception_type(StaleDataError),
    #     stop=stop_after_attempt(100),
    # )
    @db_retry
    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._fifo_dequeue_rollout()

    @db_retry
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = await session.get(RolloutInDB, rollout_id)
                if rollout_obj is None:
                    raise ValueError(f"Rollout {rollout_id} does not exist. Cannot start new attempt.")
                attempted_rollout = await self._start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    @db_retry
    async def add_span(self, span: Span) -> Span:
        seq_id = await SpanSeqIdInDB.get_next_sequence_id(self._async_session, span.rollout_id, span.attempt_id)
        return await self._add_span(span.model_dump(), seq_id=seq_id)

    @db_retry
    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        sequence_id = await SpanSeqIdInDB.get_next_sequence_id(self._async_session, rollout_id, attempt_id, sequence_id)
        span = Span.from_opentelemetry(
            src=readable_span,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
        )
        return await self._add_span(span.model_dump(), seq_id=sequence_id)

    @db_retry
    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        return await RolloutInDB.query_rollouts(self._async_session, statuses=status, ids=rollout_ids) # type: ignore

    @db_retry
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await AttemptInDB.get_attempts_for_rollout(self._async_session, rollout_id) # type: ignore

    @db_retry
    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        return await RolloutInDB.get_rollout_by_id(self._async_session, rollout_id)

    @db_retry
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await AttemptInDB.get_latest_attempt_for_rollout(self._async_session, rollout_id)

    @db_retry
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await ResourcesUpdateInDB.get_resources_by_id(self._async_session, resources_id)

    @db_retry
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        if self._latest_resources_id is None:
            return None
        return await ResourcesUpdateInDB.get_resources_by_id(self._async_session, self._latest_resources_id)

    @db_retry
    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await SpanSeqIdInDB.get_next_sequence_id(self._async_session, rollout_id, attempt_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        # implementation the timeout via tenacity retry mechanism, by a `with` context
        wait_min = 0.1 if timeout is None else min(0.1, timeout / 10) # at least one tenth of the timeout or 0.1s
        wait_max = 60 if timeout is None else min(60, timeout / 2) # at most half of the timeout or 60s
        retry_config: Dict[str, Any] = {
            "wait": wait_exponential_jitter(initial=wait_min, max=wait_max, jitter=0.1 * wait_min),
            "reraise": False,
        }
        if timeout is not None:
            retry_config["stop"] = stop_before_delay(timeout)
        logger.debug(f"wait_for_rollouts with the following retry config {retry_config}")
        time_start = time.time_ns()
        completed_rollouts: List[Rollout] = []
        try:
            async for retry_attempt in AsyncRetrying(**retry_config):
                with retry_attempt:
                    async with self._async_session() as session:
                        async with session.begin():
                            current_time = time.time_ns()
                            logger.debug(f"Begin to query rollouts at {(current_time - time_start)*1e-9} seconds")
                            result = await session.scalars(
                                select(RolloutInDB).where(RolloutInDB.rollout_id.in_(rollout_ids))
                            )
                            rollouts = result.all()
                            if len(rollouts) != len(rollout_ids):
                                existing_ids = {rollout.rollout_id for rollout in rollouts}
                                missing_ids = set(rollout_ids) - existing_ids
                                # FIXME ignore nonexisting rollout_ids to follow the behavior of InMemoryLightningStore
                                logger.warning(f"Some rollouts do not exist: {missing_ids}")
                                # raise ValueError(f"Some rollouts do not exist: {missing_ids}")
                            completed_rollouts = [
                                rollout.as_rollout() for rollout in rollouts
                                if is_finished(rollout)  # type: ignore
                            ]
                            if len(completed_rollouts) == len(rollout_ids):
                                return completed_rollouts
                            else:
                                raise Exception("Not all rollouts have reached terminal status yet.")
        except RetryError:
            return completed_rollouts

    @db_retry
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

    @db_retry
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

    @db_retry
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

    @db_retry
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
                    rollout_obj.update_status(dict(event="user_update", new_status=status))
                if not isinstance(config, Unset):
                    rollout_obj.config = config
                if not isinstance(metadata, Unset):
                    rollout_obj.rollout_metadata = metadata
                await session.flush()  # ensure the object is written to the DB
                return rollout_obj.as_rollout()

    @db_retry
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
                    msg = attempt_obj.update_status(dict(event="user_update", new_status=status))
                    if msg is not None:
                        rollout_obj.update_status(msg)
                if not isinstance(worker_id, Unset):
                    attempt_obj.worker_id = worker_id
                if not isinstance(last_heartbeat_time, Unset):
                    attempt_obj.last_heartbeat_time = last_heartbeat_time
                if not isinstance(metadata, Unset):
                    attempt_obj.attempt_metadata = metadata
                await session.flush()  # ensure the object is written to the DB
                return attempt_obj.as_attempt()

    # internal helper methods can be added here
    async def _add_span(self, span: Dict[str, Any], seq_id: Optional[int] = None) -> Span:
        """Add a new span to the database."""
        if seq_id is not None:
            span['sequence_id'] = seq_id
        extra_dic: Dict[str, Any] = {}
        for k in list(span.keys()):
            if k not in SpanInDB.__table__.columns.keys():
                extra_dic[k] = span.pop(k)
        span["extra"] = extra_dic if extra_dic else None

        async with self._async_session() as session:
            async with session.begin():
                # create SpanInDB object
                span_obj = SpanInDB(**span)
                session.add(span_obj)
                # update attempt's last_heartbeat_time and status
                attempt_obj = await session.get(AttemptInDB, span["attempt_id"])
                if attempt_obj is None:
                    raise ValueError(f"AttemptInDB not found for attempt_id={span['attempt_id']}")
                # ensure the attempt and rollout are in running status
                msg = attempt_obj.update_status(dict(event="span_received"))
                if msg is not None:
                    rollout_obj = await session.get(RolloutInDB, attempt_obj.rollout_id)
                    if rollout_obj is None:
                        raise ValueError(f"RolloutInDB not found for rollout_id={attempt_obj.rollout_id}")
                    rollout_obj.update_status(msg)
                await session.flush()  # ensure the object is written to the DB
                return span_obj.as_span()

    async def _fifo_dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """Dequeue the next rollout in FIFO order (the one with the earliest enqueue_time).
        Returns the RolloutInDB object if found, else None.
        Note: This method does not update the status of the rollout. The caller should handle that.
        """
        async with self._async_session() as session:
            async with session.begin():
                # use the update...returning to atomically select the next rollout and claim it by updating its status to 'preparing'
                result = await session.scalars(
                    select(RolloutInDB)
                    .where(RolloutInDB.status.in_(["queuing", "requeuing"]), RolloutInDB.enqueue_time.isnot(None))
                    .order_by(RolloutInDB.enqueue_time.asc())
                    .limit(1)
                )
                rollout_obj = result.one_or_none()
                if rollout_obj is None:
                    return None  # no rollout available
                # update the status of the rollout to 'preparing' via Compare-and-Swap to avoid race
                attempted_rollout = await self._start_attempt_for_rollout(session, rollout_obj)
                await session.flush()  # ensure the object is written to the DB
                return attempted_rollout

    async def _start_attempt_for_rollout(self, session: AsyncSession, rollout_obj: RolloutInDB) -> AttemptedRollout:
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
        # FIXME currently InMemoryLightningStore let all attempts under the same rollout share the same span sequence for sorting
        # create a sequence id tracker for this rollout, only if not exists
        existing = await session.get(SpanSeqIdInDB, rollout_obj.rollout_id)
        if existing is None:
            seq_obj = SpanSeqIdInDB(
                rollout_id=rollout_obj.rollout_id,
                attempt_id=attempt_obj.attempt_id,
            )
            session.add(seq_obj)

        return AttemptedRollout(**rollout_obj.as_rollout().model_dump(), attempt=attempt_obj.as_attempt())
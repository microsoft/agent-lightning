# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tenacity import RetryError

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
from .orm import SqlAlchemyBase, AttemptStatusUpdateMessage
from .retry_helper import AsyncRetryBlock, AsyncTypeBasedRetry, ExceptionRegistry, RetryStrategy
from .sqlite import AttemptInDB, ResourcesUpdateInDB, RolloutInDB, SpanInDB, SpanSeqIdInDB

logger = logging.getLogger(__name__)

# TODO add periodic cleanup of old rollouts/attempts/spans

ExceptionRegistry.register("sqlalchemy.orm.exc.StaleDataError")
ExceptionRegistry.register("sqlalchemy.exc.OperationalError")

db_retry = AsyncTypeBasedRetry({
    "sqlalchemy.exc.OperationalError": RetryStrategy(max_attempts=5, wait_seconds=1, backoff=1.5, jitter=0.3, log=True),
    "sqlalchemy.orm.exc.StaleDataError": RetryStrategy(max_attempts=100, wait_seconds=1e-3, backoff=1.0, jitter=0.1, log=True)
})


class _WaitForRolloutsCompleted(Exception):
    """Internal exception to signal that not all rollouts have completed yet."""
    pass


class BackgroundTaskConfig(BaseModel):
    name: str # unique name for the task
    method: str # method name to call, currently only supports methods of DatabaseLightningStore
    interval: Dict[Literal["seconds", "minutes", "hours"], float] # interval for the task
    is_async: bool = True # whether the task method is async, default to True


class DatabaseLightningStore(LightningStore):
    """
    A LightningStore implementation that uses a database backend to store and manage rollouts and attempts.
    The database backend is expected to support asynchronous operations.
    The store uses SQLAlchemy ORM models to interact with the database
    Args:
        database_url (string):
            The database URL for connecting to the database.
            If None, will read from the 'DATABASE_URL' environment variable.
        retry_for_waiting (RetryStrategy):
            Retry strategy for polling when waiting for rollouts to complete.
            If None, a default strategy will be used.
        wait_for_nonexistent_rollout (Bool):
            If True, when waiting for rollouts, will wait for all specified rollouts to complete, including non-existing ones.
            If False, will ignore non-existing rollouts as completed. (Default: False)
        background_tasks_cfg (list[Dict[str, Any]]):
            The configuration for in-process periodic tasks, following the definition of `BackgroundTaskConfig`.
            IF not provided (None as default), the dbstore will incorporate a default set of periodic tasks as follows:
            [
                BackgroundTaskConfig(name="check_attempt_timeout", method="check_attempt_timeout", interval={"seconds": 10.0}),
            ]
            To disable all periodic tasks, provide an empty list `[]`.
    Note:
        Explicitly use async `start()` and `stop()` methods to manage the database connection lifecycle.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        retry_for_waiting: Optional[dict[str, Any]|RetryStrategy] = None,
        wait_for_nonexistent_rollout: bool = False,
        background_tasks_cfg: list[Dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", None)
        if database_url is None:
            raise ValueError("A database URL must be provided either via the 'database_url' parameter or the 'DATABASE_URL' environment variable.")

        self._engine = create_async_engine(database_url, echo=False)
        self._async_session = async_sessionmaker(self._engine, expire_on_commit=False)

        self._latest_resources_id = None

        # special handling for retry strategy
        retry_for_waiting = retry_for_waiting or RetryStrategy(
            max_attempts=10,  # set a limit for retries if timeout is specified, otherwise will change to None later
            max_retry_delay=None, # set later
            wait_seconds=10.0, # poll every 10 seconds
            max_wait_seconds=60.0, # at most wait 60 seconds between retries
            backoff=1.0,
            jitter=0.0,
            log=True,
        )
        self.retry_for_waiting = retry_for_waiting if isinstance(retry_for_waiting, RetryStrategy) else RetryStrategy(**retry_for_waiting)
        self.wait_for_nonexistent_rollout = wait_for_nonexistent_rollout

        # setup in-process periodic tasks
        if background_tasks_cfg is None:
            self.background_tasks_cfg = [
                BackgroundTaskConfig(name="check_attempt_timeout", method="check_attempt_timeout", interval={"seconds": 10.0}),
            ]
        else:
            self.background_tasks_cfg = [
                BackgroundTaskConfig(**cfg) for cfg in background_tasks_cfg
            ]
        self._background_scheduler = BackgroundScheduler()

    async def start(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(SqlAlchemyBase.metadata.create_all)
        for task_cfg in self.background_tasks_cfg:
            self.add_background_task(task_cfg, to_scheduler_only=True)
        self._background_scheduler.start() # type: ignore

    async def stop(self):
        await self._engine.dispose()
        self._background_scheduler.shutdown() # type: ignore

    def add_background_task(self, task_cfg: Dict[str, Any] | BackgroundTaskConfig, to_scheduler_only: bool = False) -> None:
        """Add a new periodic background task to the scheduler.
        Args:
            task_cfg (Dict[str, Any] | BackgroundTaskConfig): The configuration for the background task.
            to_scheduler_only (bool): If True, only add the task to the scheduler without updating the configuration list.
        Raises:
            ValueError: If the task method is not defined in DatabaseLightningStore.
        """
        config = task_cfg if isinstance(task_cfg, BackgroundTaskConfig) else BackgroundTaskConfig(**task_cfg)
        if not to_scheduler_only:
            # check existing tasks
            for existing in self.background_tasks_cfg:
                if existing.name == config.name:
                    logger.warning(f"Background task {config.name} is already scheduled, will update its configuration.")
            self.background_tasks_cfg.append(config)
        delta_t = timedelta(**config.interval)
        if not hasattr(self, config.method):
            raise ValueError(f"Periodic task method {config.method} is not defined in DatabaseLightningStore.")
        if config.is_async:
            func = lambda: asyncio.run(getattr(self, config.method)())
        else:
            func = lambda: getattr(self, config.method)()

        self._background_scheduler.add_job( # type: ignore
            func=func,
            trigger=IntervalTrigger(**config.interval), # type: ignore
            name=f"DatabaseLightningStore.{config.name}",
            replace_existing=True,
            next_run_time=datetime.now() + delta_t,  # schedule the first run after the interval
        )

    # ------------------------------------------------------
    # Public methods defined in LightningStore
    # ------------------------------------------------------

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

    @db_retry
    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._fifo_dequeue_rollout()

    @db_retry
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        async with self._async_session() as session:
            async with session.begin():
                rollout_obj = await session.get(RolloutInDB, rollout_id)
                if rollout_obj is None:
                    raise ValueError(f"Rollout {rollout_id} not found")
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
        strategy = RetryStrategy(**self.retry_for_waiting.asdict())
        if timeout is not None:
            strategy.max_retry_delay = timeout
            if strategy.max_attempts is not None:
                strategy.wait_seconds = min(strategy.wait_seconds, timeout / (strategy.max_attempts+1))
        else:
            strategy.max_attempts = None  # infinite retries

        non_completed_ids, non_existing_ids = set(rollout_ids), set(rollout_ids)
        completed_rollouts: Dict[str, Rollout] = {}
        if len(non_completed_ids) < len(rollout_ids):
            logger.warning("Duplicate rollout_ids found in wait_for_rollouts input. Duplicates will be ignored.")

        try:
            async for attempt in AsyncRetryBlock(
                strategy,
                reraise=True,
            ):
                with attempt:
                    async with self._async_session() as session:
                        async with session.begin():
                            result = await session.scalars(
                                select(RolloutInDB).where(RolloutInDB.rollout_id.in_(non_completed_ids))
                            )
                            rollouts = [r.as_rollout() for r in result.all()]
                            for r in rollouts:
                                if r.rollout_id in non_existing_ids:
                                    non_existing_ids.discard(r.rollout_id) # found existing rollout
                                if is_finished(r):
                                    completed_rollouts[r.rollout_id] = r
                                    non_completed_ids.discard(r.rollout_id)
                            # check termination conditions
                            if self.wait_for_nonexistent_rollout:
                                if len(non_completed_ids) == 0:
                                    return [completed_rollouts[rid] for rid in rollout_ids if rid in completed_rollouts]
                                raise _WaitForRolloutsCompleted(f"WaitForRolloutsCompleted: requested={len(rollout_ids)}, completed={len(completed_rollouts)}, non_existing={len(non_existing_ids)}")
                            else:
                                if len(non_completed_ids) == len(non_existing_ids):
                                    logger.warning(f"All remaining rollouts are non-existing: {non_existing_ids}.")
                                    return [completed_rollouts[rid] for rid in rollout_ids if rid in completed_rollouts]
                                raise _WaitForRolloutsCompleted(f"WaitForRolloutsCompleted: requested={len(rollout_ids)}, completed={len(completed_rollouts)}, non_existing={len(non_existing_ids)}")

        except (RetryError, _WaitForRolloutsCompleted):
            return [completed_rollouts[rid] for rid in rollout_ids if rid in completed_rollouts]

    @db_retry
    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        async with self._async_session() as session:
            async with session.begin():
                conditions: List[Any] = [SpanInDB.rollout_id == rollout_id]
                if attempt_id is not None:
                    if attempt_id == "latest":
                        rollout_obj = await session.get(RolloutInDB, rollout_id)
                        if rollout_obj is None:
                            logger.warning(f"Rollout {rollout_id} does not exist. Cannot query latest attempt spans.")
                            return []
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
                    raise ValueError(f"Rollout {rollout_id} not found")
                # udpate fields
                if not isinstance(input, Unset):
                    rollout_obj.input = input
                if not isinstance(mode, Unset):
                    rollout_obj.mode = mode
                if not isinstance(resources_id, Unset):
                    rollout_obj.resources_id = resources_id
                if not isinstance(status, Unset):
                    await rollout_obj.update_status(dict(event="user_update", new_status=status), session)
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
                    raise ValueError(f"Rollout {rollout_id} not found")
                if attempt_id == "latest":
                    if rollout_obj.latest_attempt_id is None:
                        raise ValueError(f"Rollout {rollout_id} has no attempts. Cannot update latest attempt.")
                    attempt_id = rollout_obj.latest_attempt_id
                if attempt_id != rollout_obj.latest_attempt_id:
                    logger.warning(f"Updating attempt {attempt_id} which is not the latest attempt for rollout {rollout_id}. Latest is {rollout_obj.latest_attempt_id}.")
                attempt_obj = await session.get(AttemptInDB, attempt_id)
                if attempt_obj is None:
                    raise ValueError(f"No attempts found")
                if attempt_obj.rollout_id != rollout_id:
                    raise ValueError(f"Attempt {attempt_id} does not belong to rollout {rollout_id}.")
                # update fields
                if not isinstance(status, Unset):
                    msg = attempt_obj.update_status(dict(event="user_update", new_status=status))
                    if msg is not None:
                        await rollout_obj.update_status(msg, session)
                if not isinstance(worker_id, Unset):
                    attempt_obj.worker_id = worker_id
                if not isinstance(last_heartbeat_time, Unset):
                    attempt_obj.last_heartbeat_time = last_heartbeat_time
                if not isinstance(metadata, Unset):
                    attempt_obj.attempt_metadata = metadata
                await session.flush()  # ensure the object is written to the DB
                return attempt_obj.as_attempt()

    # ------------------------------------------------------
    # periodic background tasks can be added here
    # ------------------------------------------------------

    async def check_attempt_timeout(self):
        """Periodically check for attempts that have timed out and update their status accordingly."""
        # use update with where condition to find and update timed-out attempts
        current_time = time.time()
        attempts_timed_out: list[AttemptInDB] = []

        # Step 1: Filter and update timed-out attempts
        async with self._async_session() as session:
            async with session.begin():
                for mode in ["max_heartbeat_interval", "max_duration"]: # max_duration has higher priority
                    attempts_timed_out.extend(await self._attempt_timeout_check(session, mode, current_time))

        # Step 2: Create messages to update rollout
        messages: Dict[str, AttemptStatusUpdateMessage] = {}
        rollout_ids: set[str] = set()
        for attempt in attempts_timed_out:
            messages[attempt.attempt_id] = AttemptStatusUpdateMessage(
                timestamp=current_time,
                new_status=attempt.status,
                attempt_id=attempt.attempt_id,
                rollout_id=attempt.rollout_id,
            )
            rollout_ids.add(attempt.rollout_id)

        # Step 3: Update rollouts
        async with self._async_session() as session:
            async with session.begin():
                result = await session.scalars(
                    select(RolloutInDB).where(RolloutInDB.rollout_id.in_(rollout_ids))
                )
                rollout_objs = {r.rollout_id: r for r in result.all()}
                for msg in messages.values():
                    rollout_obj = rollout_objs[msg.rollout_id]
                    await rollout_obj.update_status(msg, session)

    # ------------------------------------------------------
    # internal helper methods can be added here
    # ------------------------------------------------------

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
                    raise ValueError(f"Attempt {span['attempt_id']} not found")
                # ensure the attempt and rollout are in running status
                msg = attempt_obj.update_status(dict(event="span_received"))
                if msg is not None:
                    rollout_obj = await session.get(RolloutInDB, attempt_obj.rollout_id)
                    if rollout_obj is None:
                        raise ValueError(f"Rollout {attempt_obj.rollout_id} not found")
                    await rollout_obj.update_status(msg, session)
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
        rollout_config = rollout_obj.config if rollout_obj.config is not None else RolloutConfig()
        attempt_obj = AttemptInDB(
            rollout_id=rollout_obj.rollout_id,
            sequence_id=rollout_obj.num_attempts + 1,
            status="preparing",
            max_duration=rollout_config.timeout_seconds,
            max_heartbeat_interval=rollout_config.unresponsive_seconds,
        )
        session.add(attempt_obj)
        # pre-update the rollout_obj fields for CAS
        if rollout_obj.status in ["queuing", "requeuing"]:
            rollout_obj.status = "running"  # type: ignore pre-update the status in the object for CAS
        rollout_obj.enqueue_time = None  # pre-update the enqueue_time in the object for CAS
        rollout_obj.num_attempts += 1  # pre-update the num_attempts in the object for CAS
        rollout_obj.latest_attempt_id = attempt_obj.attempt_id  # pre-update the latest_attempt_id in the object for CAS
        rollout_obj.latest_attempt_status = attempt_obj.status  # type: ignore

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

    async def _attempt_timeout_check(self, session: AsyncSession, mode: str, current_time: float) -> list[AttemptInDB]:
        if mode == "max_duration":
            new_status = "timeout"
            conditions = and_(
                AttemptInDB.status.in_(["preparing", "running"]),
                AttemptInDB.max_duration.isnot(None),
                (current_time - AttemptInDB.start_time) > AttemptInDB.max_duration,
            )
        elif mode == "max_heartbeat_interval":
            new_status = "unresponsive"
            conditions = and_(
                AttemptInDB.status.in_(["preparing", "running"]),
                AttemptInDB.max_heartbeat_interval.isnot(None),
                (current_time - AttemptInDB.last_heartbeat_time) > AttemptInDB.max_heartbeat_interval,
            )
        else:
            raise ValueError(f"Unsupported timeout checking mode {mode}")
        result = await session.scalars(
            update(AttemptInDB)
            .where(conditions)
            .values(status=new_status)
            .returning(AttemptInDB)
        )
        return list(result.all())
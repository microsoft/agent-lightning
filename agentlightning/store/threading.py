# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import ReadableSpan

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
    Worker,
    WorkerStatus,
)

from .base import UNSET, LightningStore, LightningStoreCapabilities, LightningStoreStatistics, Unset


class LightningStoreThreaded(LightningStore):
    """Facade that delegates all store operations to a underlying store instance.

    The operations are guaranteed to be thread-safe.
    Make sure the threaded stores are instantiated before initializing the threads.
    """

    def __init__(self, store: LightningStore) -> None:
        super().__init__()  # watchdog relies on the underlying store
        self.store = store
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        capabilities = self.store.capabilities
        return {
            **capabilities,
            "async_safe": True,
            "thread_safe": True,
        }

    async def statistics(self) -> LightningStoreStatistics:
        """Return the statistics of the store."""
        with self._lock:
            return await self.store.statistics()

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        with self._lock:
            return await self.store.start_rollout(input, mode, resources_id, config, metadata)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        with self._lock:
            return await self.store.enqueue_rollout(input, mode, resources_id, config, metadata)

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        with self._lock:
            return await self.store.dequeue_rollout(worker_id=worker_id)

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        with self._lock:
            return await self.store.start_attempt(rollout_id)

    async def query_rollouts(
        self,
        *,
        status_in: Optional[Sequence[RolloutStatus]] = None,
        rollout_id_in: Optional[Sequence[str]] = None,
        rollout_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
        status: Optional[Sequence[RolloutStatus]] = None,
        rollout_ids: Optional[Sequence[str]] = None,
    ) -> Sequence[Rollout]:
        with self._lock:
            return await self.store.query_rollouts(
                status_in=status_in,
                rollout_id_in=rollout_id_in,
                rollout_id_contains=rollout_id_contains,
                filter_logic=filter_logic,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
                status=status,
                rollout_ids=rollout_ids,
            )

    async def query_attempts(
        self,
        rollout_id: str,
        *,
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[Attempt]:
        with self._lock:
            return await self.store.query_attempts(
                rollout_id,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        with self._lock:
            return await self.store.get_rollout_by_id(rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        with self._lock:
            return await self.store.get_latest_attempt(rollout_id)

    async def query_resources(
        self,
        *,
        resources_id: Optional[str] = None,
        resources_id_contains: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[ResourcesUpdate]:
        with self._lock:
            return await self.store.query_resources(
                resources_id=resources_id,
                resources_id_contains=resources_id_contains,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        with self._lock:
            return await self.store.add_resources(resources)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        with self._lock:
            return await self.store.update_resources(resources_id, resources)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        with self._lock:
            return await self.store.get_resources_by_id(resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        with self._lock:
            return await self.store.get_latest_resources()

    async def add_many_spans(self, spans: Sequence[Span]) -> Sequence[Span]:
        with self._lock:
            return await self.store.add_many_spans(spans)

    async def add_span(self, span: Span) -> Optional[Span]:
        with self._lock:
            return await self.store.add_span(span)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Optional[Span]:
        with self._lock:
            return await self.store.add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        # This method does not change the state of the store, and it's not thread-safe.
        return await self.store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=timeout)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        with self._lock:
            return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

    async def get_many_span_sequence_ids(self, rollout_attempt_ids: Sequence[Tuple[str, str]]) -> Sequence[int]:
        with self._lock:
            return await self.store.get_many_span_sequence_ids(rollout_attempt_ids)

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
        *,
        trace_id: Optional[str] = None,
        trace_id_contains: Optional[str] = None,
        span_id: Optional[str] = None,
        span_id_contains: Optional[str] = None,
        parent_id: Optional[str] = None,
        parent_id_contains: Optional[str] = None,
        name: Optional[str] = None,
        name_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        limit: int = -1,
        offset: int = 0,
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> Sequence[Span]:
        with self._lock:
            return await self.store.query_spans(
                rollout_id,
                attempt_id,
                trace_id=trace_id,
                trace_id_contains=trace_id_contains,
                span_id=span_id,
                span_id_contains=span_id_contains,
                parent_id=parent_id,
                parent_id_contains=parent_id_contains,
                name=name,
                name_contains=name_contains,
                filter_logic=filter_logic,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        with self._lock:
            return await self.store.update_rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode,
                resources_id=resources_id,
                status=status,
                config=config,
                metadata=metadata,
            )

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        with self._lock:
            return await self.store.update_attempt(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                status=status,
                worker_id=worker_id,
                last_heartbeat_time=last_heartbeat_time,
                metadata=metadata,
            )

    async def query_workers(
        self,
        *,
        status_in: Optional[Sequence[WorkerStatus]] = None,
        worker_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[Worker]:
        with self._lock:
            return await self.store.query_workers(
                status_in=status_in,
                worker_id_contains=worker_id_contains,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )

    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        with self._lock:
            return await self.store.get_worker_by_id(worker_id)

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        with self._lock:
            return await self.store.update_worker(
                worker_id=worker_id,
                heartbeat_stats=heartbeat_stats,
            )

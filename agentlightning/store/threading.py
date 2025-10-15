# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional, Sequence

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
)

from .base import UNSET, LightningStore, Unset


class LightningStoreThreaded(LightningStore):
    """Facade that delegates all store operations to a underlying store instance.

    The operations are guaranteed to be thread-safe.
    Make sure the threaded stores are instantiated before initializing the threads.
    """

    def __init__(self, store: LightningStore) -> None:
        super().__init__()  # watchdog relies on the underlying store
        self.store = store
        self._lock = threading.Lock()

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        with self._lock:
            return await self.store.start_rollout(input, mode, resources_id, metadata)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        with self._lock:
            return await self.store.enqueue_rollout(input, mode, resources_id, metadata)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        with self._lock:
            return await self.store.dequeue_rollout()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        with self._lock:
            return await self.store.start_attempt(rollout_id)

    async def query_rollouts(
        self,
        *,
        status: Optional[Sequence[RolloutStatus]] = None,
        rollout_ids: Optional[Sequence[str]] = None,
    ) -> List[Rollout]:
        with self._lock:
            return await self.store.query_rollouts(status=status, rollout_ids=rollout_ids)

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        with self._lock:
            return await self.store.query_attempts(rollout_id)

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        with self._lock:
            return await self.store.get_rollout_by_id(rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        with self._lock:
            return await self.store.get_latest_attempt(rollout_id)

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

    async def add_span(self, span: Span) -> Span:
        with self._lock:
            return await self.store.add_span(span)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        with self._lock:
            return await self.store.add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        # This method does not change the state of the store, and it's not thread-safe.
        return await self.store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=timeout)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        with self._lock:
            return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        with self._lock:
            return await self.store.query_spans(rollout_id, attempt_id)

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

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional, Sequence, TypeVar

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import NamedResources, ResourcesUpdate, RolloutStatus, RolloutV2

from .base import LightningStore

T_co = TypeVar("T_co")


class LightningStoreThreaded(LightningStore):
    """Facade that delegates all store operations to a underlying store instance.

    The operations are guaranteed to be thread-safe.
    Make sure the threaded stores are instantiated before initializing the threads.
    """

    def __init__(self, store: LightningStore) -> None:
        self.store = store
        self._lock = threading.Lock()

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        with self._lock:
            return await self.store.add_task(sample, mode, resources_id, metadata)

    async def add_rollout(self, rollout: RolloutV2) -> RolloutV2:
        with self._lock:
            return await self.store.add_rollout(rollout)

    async def pop_rollout(self) -> Optional[RolloutV2]:
        with self._lock:
            return await self.store.pop_rollout()

    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        with self._lock:
            return await self.store.query_rollouts(status)

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
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        with self._lock:
            return await self.store.add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        with self._lock:
            return await self.store.wait_for_rollouts(rollout_ids, timeout)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        with self._lock:
            return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        with self._lock:
            return await self.store.query_spans(rollout_id, attempt_id)

    async def update_rollout(
        self,
        rollout_id: str,
        status: RolloutStatus,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        attempt_start_time: Optional[float] = None,
        last_attempt_status: Optional[RolloutStatus] = None,
        **kwargs: Any,
    ) -> RolloutV2:
        with self._lock:
            return await self.store.update_rollout(
                rollout_id=rollout_id,
                status=status,
                worker_id=worker_id,
                attempt_sequence_id=attempt_sequence_id,
                attempt_id=attempt_id,
                attempt_start_time=attempt_start_time,
                last_attempt_status=last_attempt_status,
                **kwargs,
            )

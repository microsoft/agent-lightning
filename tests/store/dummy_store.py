# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Dict, List, Literal, Optional

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store import LightningStoreCapabilities
from agentlightning.store.base import UNSET, LightningStore
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
)


class DummyLightningStore(LightningStore):
    def __init__(self, return_values: Dict[str, Any]) -> None:
        super().__init__()
        self.calls: List[tuple[str, tuple[Any, ...], Dict[str, Any]]] = []
        self.return_values = return_values

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        return LightningStoreCapabilities(
            async_safe=True,
            thread_safe=False,
            zero_copy=False,
        )

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Optional[str] = None,
        resources_id: Optional[str] = None,
        config: Optional[RolloutConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttemptedRollout:
        self.calls.append(("start_rollout", (input, mode, resources_id, config, metadata), {}))
        return self.return_values["start_rollout"]

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Optional[str] = None,
        resources_id: Optional[str] = None,
        config: Optional[RolloutConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Rollout:
        self.calls.append(("enqueue_rollout", (input, mode, resources_id, config, metadata), {}))
        return self.return_values["enqueue_rollout"]

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        self.calls.append(("dequeue_rollout", (worker_id,), {}))
        return self.return_values["dequeue_rollout"]

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        self.calls.append(("start_attempt", (rollout_id,), {}))
        return self.return_values["start_attempt"]

    async def query_rollouts(self, *args: Any, **kwargs: Any) -> List[Rollout]:
        self.calls.append(("query_rollouts", args, kwargs))
        return self.return_values["query_rollouts"]

    async def query_attempts(self, *args: Any, **kwargs: Any) -> List[Attempt]:
        self.calls.append(("query_attempts", args, kwargs))
        return self.return_values["query_attempts"]

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        self.calls.append(("get_rollout_by_id", (rollout_id,), {}))
        return self.return_values["get_rollout_by_id"]

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        self.calls.append(("get_latest_attempt", (rollout_id,), {}))
        return self.return_values["get_latest_attempt"]

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        self.calls.append(("add_resources", (resources,), {}))
        return self.return_values["add_resources"]

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        self.calls.append(("update_resources", (resources_id, resources), {}))
        return self.return_values["update_resources"]

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        self.calls.append(("get_resources_by_id", (resources_id,), {}))
        return self.return_values["get_resources_by_id"]

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        self.calls.append(("get_latest_resources", (), {}))
        return self.return_values["get_latest_resources"]

    async def query_resources(self, *args: Any, **kwargs: Any) -> List[ResourcesUpdate]:
        self.calls.append(("query_resources", args, kwargs))
        return self.return_values["query_resources"]

    async def add_span(self, span: Span) -> Span:
        self.calls.append(("add_span", (span,), {}))
        return self.return_values["add_span"]

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: Optional[int] = None,
    ) -> Span:
        self.calls.append(("add_otel_span", (rollout_id, attempt_id, readable_span, sequence_id), {}))
        return self.return_values["add_otel_span"]

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        self.calls.append(("wait_for_rollouts", (), {"rollout_ids": rollout_ids, "timeout": timeout}))
        return self.return_values["wait_for_rollouts"]

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        self.calls.append(("get_next_span_sequence_id", (rollout_id, attempt_id), {}))
        return self.return_values["get_next_span_sequence_id"]

    async def query_spans(self, *args: Any, **kwargs: Any) -> List[Span]:
        self.calls.append(("query_spans", args, kwargs))
        return self.return_values["query_spans"]

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Any = UNSET,
        mode: Optional[str] | Any = UNSET,
        resources_id: Optional[str] | Any = UNSET,
        status: RolloutStatus | Any = UNSET,
        config: Any = UNSET,
        metadata: Optional[Dict[str, Any]] | Any = UNSET,
    ) -> Rollout:
        self.calls.append(
            (
                "update_rollout",
                (rollout_id, input, mode, resources_id, status, config, metadata),
                {},
            )
        )
        return self.return_values["update_rollout"]

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Any = UNSET,
        worker_id: str | Any = UNSET,
        last_heartbeat_time: float | Any = UNSET,
        metadata: Optional[Dict[str, Any]] | Any = UNSET,
    ) -> Attempt:
        self.calls.append(
            (
                "update_attempt",
                (rollout_id, attempt_id, status, worker_id, last_heartbeat_time, metadata),
                {},
            )
        )
        return self.return_values["update_attempt"]

    async def query_workers(self, *args: Any, **kwargs: Any) -> List[Worker]:
        self.calls.append(("query_workers", args, kwargs))
        return self.return_values["query_workers"]

    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        self.calls.append(("get_worker_by_id", (worker_id,), {}))
        return self.return_values["get_worker_by_id"]

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Any = UNSET,
    ) -> Worker:
        self.calls.append(
            (
                "update_worker",
                (
                    worker_id,
                    heartbeat_stats,
                ),
                {},
            )
        )
        return self.return_values["update_worker"]


def minimal_dummy_store() -> DummyLightningStore:
    # Provide minimal return values
    return DummyLightningStore(
        return_values={
            "start_rollout": None,
            "enqueue_rollout": None,
            "dequeue_rollout": None,
            "start_attempt": None,
            "query_rollouts": [],
            "query_attempts": [],
            "get_rollout_by_id": None,
            "get_latest_attempt": None,
            "add_resources": None,
            "update_resources": None,
            "get_resources_by_id": None,
            "get_latest_resources": None,
            "add_span": None,
            "add_otel_span": None,
            "wait_for_rollouts": [],
            "get_next_span_sequence_id": 0,
            "query_spans": [],
            "update_rollout": None,
            "update_attempt": None,
            "query_workers": [],
            "get_worker_by_id": None,
            "update_worker": Worker(worker_id="worker-0"),
        }
    )

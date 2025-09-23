from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.tracer import Span
from agentlightning.types import ResourcesUpdate, RolloutStatus, RolloutV2


def is_queuing(rollout: RolloutV2) -> bool:
    return rollout.status == "queuing" or rollout.status == "requeuing"


def is_running(rollout: RolloutV2) -> bool:
    return rollout.status == "preparing" or rollout.status == "running"


def is_finished(rollout: RolloutV2) -> bool:
    return rollout.status == "error" or rollout.status == "success"


class LightningStore:
    """
    A centralized, thread-safe, async, data store for the lightning's state.
    This holds the task queue, versioned resources, and completed rollouts.
    """

    def __init__(self, watchdog: LightningStoreWatchDog | None = None):
        self.watchdog = watchdog

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        ...

    async def pop_rollout(self) -> Optional[RolloutV2]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.

        Will set the rollout status to preparing.
        """
        ...

    async def update_resources(self, update: ResourcesUpdate):
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        ...

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves a specific version of named resources by its ID.
        """
        ...

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves the latest version of named resources.
        """
        ...

    async def add_span(self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan) -> Span:
        """
        Add a span to the store.
        """
        ...

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: float) -> List[RolloutV2]: ...

    async def query_spans(self, rollout_id: str) -> List[Span]: ...

    async def _update_rollout(
        self,
        status: RolloutStatus,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        acknowledge_time: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Update the rollout status.
        This should only be used internally or by watchdog.
        """
        ...


class LightningStoreWatchDog:

    def __init__(self, timeout_seconds: float, unresponsive_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.unresponsive_seconds = unresponsive_seconds

    def healthcheck(self, store: LightningStore): ...

from typing import *
from agentlightning.types import *

from opentelemetry.sdk.trace import ReadableSpan


def is_queuing(rollout: Rollout):
    return rollout.status == "queuing" or rollout.status == "requeuing"


class LightningStore:
    """
    A centralized, thread-safe, async, in-memory data store for the server's state.
    This holds the task queue, versioned resources, and completed rollouts.
    """

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        ...

    async def pop_rollout(self) -> Optional[Rollout]:
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

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: float) -> List[Rollout]:
        ...

    async def get_spans(self, rollout_id: str) -> List[Span]:
        ...

    async def _update_rollout(self, rollout_data: Partial[Rollout]) -> Rollout:
        """
        Update the rollout status.
        This should only be used internally or by watchdog.
        """
        ...


class LightningStoreWatchDog:

    def __init__(self, timeout_seconds: float, unresponsive_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.unresponsive_seconds = unresponsive_seconds

    def healthcheck(self, store: LightningStore):
        ...
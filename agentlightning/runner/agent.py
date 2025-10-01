# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING, Any, Optional, TypeVar

from agentlightning.litagent import LitAgent
from agentlightning.store.base import LightningStore
from agentlightning.tracer.base import BaseTracer

if TYPE_CHECKING:
    from agentlightning.execution.events import Event

from .base import BaseRunner

T_task = TypeVar("T_task")


class AgentRunnerV2(BaseRunner[T_task]):
    """The runner for the agent."""

    def __init__(self, tracer: BaseTracer, max_tasks: Optional[int] = None) -> None:
        super().__init__()
        self._tracer = tracer
        self._max_tasks = max_tasks

        # Set later
        self._agent: Optional[LitAgent[T_task]] = None
        self._store: Optional[LightningStore] = None
        self._worker_id: Optional[int] = None

    def init(self, agent: LitAgent[T_task], **kwargs: Any) -> None:
        """Initialize the runner with the agent."""
        self._agent = agent

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store."""
        self._store = store
        self._worker_id = worker_id

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner."""
        self._agent = None
        self._store = None
        self._worker_id = None

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner for each worker."""
        self._worker_id = None

    def get_agent(self) -> LitAgent[T_task]:
        """Get the agent."""
        if self._agent is None:
            raise ValueError("Agent not initialized. Call init() first.")
        return self._agent

    def get_store(self) -> LightningStore:
        """Get the store."""
        if self._store is None:
            raise ValueError("Store not initialized. Call init_worker() first.")
        return self._store

    async def iter(self, store: LightningStore, event: Event) -> None:
        """Run the runner, iterate over the tasks in the store. Abort if the event is set."""
        pass

    async def step(self, input: T_task, store: LightningStore, event: Event) -> None:
        """Step the runner, execute one task."""
        store = self.get_store()
        agent = self.get_agent()

        attempted_rollout = await store.start_rollout(input=input)

        if agent.is_async():
            await agent.roll(input=input, resources=attempted_rollout.resources)
        else:
            await agent.training_rollout(input=input, resources=attempted_rollout.resources)

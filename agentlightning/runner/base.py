# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from agentlightning.litagent import LitAgent
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, ParallelWorkerBase, RolloutMode

if TYPE_CHECKING:
    from agentlightning.execution.events import Event


T_task = TypeVar("T_task")


class BaseRunner(ParallelWorkerBase, Generic[T_task]):
    """Base class for all runners."""

    def init(self, agent: LitAgent[T_task], **kwargs: Any) -> None:
        """Initialize the runner with the agent."""
        raise NotImplementedError()

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store."""
        raise NotImplementedError()

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("The behavior of run() of Runner is undefined. Use iter() or step() instead.")

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    async def iter(self, *, event: Optional[Event] = None) -> None:
        """Run the runner, iterate over the tasks in the store. Abort if the event is set."""
        raise NotImplementedError()

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[NamedResources] = None,
        mode: Optional[RolloutMode] = None,
        event: Optional[Event] = None,
    ) -> None:
        """Step the runner, execute one task."""
        raise NotImplementedError()

# Copyright (c) Microsoft. All rights reserved.

"""Abstract runner interface for executing agent tasks."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, Sequence, TypeVar

from agentlightning.execution.events import ExecutionEvent
from agentlightning.litagent import LitAgent
from agentlightning.store.base import LightningStore
from agentlightning.types import Hook, NamedResources, ParallelWorkerBase, Rollout, RolloutMode

if TYPE_CHECKING:
    from agentlightning.execution.events import ExecutionEvent


T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


class Runner(ParallelWorkerBase, Generic[T_task]):
    """Abstract base class for long-running agent executors.

    Runner implementations coordinate [`LitAgent`][agentlightning.LitAgent]
    instances, acquire work from a [`LightningStore`][agentlightning.LightningStore],
    and emit [`Rollout`][agentlightning.Rollout] objects. Subclasses decide how
    to schedule work (polling, streaming, etc.) while this base class provides a
    minimal lifecycle contract.
    """

    def init(self, agent: LitAgent[T_task], **kwargs: Any) -> None:
        """Prepare the runner to execute tasks for `agent`.

        This method is called only once during the setup for all workers, not for each worker.

        Args:
            agent: Agent instance providing task-specific logic.
            **kwargs: Optional runner-specific configuration.

        Raises:
            NotImplementedError: Subclasses must supply the initialization
                routine.
        """
        raise NotImplementedError()

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Configure worker-local state before processing tasks.

        This method is called for **each** worker during the setup.

        Args:
            worker_id: Unique identifier for this worker process or thread.
            store: Shared [`LightningStore`][agentlightning.LightningStore]
                backing task coordination.
            **kwargs: Optional worker-specific configuration.

        Raises:
            NotImplementedError: Subclasses must prepare per-worker resources.
        """
        raise NotImplementedError()

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Deprecated synchronous entry point.

        Use [`iter()`][agentlightning.Runner.iter] or [`step()`][agentlightning.Runner.step] instead.

        Raises:
            RuntimeError: Always raised to direct callers to
                [iter()][agentlightning.Runner.iter] or
                [step()][agentlightning.Runner.step].
        """
        raise RuntimeError("The behavior of run() of Runner is undefined. Use iter() or step() instead.")

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        """Release resources acquired during [`init()`][agentlightning.Runner.init].

        Raises:
            NotImplementedError: Subclasses must implement the shutdown routine.
        """
        raise NotImplementedError()

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        """Release per-worker resources allocated by [`init_worker()`][agentlightning.Runner.init_worker].

        Args:
            worker_id: Identifier of the worker being torn down.

        Raises:
            NotImplementedError: Subclasses must implement the shutdown routine.
        """
        raise NotImplementedError()

    @contextmanager
    def run_context(
        self,
        *,
        agent: LitAgent[T_task],
        store: LightningStore,
        hooks: Optional[Sequence[Hook]] = None,
        worker_id: Optional[int] = None,
    ) -> Iterator[Runner[T_task]]:
        """Initialize and tear down a runner within a simple context manager.

        The helper is primarily intended for debugging runner implementations
        outside of a full [`Trainer`][agentlightning.Trainer] stack.

        Args:
            agent: Agent executed by this runner.
            store: Backing [`LightningStore`][agentlightning.LightningStore].
                If you don't have one, you can easily create one with
                [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore].
            hooks: Optional sequence of hooks recognised by the runner.
                Not all runners support hooks.
            worker_id: Override the worker identifier used during setup. Defaults
                to `0`.
        """
        _initialized: bool = False
        _worker_initialized: bool = False
        try:
            self.init(agent=agent, hooks=hooks)
            _initialized = True
            self.init_worker(worker_id=0, store=store)
            _worker_initialized = True
            yield self
        finally:
            try:
                if _worker_initialized:
                    self.teardown_worker(worker_id=worker_id if worker_id is not None else 0)
            except Exception:
                logger.error("Error during runner worker teardown", exc_info=True)

            try:
                if _initialized:
                    self.teardown()
            except Exception:
                logger.error("Error during runner teardown", exc_info=True)

    async def iter(self, *, event: Optional[ExecutionEvent] = None) -> None:
        """Run the runner, continuously iterating over tasks in the store.

        This method runs in a loop, polling the store for new tasks and executing
        them until interrupted by the event or when no more tasks are available.

        Args:
            event: Cooperative stop signal. When set, the runner should complete
                the current unit of work and exit the loop.

        Raises:
            NotImplementedError: Subclasses provide the iteration behavior.
        """
        raise NotImplementedError()

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[NamedResources] = None,
        mode: Optional[RolloutMode] = None,
        event: Optional[ExecutionEvent] = None,
    ) -> Rollout:
        """Execute a single task with the given input.

        This method provides fine-grained control for executing individual tasks
        directly, bypassing the store's task queue.

        Args:
            input: Task payload consumed by the agent.
            resources: Optional named resources scoped to this invocation.
            mode: Optional rollout mode such as `"train"` or `"eval"`.
            event: Cooperative stop signal for long-running tasks.

        Returns:
            Completed rollout produced by the agent.

        Raises:
            NotImplementedError: Subclasses provide the execution behavior.
        """
        raise NotImplementedError()

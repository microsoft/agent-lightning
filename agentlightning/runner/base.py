# Copyright (c) Microsoft. All rights reserved.

"""Base runner interface for executing agent tasks.

This module defines the abstract base class for all runner implementations
in the agent-lightning framework. Runners are responsible for managing the
execution lifecycle of agents and coordinating with the store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from agentlightning.litagent import LitAgent
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, ParallelWorkerBase, RolloutMode

if TYPE_CHECKING:
    from agentlightning.execution.events import Event


T_task = TypeVar("T_task")


class BaseRunner(ParallelWorkerBase, Generic[T_task]):
    """Base class for all runners.

    This abstract base class defines the interface that all runner implementations
    must follow. Runners are responsible for executing agent tasks, managing the
    execution lifecycle, and coordinating with the store.
    """

    def init(self, agent: LitAgent[T_task], **kwargs: Any) -> None:
        """Initialize the runner with the agent.

        This method is called once during setup to configure the runner with
        the agent it will execute.

        Args:
            agent: The LitAgent instance to be managed by this runner.
            **kwargs: Additional initialization arguments specific to the runner implementation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store.

        This method is called once per worker process in a distributed setup.
        It provides the worker with its unique ID and the store instance for
        task coordination.

        Args:
            worker_id: Unique identifier for this worker process.
            store: The LightningStore instance for task coordination and data persistence.
            **kwargs: Additional worker-specific initialization arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Undefined method - use iter() or step() instead.

        This method is intentionally not implemented as the execution behavior
        should be defined through iter() for continuous execution or step()
        for single-task execution.

        Args:
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Raises:
            RuntimeError: Always raised to indicate this method should not be used.
        """
        raise RuntimeError("The behavior of run() of Runner is undefined. Use iter() or step() instead.")

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        """Clean up runner resources and reset state.

        This method is called once during shutdown to clean up any resources
        allocated during initialization and reset the runner state.

        Args:
            *args: Additional teardown arguments.
            **kwargs: Additional teardown keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        """Clean up worker-specific resources.

        This method is called once per worker during shutdown to clean up
        any resources specific to that worker.

        Args:
            worker_id: The unique identifier of the worker being torn down.
            *args: Additional teardown arguments.
            **kwargs: Additional teardown keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    async def iter(self, *, event: Optional[Event] = None) -> None:
        """Run the runner, continuously iterating over tasks in the store.

        This method runs in a loop, polling the store for new tasks and executing
        them until interrupted by the event or when no more tasks are available.

        Args:
            event: Optional Event object that can be used to signal the runner
                to stop gracefully. When set, the runner should finish its current
                task and exit the iteration loop.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[NamedResources] = None,
        mode: Optional[RolloutMode] = None,
        event: Optional[Event] = None,
    ) -> None:
        """Execute a single task with the given input.

        This method provides fine-grained control for executing individual tasks
        directly, bypassing the store's task queue.

        Args:
            input: The task input to be processed by the agent.
            resources: Optional named resources to be used for this specific task.
                If not provided, the latest resources from the store will be used.
            mode: Optional rollout mode (e.g., "train", "test"). If not provided,
                the default mode will be used.
            event: Optional Event object to signal interruption. When set, the
                runner may abort the current execution.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

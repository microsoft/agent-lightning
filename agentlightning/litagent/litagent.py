# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
import logging
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

from agentlightning.types import NamedResources, Rollout, RolloutRawResult, Task

if TYPE_CHECKING:
    from agentlightning.runner import Runner
    from agentlightning.tracer import Tracer
    from agentlightning.trainer import Trainer


logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "LitAgent",
]


def is_v0_1_rollout_api(func: Callable[..., Any]) -> bool:
    """Check if the rollout API is v0.1.
    Inspect the function signature to see if it has a rollout_id parameter.

    Args:
        func: The function to check.
    """
    return "rollout_id" in inspect.signature(func).parameters


class LitAgent(Generic[T]):
    """Base class for the training and validation logic of an agent.

    Developers should subclass this class and implement the rollout methods
    to define the agent's behavior for a single task. The agent's logic
    is completely decoupled from the server communication and training
    infrastructure.
    """

    def __init__(self, *, trained_agents: Optional[str] = None) -> None:  # FIXME: str | None won't work for cli
        """
        Initialize the LitAgent.

        Args:
            trained_agents: Optional string representing the trained agents.
                            This can be used to track which agents have been trained by this instance.
                            Deprecated. Configure `agent_match` in adapter instead.
        """
        if trained_agents is not None:
            warnings.warn(
                "`trained_agents` is deprecated. Configure `agent_match` in adapter instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.trained_agents = trained_agents

        self._trainer_ref: weakref.ReferenceType[Trainer] | None = None
        self._runner_ref: weakref.ReferenceType[Runner[T]] | None = None

    def is_async(self) -> bool:
        """
        Check if the agent implements asynchronous rollout methods.
        Override this property for customized async detection logic.

        Returns:
            True if the agent has custom async rollout methods, False otherwise.
        """
        return (
            (
                hasattr(self, "training_rollout_async")
                and self.__class__.training_rollout_async is not LitAgent.training_rollout_async  # type: ignore
            )
            or (
                hasattr(self, "validation_rollout_async")
                and self.__class__.validation_rollout_async is not LitAgent.validation_rollout_async  # type: ignore
            )
            or (hasattr(self, "rollout_async") and self.__class__.rollout_async is not LitAgent.rollout_async)  # type: ignore
        )

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this agent.

        Args:
            trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer_ref = weakref.ref(trainer)

    def get_trainer(self) -> Trainer:
        """
        Get the trainer for this agent.

        Returns:
            The Trainer instance associated with this agent.
        """
        if self._trainer_ref is None:
            raise ValueError("Trainer has not been set for this agent.")
        trainer = self._trainer_ref()
        if trainer is None:
            raise ValueError("Trainer reference is no longer valid (object has been garbage collected).")
        return trainer

    @property
    def trainer(self) -> Trainer:
        """Convenient shortcut of self.get_trainer()."""
        return self.get_trainer()

    def get_tracer(self) -> Tracer:
        """
        Get the tracer for this agent.

        Returns:
            The Tracer instance associated with this agent.
        """
        if hasattr(self.runner, "tracer"):
            return self.runner.tracer  # type: ignore
        else:
            return self.trainer.tracer

    @property
    def tracer(self) -> Tracer:
        """Convenient shortcut of self.get_tracer()."""
        return self.get_tracer()

    def set_runner(self, runner: Runner[T]) -> None:
        """
        Set the runner for this agent.

        Args:
            runner: The runner instance that will handle the execution of rollouts.
        """
        self._runner_ref = weakref.ref(runner)

    def get_runner(self) -> Runner[T]:
        """
        Get the runner for this agent.

        Returns:
            The runner instance associated with this agent.
        """
        if self._runner_ref is None:
            raise ValueError("Runner has not been set for this agent.")
        runner = self._runner_ref()
        if runner is None:
            raise ValueError("Runner reference is no longer valid (object has been garbage collected).")
        return runner

    @property
    def runner(self) -> Runner[T]:
        """Convenient shortcut of self.get_runner()."""
        return self.get_runner()

    def on_rollout_start(self, task: Task, runner: Runner[T], tracer: Tracer) -> None:
        """Hook called immediately before a rollout begins.

        Args:
            task: The `Task` object that will be processed.
            runner: The [`Runner`][agentlightning.Runner] managing the rollout.
            tracer: The [`Tracer`][agentlightning.Tracer] instance associated with the runner.

        Deprecated:
            In favor of `on_rollout_start` in the [`Hook`][agentlightning.Hook] interface.

        Subclasses can override this method to implement custom logic such as
        logging, metric collection, or resource setup. By default, this is a
        no-op.
        """

    def on_rollout_end(self, task: Task, rollout: Rollout, runner: Runner[T], tracer: Tracer) -> None:
        """Hook called after a rollout completes.

        Deprecated in favor of `on_rollout_end` in the `Hook` interface.

        Args:
            task: The `Task` object that was processed.
            rollout: The resulting [`Rollout`][agentlightning.Rollout] object.
            runner: The [`Runner`][agentlightning.Runner] managing the rollout.
            tracer: The [`Tracer`][agentlightning.Tracer] instance associated with the runner.

        Subclasses can override this method for cleanup or additional
        logging. By default, this is a no-op.
        """

    def rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Main entry point for executing a rollout.

        This method determines whether to call the synchronous or
        asynchronous rollout method based on the agent's implementation.

        If you don't wish to implement both training rollout and validation
        rollout separately, you can just implement `rollout` which will work for both.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            resources: A dictionary of named resources (e.g., LLMs, prompt
                       templates) for the agent to use.
            rollout: The full rollout object, please avoid from directly modifying it.
                     Most agents should only use `task` and `resources`. Use `rollout`
                     only if you need to access metadata like `rollout_id`.

        Returns:
            The result of the rollout, which can be one of:
            - None. The tracing should be handled by the agent runner.
            - A float representing the final reward.
            - A list of `Triplet` objects for detailed, step-by-step feedback.
            - A list of `ReadableSpan` objects for OpenTelemetry tracing.
            - A list of dictionaries for any trace spans.
            - A complete `Rollout` object for full control over reporting.
        """
        raise NotImplementedError("Agents must implement the `rollout` method.")

    async def rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Asynchronous version of the main rollout method.

        This method determines whether to call the synchronous or
        asynchronous rollout method based on the agent's implementation.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            resources: A dictionary of named resources (e.g., LLMs, prompt
                       templates) for the agent to use.
            rollout: The full rollout object, please avoid from directly modifying it.
                     Most agents should only use `task` and `resources`. Use `rollout`
                     only if you need to access metadata like `rollout_id`.

        Returns:
            The result of the rollout, which can be one of:
            - None. The tracing should be handled by the agent runner.
            - A float representing the final reward.
            - A list of `Triplet` objects for detailed, step-by-step feedback.
            - A list of `ReadableSpan` objects for OpenTelemetry tracing.
            - A list of dictionaries for any trace spans.
            - A complete `Rollout` object for full control over reporting.
        """
        raise NotImplementedError("Agents must implement the `rollout_async` method for async operations.")

    def training_rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Defines the agent's behavior for a single training task.

        This method should contain the logic for how the agent processes an
        input, uses the provided resources (like LLMs or prompts), and
        produces a result.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            resources: A dictionary of named resources (e.g., LLMs, prompt
                       templates) for the agent to use.
            rollout: The full rollout object, please avoid from directly modifying it.
        """
        return self.rollout(task, resources, rollout)

    def validation_rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Defines the agent's behavior for a single validation task.

        By default, this method redirects to `training_rollout`. Override it
        if the agent should behave differently during validation.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            resources: A dictionary of named resources for the agent to use.
            rollout: The full rollout object, avoid from modifying it.

        Returns:
            The result of the validation rollout. See `rollout` for
            possible return types.
        """
        return self.rollout(task, resources, rollout)

    async def training_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Asynchronous version of `training_rollout`.

        This method should be implemented by agents that perform asynchronous
        operations (e.g., non-blocking I/O, concurrent API calls).

        Args:
            task: The task object received from the server.
            resources: A dictionary of named resources for the agent to use.
            rollout: The full rollout object, avoid from modifying it.

        Returns:
            The result of the asynchronous training rollout. See `rollout` for
            possible return types.
        """
        return await self.rollout_async(task, resources, rollout)

    async def validation_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Asynchronous version of `validation_rollout`.

        By default, this method redirects to `training_rollout_async`.
        Override it for different asynchronous validation behavior.

        Args:
            task: The task object received from the server.
            resources: A dictionary of named resources for the agent to use.
            rollout: The full rollout object, avoid from modifying it.

        Returns:
            The result of the asynchronous validation rollout. See `rollout` for
            possible return types.
        """
        return await self.rollout_async(task, resources, rollout)

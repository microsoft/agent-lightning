# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
import logging
import weakref
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Generic, Optional, TypeVar, Union

from .types import LLM, NamedResources, Rollout, RolloutRawResultV2, RolloutV2, Task

if TYPE_CHECKING:
    from .runner import BaseRunner
    from .tracer import BaseTracer
    from .trainer import Trainer


logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "LitAgent",
    "LitAgentLLM",
    "llm_rollout",
    "rollout",
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
        """
        self.trained_agents = trained_agents
        self._trainer_ref: weakref.ReferenceType[Trainer] | None = None
        self._runner_ref: weakref.ReferenceType[BaseRunner[T]] | None = None

    @property
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

    @property
    def trainer(self) -> Trainer:
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
    def tracer(self) -> BaseTracer:
        """
        Get the tracer for this agent.

        Returns:
            The BaseTracer instance associated with this agent.
        """
        return self.trainer.tracer

    def set_runner(self, runner: BaseRunner[T]) -> None:
        """
        Set the runner for this agent.

        Args:
            runner: The runner instance that will handle the execution of rollouts.
        """
        self._runner_ref = weakref.ref(runner)

    @property
    def runner(self) -> BaseRunner[T]:
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

    def on_rollout_start(self, task: Task, runner: BaseRunner[T], tracer: BaseTracer) -> None:
        """Hook called immediately before a rollout begins.

        Deprecated in favor of `on_rollout_start` in the `Hook` interface.

        Args:
            task: The :class:`Task` object that will be processed.
            runner: The :class:`BaseRunner` managing the rollout.
            tracer: The tracer instance associated with the runner.

        Subclasses can override this method to implement custom logic such as
        logging, metric collection, or resource setup. By default, this is a
        no-op.
        """

    def on_rollout_end(self, task: Task, rollout: RolloutV2, runner: BaseRunner[T], tracer: BaseTracer) -> None:
        """Hook called after a rollout completes.

        Deprecated in favor of `on_rollout_end` in the `Hook` interface.

        Args:
            task: The :class:`Task` object that was processed.
            rollout: The resulting :class:`Rollout` object.
            runner: The :class:`BaseRunner` managing the rollout.
            tracer: The tracer instance associated with the runner.

        Subclasses can override this method for cleanup or additional
        logging. By default, this is a no-op.
        """

    def rollout(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
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

    async def rollout_async(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
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

    def training_rollout(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
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

    def validation_rollout(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
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

    async def training_rollout_async(
        self, task: T, resources: NamedResources, rollout: RolloutV2
    ) -> RolloutRawResultV2:
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

    async def validation_rollout_async(
        self, task: T, resources: NamedResources, rollout: RolloutV2
    ) -> RolloutRawResultV2:
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


LlmRolloutFunc = Union[
    Callable[[T, LLM, Rollout], RolloutRawResultV2],
    Callable[[T, LLM], RolloutRawResultV2],
    Callable[[T, LLM, Rollout], Coroutine[Any, Any, RolloutRawResultV2]],
    Callable[[T, LLM], Coroutine[Any, Any, RolloutRawResultV2]],
]


class LitAgentLLM(LitAgent[T]):
    """A specialized LitAgent that wraps a function-based rollout that accepts
    dynamically a task input and a configured LLM.

    This class allows users to define agent behavior using a simple function
    that takes task input and an LLM resource, rather than implementing a full
    LitAgent subclass.
    """

    def __init__(self, llm_rollout_func: LlmRolloutFunc[T], *, trained_agents: Optional[str] = None) -> None:
        """
        Initialize the LitAgentLLM with an LLM rollout function.

        Args:
            llm_rollout_func: A function that defines the agent's behavior.
                              Can be sync or async, and can optionally accept a Rollout parameter.
            trained_agents: Optional string representing the trained agents.
                            This can be used to track which agents have been trained by this instance.
        """
        super().__init__(trained_agents=trained_agents)
        self.llm_rollout_func = llm_rollout_func
        self._is_async = inspect.iscoroutinefunction(llm_rollout_func)
        self._accepts_rollout = "rollout" in inspect.signature(llm_rollout_func).parameters

        # Copy function metadata to preserve type hints and other attributes
        functools.update_wrapper(self, llm_rollout_func)  # type: ignore

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the agent instance callable, preserving the original function behavior."""
        return self.llm_rollout_func(*args, **kwargs)

    @property
    def is_async(self) -> bool:
        return self._is_async

    def rollout(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
        """Execute a synchronous rollout using the wrapped function.

        Args:
            task: The task input data.
            resources: Dictionary of named resources including LLMs.
            rollout: The rollout object with metadata.

        Returns:
            The result from the wrapped rollout function.
        """
        if self._is_async:
            raise RuntimeError("This LitAgentLLM uses an async function. Use rollout_async instead.")

        # Find the first LLM resource
        llm = self._get_llm_resource(resources)

        if self._accepts_rollout:
            return self.llm_rollout_func(task, llm=llm, rollout=rollout)  # type: ignore
        else:
            return self.llm_rollout_func(task, llm=llm)  # type: ignore

    async def rollout_async(self, task: T, resources: NamedResources, rollout: RolloutV2) -> RolloutRawResultV2:
        """Execute an asynchronous rollout using the wrapped function.

        Args:
            task: The task input data.
            resources: Dictionary of named resources including LLMs.
            rollout: The rollout object with metadata.

        Returns:
            The result from the wrapped rollout function.
        """
        if not self._is_async:
            raise RuntimeError("This LitAgentLLM uses a sync function. Use rollout instead.")

        # Find the first LLM resource
        llm = self._get_llm_resource(resources)

        if self._accepts_rollout:
            return await self.llm_rollout_func(task, llm=llm, rollout=rollout)  # type: ignore
        else:
            return await self.llm_rollout_func(task, llm=llm)  # type: ignore

    def _get_llm_resource(self, resources: NamedResources) -> LLM:
        """Extract the first LLM resource from the resources dictionary.

        Args:
            resources: Dictionary of named resources.

        Returns:
            The first LLM resource found.

        Raises:
            ValueError: If no LLM resource is found.
        """
        resource_found: LLM | None = None
        for name, resource in resources.items():
            if isinstance(resource, LLM):
                if resource_found is not None:
                    logger.warning(f"Multiple LLM resources found in resources. Using the first one: '{name}'.")
                    break
                resource_found = resource

        if resource_found is None:
            raise ValueError("No LLM resource found in the provided resources.")
        return resource_found


def llm_rollout(func: LlmRolloutFunc[T], *, trained_agents: Optional[str] = None) -> LitAgentLLM[T]:
    """Create a LitAgentLLM from a function that takes (task, llm[, rollout]).

    This decorator allows you to define an agent using a simple function
    instead of creating a full LitAgent subclass. The returned LitAgentLLM
    instance is callable, preserving the original function's behavior.

    Args:
        func: A function that defines the agent's behavior. Can be:
              - sync: (task, llm) -> result
              - sync with rollout: (task, llm, rollout) -> result
              - async: async (task, llm) -> result
              - async with rollout: async (task, llm, rollout) -> result
        trained_agents: Optional string representing trained agents.

    Returns:
        A callable LitAgentLLM instance that preserves the original function's
        type hints and behavior while providing all agent functionality.

    Example:
        @llm_rollout
        def my_agent(task, llm):
            # Agent logic here
            return response

        # Function is still callable with original behavior
        result = my_agent(task, llm)

        # Agent methods are also available
        result = my_agent.rollout(task, resources, rollout)
    """
    return LitAgentLLM(func, trained_agents=trained_agents)


def rollout(func: Union[LlmRolloutFunc[T], Callable[..., Any]], *, trained_agents: Optional[str] = None) -> LitAgent[T]:
    """Create a LitAgent from a function, automatically detecting the appropriate type.

    This function inspects the provided callable and creates the appropriate
    agent type based on its signature. The returned agent instance is callable,
    preserving the original function's behavior and type hints.

    Args:
        func: A function that defines the agent's behavior.
        trained_agents: Optional string representing trained agents.

    Returns:
        A callable LitAgent subclass instance that preserves the original function's
        type hints and behavior while providing all agent functionality.

    Example:
        @rollout
        def my_agent(task, llm):
            client = OpenAI(base_url=llm.endpoint)
            response = client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": task.input}],
            )

        # Function is still callable with original behavior
        result = my_agent(task, llm)

        # Agent methods are also available
        result = my_agent.rollout(task, resources, rollout)

    Raises:
        NotImplementedError: If the function signature doesn't match any known patterns.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Check if it matches the LLM rollout API pattern
    # Should have at least 2 params, with the second one being 'llm' or typed as LLM
    if len(params) >= 2:
        second_param = sig.parameters[params[1]]
        # Check if the second parameter is named 'llm' or has LLM type annotation
        if second_param.name == "llm" or (
            second_param.annotation != inspect.Parameter.empty
            and (second_param.annotation == LLM or str(second_param.annotation).endswith("LLM"))
        ):
            return llm_rollout(func, trained_agents=trained_agents)

    raise NotImplementedError(
        f"Function signature {sig} does not match any known agent patterns. "
        "Expected signatures: (task, llm[, rollout]) or async (task, llm[, rollout])"
    )

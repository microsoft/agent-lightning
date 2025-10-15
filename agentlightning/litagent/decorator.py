# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Protocol, TypeGuard, TypeVar, Union, overload

from agentlightning.types import (
    LLM,
    AttemptedRollout,
    NamedResources,
    PromptTemplate,
    ProxyLLM,
    Rollout,
    RolloutRawResult,
)

from .litagent import LitAgent

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "llm_rollout",
    "prompt_rollout",
    "rollout",
]


T_contra = TypeVar("T_contra", contravariant=True)


class LlmRolloutFuncSync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM) -> RolloutRawResult: ...


class LlmRolloutFuncSync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM, rollout: Rollout) -> RolloutRawResult: ...


class LlmRolloutFuncAsync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM) -> Awaitable[RolloutRawResult]: ...


class LlmRolloutFuncAsync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM, rollout: Rollout) -> Awaitable[RolloutRawResult]: ...


LlmRolloutFunc = Union[
    LlmRolloutFuncSync2[T_contra],
    LlmRolloutFuncSync3[T_contra],
    LlmRolloutFuncAsync2[T_contra],
    LlmRolloutFuncAsync3[T_contra],
]


class PromptRolloutFuncSync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate) -> RolloutRawResult: ...


class PromptRolloutFuncAsync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate) -> Awaitable[RolloutRawResult]: ...


class PromptRolloutFuncSync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate, rollout: Rollout) -> RolloutRawResult: ...


class PromptRolloutFuncAsync3(Protocol[T_contra]):
    def __call__(
        self, task: T_contra, prompt_template: PromptTemplate, rollout: Rollout
    ) -> Awaitable[RolloutRawResult]: ...


PromptRolloutFunc = Union[
    PromptRolloutFuncSync2[T_contra],
    PromptRolloutFuncSync3[T_contra],
    PromptRolloutFuncAsync2[T_contra],
    PromptRolloutFuncAsync3[T_contra],
]


class FunctionalLitAgentFunc(Protocol[T_contra]):
    def __call__(
        self, task: T_contra, *args: Any, **kwargs: Any
    ) -> Union[RolloutRawResult, Awaitable[RolloutRawResult]]: ...


class FunctionalLitAgent(LitAgent[T]):
    """A specialized LitAgent that wraps a function-based rollout that accepts
    dynamically a task input and a configured resource (LLM / prompt template / ...).

    This class allows users to define agent behavior using a simple function
    that takes task input and a resource, rather than implementing a full
    LitAgent subclass.
    """

    def __init__(self, rollout_func: FunctionalLitAgentFunc[T], *, strip_proxy: bool = True) -> None:
        """
        Initialize the FunctionalLitAgent with a functional rollout function.

        Args:
            rollout_func: A function that defines the agent's behavior.
                          Can be sync or async, and can optionally accept a Rollout parameter.
                          The function signature determines which resources are injected (llm, prompt_template, etc.).
            strip_proxy: Whether to strip the ProxyLLM resource into a LLM resource when the function accepts an llm parameter.
                         Defaults to True.
        """
        super().__init__()
        self._rollout_func = rollout_func
        self._strip_proxy = strip_proxy
        self._is_async = inspect.iscoroutinefunction(rollout_func)
        self._sig = inspect.signature(rollout_func)

        # Copy function metadata to preserve type hints and other attributes
        functools.update_wrapper(self, rollout_func)  # type: ignore

    def _accepts_rollout(self) -> bool:
        return "rollout" in self._sig.parameters

    def _accepts_llm(self) -> bool:
        return "llm" in self._sig.parameters

    def _accepts_prompt_template(self) -> bool:
        return "prompt_template" in self._sig.parameters

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the agent instance callable, preserving the original function behavior."""
        return self._rollout_func(*args, **kwargs)  # type: ignore

    def is_async(self) -> bool:
        return self._is_async

    def rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Execute a synchronous rollout using the wrapped function.

        Args:
            task: The task input data.
            resources: Dictionary of named resources including LLMs.
            rollout: The rollout object with metadata.

        Returns:
            The result from the wrapped rollout function.
        """
        if self._is_async:
            raise RuntimeError(f"{self._rollout_func} is asynchronous. Use rollout_async instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return self._rollout_func(task, **kwargs)  # type: ignore

    async def rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Execute an asynchronous rollout using the wrapped function.

        Args:
            task: The task input data.
            resources: Dictionary of named resources including LLMs.
            rollout: The rollout object with metadata.

        Returns:
            The result from the wrapped rollout function.
        """
        if not self._is_async:
            raise RuntimeError(f"{self._rollout_func} is synchronous. Use rollout instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return await self._rollout_func(task, **kwargs)  # type: ignore

    def _get_kwargs(self, resources: NamedResources, rollout: Rollout) -> Dict[str, Any]:
        """Extract the kwargs needed for the rollout function based on its signature.

        Dynamically builds the kwargs dictionary by inspecting the function signature and
        including only the parameters the function accepts. This allows flexible function
        signatures that can request any combination of: rollout, llm, and/or prompt_template.

        Args:
            resources: Dictionary of named resources available for the rollout.
            rollout: The rollout object with metadata.

        Returns:
            A dictionary of kwargs to pass to the rollout function.
        """

        kwargs: Dict[str, Any] = {}
        if self._accepts_rollout():
            kwargs["rollout"] = rollout
        if self._accepts_llm():
            kwargs["llm"] = self._get_llm_resource(resources, rollout)
        if self._accepts_prompt_template():
            kwargs["prompt_template"] = self._get_prompt_template_resource(resources, rollout)

        return kwargs

    def _get_llm_resource(self, resources: NamedResources, rollout: Rollout) -> LLM:
        """Extract the first LLM resource from the resources dictionary.

        Strip the ProxyLLM resource into a LLM resource if needed.

        Args:
            resources: Dictionary of named resources.
            rollout: The rollout object with metadata.

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

        if self._strip_proxy:
            resource_found = self._strip_proxy_helper(resource_found, rollout)

        return resource_found

    def _get_prompt_template_resource(self, resources: NamedResources, rollout: Rollout) -> PromptTemplate:
        """Extract the first PromptTemplate resource from the resources dictionary.

        Args:
            resources: Dictionary of named resources.
            rollout: The rollout object with metadata. Not used in this method.

        Returns:
            The first PromptTemplate resource found.

        Raises:
            ValueError: If no PromptTemplate resource is found.
        """
        resource_found: PromptTemplate | None = None
        for name, resource in resources.items():
            if isinstance(resource, PromptTemplate):
                if resource_found is not None:
                    logger.warning(
                        f"Multiple prompt template resources found in resources. Using the first one: '{name}'."
                    )
                    break
                resource_found = resource

        if resource_found is None:
            raise ValueError("No prompt template resource found in the provided resources.")

        return resource_found

    def _strip_proxy_helper(self, proxy_llm: LLM, rollout: Rollout) -> LLM:
        """Strip the ProxyLLM resource into a concrete LLM resource.

        This method resolves ProxyLLM instances to their concrete LLM implementation
        by attaching the attempted rollout context. This is only used when the function
        signature accepts an 'llm' parameter and strip_proxy is True.

        Args:
            proxy_llm: The LLM resource, which may be a ProxyLLM.
            rollout: The rollout object with metadata.

        Returns:
            The concrete LLM resource.

        Raises:
            ValueError: If the rollout is not an AttemptedRollout (required for stripping ProxyLLM).
        """

        if not isinstance(proxy_llm, ProxyLLM):
            # Not a ProxyLLM, nothing to strip here.
            return proxy_llm

        # Rollout is still a Rollout here because API is not stabilized yet.
        # In practice, it must be an AttemptedRollout.
        if not isinstance(rollout, AttemptedRollout):
            raise ValueError("Rollout is not an AttemptedRollout.")

        return proxy_llm.with_attempted_rollout(rollout)


@overload
def llm_rollout(func: LlmRolloutFunc[T]) -> FunctionalLitAgent[T]: ...


@overload
def llm_rollout(*, strip_proxy: bool = True) -> Callable[[LlmRolloutFunc[T]], FunctionalLitAgent[T]]: ...


def llm_rollout(
    func: LlmRolloutFunc[T] | None = None, *, strip_proxy: bool = True
) -> FunctionalLitAgent[T] | Callable[[LlmRolloutFunc[T]], FunctionalLitAgent[T]]:
    """Create a FunctionalLitAgent from a function that takes (task, llm[, rollout]).

    This decorator allows you to define an agent using a simple function
    instead of creating a full LitAgent subclass. The returned FunctionalLitAgent
    instance is callable, preserving the original function's behavior.

    Args:
        func: A function that defines the agent's behavior. Can be:
              - sync: (task, llm) -> result
              - sync with rollout: (task, llm, rollout) -> result
              - async: async (task, llm) -> result
              - async with rollout: async (task, llm, rollout) -> result
        strip_proxy: Whether to strip the ProxyLLM resource into a LLM resource.
                     Defaults to True.

    Returns:
        A callable FunctionalLitAgent instance that preserves the original function's
        type hints and behavior while providing all agent functionality.

    Example:
        @llm_rollout
        def my_agent(task, llm):
            # Agent logic here
            return response

        @llm_rollout(strip_proxy=False)
        def my_agent_no_strip(task, llm):
            # Agent logic here
            return response

        # Function is still callable with original behavior
        result = my_agent(task, llm)

        # Agent methods are also available
        result = my_agent.rollout(task, resources, rollout)
    """

    def decorator(f: LlmRolloutFunc[T]) -> FunctionalLitAgent[T]:
        _validate_llm_rollout_func(f)
        return FunctionalLitAgent(f, strip_proxy=strip_proxy)

    if func is None:
        # Called with arguments: @llm_rollout(strip_proxy=False)
        return decorator
    else:
        # Called without arguments: @llm_rollout
        return decorator(func)


def _validate_llm_rollout_func(func: Any) -> TypeGuard[LlmRolloutFunc[Any]]:
    """Validate the function signature of a LLM rollout function.

    Ensures the function follows the expected pattern for LLM-based rollouts:
    - Must have at least 2 parameters
    - First parameter must be named 'task'
    - Must have a parameter named 'llm'
    - Optionally can have a 'rollout' parameter

    Args:
        func: The function to validate.

    Returns:
        True if the function signature is valid.

    Raises:
        ValueError: If the function signature does not match the expected pattern.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if len(params) < 2:
        raise ValueError(f"Function {func} must have at least 2 parameters.")
    if params[0] != "task":
        raise ValueError(f"Function {func} must be a positional parameter called 'task'.")
    if "llm" not in params:
        raise ValueError(f"Function {func} must have a positional parameter called 'llm'.")

    return True


@overload
def prompt_rollout(func: PromptRolloutFunc[T]) -> FunctionalLitAgent[T]: ...


@overload
def prompt_rollout() -> Callable[[PromptRolloutFunc[T]], FunctionalLitAgent[T]]: ...


def prompt_rollout(
    func: PromptRolloutFunc[T] | None = None,
) -> FunctionalLitAgent[T] | Callable[[PromptRolloutFunc[T]], FunctionalLitAgent[T]]:
    """Create a FunctionalLitAgent from a function that takes (task, prompt_template[, rollout]).

    This decorator is designed for agents that work with tunable prompt templates. It enables
    a workflow where algorithms manage and optimize the prompt template, while agents consume
    the template to perform rollouts. This is particularly useful for prompt optimization scenarios.

    Args:
        func: A function that defines the agent's behavior. Can be:
              - sync: (task, prompt_template) -> result
              - sync with rollout: (task, prompt_template, rollout) -> result
              - async: async (task, prompt_template) -> result
              - async with rollout: async (task, prompt_template, rollout) -> result

    Returns:
        A callable FunctionalLitAgent instance that preserves the original function's
        type hints and behavior while providing all agent functionality.

    Example:
        @prompt_rollout
        def my_agent(task, prompt_template):
            # Use the prompt template to generate a response
            messages = prompt_template.format(task=task.input)
            # ... perform rollout with the formatted prompt
            return response

        # Function is still callable with original behavior
        result = my_agent(task, prompt_template)

        # Agent methods are also available
        result = my_agent.rollout(task, resources, rollout)
    """

    def decorator(f: PromptRolloutFunc[T]) -> FunctionalLitAgent[T]:
        _validate_prompt_rollout_func(f)
        return FunctionalLitAgent(f)

    if func is None:
        return decorator
    else:
        return decorator(func)


def _validate_prompt_rollout_func(func: Any) -> TypeGuard[PromptRolloutFunc[Any]]:
    """Validate the function signature of a prompt rollout function.

    Ensures the function follows the expected pattern for prompt-template-based rollouts:
    - Must have at least 2 parameters
    - First parameter must be named 'task'
    - Must have a parameter named 'prompt_template'
    - Optionally can have a 'rollout' parameter

    Args:
        func: The function to validate.

    Returns:
        True if the function signature is valid.

    Raises:
        ValueError: If the function signature does not match the expected pattern.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if len(params) < 2:
        raise ValueError(f"Function {func} must have at least 2 parameters.")
    if params[0] != "task":
        raise ValueError(f"Function {func} must be a positional parameter called 'task'.")
    if "prompt_template" not in params:
        raise ValueError(f"Function {func} must have a positional parameter called 'prompt_template'.")

    return True


def rollout(func: Union[LlmRolloutFunc[T], PromptRolloutFunc[T], Callable[..., Any]]) -> FunctionalLitAgent[T]:
    """Create a LitAgent from a function, automatically detecting the appropriate type.

    This function inspects the provided callable and creates the appropriate
    agent type based on its signature. It supports both LLM-based and prompt-template-based
    agents. The returned agent instance is callable, preserving the original function's
    behavior and type hints.

    Args:
        func: A function that defines the agent's behavior. Supported signatures:
              - (task, llm[, rollout]) for LLM-based agents
              - (task, prompt_template[, rollout]) for prompt-template-based agents

    Returns:
        A callable FunctionalLitAgent instance that preserves the original function's
        type hints and behavior while providing all agent functionality.

    Example:
        # LLM-based agent
        @rollout
        def my_llm_agent(task, llm):
            client = OpenAI(base_url=llm.endpoint)
            response = client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": task.input}],
            )
            return response

        # Prompt-template-based agent
        @rollout
        def my_prompt_agent(task, prompt_template):
            messages = prompt_template.format(task=task.input)
            # ... perform rollout with the formatted prompt
            return response

        # Function is still callable with original behavior
        result = my_llm_agent(task, llm)

        # Agent methods are also available
        result = my_llm_agent.rollout(task, resources, rollout)

    Raises:
        NotImplementedError: If the function signature doesn't match any known patterns.
    """
    # Check if it matches the LLM rollout API pattern
    sig = inspect.signature(func)

    try:
        if _validate_llm_rollout_func(func):
            return llm_rollout(func)
    except ValueError:
        pass

    try:
        if _validate_prompt_rollout_func(func):
            return prompt_rollout(func)
    except ValueError:
        pass

    raise NotImplementedError(
        f"Function signature {sig} does not match any known agent patterns. "
        "Expected signatures: (task, llm[, rollout]) or (task, prompt_template[, rollout]). "
        "Functions can be sync or async."
    )

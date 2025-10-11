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
    RolloutRawResultV2,
    RolloutV2,
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
    def __call__(self, task: T_contra, llm: LLM) -> RolloutRawResultV2: ...


class LlmRolloutFuncSync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM, rollout: RolloutV2) -> RolloutRawResultV2: ...


class LlmRolloutFuncAsync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM) -> Awaitable[RolloutRawResultV2]: ...


class LlmRolloutFuncAsync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, llm: LLM, rollout: RolloutV2) -> Awaitable[RolloutRawResultV2]: ...


LlmRolloutFunc = Union[
    LlmRolloutFuncSync2[T_contra],
    LlmRolloutFuncSync3[T_contra],
    LlmRolloutFuncAsync2[T_contra],
    LlmRolloutFuncAsync3[T_contra],
]


class PromptRolloutFuncSync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate) -> RolloutRawResultV2: ...


class PromptRolloutFuncAsync2(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate) -> Awaitable[RolloutRawResultV2]: ...


class PromptRolloutFuncSync3(Protocol[T_contra]):
    def __call__(self, task: T_contra, prompt_template: PromptTemplate, rollout: RolloutV2) -> RolloutRawResultV2: ...


class PromptRolloutFuncAsync3(Protocol[T_contra]):
    def __call__(
        self, task: T_contra, prompt_template: PromptTemplate, rollout: RolloutV2
    ) -> Awaitable[RolloutRawResultV2]: ...


PromptRolloutFunc = Union[
    PromptRolloutFuncSync2[T_contra],
    PromptRolloutFuncSync3[T_contra],
    PromptRolloutFuncAsync2[T_contra],
    PromptRolloutFuncAsync3[T_contra],
]


class FunctionalLitAgentFunc(Protocol[T_contra]):
    def __call__(
        self, task: T_contra, *args: Any, **kwargs: Any
    ) -> Union[RolloutRawResultV2, Awaitable[RolloutRawResultV2]]: ...


class FunctionalLitAgent(LitAgent[T]):
    """A specialized LitAgent that wraps a function-based rollout that accepts
    dynamically a task input and a configured resource (LLM / prompt template / ...).

    This class allows users to define agent behavior using a simple function
    that takes task input and a resource, rather than implementing a full
    LitAgent subclass.
    """

    def __init__(self, rollout_func: FunctionalLitAgentFunc[T], *, strip_proxy: bool = True) -> None:
        """
        Initialize the FunctionalLitAgent with an functional rollout function.

        Args:
            rollout_func: A function that defines the agent's behavior.
                          Can be sync or async, and can optionally accept a Rollout parameter.
            strip_proxy: Whether to strip the ProxyLLM resource into a LLM resource.
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
            raise RuntimeError(f"{self._rollout_func} is asynchronous. Use rollout_async instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return self._rollout_func(task, **kwargs)  # type: ignore

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
            raise RuntimeError(f"{self._rollout_func} is synchronous. Use rollout instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return await self._rollout_func(task, **kwargs)  # type: ignore

    def _get_kwargs(self, resources: NamedResources, rollout: RolloutV2) -> Dict[str, Any]:
        """Extract the kwargs needed for the rollout function."""

        kwargs: Dict[str, Any] = {}
        if self._accepts_rollout():
            kwargs["rollout"] = rollout
        if self._accepts_llm():
            kwargs["llm"] = self._get_llm_resource(resources, rollout)
        if self._accepts_prompt_template():
            kwargs["prompt_template"] = self._get_prompt_template_resource(resources, rollout)

        return kwargs

    def _get_llm_resource(self, resources: NamedResources, rollout: RolloutV2) -> LLM:
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

    def _get_prompt_template_resource(self, resources: NamedResources, rollout: RolloutV2) -> PromptTemplate:
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

    def _strip_proxy_helper(self, proxy_llm: LLM, rollout: RolloutV2) -> LLM:
        """Strip the ProxyLLM resource into a LLM resource."""

        if not isinstance(proxy_llm, ProxyLLM):
            # Not a ProxyLLM, nothing to strip here.
            return proxy_llm

        # Rollout is still a RolloutV2 here because API is not stabilized yet.
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
    """Validate the function signature of a LLM rollout function."""
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

    This decorator helps users who want to tune the prompt template within their agents.
    Algorithms manage and update the prompt template, agents use the prompt template and then rollout.
    """

    def decorator(f: PromptRolloutFunc[T]) -> FunctionalLitAgent[T]:
        _validate_prompt_rollout_func(f)
        return FunctionalLitAgent(f)

    if func is None:
        return decorator
    else:
        return decorator(func)


def _validate_prompt_rollout_func(func: Any) -> TypeGuard[PromptRolloutFunc[Any]]:
    """Validate the function signature of a prompt rollout function."""
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
    agent type based on its signature. The returned agent instance is callable,
    preserving the original function's behavior and type hints.

    Args:
        func: A function that defines the agent's behavior.

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
        "Expected signatures: (task, llm[, rollout]) or async (task, llm[, rollout])"
    )

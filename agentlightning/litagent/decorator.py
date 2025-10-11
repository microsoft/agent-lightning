# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Protocol, TypeVar, Union, cast, overload

from agentlightning.types import LLM, AttemptedRollout, NamedResources, ProxyLLM, RolloutRawResultV2, RolloutV2

from .litagent import LitAgent

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "LitAgentLLM",
    "llm_rollout",
    # "prompt_rollout",
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


class LitAgentLLM(LitAgent[T]):
    """A specialized LitAgent that wraps a function-based rollout that accepts
    dynamically a task input and a configured LLM.

    This class allows users to define agent behavior using a simple function
    that takes task input and an LLM resource, rather than implementing a full
    LitAgent subclass.
    """

    def __init__(self, llm_rollout_func: LlmRolloutFunc[T], *, strip_proxy: bool = True) -> None:
        """
        Initialize the LitAgentLLM with an LLM rollout function.

        Args:
            llm_rollout_func: A function that defines the agent's behavior.
                              Can be sync or async, and can optionally accept a Rollout parameter.
            strip_proxy: Whether to strip the ProxyLLM resource into a LLM resource.
        """
        super().__init__()
        self.llm_rollout_func = llm_rollout_func
        self.strip_proxy = strip_proxy
        self._is_async = inspect.iscoroutinefunction(llm_rollout_func)
        self._accepts_rollout = "rollout" in inspect.signature(llm_rollout_func).parameters

        # Copy function metadata to preserve type hints and other attributes
        functools.update_wrapper(self, llm_rollout_func)  # type: ignore

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the agent instance callable, preserving the original function behavior."""
        return self.llm_rollout_func(*args, **kwargs)  # type: ignore

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

        # Strip ProxyLLM if needed
        if self.strip_proxy:
            llm = self._strip_proxy(llm, rollout)

        if self._accepts_rollout:
            llm_rollout_func = cast(LlmRolloutFuncSync3[T], self.llm_rollout_func)
            return llm_rollout_func(task, llm=llm, rollout=rollout)
        else:
            llm_rollout_func = cast(LlmRolloutFuncSync2[T], self.llm_rollout_func)
            return llm_rollout_func(task, llm=llm)

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

        # Strip ProxyLLM if needed
        if self.strip_proxy:
            llm = self._strip_proxy(llm, rollout)

        if self._accepts_rollout:
            llm_rollout_func = cast(LlmRolloutFuncAsync3[T], self.llm_rollout_func)
            return await llm_rollout_func(task, llm=llm, rollout=rollout)
        else:
            llm_rollout_func = cast(LlmRolloutFuncAsync2[T], self.llm_rollout_func)
            return await llm_rollout_func(task, llm=llm)

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

    def _strip_proxy(self, proxy_llm: LLM, rollout: RolloutV2) -> LLM:
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
def llm_rollout(func: LlmRolloutFunc[T]) -> LitAgentLLM[T]: ...


@overload
def llm_rollout(*, strip_proxy: bool = True) -> Callable[[LlmRolloutFunc[T]], LitAgentLLM[T]]: ...


def llm_rollout(
    func: LlmRolloutFunc[T] | None = None, *, strip_proxy: bool = True
) -> LitAgentLLM[T] | Callable[[LlmRolloutFunc[T]], LitAgentLLM[T]]:
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
        strip_proxy: Whether to strip the ProxyLLM resource into a LLM resource.
                     Defaults to True.

    Returns:
        A callable LitAgentLLM instance that preserves the original function's
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

    def decorator(f: LlmRolloutFunc[T]) -> LitAgentLLM[T]:
        return LitAgentLLM(f, strip_proxy=strip_proxy)

    if func is None:
        # Called with arguments: @llm_rollout(strip_proxy=False)
        return decorator
    else:
        # Called without arguments: @llm_rollout
        return decorator(func)


# def prompt_rollout(func: PromptRolloutFunc[T]) -> LitAgent[T]:


def rollout(func: Union[LlmRolloutFunc[T], Callable[..., Any]]) -> LitAgent[T]:
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
            return llm_rollout(func)

    raise NotImplementedError(
        f"Function signature {sig} does not match any known agent patterns. "
        "Expected signatures: (task, llm[, rollout]) or async (task, llm[, rollout])"
    )

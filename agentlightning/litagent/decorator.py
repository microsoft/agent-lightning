# Copyright (c) Microsoft. All rights reserved.

"""Convenience decorators for building lightweight `LitAgent` implementations."""

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
    """Adapter that turns plain rollout functions into [`LitAgent`][agentlightning.litagent.litagent.LitAgent] instances.

    The helper inspects the wrapped function to determine which resources to
    inject, allowing both synchronous and asynchronous callables to participate
    in the training loop without writing a dedicated subclass.
    """

    def __init__(self, rollout_func: FunctionalLitAgentFunc[T], *, strip_proxy: bool = True) -> None:
        """Initialize the wrapper around a rollout function.

        Args:
            rollout_func: Callable that implements the rollout. It may be synchronous
                or asynchronous and can optionally receive a
                [`Rollout`][agentlightning.types.core.Rollout] alongside resources such as
                ``llm`` or ``prompt_template``.
            strip_proxy: When ``True``, convert
                [`ProxyLLM`][agentlightning.types.resources.ProxyLLM] inputs into
                [`LLM`][agentlightning.types.resources.LLM] instances before calling the
                rollout function. Defaults to ``True``.
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
            task: Task input data.
            resources: Mapping of named resources available to the agent.
            rollout: Rollout metadata provided by the runtime.

        Returns:
            Result produced by the wrapped rollout function.

        Raises:
            RuntimeError: If the wrapped function is asynchronous.
        """
        if self._is_async:
            raise RuntimeError(f"{self._rollout_func} is asynchronous. Use rollout_async instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return self._rollout_func(task, **kwargs)  # type: ignore

    async def rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Execute an asynchronous rollout using the wrapped function.

        Args:
            task: Task input data.
            resources: Mapping of named resources available to the agent.
            rollout: Rollout metadata provided by the runtime.

        Returns:
            Result produced by the wrapped rollout coroutine.

        Raises:
            RuntimeError: If the wrapped function is synchronous.
        """
        if not self._is_async:
            raise RuntimeError(f"{self._rollout_func} is synchronous. Use rollout instead.")

        kwargs = self._get_kwargs(resources, rollout)
        return await self._rollout_func(task, **kwargs)  # type: ignore

    def _get_kwargs(self, resources: NamedResources, rollout: Rollout) -> Dict[str, Any]:
        """Prepare keyword arguments expected by the wrapped rollout function.

        Args:
            resources: Mapping of named resources available for the rollout.
            rollout: Rollout metadata provided by the runtime.

        Returns:
            Dictionary of keyword arguments to forward to the rollout function.
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
        """Retrieve the first LLM resource from the available resources.

        Args:
            resources: Mapping of named resources.
            rollout: Rollout metadata used when stripping proxy endpoints.

        Returns:
            First [`LLM`][agentlightning.types.resources.LLM] resource encountered.

        Raises:
            ValueError: If no LLM resource is present.
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
        """Retrieve the first prompt template resource from the available resources.

        Args:
            resources: Mapping of named resources.
            rollout: Rollout metadata (unused).

        Returns:
            First [`PromptTemplate`][agentlightning.types.resources.PromptTemplate] resource encountered.

        Raises:
            ValueError: If no prompt template resource is present.
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
        """Convert [`ProxyLLM`][agentlightning.types.resources.ProxyLLM] instances into concrete LLMs.

        Args:
            proxy_llm: Candidate LLM resource.
            rollout: Rollout metadata that provides rollout and attempt identifiers.

        Returns:
            [`LLM`][agentlightning.types.resources.LLM] with rollout context baked into the endpoint.

        Raises:
            ValueError: If the rollout is not an
                [`AttemptedRollout`][agentlightning.types.core.AttemptedRollout].
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
    """Create a [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] for LLM-based rollouts.

    Args:
        func: Callable defining the agent's behaviour. Supported signatures include:

            * ``(task, llm) -> result``
            * ``(task, llm, rollout) -> result``
            * ``async (task, llm) -> result``
            * ``async (task, llm, rollout) -> result``

        strip_proxy: When ``True``, convert proxy resources into concrete
            [`LLM`][agentlightning.types.resources.LLM] instances before calling the
            function. Defaults to ``True``.

    Returns:
        [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] that
        wraps the supplied function.

    Examples:
        ```python
        @llm_rollout
        def my_agent(task, llm):
            return llm.endpoint

        @llm_rollout(strip_proxy=False)
        def my_agent_no_strip(task, llm):
            return llm.model

        result = my_agent(task, llm)
        result = my_agent.rollout(task, resources, rollout)
        ```
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
    """Validate the function signature of an LLM rollout function.

    Args:
        func: Function to inspect.

    Returns:
        ``True`` when the signature matches the supported patterns.

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
    """Create a [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] for prompt-based rollouts.

    Args:
        func: Callable defining the agent's behaviour. Supported signatures include:

            * ``(task, prompt_template) -> result``
            * ``(task, prompt_template, rollout) -> result``
            * ``async (task, prompt_template) -> result``
            * ``async (task, prompt_template, rollout) -> result``

    Returns:
        [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] that
        wraps the supplied function.

    Examples:
        ```python
        @prompt_rollout
        def my_agent(task, prompt_template):
            messages = prompt_template.format(task=task.input)
            return messages

        result = my_agent(task, prompt_template)
        result = my_agent.rollout(task, resources, rollout)
        ```
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

    Args:
        func: Function to inspect.

    Returns:
        ``True`` when the signature matches the supported patterns.

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
    """Create a [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] from an arbitrary rollout function.

    Args:
        func: Callable that implements the rollout. Supported signatures include
            those listed in [`llm_rollout`][agentlightning.litagent.decorator.llm_rollout]
            and [`prompt_rollout`][agentlightning.litagent.decorator.prompt_rollout].

    Returns:
        [`FunctionalLitAgent`][agentlightning.litagent.decorator.FunctionalLitAgent] that
        wraps the supplied function.

    Examples:
        ```python
        @rollout
        def my_llm_agent(task, llm):
            return llm.model

        @rollout
        def my_prompt_agent(task, prompt_template):
            return prompt_template.format(task=task.input)

        result = my_llm_agent(task, llm)
        result = my_llm_agent.rollout(task, resources, rollout)
        ```

    Raises:
        NotImplementedError: If the function signature does not match a supported pattern.
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

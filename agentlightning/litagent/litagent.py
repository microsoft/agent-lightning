# Copyright (c) Microsoft. All rights reserved.

"""Base abstractions for building agents that plug into Agent Lightning."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Generic, TypeVar

from agentlightning.types import NamedResources, Rollout, RolloutResult

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "LitAgent",
]


def is_v0_1_rollout_api(func: Callable[..., Any]) -> bool:
    """Return `True` when the rollout function uses the deprecated v0.1 signature.

    The helper inspects the callable's signature to detect whether a `rollout_id`
    parameter is present, which indicates the legacy API.

    Args:
        func: Function to analyze.

    Returns:
        `True` if the callable exposes a `rollout_id` parameter.
    """
    return "rollout_id" in inspect.signature(func).parameters


class LitAgent(Generic[T]):
    """Base class for implementing agent rollouts.

    Subclasses override the rollout methods to process tasks while the trainer and
    runner infrastructure manages orchestration, tracing, and persistence.
    """

    def __init__(self) -> None:
        """Initialize the agent instance."""

    def is_async(self) -> bool:
        """Return `True` when the agent overrides any asynchronous rollout methods.

        Override this method for customized async detection logic.
        """
        return any(
            method_name in base.__dict__
            for base in type(self).__mro__
            if base is not LitAgent
            for method_name in ("training_rollout_async", "validation_rollout_async", "rollout_async")
        )

    def rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Execute a rollout synchronously.


        If you don't wish to implement both training rollout and validation
        rollout separately, you can just implement `rollout` which will work for both.

        Args:
            task: Task payload provided by the scheduler.
            resources: Mapping of named resources (for example LLMs or prompt templates).
            rollout: Rollout metadata. Avoid mutating this object directly unless a
                subclass needs to override defaults.

        Returns:
            One of the following values:

            * `None` when tracing is handled by the runner.
            * `float` representing the final reward.
            * `list[AgentSpanPayload]` with agent-defined spans.
        """
        raise NotImplementedError("Agents must implement the `rollout` method.")

    async def rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Execute a rollout asynchronously.

        Args:
            task: Task payload provided by the scheduler.
            resources: Mapping of named resources (for example LLMs or prompt templates).
            rollout: Rollout metadata. Avoid mutating this object directly unless a
                subclass needs to override defaults.

        Returns:
            Same possible return values as
            [`rollout`][agentlightning.LitAgent.rollout].
        """
        raise NotImplementedError("Agents must implement the `rollout_async` method for async operations.")

    def training_rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Process a single training task synchronously.

        By default, this method delegates to
        [`rollout`][agentlightning.LitAgent.rollout].
        """
        return self.rollout(task, resources, rollout)

    def validation_rollout(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Process a single validation task synchronously.

        Override this method when validation should differ from training. The default
        implementation delegates to
        [`training_rollout`][agentlightning.LitAgent.training_rollout].
        """
        return self.rollout(task, resources, rollout)

    async def training_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Process a single training task asynchronously.

        By default, this method delegates to
        [`rollout_async`][agentlightning.LitAgent.rollout_async].
        """
        return await self.rollout_async(task, resources, rollout)

    async def validation_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutResult:
        """Process a single validation task asynchronously.

        Override this method when validation should differ from training. The default
        implementation delegates to
        [`training_rollout_async`][agentlightning.LitAgent.training_rollout_async].
        """
        return await self.rollout_async(task, resources, rollout)

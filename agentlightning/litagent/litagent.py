# Copyright (c) Microsoft. All rights reserved.

"""Base abstractions for building agents that plug into Agent Lightning."""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

from agentlightning.types import NamedResources, Rollout, RolloutResult, Task

if TYPE_CHECKING:
    from agentlightning.runner import Runner
    from agentlightning.tracer import Tracer


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

    def __init__(self, *, trained_agents: Optional[str] = None) -> None:  # FIXME: str | None won't work for cli
        """Initialize the agent instance.

        Args:
            trained_agents: Optional identifier used by legacy tooling to mark trained
                agents.

        !!! warning "Deprecated"
            The `trained_agents` flag is deprecated. Configure `agent_match` in the adapter
            layer instead. See [`TracerTraceToTriplet`][agentlightning.TracerTraceToTriplet]
            for more details.
        """
        if trained_agents is not None:
            warnings.warn(
                "`trained_agents` is deprecated. Configure `agent_match` in adapter instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.trained_agents = trained_agents

    def is_async(self) -> bool:
        """Return `True` when the agent overrides any asynchronous rollout methods.

        Override this method for customized async detection logic.
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

    def on_rollout_start(self, task: Task, runner: Runner[T], tracer: Tracer) -> None:
        """Hook invoked immediately before a rollout begins.

        Subclasses can override this method to implement custom logic such as logging,
        metric collection, or resource setup. The default implementation is a no-op.

        Args:
            task: [`Task`][agentlightning.Task] that will be processed.
            runner: [`Runner`][agentlightning.Runner] managing the rollout.
            tracer: [`Tracer`][agentlightning.Tracer] associated with the runner.

        !!! warning "Deprecated"
            Override [`Hook.on_rollout_start`][agentlightning.Hook.on_rollout_start]
            instead of this method when extending agents.
        """

    def on_rollout_end(self, task: Task, rollout: Rollout, runner: Runner[T], tracer: Tracer) -> None:
        """Hook invoked after a rollout completes.

        Subclasses can override this method for cleanup or additional logging. The default
        implementation is a no-op.

        Args:
            task: [`Task`][agentlightning.Task] that was processed.
            rollout: Resulting [`Rollout`][agentlightning.Rollout].
            runner: [`Runner`][agentlightning.Runner] managing the rollout.
            tracer: [`Tracer`][agentlightning.Tracer] associated with the runner.

        !!! warning "Deprecated"
            Override [`Hook.on_rollout_end`][agentlightning.Hook.on_rollout_end]
            instead of this method when extending agents.
        """

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

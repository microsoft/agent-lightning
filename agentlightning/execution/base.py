# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
from typing import Protocol

from agentlightning.store.base import LightningStore

from .events import ExecutionEvent

logger = logging.getLogger(__name__)


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"0", "false", "no", "off"}


def resolve_managed_store_flag(value: bool | None) -> bool:
    """Determine whether execution helpers should wrap the provided store.

    The helper first honours an explicit `value`. When `None` it falls back
    to the `AGL_MANAGED_STORE` environment variable, accepting a variety
    of truthy and falsy spellings. Missing environment configuration defaults to
    `True` so that higher-level strategies create the appropriate client or
    server wrappers automatically.

    Args:
        value: Optional override supplied by the caller.

    Returns:
        `True` when a managed store should be created around the provided
        instance, otherwise `False`.

    Raises:
        ValueError: If `AGL_MANAGED_STORE` is set to an unsupported
            value.
    """

    if value is not None:
        return value

    env_value = os.getenv("AGL_MANAGED_STORE")
    if env_value is None:
        return True

    normalized = env_value.strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False

    raise ValueError("AGL_MANAGED_STORE must be one of 1, 0, true, false, yes, no, on, or off")


class AlgorithmBundle(Protocol):
    """Callable bundle produced by [`Trainer`][agentlightning.Trainer].

    Execution strategies treat the returned coroutine as opaque, only providing
    the shared store instance and cooperative stop event. Bundles typically
    encapsulate algorithm setup plus adapter and LLM proxy, etc.
    """

    async def __call__(self, store: LightningStore, event: ExecutionEvent) -> None:
        """Execute algorithm logic using ``store`` until completion or stop."""


class RunnerBundle(Protocol):
    """Callable bundle wrapping runner setup and the worker loop, as opposed to the
    [`AlgorithmBundle`][agentlightning.AlgorithmBundle]."""

    async def __call__(self, store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        """Execute runner logic for ``worker_id`` using ``store`` and ``event``."""


class ExecutionStrategy:
    """Coordinate algorithm and runner bundles within a single process abstraction.

    Strategies decide how many worker bundles to launch, whether to communicate
    through shared memory or an HTTP boundary, and how to react to shutdown
    signals. They intentionally avoid inspecting the bundle internals; instead,
    each bundle remains responsible for its own scheduling semantics.

    !!! note
        Implementations must honor the [execute()][agentlightning.ExecutionStrategy.execute]
        contract by propagating `KeyboardInterrupt` and ensuring resources are
        released when an error occurs on either side of the algorithm/runner
        pair.
    """

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        """Run the provided bundles using the configured orchestration model.

        Args:
            algorithm: Callable bundle responsible for algorithm execution.
            runner: Callable bundle for runner workers.
            store: Concrete [`LightningStore`][agentlightning.LightningStore]
                shared across bundles.

        Raises:
            NotImplementedError: Subclasses must provide the orchestration
                implementation.
        """

        raise NotImplementedError()

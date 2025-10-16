# Copyright (c) Microsoft. All rights reserved.

import logging
import os
from typing import Protocol

from agentlightning.store.base import LightningStore

from .events import ExecutionEvent

logger = logging.getLogger(__name__)


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"0", "false", "no", "off"}


def resolve_managed_store_flag(value: bool | None) -> bool:
    """Resolve the managed_store flag from an explicit value or environment."""

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
    async def __call__(self, store: LightningStore, event: ExecutionEvent) -> None:
        """Initalization and execution logic."""


class RunnerBundle(Protocol):
    async def __call__(self, store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        """Initalization and execution logic."""


class ExecutionStrategy:
    """When trainer has created the executable of algorithm and runner in two bundles,
    the execution strategy defines how to run them together, and how many parallel runners to run.

    The store is the centric place for the two bundles to communicate.

    The algorithm and runner's behavior (whether runner should perform one step or run forever,
    whether the algo would send out the tasks or not) are defined inside the bundle,
    and does not belong to the execution strategy.

    The execute should support Ctrl+C to exit gracefully.
    """

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        raise NotImplementedError()

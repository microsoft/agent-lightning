# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Protocol

from agentlightning.store.base import LightningStore

from .events import ExecutionEvent

logger = logging.getLogger(__name__)


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

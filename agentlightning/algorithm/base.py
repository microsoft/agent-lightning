# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
from typing import Awaitable, Union

from agentlightning.types import AlgorithmContext


class Algorithm:
    """Algorithm is the strategy, or tuner to train the agent."""

    def is_async(self) -> bool:
        """Return True if the algorithm is asynchronous."""
        return inspect.iscoroutinefunction(self.run)

    def run(self, context: AlgorithmContext) -> Union[None, Awaitable[None]]:
        """Subclasses should implement this method to implement the algorithm.

        Args:
            context: Runtime input for the algorithm execution.

        Returns:
            Algorithm should refrain from returning anything. It should just run the algorithm.
        """
        raise NotImplementedError("Subclasses must implement run().")

# Copyright (c) Microsoft. All rights reserved.

"""Agent-lightning glue around the Tinker reinforcement-learning algorithm.

This implements Agent-lightning's [`Algorithm`][agentlightning.Algorithm] interface
for quick one-click running.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import chz

from agentlightning.adapter import TracerTraceToTriplet
from agentlightning.algorithm import Algorithm
from agentlightning.llm_proxy import LLMProxy
from agentlightning.types import Dataset

from .train import Config, main_training_loop

logger = logging.getLogger(__name__)


class Tinker(Algorithm):
    """A wrapper around `agl_tinker.train` that uses Agent-lightning resources.

    Compared to the `agl_tinker.train` function, this class:

    * Pulls the store, tracer adapter, and LiteLLM proxy from the ambient
      Agent-lightning runtime instead of constructing its own.
    * Replaces the dataset configured in ``Config`` with the datasets provided
      by Agent-lightning so existing resource loaders (e.g., `agl.Dataset`)
      keep working.
    * Ensures the adapter is `TracerTraceToTriplet` because rollouts are
      reconstructed from spans rather than via Tinker's native data construction.
    """

    def __init__(self, config: Config) -> None:
        """Store the training configuration."""
        self.config = config

    async def run(
        self, train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None
    ) -> None:
        """Execute the Tinker training loop with Agent-lightning resources.

        Args:
            train_dataset: Dataset injected by Agent-lightning for training.
            val_dataset: Dataset injected by Agent-lightning for evaluation.

        Raises:
            ValueError: If mandatory datasets are missing or if the adapter is
                not a [`TracerTraceToTriplet`][agentlightning.TracerTraceToTriplet] instance.

        This mirrors `agl_tinker.train.main` but instead of launching
        a brand-new LiteLLM proxy it reuses (or lazily creates) the proxy
        managed by the Algorithm base class, so rollouts stay visible to the
        Agent-lightning store.
        """
        if train_dataset is None or val_dataset is None:
            raise ValueError("train_dataset and val_dataset are required")

        config = chz.replace(  # type: ignore
            self.config,
            dataset_builder=chz.replace(  # type: ignore
                self.config.dataset_builder, train_dataset=train_dataset, val_dataset=val_dataset
            ),
        )

        store = self.get_store()
        adapter = self.get_adapter()
        if not isinstance(adapter, TracerTraceToTriplet):
            raise ValueError("Adapter must be a TracerTraceToTriplet")
        llm_proxy = self.get_llm_proxy()
        if llm_proxy is None:
            logger.warning("No LLM proxy found, creating one for you.")

            llm_proxy = LLMProxy(
                port=config.llm_proxy_port,
                model_list=[],
                store=store,
                launch_mode="thread",
            )

        await main_training_loop(config, store, adapter, llm_proxy)  # type: ignore

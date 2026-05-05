# Copyright (c) Microsoft. All rights reserved.

"""GEPA algorithm integration for Agent Lightning."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentlightning.algorithm.base import Algorithm
from agentlightning.algorithm.utils import with_llm_proxy, with_store
from agentlightning.types import Dataset, NamedResources, PromptTemplate

from .callbacks import LightningGEPACallback
from .config import GEPAConfig
from .resources import PromptResourceCodec
from .rollout_adapter import LightningGEPAAdapter

if TYPE_CHECKING:
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.store.base import LightningStore

logger = logging.getLogger(__name__)


class GEPA(Algorithm):
    """Evolutionary prompt optimizer powered by GEPA's reflective mutation engine.

    GEPA evaluates prompt candidates via batched rollouts, builds reflective
    datasets from execution traces, and proposes improved candidates through
    LLM-driven reflection while tracking a Pareto frontier of per-example
    performance.

    The algorithm runs GEPA's synchronous ``optimize()`` in a worker thread
    via ``asyncio.to_thread``, while the `LightningGEPAAdapter` calls back
    to the async event loop for store operations.

    Args:
        config: GEPA optimizer and rollout configuration. Uses sensible
            defaults when ``None``.
        resource_name: Explicit resource key in ``initial_resources`` to
            optimize. When ``None``, auto-detects the first `PromptTemplate`.
    """

    def __init__(
        self,
        config: GEPAConfig | None = None,
        *,
        resource_name: str | None = None,
    ) -> None:
        self._config = config or GEPAConfig()
        self._resource_name = resource_name
        self._result: Any = None
        self._best_prompt: Optional[PromptTemplate] = None
        self._codec: Optional[PromptResourceCodec] = None

    @property
    def config(self) -> GEPAConfig:
        """The active configuration."""
        return self._config

    @property
    def result(self) -> Optional[Any]:
        """The full ``GEPAResult`` after ``run()`` completes, or ``None``."""
        return self._result

    def get_best_prompt(self) -> PromptTemplate:
        """Retrieve the best prompt discovered during optimization.

        Returns:
            The `PromptTemplate` corresponding to GEPA's best candidate.

        Raises:
            ValueError: If ``run()`` has not been called yet.
        """
        if self._best_prompt is None:
            raise ValueError("No best prompt available — run() has not been called yet")
        return self._best_prompt

    @with_llm_proxy()
    @with_store
    async def run(
        self,
        store: LightningStore,
        llm_proxy: Optional[LLMProxy],
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        """Execute GEPA optimization over the configured prompt resource.

        The method:

        1. Discovers the optimizable `PromptTemplate` from ``initial_resources``.
        2. Materializes AGL datasets into plain lists.
        3. Builds the adapter, callback, and reflection LM.
        4. Runs ``gepa.optimize()`` in a worker thread.
        5. Publishes the best candidate back to the store.

        Args:
            store: Injected by ``@with_store`` — callers should not provide this.
            llm_proxy: Injected by ``@with_llm_proxy()`` — callers should not provide this.
            train_dataset: Training examples for gradient computation. Required.
            val_dataset: Validation examples for candidate evaluation. Optional
                for GEPA (when ``None``, GEPA skips validation-based selection).

        Raises:
            ValueError: If ``train_dataset`` is ``None`` or ``initial_resources``
                are not set.
        """
        if train_dataset is None:
            raise ValueError("train_dataset is required for GEPA algorithm")

        initial_resources = self.get_initial_resources()
        if initial_resources is None:
            raise ValueError(
                "initial_resources are not set for GEPA algorithm. "
                "Use algorithm.set_initial_resources() or set it in Trainer()"
            )

        import gepa as gepa_lib

        # Build codec and seed candidate
        codec, seed_candidate = PromptResourceCodec.from_initial_resources(
            initial_resources, resource_name=self._resource_name
        )
        self._codec = codec

        # Materialize datasets into plain lists
        train_list: List[Any] = [train_dataset[i] for i in range(len(train_dataset))]
        val_list: Optional[List[Any]] = (
            [val_dataset[i] for i in range(len(val_dataset))] if val_dataset is not None else None
        )

        # Prepare adapter
        loop = asyncio.get_running_loop()
        version_counter: List[int] = [0]
        adapter = LightningGEPAAdapter(
            store=store,
            codec=codec,
            loop=loop,
            version_counter=version_counter,
            rollout_batch_timeout=self._config.rollout_batch_timeout,
            rollout_poll_interval=self._config.rollout_poll_interval,
        )

        # Prepare reflection LM
        reflection_lm = self._build_reflection_lm()

        # Prepare callback
        callback = LightningGEPACallback(use_wandb=self._config.use_wandb)

        # Build kwargs for gepa.optimize
        optimize_kwargs: Dict[str, Any] = {
            "seed_candidate": seed_candidate,
            "trainset": train_list,
            "adapter": adapter,
            "callbacks": [callback],
            "candidate_selection_strategy": self._config.candidate_selection_strategy,
            "frontier_type": self._config.frontier_type,
            "module_selector": self._config.module_selector,
            "seed": self._config.seed,
            "use_merge": self._config.use_merge,
            "max_merge_invocations": self._config.max_merge_invocations,
            "skip_perfect_score": self._config.skip_perfect_score,
            "perfect_score": self._config.perfect_score,
            "display_progress_bar": self._config.display_progress_bar,
            "raise_on_exception": self._config.raise_on_exception,
        }
        if val_list is not None:
            optimize_kwargs["valset"] = val_list
        if self._config.max_metric_calls is not None:
            optimize_kwargs["max_metric_calls"] = self._config.max_metric_calls
        if self._config.reflection_minibatch_size is not None:
            optimize_kwargs["reflection_minibatch_size"] = self._config.reflection_minibatch_size
        if reflection_lm is not None:
            optimize_kwargs["reflection_lm"] = reflection_lm
        if self._config.use_wandb:
            optimize_kwargs["use_wandb"] = True
            if self._config.wandb_api_key is not None:
                optimize_kwargs["wandb_api_key"] = self._config.wandb_api_key
            if self._config.wandb_init_kwargs:
                optimize_kwargs["wandb_init_kwargs"] = self._config.wandb_init_kwargs

        # Merge any extra kwargs
        optimize_kwargs.update(self._config.extra_kwargs)

        logger.info(
            "Starting GEPA optimization: seed_candidate keys=%s, train_size=%d, val_size=%s, max_metric_calls=%s",
            list(seed_candidate.keys()),
            len(train_list),
            len(val_list) if val_list else "N/A",
            self._config.max_metric_calls,
        )

        # Run synchronous GEPA in a worker thread
        gepa_result: Any = await asyncio.to_thread(gepa_lib.optimize, **optimize_kwargs)  # type: ignore[reportUnknownMemberType]

        self._result = gepa_result

        # Extract and publish the best candidate
        best_candidate: Any = gepa_result.best_candidate
        if isinstance(best_candidate, str):
            # Single-component case: GEPA returns a plain string
            best_candidate = {codec.resource_name: best_candidate}

        best_resources: NamedResources = codec.candidate_to_resources(best_candidate)
        self._best_prompt = best_resources[codec.resource_name]  # type: ignore[assignment]

        # Publish final best resources to the store
        best_version = f"gepa-best-v{version_counter[0]}"
        version_counter[0] += 1
        await store.update_resources(best_version, best_resources)

        best_score: float = 0.0
        if gepa_result.val_aggregate_scores:
            best_score = float(gepa_result.val_aggregate_scores[gepa_result.best_idx])
        logger.info(
            "GEPA optimization complete. Best candidate score: %.4f, published as resources_id=%s",
            best_score,
            best_version,
        )

    def _build_reflection_lm(self) -> Optional[Any]:
        """Build a reflection language model callable for GEPA if configured.

        Returns a callable matching GEPA's ``LanguageModel`` protocol::

            def __call__(self, prompt: str | list[dict[str, Any]]) -> str
        """
        if self._config.reflection_model is None:
            return None

        model_name = self._config.reflection_model
        extra_kwargs = dict(self._config.reflection_model_kwargs)

        def _litellm_completion(prompt: Any) -> str:
            import litellm  # type: ignore[reportUnknownMemberType]

            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            response: Any = litellm.completion(model=model_name, messages=messages, **extra_kwargs)  # type: ignore[reportUnknownMemberType]
            return str(response.choices[0].message.content)

        return _litellm_completion

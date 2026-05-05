# Copyright (c) Microsoft. All rights reserved.

"""Configuration for the GEPA algorithm integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class GEPAConfig:
    """Configuration for the GEPA evolutionary prompt optimizer.

    Groups GEPA optimizer knobs, rollout execution parameters, and
    reflection LLM configuration into a single configuration object.

    Args:
        max_metric_calls: Maximum number of evaluation calls before stopping.
            Maps to GEPA's budget constraint. ``None`` means unlimited.
        candidate_selection_strategy: Strategy for selecting candidates.
            ``"pareto"`` tracks per-instance best, ``"current_best"`` keeps the
            single highest scorer, ``"epsilon_greedy"`` adds exploration.
        frontier_type: Pareto frontier granularity. ``"instance"`` tracks
            per-example bests; ``"aggregate"`` tracks only the overall best.
        reflection_minibatch_size: Number of examples sampled for each
            reflection step. ``None`` lets GEPA choose automatically.
        module_selector: Strategy for choosing which component to update.
            ``"round_robin"`` cycles through components sequentially.
        seed: Random seed for reproducibility.
        use_merge: Whether to attempt merging top candidates.
        max_merge_invocations: Maximum number of merge attempts when
            ``use_merge`` is enabled.
        skip_perfect_score: Skip further evaluation when a candidate
            achieves perfect score.
        perfect_score: Value considered a perfect score.
        display_progress_bar: Show a progress bar during optimization.
        raise_on_exception: Raise exceptions from GEPA instead of logging.
        rollout_batch_timeout: Maximum seconds to wait for a rollout batch
            to complete before scoring incomplete rollouts as 0.0.
        rollout_poll_interval: Seconds between polling the store for
            rollout completion.
        reflection_model: Model identifier passed to ``litellm.completion``
            for GEPA's reflection/proposal calls. When ``None``, GEPA uses
            its own default model.
    """

    # GEPA optimizer knobs
    max_metric_calls: Optional[int] = None
    candidate_selection_strategy: Literal["pareto", "current_best", "epsilon_greedy"] = "pareto"
    frontier_type: Literal["instance", "aggregate"] = "instance"
    reflection_minibatch_size: Optional[int] = None
    module_selector: str = "round_robin"
    seed: int = 0
    use_merge: bool = False
    max_merge_invocations: int = 5
    skip_perfect_score: bool = True
    perfect_score: float = 1.0
    display_progress_bar: bool = False
    raise_on_exception: bool = True

    # Rollout execution parameters
    rollout_batch_timeout: float = 3600.0
    rollout_poll_interval: float = 2.0

    # Reflection LLM configuration
    reflection_model: Optional[str] = None
    reflection_model_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {}
    )  # pyright: ignore[reportUnknownVariableType]
    """Extra keyword arguments forwarded to ``litellm.completion`` for reflection
    calls. Useful for passing authentication parameters such as
    ``azure_ad_token_provider`` for Azure Entra ID."""

    # Experiment tracking
    use_wandb: bool = False
    """Enable Weights & Biases experiment tracking during optimization."""
    wandb_api_key: Optional[str] = None
    """W&B API key. When ``None``, relies on ``WANDB_API_KEY`` env var or prior ``wandb login``."""
    wandb_init_kwargs: Dict[str, Any] = field(default_factory=lambda: {})  # pyright: ignore[reportUnknownVariableType]
    """Extra keyword arguments forwarded to ``wandb.init()`` (e.g. ``project``, ``name``, ``tags``)."""

    # Extra kwargs forwarded to gepa.optimize()
    extra_kwargs: Dict[str, Any] = field(default_factory=lambda: {})  # pyright: ignore[reportUnknownVariableType]

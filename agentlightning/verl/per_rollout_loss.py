"""Rollout-level mean policy loss for VERL."""

from __future__ import annotations

from typing import Any

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss

PER_ROLLOUT_MEAN_LOSS_MODE = "per_rollout_mean"


def normalize_advantages_by_rollout(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_ids: Any,
    *,
    num_trained_rows: int,
) -> torch.Tensor:
    """Normalize each row by its rollout's token count and batch size."""
    if len(rollout_ids) != advantages.shape[0]:
        raise ValueError(f"rollout_ids length ({len(rollout_ids)}) must match advantages rows ({advantages.shape[0]})")
    if num_trained_rows <= 0:
        raise ValueError("num_trained_rows must be positive")

    row_token_counts = response_mask.sum(dim=-1).to(dtype=advantages.dtype)
    rollout_token_counts: dict[Any, float] = {}
    for row_index, rollout_id in enumerate(rollout_ids):
        rollout_token_counts[rollout_id] = rollout_token_counts.get(rollout_id, 0.0) + float(
            row_token_counts[row_index].item()
        )

    row_divisors = torch.tensor(
        [rollout_token_counts[rollout_id] * num_trained_rows for rollout_id in rollout_ids],
        dtype=advantages.dtype,
        device=advantages.device,
    ).clamp_min(1.0)
    return advantages / row_divisors.unsqueeze(-1)


@register_policy_loss(PER_ROLLOUT_MEAN_LOSS_MODE)
def compute_policy_loss_per_rollout_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clipped PPO loss from rollout-normalized advantages."""
    assert config is not None, "per_rollout_mean loss requires the actor config"

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0, f"clip_ratio_c must be greater than 1.0, got {clip_ratio_c}"

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    dp_size = config.global_batch_info.get("dp_size", 1) if config.global_batch_info else 1
    pg_loss = verl_F.masked_sum(pg_losses, response_mask) * (dp_size or 1)
    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, metrics


def register_in_worker() -> None:
    """Import hook used by Ray actor processes."""

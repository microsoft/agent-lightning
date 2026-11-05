# Copyright (c) Microsoft. All rights reserved.

"""Registered direct on-policy process REINFORCE loss for the locked FSDP engine."""

from __future__ import annotations

from typing import Any

import torch
from verl.trainer.ppo.core_algos import register_policy_loss

from agentlightning.verl.process_credit import PROCESS_LOSS_MODE


def compute_policy_loss_process_reinforce(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sum fixed-K/H action coefficients, cancelling native DP gradient averaging.

    This is a score-function loss at the collection policy, not a PPO surrogate.
    Native microbatches add without further row/token normalization. Inactive
    values are replaced before multiplication, including NaNs on padding/tools.
    """
    del loss_agg_mode
    if rollout_is_weights is not None:
        raise ValueError("process_reinforce forbids rollout importance weights")
    info = getattr(config, "global_batch_info", None)
    dp_size = info.get("dp_size") if info is not None else None
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("process_reinforce requires explicit global dp_size metadata")
    if not (old_log_prob.shape == log_prob.shape == advantages.shape == response_mask.shape):
        raise ValueError("process loss token geometries must match")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("process response mask must be binary")
    active = response_mask.bool()
    if any(not torch.isfinite(value[active]).all() for value in (log_prob, old_log_prob, advantages)):
        raise ValueError("process loss requires finite active logprobs/coefficients")
    safe_log_prob = torch.where(active, log_prob, torch.zeros_like(log_prob))
    safe_coefficients = torch.where(active, advantages.detach(), torch.zeros_like(advantages))
    if not torch.isfinite(safe_coefficients * dp_size).all():
        raise ValueError("process coefficient overflows native DP gradient scaling")
    loss = -(safe_log_prob * safe_coefficients).sum() * dp_size
    if not torch.isfinite(loss):
        raise ValueError("process loss accumulation overflows native tensor dtype")
    count = int(active.sum().item())
    return loss, {
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
        "actor/ppo_kl": float((old_log_prob[active] - log_prob[active]).detach().mean().item()) if count else 0.0,
        "actor/process_active_tokens": count,
        "actor/process_coefficient_l1": float(safe_coefficients.abs().sum().item()),
    }


register_policy_loss(PROCESS_LOSS_MODE)(compute_policy_loss_process_reinforce)


def build_process_actor_payload(batch, *, seed: int):
    """Use native padding conversion with exactly one complete optimizer minibatch."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    if batch.batch is None or not len(batch) or batch.meta_info.get("process_zero_action"):
        raise ValueError("zero-action process collections must skip actor updates explicitly")
    if not batch.meta_info.get("process_scheduled_count") or "old_log_probs" not in batch.batch:
        raise ValueError("process actor payload requires fixed-K coefficients and current-policy logprobs")
    payload = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(
        payload,
        mini_batch_size=len(batch),
        global_batch_size=len(batch),
        epochs=1,
        seed=seed,
        dataloader_kwargs={"shuffle": False},
        compute_loss=True,
        calculate_entropy=False,
        distillation_use_topk=False,
    )
    return payload

# Copyright (c) Microsoft. All rights reserved.

"""Rollout-level advantage computation for Agent Lightning training batches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage


def compute_rollout_level_advantage(
    batch: DataProto,
    *,
    adv_estimator: Any,
    gamma: float,
    lam: float,
    num_repeat: int,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any | None = None,
    compute_advantage_fn: Callable[..., DataProto] = compute_advantage,
) -> tuple[DataProto, dict[str, int]]:
    """Compute advantages once per rollout, then broadcast to rollout triplets."""
    rollout_ids = _required_non_tensor(batch, "rollout_id_list")
    if len(rollout_ids) != len(batch):
        raise RuntimeError(f"rollout_id_list length ({len(rollout_ids)}) must match batch length ({len(batch)})")

    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        uid_values = batch.non_tensor_batch.get("data_id_list")
        if uid_values is None:
            raise RuntimeError("rollout-level advantage requires uid or data_id_list in non_tensor_batch")
        batch.non_tensor_batch["uid"] = uid_values

    response_mask = _required_tensor(batch, "response_mask")
    token_level_rewards = _required_tensor(batch, "token_level_rewards")

    rollout_to_indices: dict[Any, list[int]] = {}
    for row_index, rollout_id in enumerate(rollout_ids):
        rollout_to_indices.setdefault(rollout_id, []).append(row_index)

    reward_sums = token_level_rewards.sum(dim=-1).detach().float()
    representative_indices: list[int] = []
    for rollout_id, row_indices in rollout_to_indices.items():
        representative_indices.append(row_indices[0])
        _validate_same_uid(rollout_id, row_indices, uid_values)
        _validate_same_reward(rollout_id, row_indices, reward_sums)

    rollout_batch = batch[representative_indices]
    rollout_batch = compute_advantage_fn(
        rollout_batch,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    rollout_scalars = _extract_rollout_scalars(
        rollout_batch,
        key="advantages",
        response_mask=rollout_batch.batch["response_mask"],
    )
    batch.batch["advantages"] = _broadcast_rollout_scalars(
        rollout_ids=rollout_ids,
        rollout_to_scalar=rollout_scalars,
        response_mask=response_mask,
    )

    if "returns" in rollout_batch.batch:
        return_scalars = _extract_rollout_scalars(
            rollout_batch,
            key="returns",
            response_mask=rollout_batch.batch["response_mask"],
        )
        batch.batch["returns"] = _broadcast_rollout_scalars(
            rollout_ids=rollout_ids,
            rollout_to_scalar=return_scalars,
            response_mask=response_mask,
        )

    metrics = {
        "training/rollout_level_advantage/n_rows": len(batch),
        "training/rollout_level_advantage/n_rollouts": len(rollout_to_indices),
        "training/rollout_level_advantage/n_multi_row_rollouts": sum(
            1 for row_indices in rollout_to_indices.values() if len(row_indices) > 1
        ),
        "training/rollout_level_advantage/max_rows_per_rollout": max(
            (len(row_indices) for row_indices in rollout_to_indices.values()),
            default=0,
        ),
    }
    return batch, metrics


def _required_non_tensor(batch: DataProto, key: str) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        raise RuntimeError(f"rollout-level advantage requires {key} in non_tensor_batch")
    return values


def _required_tensor(batch: DataProto, key: str) -> torch.Tensor:
    value = batch.batch.get(key)
    if value is None:
        raise RuntimeError(f"rollout-level advantage requires {key} in batch")
    return value


def _validate_same_uid(rollout_id: Any, row_indices: list[int], uid_values: Any) -> None:
    first_uid = uid_values[row_indices[0]]
    if any(uid_values[row_index] != first_uid for row_index in row_indices[1:]):
        raise RuntimeError(f"rollout-level advantage found multiple uid values for rollout_id={rollout_id!r}")


def _validate_same_reward(rollout_id: Any, row_indices: list[int], reward_sums: torch.Tensor) -> None:
    rollout_rewards = reward_sums[row_indices]
    if not torch.allclose(rollout_rewards, rollout_rewards[0].expand_as(rollout_rewards)):
        raise RuntimeError(
            "rollout-level advantage requires all triplets for the same rollout_id "
            f"to share the same scalar token_level_rewards sum; got rollout_id={rollout_id!r}"
        )


def _extract_rollout_scalars(
    batch: DataProto,
    *,
    key: str,
    response_mask: torch.Tensor,
) -> dict[Any, torch.Tensor]:
    values = _required_tensor(batch, key)
    rollout_ids = _required_non_tensor(batch, "rollout_id_list")
    scalars: dict[Any, torch.Tensor] = {}
    for row_index, rollout_id in enumerate(rollout_ids):
        masked_values = values[row_index][response_mask[row_index].bool()]
        if masked_values.numel() == 0:
            raise RuntimeError(f"rollout-level advantage cannot extract {key} for empty rollout_id={rollout_id!r}")
        first_value = masked_values[0]
        if not torch.allclose(masked_values, first_value.expand_as(masked_values)):
            raise RuntimeError(
                f"rollout-level advantage requires scalar outcome-style {key}; "
                f"got non-constant token values for rollout_id={rollout_id!r}"
            )
        scalars[rollout_id] = first_value.detach()
    return scalars


def _broadcast_rollout_scalars(
    *,
    rollout_ids: Any,
    rollout_to_scalar: dict[Any, torch.Tensor],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    row_scalars = torch.stack([rollout_to_scalar[rollout_id] for rollout_id in rollout_ids])
    row_scalars = row_scalars.to(device=response_mask.device)
    return row_scalars.unsqueeze(-1) * response_mask.to(dtype=row_scalars.dtype)

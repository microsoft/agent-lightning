from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch
from tensordict import TensorDict
from verl import DataProto

from agl_lite.verl.rollout_level_advantage import compute_rollout_level_advantage


def _batch(
    *,
    rewards: list[float] | None = None,
    rollout_ids: list[str] | None = None,
    data_ids: list[str] | None = None,
) -> DataProto:
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )
    rewards = rewards or [1.0, 1.0, 3.0]
    rollout_ids = rollout_ids or ["rollout-1", "rollout-1", "rollout-2"]
    data_ids = data_ids or ["data-1", "data-1", "data-1"]

    token_level_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    for row_index, reward in enumerate(rewards):
        token_level_rewards[row_index, 0] = reward

    batch = DataProto(
        batch=TensorDict(
            {
                "response_mask": response_mask,
                "token_level_rewards": token_level_rewards,
            },
            batch_size=len(rewards),
        )
    )
    batch.non_tensor_batch["rollout_id_list"] = np.array(rollout_ids, dtype=object)
    batch.non_tensor_batch["data_id_list"] = np.array(data_ids, dtype=object)
    batch.non_tensor_batch["uid"] = np.array(data_ids, dtype=object)
    return batch


def _compute_rollout_level_advantage(
    batch: DataProto,
    compute_advantage_fn: Any,
) -> tuple[DataProto, dict[str, int]]:
    return compute_rollout_level_advantage(
        batch,
        adv_estimator="grpo",
        gamma=1.0,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config={},
        compute_advantage_fn=compute_advantage_fn,
    )


def test_rollout_level_advantage_collapses_rollouts_and_broadcasts_triplets() -> None:
    captured: dict[str, Any] = {}

    def fake_compute_advantage(rollout_batch: DataProto, **kwargs: Any) -> DataProto:
        captured["rollout_ids"] = list(rollout_batch.non_tensor_batch["rollout_id_list"])
        captured["uids"] = list(rollout_batch.non_tensor_batch["uid"])
        captured["kwargs"] = kwargs

        response_mask = rollout_batch.batch["response_mask"]
        advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        returns = torch.zeros_like(response_mask, dtype=torch.float32)
        advantages[0] = response_mask[0].float() * 2.0
        advantages[1] = response_mask[1].float() * -1.0
        returns[0] = response_mask[0].float() * 3.0
        returns[1] = response_mask[1].float() * -2.0
        rollout_batch.batch["advantages"] = advantages
        rollout_batch.batch["returns"] = returns
        return rollout_batch

    batch, metrics = _compute_rollout_level_advantage(_batch(), fake_compute_advantage)

    assert captured["rollout_ids"] == ["rollout-1", "rollout-2"]
    assert captured["uids"] == ["data-1", "data-1"]
    assert captured["kwargs"]["adv_estimator"] == "grpo"
    assert captured["kwargs"]["num_repeat"] == 2
    assert metrics == {
        "training/rollout_level_advantage/n_rows": 3,
        "training/rollout_level_advantage/n_rollouts": 2,
        "training/rollout_level_advantage/n_multi_row_rollouts": 1,
        "training/rollout_level_advantage/max_rows_per_rollout": 2,
    }
    assert batch.batch["advantages"].tolist() == [
        [2.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0],
    ]
    assert batch.batch["returns"].tolist() == [
        [3.0, 3.0, 0.0],
        [3.0, 0.0, 0.0],
        [-2.0, -2.0, -2.0],
    ]


def test_rollout_level_advantage_requires_rollout_ids() -> None:
    batch = _batch()
    del batch.non_tensor_batch["rollout_id_list"]

    with pytest.raises(RuntimeError, match="rollout_id_list"):
        _compute_rollout_level_advantage(batch, lambda rollout_batch, **_: rollout_batch)


def test_rollout_level_advantage_rejects_inconsistent_rollout_rewards() -> None:
    batch = _batch(rewards=[1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError, match="same scalar token_level_rewards"):
        _compute_rollout_level_advantage(batch, lambda rollout_batch, **_: rollout_batch)


def test_rollout_level_advantage_rejects_non_scalar_token_advantages() -> None:
    def fake_compute_advantage(rollout_batch: DataProto, **_: Any) -> DataProto:
        rollout_batch.batch["advantages"] = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [-1.0, -1.0, -1.0],
            ]
        )
        return rollout_batch

    with pytest.raises(RuntimeError, match="scalar outcome-style advantages"):
        _compute_rollout_level_advantage(_batch(), fake_compute_advantage)

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch
from tensordict import TensorDict
from verl import DataProto

from agentlightning.verl.trainer import _grpo_group_metrics, _same_reward_uid_indices


def _make_batch(
    uids: list[str],
    token_rewards: list[list[float]],
) -> DataProto:
    batch_size = len(uids)
    max_len = max(len(row) for row in token_rewards)
    padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, row in enumerate(token_rewards):
        for j, val in enumerate(row):
            padded[i, j] = val

    batch = DataProto(
        batch=TensorDict(
            {"token_level_scores": padded},
            batch_size=batch_size,
        )
    )
    batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)
    return batch


def test_same_reward_uid_indices_ignores_singleton_groups() -> None:
    # prompt-1 has only 1 sample with reward 1.0 (singleton)
    # prompt-2 has only 1 sample with reward 0.0 (singleton)
    # prompt-3 has only 1 sample with reward -0.5 (singleton)
    batch = _make_batch(
        uids=["prompt-1", "prompt-2", "prompt-3"],
        token_rewards=[[1.0], [0.0], [-0.5]],
    )

    same_reward = _same_reward_uid_indices(batch)
    assert same_reward == []


def test_same_reward_uid_indices_detects_multi_sample_same_reward_groups() -> None:
    # prompt-1 has 2 samples with identical reward 1.0 (indices 0, 1) -> same reward
    # prompt-2 has 1 sample with reward 1.0 (index 2) -> singleton, must NOT be included
    # prompt-3 has 2 samples with different rewards 0.0, 1.0 (indices 3, 4) -> diverse
    # prompt-4 has 3 samples with identical reward -2.0 (indices 5, 6, 7) -> same reward
    batch = _make_batch(
        uids=[
            "prompt-1",
            "prompt-1",
            "prompt-2",
            "prompt-3",
            "prompt-3",
            "prompt-4",
            "prompt-4",
            "prompt-4",
        ],
        token_rewards=[
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [1.0],
            [-2.0],
            [-2.0],
            [-2.0],
        ],
    )

    same_reward = _same_reward_uid_indices(batch)
    assert set(same_reward) == {0, 1, 5, 6, 7}


def test_same_reward_uid_indices_handles_empty_and_missing_uid() -> None:
    assert _same_reward_uid_indices(DataProto()) == []

    batch_no_uid = DataProto(
        batch=TensorDict(
            {"token_level_scores": torch.ones(2, 4)},
            batch_size=2,
        )
    )
    assert _same_reward_uid_indices(batch_no_uid) == []


def test_grpo_group_metrics_excludes_singletons_from_zero_variance() -> None:
    # 3 prompts, all singletons: intra-group reward variance does not exist for N=1
    batch = _make_batch(
        uids=["p1", "p2", "p3"],
        token_rewards=[[1.0], [2.0], [3.0]],
    )
    metrics = _grpo_group_metrics(batch)
    assert metrics == {
        "training/n_groups": 3,
        "training/n_zero_adv_groups": 0,
    }


def test_grpo_group_metrics_detects_zero_variance_multi_sample_groups() -> None:
    # p1: 2 samples, rewards 1.0, 1.0 -> zero variance (zero adv group)
    # p2: 1 sample, reward 1.0 -> singleton (not zero adv group)
    # p3: 2 samples, rewards 0.0, 1.0 -> non-zero variance
    # p4: 2 samples, fractional rewards (1/3) -> zero variance with tolerance
    batch = _make_batch(
        uids=["p1", "p1", "p2", "p3", "p3", "p4", "p4"],
        token_rewards=[
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [1.0],
            [1.0 / 3.0],
            [1.0 / 3.0],
        ],
    )
    metrics = _grpo_group_metrics(batch)
    assert metrics == {
        "training/n_groups": 4,
        "training/n_zero_adv_groups": 2,  # p1 and p4
    }


def test_grpo_group_metrics_missing_keys_returns_empty() -> None:
    assert _grpo_group_metrics(DataProto()) == {}

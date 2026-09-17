# Copyright (c) Microsoft. All rights reserved.

# pyright: reportCallIssue=false

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch

from agentlightning.verl.per_rollout_loss import (
    CISPO_PER_ROLLOUT_MEAN_LOSS_MODE,
    PER_ROLLOUT_MEAN_LOSS_MODE,
    compute_policy_loss_cispo_per_rollout_mean,
    compute_policy_loss_per_rollout_mean,
    normalize_advantages_by_rollout,
)


class _Config:
    def __init__(self, dp_size: int = 1) -> None:
        self.clip_ratio = 0.2
        self.clip_ratio_low = None
        self.clip_ratio_high = None
        self.global_batch_info = {"dp_size": dp_size}

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_loss_is_registered() -> None:
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    assert PER_ROLLOUT_MEAN_LOSS_MODE in POLICY_LOSS_REGISTRY
    assert CISPO_PER_ROLLOUT_MEAN_LOSS_MODE in POLICY_LOSS_REGISTRY


def test_normalize_advantages_by_rollout() -> None:
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )
    advantages = torch.ones_like(response_mask, dtype=torch.float32)

    scaled = normalize_advantages_by_rollout(
        advantages,
        response_mask,
        ["A", "A", "B"],
        num_trained_rows=3,
    )

    a_mass = (scaled[:2] * response_mask[:2]).sum().item()
    b_mass = (scaled[2:] * response_mask[2:]).sum().item()
    assert a_mass == pytest.approx(1 / 3)
    assert b_mass == pytest.approx(1 / 3)


def test_policy_loss_matches_masked_sum() -> None:
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    advantages = torch.tensor([[0.5, 0.5, 0.0], [-0.2, -0.2, -0.2]])
    log_prob = torch.zeros(2, 3)

    loss, metrics = compute_policy_loss_per_rollout_mean(
        old_log_prob=log_prob,  # pyright: ignore[reportCallIssue]
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_Config(dp_size=2),
    )

    assert loss.item() == pytest.approx((-(advantages * response_mask).sum() * 2).item())
    assert metrics["actor/ppo_kl"] == pytest.approx(0.0)


def test_cispo_keeps_clipped_token_gradients_and_stops_ratio_gradient() -> None:
    ratios = torch.tensor([[2.0, 0.5, 1.1]])
    log_prob = ratios.log().requires_grad_()
    old_log_prob = torch.zeros_like(log_prob)
    advantages = torch.tensor([[2.0, -3.0, 4.0]])
    response_mask = torch.ones_like(log_prob, dtype=torch.bool)

    loss, metrics = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_Config(),  # pyright: ignore[reportCallIssue]
    )
    loss.backward()

    expected_gradient = -torch.tensor([[1.2, 0.8, 1.1]]) * advantages
    assert log_prob.grad is not None
    assert torch.allclose(log_prob.grad, expected_gradient)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(2 / 3)


def test_cispo_respects_mask_and_rollout_is_weights() -> None:
    log_prob = torch.tensor([[torch.log(torch.tensor(2.0)), torch.log(torch.tensor(2.0))]], requires_grad=True)
    old_log_prob = torch.zeros_like(log_prob)
    advantages = torch.ones_like(log_prob)
    response_mask = torch.tensor([[True, False]])

    loss, _ = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_Config(dp_size=2),
        rollout_is_weights=torch.tensor([[0.5, 100.0]]),  # pyright: ignore[reportCallIssue]
    )
    loss.backward()

    assert log_prob.grad is not None
    assert torch.allclose(log_prob.grad, torch.tensor([[-1.2, 0.0]]))


def test_normalize_advantages_validates_inputs() -> None:
    mask = torch.ones(2, 3, dtype=torch.long)
    advantages = torch.ones(2, 3)

    with pytest.raises(ValueError, match="rollout_ids length"):
        normalize_advantages_by_rollout(advantages, mask, ["A"], num_trained_rows=2)
    with pytest.raises(ValueError, match="num_trained_rows"):
        normalize_advantages_by_rollout(advantages, mask, ["A", "B"], num_trained_rows=0)

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch
from omegaconf import DictConfig

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


def _cispo_config(
    dp_size: int = 1,
    *,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
) -> DictConfig:
    return DictConfig(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": clip_ratio_low,
            "clip_ratio_high": clip_ratio_high,
            "global_batch_info": {"dp_size": dp_size},
        }
    )


def test_loss_is_registered() -> None:
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    assert PER_ROLLOUT_MEAN_LOSS_MODE in POLICY_LOSS_REGISTRY
    assert POLICY_LOSS_REGISTRY[CISPO_PER_ROLLOUT_MEAN_LOSS_MODE] is compute_policy_loss_cispo_per_rollout_mean


def test_cispo_detaches_importance_ratio() -> None:
    log_prob = torch.tensor([[-1.0]], dtype=torch.float64, requires_grad=True)
    old_log_prob = torch.tensor([[-1.0]], dtype=torch.float64, requires_grad=True)

    loss, _ = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob,
        log_prob,
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[True]]),
        "token-mean",
        _cispo_config(),
        None,
    )
    loss.backward()

    # Without detach, the forward loss is still 2 but the current-policy gradient becomes 0.
    assert loss.item() == pytest.approx(2.0)
    torch.testing.assert_close(log_prob.grad, torch.tensor([[-2.0]], dtype=torch.float64))
    assert old_log_prob.grad is None


def test_cispo_clipped_tokens_keep_policy_gradients() -> None:
    log_prob = torch.full((1, 3), -2.0, dtype=torch.float64, requires_grad=True)
    old_log_prob = log_prob.detach() - torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float64).log()

    loss, metrics = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob,
        log_prob,
        torch.tensor([[-2.0, 3.0, 4.0]], dtype=torch.float64),
        torch.tensor([[True, True, True]]),
        "token-mean",
        _cispo_config(),
        None,
    )
    loss.backward()

    # Clipped ratios [0.8, 1, 1.2] remain coefficients of the log-probability gradient.
    assert loss.item() == pytest.approx(12.4)
    torch.testing.assert_close(log_prob.grad, torch.tensor([[1.6, -3.0, -4.8]], dtype=torch.float64))
    assert metrics["actor/pg_clipfrac"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("clip_ratio_low", "clip_ratio_high", "expected_loss", "expected_gradient"),
    [
        pytest.param(None, None, 6.0, [-0.8, -1.0, -1.2], id="defaults"),
        pytest.param(0.1, None, 6.2, [-0.9, -1.0, -1.2], id="default-high"),
        pytest.param(None, 0.4, 6.4, [-0.8, -1.0, -1.4], id="default-low"),
        pytest.param(0.1, 0.4, 6.6, [-0.9, -1.0, -1.4], id="asymmetric"),
        pytest.param(10.0, 0.2, 4.9, [-0.25, -1.0, -1.2], id="high-only-clipping"),
    ],
)
def test_cispo_clip_configuration(
    clip_ratio_low: float | None,
    clip_ratio_high: float | None,
    expected_loss: float,
    expected_gradient: list[float],
) -> None:
    log_prob = torch.full((1, 3), -2.0, dtype=torch.float64, requires_grad=True)
    old_log_prob = log_prob.detach() - torch.tensor([[0.25, 1.0, 2.0]], dtype=torch.float64).log()

    loss, _ = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob,
        log_prob,
        torch.ones((1, 3), dtype=torch.float64),
        torch.tensor([[True, True, True]]),
        "token-mean",
        _cispo_config(clip_ratio_low=clip_ratio_low, clip_ratio_high=clip_ratio_high),
        None,
    )
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss)
    torch.testing.assert_close(log_prob.grad, torch.tensor([expected_gradient], dtype=torch.float64))


@pytest.mark.parametrize(
    ("weights", "dp_size", "expected_loss", "expected_gradient"),
    [
        pytest.param(None, 1, -0.8, [-2.0, 0.0, 2.4], id="masked"),
        pytest.param([0.5, 123.0, 2.0], 3, -22.8, [-3.0, 0.0, 14.4], id="weighted-and-scaled"),
    ],
)
def test_cispo_mask_and_importance_weights(
    weights: list[float] | None,
    dp_size: int,
    expected_loss: float,
    expected_gradient: list[float],
) -> None:
    log_prob = torch.full((1, 3), -2.0, dtype=torch.float64, requires_grad=True)
    old_log_prob = log_prob.detach() - torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64).log()

    loss, metrics = compute_policy_loss_cispo_per_rollout_mean(
        old_log_prob,
        log_prob,
        torch.tensor([[2.0, 99.0, -3.0]], dtype=torch.float64),
        torch.tensor([[True, False, True]]),
        "token-mean",
        _cispo_config(dp_size=dp_size),
        None if weights is None else torch.tensor([weights], dtype=torch.float64),
    )
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss)
    torch.testing.assert_close(log_prob.grad, torch.tensor([expected_gradient], dtype=torch.float64))
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)


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


def test_normalize_advantages_validates_inputs() -> None:
    mask = torch.ones(2, 3, dtype=torch.long)
    advantages = torch.ones(2, 3)

    with pytest.raises(ValueError, match="rollout_ids length"):
        normalize_advantages_by_rollout(advantages, mask, ["A"], num_trained_rows=2)
    with pytest.raises(ValueError, match="num_trained_rows"):
        normalize_advantages_by_rollout(advantages, mask, ["A", "B"], num_trained_rows=0)

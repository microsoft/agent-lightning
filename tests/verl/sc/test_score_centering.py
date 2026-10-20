# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch

from agentlightning.verl.sc.score_centering import ScoreCenteringConfig, global_token_mean, support_scores, token_loss


@pytest.mark.parametrize("mode", ["none", "mis", "tis"])
@pytest.mark.parametrize("centering", [False, True])
def test_gradient_matches_full_distribution_oracle(mode, centering):
    initial = torch.tensor([0.2, -0.7, 1.1, 0.9, -0.4, 0.1], dtype=torch.float64)
    ids = torch.tensor([[0, 2, 4, -1]])
    config = ScoreCenteringConfig(centering=centering, is_mode=mode)
    p = initial.softmax(0)
    q = torch.tensor([0.02, 0, 0.78, 0, 0.10, 0], dtype=torch.float64)
    tail = torch.tensor([False, True, False, True, False, True])
    q[tail] = (1 - q.sum()) * p[tail] / p[tail].sum()
    ratio = p / q
    weights = {"none": torch.ones_like(q), "mis": ratio * ((ratio >= 0.5) & (ratio <= 5)), "tis": ratio.clamp(max=2)}[
        mode
    ]
    expectation = torch.zeros_like(initial)
    for y in range(6):
        logits = initial.clone().requires_grad_()
        log_p = logits.log_softmax(-1)
        advantage = torch.tensor([1.7], dtype=torch.float64, requires_grad=True)
        scores = support_scores(
            log_p[ids.clamp_min(0)], ids, q[ids.clamp_min(0)].log(), log_p[y : y + 1], q[y : y + 1].log(), config
        )
        grad, adv_grad = torch.autograd.grad(
            token_loss(scores, advantage, config).sum(), (logits, advantage), allow_unused=True
        )
        basis = torch.eye(6, dtype=torch.float64)[y]
        expected = -1.7 * (weights[y] * (basis - p) - centering * (q * weights - (q * weights).sum() * p))
        torch.testing.assert_close(grad, expected, atol=1e-12, rtol=1e-12)
        assert adv_grad is None
        expectation += q[y] * grad
    if centering:
        torch.testing.assert_close(expectation, torch.zeros_like(expectation), atol=1e-12, rtol=0)


def test_matching_policies_have_zero_correction_gradient():
    logits = torch.tensor([[1.0, 0.2, -0.3, 0.5]], dtype=torch.float64, requires_grad=True)
    log_p = logits.log_softmax(-1)
    ids = torch.tensor([[0, 2]])
    scores = support_scores(
        log_p[:, [0, 2]], ids, log_p[:, [0, 2]].detach(), log_p[:, 1], log_p[:, 1].detach(), ScoreCenteringConfig()
    )
    gradient = torch.autograd.grad(scores["sc_correction"].sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_sample_outside_head_uses_true_behavior_logprob(mode):
    log_p = torch.tensor([[0.04, 0.3, 0.56, 0.1]], dtype=torch.float64).log().requires_grad_()
    config = ScoreCenteringConfig(is_mode=mode)
    scores = support_scores(
        log_p[:, :2],
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.1]]).log(),
        log_p[:, 3],
        torch.tensor([0.4]).log(),
        config,
    )
    expected = 0.25 if mode == "tis" else 0.0
    torch.testing.assert_close(
        scores["sc_sample_weight"], torch.tensor([expected], dtype=torch.float64), atol=1e-7, rtol=1e-7
    )
    if mode == "mis":
        gradient = torch.autograd.grad(token_loss(scores, torch.ones(1), config).sum(), log_p)[0]
        assert gradient[:, :2].abs().sum() > 0


def test_zero_tail_floor_matches_paper_surrogate():
    log_p = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    scores = support_scores(
        log_p[:, :2], torch.tensor([[0, 1]]), q.log(), log_p[:, 0], q[:, 0].log(), ScoreCenteringConfig()
    )
    rho = 1e-6 / 0.3
    expected = ((q - rho * log_p[:, :2].detach().exp()) * log_p[:, :2]).sum(-1)
    torch.testing.assert_close(scores["sc_correction"], expected)


@pytest.mark.parametrize("microbatch", [1, 2])
def test_four_rank_token_mean_matches_single_batch(microbatch):
    # Unequal generated-token counts on four ranks must not become rank means.
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    features = torch.arange(32, dtype=torch.float64).view(8, 4) / 10
    mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    reference = ((parameter * features).square()[mask]).mean()
    expected = torch.autograd.grad(reference, parameter)[0]
    grads = []
    for rank in range(4):
        gradient = 0
        for start in range(rank * 2, rank * 2 + 2, microbatch):
            stop = min(start + microbatch, rank * 2 + 2)
            loss = global_token_mean((parameter * features[start:stop]).square(), mask[start:stop], int(mask.sum()), 4)
            gradient = gradient + torch.autograd.grad(loss, parameter)[0]
        grads.append(gradient)
    torch.testing.assert_close(torch.stack(grads).mean(), expected)


def test_mis_masks_extreme_ratios_and_zero_terms_before_multiplication():
    log_p = torch.tensor([[-1000.0, -torch.inf, -torch.inf]], requires_grad=True)
    scores = support_scores(
        log_p,
        torch.tensor([[0, 1, -1]]),
        torch.tensor([[-0.2, -0.3, -torch.inf]]),
        torch.tensor([-torch.inf], requires_grad=True),
        torch.tensor([-1.0]),
        ScoreCenteringConfig(is_mode="mis"),
    )
    loss = token_loss(scores, torch.ones(1), ScoreCenteringConfig(is_mode="mis")).sum()
    assert torch.isfinite(loss)
    assert scores["sc_sample_weight"].item() == 0
    assert scores["sc_support_size"].item() == 2
    assert torch.isfinite(torch.autograd.grad(loss, log_p)[0]).all()

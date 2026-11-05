# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import itertools

import pytest
import torch

from agentlightning.credit.resolve import resolve_credit
from agentlightning.credit.types import ProcessCreditConfig
from agentlightning.verl.process_credit import ScheduledGroups, ScheduledRollout, compute_process_coefficients


def schedule(ids, size):
    return ScheduledGroups(
        rollouts=tuple(
            ScheduledRollout(rollout_id=rid, group_id="task", group_size=size, policy_version=7) for rid in ids
        )
    )


@pytest.mark.asyncio
async def test_fixed_h_and_every_scheduled_rollout_including_zero_action(sealed_snapshot):
    config = ProcessCreditConfig(enabled=True, gamma=0.5, normalization_tokens=10)
    active = resolve_credit(await sealed_snapshot(rollout_id="active", group_size=2), config)
    empty = resolve_credit(
        await sealed_snapshot(rollout_id="empty", group_size=2, action_count=0, rewards=(0,)), config
    )
    result = compute_process_coefficients((active, empty), schedule(("active", "empty"), 2), config)
    assert result.scheduled_count == 2
    assert result.coefficients["active"] == pytest.approx((0.02, 0.01, 0.0125))
    assert result.coefficients["empty"] == ()
    with pytest.raises(ValueError, match="scheduled"):
        compute_process_coefficients((active,), schedule(("active", "empty"), 2), config)


@pytest.mark.asyncio
@pytest.mark.parametrize("gamma", [0, 0.5, 1])
@pytest.mark.parametrize("baseline", ["zero", "task_loo"])
async def test_enumerated_score_gradient_equals_autograd_objective(sealed_snapshot, gamma, baseline):
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    p, q = theta.sigmoid(), phi.sigmoid()
    failure = 0.1
    probabilities = [
        torch.tensor(failure, dtype=torch.float64),
        (1 - failure) * (1 - p),
        (1 - failure) * p * (1 - q),
        (1 - failure) * p * q,
    ]
    rewards = [(), (0.4,), (-0.1, -0.3), (-0.1, 1)]
    # Continuation responses include a deterministic EOS token with logprob zero.
    token_logs = [(), ((1 - p).log(),), (p.log(), (1 - q).log()), (p.log(), q.log())]
    config = ProcessCreditConfig(enabled=True, gamma=gamma, normalization_tokens=2, baseline=baseline)
    episode_returns = [sum(gamma**i * r for i, r in enumerate(values)) for values in rewards]
    objective = (
        torch.stack([prob * value for prob, value in zip(probabilities, episode_returns, strict=True)]).sum() / 2
    )
    expected = torch.autograd.grad(objective, (theta, phi), retain_graph=True)
    size = 2 if baseline == "task_loo" else 1
    estimator = theta * 0 + phi * 0
    for outcomes in itertools.product(range(4), repeat=size):
        resolved = []
        for i, outcome in enumerate(outcomes):
            values = rewards[outcome]
            snapshot = await sealed_snapshot(
                rewards=values or (0,), action_count=len(values), rollout_id=f"r{i}", group_size=size
            )
            resolved.append(resolve_credit(snapshot, config))
        coefficients = compute_process_coefficients(
            tuple(resolved), schedule(tuple(f"r{i}" for i in range(size)), size), config
        )
        sampled_probability = torch.prod(torch.stack([probabilities[outcome] for outcome in outcomes])).detach()
        score = theta * 0 + phi * 0
        for i, outcome in enumerate(outcomes):
            for t, log_prob in enumerate(token_logs[outcome]):
                # A variable response length adds a deterministic EOS, not a
                # random observed-token denominator or an extra policy decision.
                score = score + coefficients.coefficients[f"r{i}"][t] * (log_prob + 0)
        estimator = estimator + sampled_probability * score
    actual = torch.autograd.grad(estimator, (theta, phi))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_realized_token_normalizer_counterexample():
    p = 0.4
    variable = p * (1 - p) / 2 + (1 - p) * (-p)
    fixed = p * (1 - p) / 2 + (1 - p) * (-p) / 2
    assert variable == pytest.approx(-0.12)
    assert fixed == 0


@pytest.mark.asyncio
async def test_nonzero_coefficients_cannot_silently_underflow_float32(sealed_snapshot):
    config = ProcessCreditConfig(enabled=True, normalization_tokens=1e100)
    resolved = resolve_credit(await sealed_snapshot(), config)
    with pytest.raises(ValueError, match="float32"):
        compute_process_coefficients((resolved,), schedule(("r",), 1), config)

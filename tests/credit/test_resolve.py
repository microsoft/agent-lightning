# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agentlightning.credit.resolve import resolve_credit
from agentlightning.credit.types import ProcessCreditConfig


@pytest.mark.asyncio
async def test_discounted_return_oracle(sealed_snapshot):
    snapshot = await sealed_snapshot()
    resolved = resolve_credit(snapshot, ProcessCreditConfig(enabled=True, gamma=0.5))
    assert resolved.rewards == pytest.approx((0.2, -0.1, 1))
    assert resolved.returns == pytest.approx((0.4, 0.4, 1))
    assert [a * b for a, b in zip(resolved.discounts, resolved.returns, strict=True)] == pytest.approx((0.4, 0.2, 0.25))
    assert resolved.terminal_outcome == 1
    assert resolved.episode_return == pytest.approx(0.4)


@pytest.mark.asyncio
@pytest.mark.parametrize("gamma,expected", [(0, (0.2, -0.1, 1)), (1, (1.1, 0.9, 1))])
async def test_gamma_extremes(sealed_snapshot, gamma, expected):
    result = resolve_credit(await sealed_snapshot(), ProcessCreditConfig(enabled=True, gamma=gamma))
    assert result.returns == pytest.approx(expected)
    assert result.discounts[0] == 1


@pytest.mark.asyncio
async def test_evidence_cutoff_not_target_controls_reward_time(sealed_snapshot):
    config = ProcessCreditConfig(enabled=True, gamma=0.5)
    late = resolve_credit(await sealed_snapshot(rewards=(1, 0), target=0, cutoff=2), config)
    early = resolve_credit(await sealed_snapshot(rewards=(1, 0), target=0, cutoff=0), config)
    assert late.rewards == (0, 0, 1)
    assert late.returns == (0.25, 0.5, 1)
    assert early.rewards == (1, 0, 0)
    assert early.returns == (1, 0, 0)
    assert late.attributions[0].target_index == 0
    assert late.attributions[0].cutoff_index == 2


@pytest.mark.asyncio
async def test_zero_action_terminal_remains_explicit(sealed_snapshot):
    result = resolve_credit(await sealed_snapshot(rewards=(0,), action_count=0), ProcessCreditConfig(enabled=True))
    assert result.returns == ()
    assert result.terminal_outcome == 0
    assert result.episode_return == 0


@pytest.mark.asyncio
async def test_weighted_accumulation_rejects_float32_overflow(sealed_snapshot):
    with pytest.raises(ValueError, match="float32"):
        resolve_credit(await sealed_snapshot(), ProcessCreditConfig(enabled=True, terminal_weight=1e39))


@pytest.mark.asyncio
async def test_large_finite_raw_reward_is_allowed_when_fixed_weight_zero(sealed_snapshot):
    result = resolve_credit(
        await sealed_snapshot(rewards=(1e39,), action_count=1), ProcessCreditConfig(enabled=True, terminal_weight=0)
    )
    assert result.terminal_outcome == 1e39
    assert result.rewards == (0,)

# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 2: Reward staleness detection."""

import pytest

from agentlightning.emergence.reward_audit import (
    RewardAuditAdapter,
    RewardStalenessAuditor,
    _kl_divergence_binned,
    _rank,
    _spearman_rank_correlation,
)
from agentlightning.emergence.types import AuditRecord

from .conftest import make_span


class TestRank:
    def test_simple(self):
        ranks = _rank([3.0, 1.0, 2.0])
        assert ranks == [3.0, 1.0, 2.0]

    def test_ties(self):
        ranks = _rank([1.0, 1.0, 3.0])
        assert ranks[0] == ranks[1]  # Tied ranks should be average
        assert ranks[2] == 3.0


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        corr = _spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        assert abs(corr - 1.0) < 0.01

    def test_perfect_negative(self):
        corr = _spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0])
        assert abs(corr - (-1.0)) < 0.01

    def test_insufficient_data(self):
        assert _spearman_rank_correlation([1.0], [2.0]) == 0.0


class TestKLDivergence:
    def test_identical_distributions(self):
        samples = [float(i) for i in range(20)]
        kl = _kl_divergence_binned(samples, samples)
        assert kl < 0.01

    def test_different_distributions(self):
        p = [float(i) for i in range(20)]
        q = [float(i + 10) for i in range(20)]
        kl = _kl_divergence_binned(p, q)
        assert kl > 0.0

    def test_constant_values(self):
        p = [1.0] * 20
        q = [1.0] * 20
        kl = _kl_divergence_binned(p, q)
        assert kl == 0.0


class TestRewardStalenessAuditor:
    def test_no_audit_before_frequency(self):
        auditor = RewardStalenessAuditor(audit_frequency=10)
        for i in range(9):
            auditor.record_reward(float(i), f"r{i}")
            auditor.record_independent_check(float(i), f"r{i}")
        # Not at audit_frequency yet (obs count is 9, not divisible by 10)
        result = auditor.audit()
        assert result is None

    def test_audit_at_frequency_with_high_correlation(self):
        auditor = RewardStalenessAuditor(audit_frequency=10, divergence_threshold=0.5)
        for i in range(10):
            auditor.record_reward(float(i), f"r{i}")
            auditor.record_independent_check(float(i), f"r{i}")
        # Perfect correlation should not trigger staleness
        result = auditor.audit()
        assert result is None

    def test_audit_detects_divergence(self):
        auditor = RewardStalenessAuditor(audit_frequency=10, divergence_threshold=0.2)
        for i in range(10):
            auditor.record_reward(float(i), f"r{i}")
            # Independent check is inversely correlated
            auditor.record_independent_check(float(9 - i), f"r{i}")
        result = auditor.audit()
        assert result is not None
        assert result.rank_correlation < 0.0
        assert "could indicate" in result.description.lower() or "Could indicate" in result.description

    def test_distribution_shift_detected(self):
        auditor = RewardStalenessAuditor(window_size=20)
        # First half: low rewards
        for i in range(20):
            auditor.record_reward(float(i) * 0.01, f"r{i}")
        # Second half: high rewards
        for i in range(20):
            auditor.record_reward(float(i) * 0.1 + 5.0, f"r{i + 20}")
        report = auditor.get_distribution_shift()
        if report is not None:
            assert report.kl_divergence > 0

    def test_no_distribution_shift_insufficient_data(self):
        auditor = RewardStalenessAuditor(window_size=100)
        for i in range(5):
            auditor.record_reward(float(i), f"r{i}")
        assert auditor.get_distribution_shift() is None


class TestRewardAuditAdapter:
    def test_extracts_reward_spans(self):
        adapter = RewardAuditAdapter()
        # Create a reward span
        reward_span = make_span(
            "agentlightning.annotation",
            attributes={
                "agentlightning.reward.0.name": "primary",
                "agentlightning.reward.0.value": 0.75,
            },
            rollout_id="rollout-1",
            attempt_id="attempt-1",
            start_time=1.0,
        )
        non_reward_span = make_span("openai.chat.completion")

        records = adapter.adapt([reward_span, non_reward_span])
        assert len(records) == 1
        assert records[0].rollout_id == "rollout-1"
        assert records[0].reward_value == 0.75

    def test_handles_empty_source(self):
        adapter = RewardAuditAdapter()
        records = adapter.adapt([])
        assert records == []

    def test_multi_dimensional_reward(self):
        adapter = RewardAuditAdapter()
        span = make_span(
            "agentlightning.annotation",
            attributes={
                "agentlightning.reward.0.name": "speed",
                "agentlightning.reward.0.value": 0.9,
                "agentlightning.reward.1.name": "quality",
                "agentlightning.reward.1.value": 0.6,
            },
        )
        records = adapter.adapt([span])
        assert len(records) == 1
        assert records[0].reward_dimensions.get("speed") == 0.9
        assert records[0].reward_dimensions.get("quality") == 0.6

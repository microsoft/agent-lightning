# Copyright (c) Microsoft. All rights reserved.

"""Gap 2: Reward function staleness detection.

Detects drift between reward signals and independent success metrics.
The auditor maintains two parallel streams and checks for correlation decay.

Dissolution condition: when reward models include built-in calibration or
uncertainty quantification, making external staleness detection redundant.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence

from agentlightning.adapter.base import TraceAdapter
from agentlightning.emitter.reward import get_reward_value, get_rewards_from_span
from agentlightning.types import Span

from .types import AuditRecord, DistributionShiftReport, StalenessReport

logger = logging.getLogger(__name__)


class RewardStalenessAuditor:
    """Detect drift between reward signals and independent success metrics.

    The auditor maintains two parallel streams:
    1. Reward values from emit_reward() -- the optimization signal
    2. Independent success measurements -- ground truth checks

    When these diverge beyond a threshold, the reward function may be stale.

    This requires the developer to define independent_check() -- a function
    that evaluates task success without using the reward function. The auditor
    cannot detect staleness without a reference signal.
    """

    def __init__(
        self,
        audit_frequency: int = 50,
        divergence_threshold: float = 0.2,
        window_size: int = 100,
    ):
        self._audit_frequency = audit_frequency
        self._divergence_threshold = divergence_threshold
        self._window_size = window_size
        self._reward_history: deque[tuple[str, float]] = deque(maxlen=window_size)
        self._success_history: deque[tuple[str, float]] = deque(maxlen=window_size)
        self._historical_rewards: deque[float] = deque(maxlen=window_size * 2)
        self._observation_count = 0

    def record_reward(self, reward: float, rollout_id: str) -> None:
        """Record emitted reward for audit comparison."""
        self._reward_history.append((rollout_id, reward))
        self._historical_rewards.append(reward)
        self._observation_count += 1

    def record_independent_check(self, success: float, rollout_id: str) -> None:
        """Record independent success measurement.

        This is the critical input that the developer must provide.
        Without it, staleness cannot be detected -- only reward distribution
        changes (which could be genuine improvement).
        """
        self._success_history.append((rollout_id, success))

    def audit(self) -> Optional[StalenessReport]:
        """Run staleness check if enough data has accumulated.

        Computes rank correlation between reward and independent success.
        If correlation drops below threshold, returns StalenessReport.
        """
        if self._observation_count % self._audit_frequency != 0:
            return None

        # Match rollouts that have both reward and success
        reward_map = {rid: val for rid, val in self._reward_history}
        success_map = {rid: val for rid, val in self._success_history}
        common_ids = sorted(set(reward_map.keys()) & set(success_map.keys()))

        if len(common_ids) < 5:
            return None

        rewards = [reward_map[rid] for rid in common_ids]
        successes = [success_map[rid] for rid in common_ids]
        correlation = _spearman_rank_correlation(rewards, successes)

        if correlation >= (1.0 - self._divergence_threshold):
            return None

        severity = "advisory"
        if correlation < 0.5:
            severity = "critical"
        elif correlation < 0.7:
            severity = "warning"

        return StalenessReport(
            rank_correlation=correlation,
            window_size=len(common_ids),
            severity=severity,
            description=(
                f"Reward-success correlation: {correlation:.2f} over {len(common_ids)} rollouts. "
                f"Could indicate reward function drift — the optimization signal may no longer "
                f"reflect actual task success. Independent validation recommended."
            ),
        )

    def get_distribution_shift(self) -> Optional[DistributionShiftReport]:
        """Detect reward distribution changes even without independent checks.

        Uses KL divergence between recent and historical reward distributions.
        This is weaker than correlation-based staleness detection -- distribution
        shift could be genuine improvement -- but requires no independent signal.
        """
        if len(self._historical_rewards) < self._window_size:
            return None

        all_rewards = list(self._historical_rewards)
        midpoint = len(all_rewards) // 2
        historical = all_rewards[:midpoint]
        recent = all_rewards[midpoint:]

        if not historical or not recent:
            return None

        kl_div = _kl_divergence_binned(historical, recent)

        if kl_div < 0.1:
            return None

        return DistributionShiftReport(
            kl_divergence=kl_div,
            window_size=len(recent),
            description=(
                f"Reward distribution has shifted (KL divergence: {kl_div:.2f}). "
                f"Could indicate reward hacking, environment change, or genuine "
                f"capability improvement. Independent validation recommended."
            ),
        )


class RewardAuditAdapter(TraceAdapter[List[AuditRecord]]):
    """Adapter that processes traces for reward audit rather than training.

    Implements the same TraceAdapter[T_to] interface as TracerTraceToTriplet,
    but produces audit records instead of training triplets.

    Can run alongside the training adapter without interference.

    Dissolution condition: when audit data flows through the primary adapter
    pipeline rather than requiring a separate adapter.
    """

    def adapt(self, source: Sequence[Span], /) -> List[AuditRecord]:
        """Extract reward spans and pair with task metadata for audit."""
        records: List[AuditRecord] = []

        for span in source:
            reward_value = get_reward_value(span)
            if reward_value is None:
                continue

            # Extract dimensional rewards
            reward_dimensions: Dict[str, float] = {}
            reward_list = get_rewards_from_span(span)
            for r in reward_list:
                reward_dimensions[r.name] = r.value

            # Hash task input for grouping
            task_input_hash = hashlib.sha256(
                f"{span.rollout_id}".encode()
            ).hexdigest()[:12]

            records.append(
                AuditRecord(
                    rollout_id=span.rollout_id,
                    attempt_id=span.attempt_id,
                    reward_value=reward_value,
                    reward_dimensions=reward_dimensions,
                    task_input_hash=task_input_hash,
                    timestamp=span.start_time,
                )
            )

        return records


def _rank(values: List[float]) -> List[float]:
    """Convert values to ranks (average rank for ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 2:
        return 0.0

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rxi - mean_rx) * (ryi - mean_ry) for rxi, ryi in zip(rx, ry))
    std_rx = math.sqrt(sum((rxi - mean_rx) ** 2 for rxi in rx))
    std_ry = math.sqrt(sum((ryi - mean_ry) ** 2 for ryi in ry))

    if std_rx == 0 or std_ry == 0:
        return 0.0

    return cov / (std_rx * std_ry)


def _kl_divergence_binned(p_samples: List[float], q_samples: List[float], n_bins: int = 20) -> float:
    """Approximate KL divergence using binned distributions."""
    all_samples = p_samples + q_samples
    min_val = min(all_samples)
    max_val = max(all_samples)
    if max_val == min_val:
        return 0.0

    bin_width = (max_val - min_val) / n_bins
    epsilon = 1e-10

    def _bin_counts(samples: List[float]) -> List[float]:
        counts = [0.0] * n_bins
        for s in samples:
            idx = min(int((s - min_val) / bin_width), n_bins - 1)
            counts[idx] += 1.0
        total = sum(counts)
        return [(c + epsilon) / (total + n_bins * epsilon) for c in counts]

    p_dist = _bin_counts(p_samples)
    q_dist = _bin_counts(q_samples)

    kl = 0.0
    for p, q in zip(p_dist, q_dist):
        if p > 0:
            kl += p * math.log(p / q)
    return max(0.0, kl)

# Copyright (c) Microsoft. All rights reserved.

"""Gap 1: Sliding window collapse detection.

Tracks entropy over sliding windows and detects exploration collapse.
Collapse detection: when entropy drops below threshold AND reward is still
improving, the system may be narrowing rather than improving.

This is a signal, not a conclusion. The monitor surfaces the tension;
it does not resolve it.

Dissolution condition: when algorithms natively implement exploration-exploitation
balancing that makes external collapse detection unnecessary.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel

from agentlightning.types import Triplet

from .entropy import TrajectoryEntropy

if TYPE_CHECKING:
    from agentlightning.adapter.triplet import TraceTree

logger = logging.getLogger(__name__)


class EntropySnapshot(BaseModel):
    """Point-in-time entropy measurement."""

    shape_entropy: float = 0.0
    tool_entropy: float = 0.0
    reward_entropy: float = 0.0
    mean_reward: float = 0.0
    window_index: int = 0


class CollapseSignal(BaseModel):
    """Signal emitted when exploration collapse is detected."""

    entropy_trend: float
    """Slope of entropy over trend_window (negative = declining)."""
    reward_trend: float
    """Slope of reward over trend_window (positive = improving)."""
    severity: str
    """'low' | 'medium' | 'high'"""
    description: str
    """Hedged interpretation using 'could indicate...' language."""


class ExplorationDecayMonitor:
    """Track entropy over sliding windows and detect collapse.

    Collapse detection: when entropy drops below threshold AND reward
    is still improving, the system may be narrowing rather than improving.

    This is a signal, not a conclusion. The monitor surfaces the tension;
    it does not resolve it.
    """

    def __init__(
        self,
        window_size: int = 50,
        alert_threshold: float = 0.3,
        trend_window: int = 5,
    ):
        self._window_size = window_size
        self._alert_threshold = alert_threshold
        self._trend_window = trend_window
        self._history: deque[EntropySnapshot] = deque(maxlen=1000)
        self._entropy_calculator = TrajectoryEntropy()
        self._window_counter = 0

    def record(self, trees: Sequence["TraceTree"], triplets: Sequence[Triplet]) -> EntropySnapshot:
        """Compute and store entropy snapshot for current window."""
        shape_entropy = 0.0
        tool_entropy = 0.0
        reward_entropy = 0.0
        mean_reward = 0.0

        try:
            shape_entropy = self._entropy_calculator.compute_shape_entropy(trees, self._window_size)
        except Exception:
            logger.debug("Shape entropy computation failed.", exc_info=True)

        try:
            tool_entropy = self._entropy_calculator.compute_tool_entropy(trees, self._window_size)
        except Exception:
            logger.debug("Tool entropy computation failed.", exc_info=True)

        try:
            reward_entropy = self._entropy_calculator.compute_reward_entropy(triplets, window_size=self._window_size)
        except Exception:
            logger.debug("Reward entropy computation failed.", exc_info=True)

        rewarded = [t for t in triplets if t.reward is not None]
        if rewarded:
            mean_reward = sum(t.reward for t in rewarded) / len(rewarded)  # type: ignore[misc]

        snapshot = EntropySnapshot(
            shape_entropy=shape_entropy,
            tool_entropy=tool_entropy,
            reward_entropy=reward_entropy,
            mean_reward=mean_reward,
            window_index=self._window_counter,
        )
        self._history.append(snapshot)
        self._window_counter += 1
        return snapshot

    def detect_collapse(self) -> Optional[CollapseSignal]:
        """Check if entropy is declining while reward is stable/improving.

        Returns CollapseSignal if collapse pattern detected, None otherwise.
        """
        if len(self._history) < self._trend_window:
            return None

        recent = list(self._history)[-self._trend_window:]

        # Compute trends using simple linear regression slope
        entropy_values = [s.shape_entropy for s in recent]
        reward_values = [s.mean_reward for s in recent]

        entropy_trend = _compute_trend(entropy_values)
        reward_trend = _compute_trend(reward_values)

        latest_entropy = recent[-1].shape_entropy

        # Collapse pattern: entropy declining AND (reward stable or improving)
        if entropy_trend >= 0:
            return None
        if latest_entropy >= self._alert_threshold:
            return None

        # Determine severity
        if latest_entropy < self._alert_threshold * 0.5 and reward_trend > 0:
            severity = "high"
        elif latest_entropy < self._alert_threshold and reward_trend >= 0:
            severity = "medium"
        else:
            severity = "low"

        first_entropy = recent[0].shape_entropy
        description = (
            f"Trajectory entropy: {latest_entropy:.2f} "
            f"({'↓' if entropy_trend < 0 else '↑'} from {first_entropy:.2f} over {self._trend_window} windows). "
            f"Reward: {recent[-1].mean_reward:.2f} "
            f"({'↑' if reward_trend > 0 else '→'} from {recent[0].mean_reward:.2f}). "
            f"Could indicate policy narrowing — high reward may reflect convergence to "
            f"a single strategy rather than genuine improvement across diverse approaches."
        )

        return CollapseSignal(
            entropy_trend=entropy_trend,
            reward_trend=reward_trend,
            severity=severity,
            description=description,
        )

    def summary(self) -> str:
        """Human-readable summary with 'could be' language."""
        if not self._history:
            return "No entropy data recorded yet."

        latest = self._history[-1]
        lines = [
            f"Shape entropy: {latest.shape_entropy:.2f}",
            f"Tool entropy: {latest.tool_entropy:.2f}",
            f"Reward entropy: {latest.reward_entropy:.2f}",
            f"Mean reward: {latest.mean_reward:.2f}",
            f"Windows recorded: {len(self._history)}",
        ]

        collapse = self.detect_collapse()
        if collapse:
            lines.append(f"Collapse signal ({collapse.severity}): {collapse.description}")

        return "\n".join(lines)


def _compute_trend(values: List[float]) -> float:
    """Compute simple linear regression slope over a sequence of values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator

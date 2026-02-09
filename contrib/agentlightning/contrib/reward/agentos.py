# Copyright (c) Microsoft. All rights reserved.

"""
PolicyReward - Convert Policy Violations to RL Penalties
=========================================================

Reward function that integrates Agent-OS governance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class PolicyReward:
    """
    Reward function that penalizes policy violations.

    Example:
        >>> from agent_os import KernelSpace
        >>>
        >>> kernel = KernelSpace(policy="strict")
        >>> reward_fn = PolicyReward(kernel, base_reward_fn=accuracy)
        >>>
        >>> reward = reward_fn(rollout)  # Base reward - violation penalties
    """

    def __init__(
        self,
        kernel: Any,
        *,
        base_reward_fn: Optional[Callable[[Any], float]] = None,
        critical_penalty: float = -100.0,
        high_penalty: float = -50.0,
        medium_penalty: float = -10.0,
        low_penalty: float = -1.0,
        clean_bonus: float = 5.0,
    ):
        """
        Initialize policy-aware reward.

        Args:
            kernel: Agent-OS KernelSpace
            base_reward_fn: Base reward function
            critical_penalty: Penalty for critical violations
            high_penalty: Penalty for high violations
            medium_penalty: Penalty for medium violations
            low_penalty: Penalty for low violations
            clean_bonus: Bonus for clean execution
        """
        self.kernel = kernel
        self.base_reward_fn = base_reward_fn or self._default_reward
        self.penalties = {
            "critical": critical_penalty,
            "high": high_penalty,
            "medium": medium_penalty,
            "low": low_penalty,
        }
        self.clean_bonus = clean_bonus

        self._total_rewards = 0
        self._total_penalties = 0.0

    def _default_reward(self, rollout: Any) -> float:
        """Default: 1.0 for success, 0.0 for failure."""
        return 1.0 if getattr(rollout, "success", False) else 0.0

    def __call__(self, rollout: Any, *, emit: bool = True) -> float:
        """
        Calculate reward with policy penalties.

        Args:
            rollout: Rollout with violations attribute
            emit: Emit reward span

        Returns:
            Final reward
        """
        base = self.base_reward_fn(rollout)

        violations = getattr(rollout, "violations", [])
        penalty = sum(self.penalties.get(v.severity, -10.0) for v in violations)

        reward = base + penalty
        if not violations:
            reward += self.clean_bonus

        self._total_rewards += 1
        self._total_penalties += penalty

        if emit:
            self._emit_reward(reward, base, penalty, len(violations))

        return reward

    def _emit_reward(
        self,
        final: float,
        base: float,
        penalty: float,
        violation_count: int,
    ) -> None:
        """Emit multi-dimensional reward."""
        try:
            from agentlightning.emitter import emit_reward

            emit_reward(
                {"final": final, "base": base, "policy_penalty": penalty},
                primary_key="final",
                attributes={"agent_os.violations": violation_count},
            )
        except ImportError:
            logger.debug(
                "agentlightning.emitter not available; skipping reward emission.",
                exc_info=True,
            )

    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        total = self._total_rewards or 1
        return {
            "total_rewards": self._total_rewards,
            "avg_penalty": self._total_penalties / total,
        }

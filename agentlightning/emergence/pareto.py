# Copyright (c) Microsoft. All rights reserved.

"""Gap 3: Multi-objective tension support.

Track Pareto fronts across reward dimensions, preserving the trade-off
structure that scalar reward collapse destroys.

Dissolution condition: when agent-lightning's Triplet model natively supports
vector rewards rather than Optional[float].
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ParetoPoint(BaseModel):
    """A point in multi-dimensional reward space."""

    rollout_id: str
    values: Dict[str, float]
    rank: int = 0
    """0 = Pareto-optimal, N = dominated by N front layers."""


class ParetoClassification(BaseModel):
    """Result of classifying a new point against the Pareto front."""

    rank: int
    dominated_by: List[str]
    dominates: List[str]
    tension_report: str = ""


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Return True if a Pareto-dominates b (at least as good in all, strictly better in one)."""
    keys = set(a.keys()) | set(b.keys())
    at_least_as_good = all(a.get(k, 0.0) >= b.get(k, 0.0) for k in keys)
    strictly_better = any(a.get(k, 0.0) > b.get(k, 0.0) for k in keys)
    return at_least_as_good and strictly_better


class ParetoTracker:
    """Track Pareto fronts across reward dimensions.

    Instead of collapsing multi-dimensional rewards to a scalar,
    maintains the full Pareto surface and surfaces the trade-offs
    that optimization would otherwise hide.

    Dissolution condition: when all reward matching policies preserve
    dimensional structure by default.
    """

    def __init__(
        self,
        dimensions: List[str],
        primary_key: Optional[str] = None,
    ):
        self._dimensions = dimensions
        self._primary_key = primary_key or (dimensions[0] if dimensions else None)
        self._front: List[ParetoPoint] = []
        self._history: deque[ParetoPoint] = deque(maxlen=5000)

    def add_point(
        self,
        rollout_id: str,
        values: Dict[str, float],
    ) -> ParetoClassification:
        """Classify a new point against the current front."""
        point = ParetoPoint(rollout_id=rollout_id, values=values)
        self._history.append(point)

        dominated_by: List[str] = []
        dominates_list: List[str] = []

        for existing in self._front:
            if _dominates(existing.values, values):
                dominated_by.append(existing.rollout_id)
            elif _dominates(values, existing.values):
                dominates_list.append(existing.rollout_id)

        # Update front
        if dominates_list:
            self._front = [p for p in self._front if p.rollout_id not in set(dominates_list)]

        rank = len(dominated_by)
        point.rank = rank

        if rank == 0:
            self._front.append(point)

        # Generate tension report
        tension_map = self.get_tension_map()
        tensions = [(k, v) for k, v in tension_map.items() if v < -0.3]
        if tensions:
            most_tense = min(tensions, key=lambda x: x[1])
            tension_report = (
                f"{most_tense[0][0]} vs {most_tense[0][1]} "
                f"(\u03c1 = {most_tense[1]:.2f})"
            )
        else:
            tension_report = "No strong dimensional tensions detected."

        return ParetoClassification(
            rank=rank,
            dominated_by=dominated_by,
            dominates=dominates_list,
            tension_report=tension_report,
        )

    def get_front(self, rank: int = 0) -> List[ParetoPoint]:
        """Get the Nth Pareto front layer."""
        if rank == 0:
            return list(self._front)

        # Compute layered fronts
        remaining = list(self._history)
        for _ in range(rank):
            front_ids = set()
            for i, p in enumerate(remaining):
                is_dominated = False
                for j, q in enumerate(remaining):
                    if i != j and _dominates(q.values, p.values):
                        is_dominated = True
                        break
                if not is_dominated:
                    front_ids.add(p.rollout_id)
            remaining = [p for p in remaining if p.rollout_id not in front_ids]

        # The remaining points form the Nth front candidates
        front: List[ParetoPoint] = []
        for p in remaining:
            is_dominated = False
            for q in remaining:
                if p.rollout_id != q.rollout_id and _dominates(q.values, p.values):
                    is_dominated = True
                    break
            if not is_dominated:
                front.append(ParetoPoint(rollout_id=p.rollout_id, values=p.values, rank=rank))

        return front

    def get_tension_map(self) -> Dict[Tuple[str, str], float]:
        """Pairwise correlation between dimensions across all points.

        Negative correlation = structural trade-off (tension).
        Positive correlation = aligned objectives (no tension).
        Near-zero = independent objectives.
        """
        if len(self._history) < 3:
            return {}

        result: Dict[Tuple[str, str], float] = {}
        for i, dim_a in enumerate(self._dimensions):
            for dim_b in self._dimensions[i + 1:]:
                values_a = [p.values.get(dim_a, 0.0) for p in self._history]
                values_b = [p.values.get(dim_b, 0.0) for p in self._history]
                corr = _pearson_correlation(values_a, values_b)
                result[(dim_a, dim_b)] = corr

        return result

    def summary(self) -> str:
        """Human-readable tension summary with 'could be' language."""
        if not self._history:
            return "No Pareto data recorded yet."

        front_size = len(self._front)
        total = len(self._history)
        n_dims = len(self._dimensions)

        lines = [
            f"Pareto front: {front_size} non-dominated solutions across {n_dims} dimensions.",
            f"Total observations: {total}.",
        ]

        tension_map = self.get_tension_map()
        tensions = sorted(tension_map.items(), key=lambda x: x[1])
        if tensions and tensions[0][1] < -0.3:
            (dim_a, dim_b), corr = tensions[0]
            lines.append(f"Primary tension: {dim_a} vs {dim_b} (\u03c1 = {corr:.2f}).")

            # Check which direction the front favors
            front_values_a = [p.values.get(dim_a, 0.0) for p in self._front]
            front_values_b = [p.values.get(dim_b, 0.0) for p in self._front]
            if front_values_a and front_values_b:
                mean_a = sum(front_values_a) / len(front_values_a)
                mean_b = sum(front_values_b) / len(front_values_b)
                if mean_a > mean_b:
                    lines.append(
                        f"Current front favors {dim_a} — {dim_b} ceiling could indicate "
                        f"unexplored strategies that sacrifice {dim_a} for depth."
                    )
                else:
                    lines.append(
                        f"Current front favors {dim_b} — {dim_a} ceiling could indicate "
                        f"unexplored strategies that sacrifice {dim_b} for speed."
                    )

        return "\n".join(lines)


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation between two lists."""
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)

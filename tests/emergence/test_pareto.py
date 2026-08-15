# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 3: Pareto tension tracking."""

import pytest

from agentlightning.emergence.pareto import (
    ParetoClassification,
    ParetoPoint,
    ParetoTracker,
    _dominates,
    _pearson_correlation,
)


class TestDominates:
    def test_a_dominates_b(self):
        assert _dominates({"x": 2.0, "y": 3.0}, {"x": 1.0, "y": 2.0})

    def test_equal_no_domination(self):
        assert not _dominates({"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0})

    def test_trade_off_no_domination(self):
        # a is better in x but worse in y
        assert not _dominates({"x": 2.0, "y": 1.0}, {"x": 1.0, "y": 2.0})

    def test_partial_domination(self):
        # a is better in x, equal in y
        assert _dominates({"x": 2.0, "y": 1.0}, {"x": 1.0, "y": 1.0})


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        corr = _pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert abs(corr - 1.0) < 0.01

    def test_perfect_negative(self):
        corr = _pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
        assert abs(corr - (-1.0)) < 0.01

    def test_zero_variance(self):
        corr = _pearson_correlation([1.0, 1.0, 1.0], [2.0, 4.0, 6.0])
        assert corr == 0.0

    def test_insufficient_data(self):
        assert _pearson_correlation([1.0], [2.0]) == 0.0


class TestParetoTracker:
    def test_first_point_is_pareto_optimal(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        result = tracker.add_point("r1", {"speed": 0.8, "quality": 0.6})
        assert result.rank == 0
        assert len(result.dominated_by) == 0

    def test_dominated_point(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        tracker.add_point("r1", {"speed": 0.8, "quality": 0.8})
        result = tracker.add_point("r2", {"speed": 0.5, "quality": 0.5})
        assert result.rank > 0
        assert "r1" in result.dominated_by

    def test_new_point_displaces_front(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        tracker.add_point("r1", {"speed": 0.5, "quality": 0.5})
        result = tracker.add_point("r2", {"speed": 0.8, "quality": 0.8})
        assert result.rank == 0
        assert "r1" in result.dominates

    def test_pareto_front_with_trade_offs(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        tracker.add_point("r1", {"speed": 0.9, "quality": 0.3})
        tracker.add_point("r2", {"speed": 0.3, "quality": 0.9})
        tracker.add_point("r3", {"speed": 0.6, "quality": 0.6})
        front = tracker.get_front(rank=0)
        # r1 and r2 should both be on the front (trade-off)
        front_ids = [p.rollout_id for p in front]
        assert "r1" in front_ids
        assert "r2" in front_ids

    def test_tension_map_negative_correlation(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        # Create negatively correlated points
        for i in range(10):
            speed = float(i) / 10
            quality = 1.0 - speed
            tracker.add_point(f"r{i}", {"speed": speed, "quality": quality})
        tension = tracker.get_tension_map()
        assert ("speed", "quality") in tension
        assert tension[("speed", "quality")] < -0.5

    def test_tension_map_insufficient_data(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        tracker.add_point("r1", {"speed": 0.5, "quality": 0.5})
        assert tracker.get_tension_map() == {}

    def test_summary_empty(self):
        tracker = ParetoTracker(dimensions=["a", "b"])
        assert "No Pareto data" in tracker.summary()

    def test_summary_with_tension(self):
        tracker = ParetoTracker(dimensions=["speed", "quality"])
        for i in range(20):
            speed = float(i) / 20
            quality = 1.0 - speed
            tracker.add_point(f"r{i}", {"speed": speed, "quality": quality})
        summary = tracker.summary()
        assert "Pareto front" in summary
        assert "tension" in summary.lower() or "ρ" in summary

    def test_get_front_layered(self):
        tracker = ParetoTracker(dimensions=["x", "y"])
        # Layer 0: non-dominated
        tracker.add_point("a", {"x": 1.0, "y": 0.0})
        tracker.add_point("b", {"x": 0.0, "y": 1.0})
        # Layer 1: dominated by a or b
        tracker.add_point("c", {"x": 0.5, "y": 0.0})
        tracker.add_point("d", {"x": 0.0, "y": 0.5})

        front_0 = tracker.get_front(rank=0)
        assert len(front_0) >= 2

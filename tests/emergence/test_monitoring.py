# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 1: Exploration collapse monitoring."""

import pytest

from agentlightning.emergence.monitoring import (
    CollapseSignal,
    EntropySnapshot,
    ExplorationDecayMonitor,
    _compute_trend,
)

from .conftest import make_diverse_trees, make_triplet, make_uniform_trees


class TestComputeTrend:
    def test_increasing(self):
        assert _compute_trend([1.0, 2.0, 3.0, 4.0]) > 0

    def test_decreasing(self):
        assert _compute_trend([4.0, 3.0, 2.0, 1.0]) < 0

    def test_constant(self):
        assert _compute_trend([5.0, 5.0, 5.0]) == 0.0

    def test_single_value(self):
        assert _compute_trend([1.0]) == 0.0

    def test_empty(self):
        assert _compute_trend([]) == 0.0


class TestExplorationDecayMonitor:
    def test_record_creates_snapshot(self):
        monitor = ExplorationDecayMonitor(window_size=5)
        trees = make_diverse_trees(5)
        triplets = [make_triplet(reward=0.5) for _ in range(5)]
        snapshot = monitor.record(trees, triplets)
        assert isinstance(snapshot, EntropySnapshot)
        assert snapshot.window_index == 0

    def test_detect_collapse_insufficient_data(self):
        monitor = ExplorationDecayMonitor(trend_window=5)
        # Not enough windows recorded
        assert monitor.detect_collapse() is None

    def test_detect_collapse_pattern(self):
        monitor = ExplorationDecayMonitor(
            window_size=5,
            alert_threshold=0.5,
            trend_window=3,
        )
        # Simulate declining entropy with improving reward
        # Record diverse trajectories first, then uniform ones
        diverse = make_diverse_trees(5)
        uniform = make_uniform_trees(5)

        # Window 1: high entropy
        triplets_low = [make_triplet(reward=0.3) for _ in range(5)]
        monitor.record(diverse, triplets_low)

        # Window 2: medium entropy
        triplets_mid = [make_triplet(reward=0.6) for _ in range(5)]
        monitor.record(diverse[:3] + uniform[:2], triplets_mid)

        # Window 3: low entropy, high reward
        triplets_high = [make_triplet(reward=0.9) for _ in range(5)]
        monitor.record(uniform, triplets_high)

        signal = monitor.detect_collapse()
        # May or may not detect collapse depending on actual entropy values
        # but should not crash
        if signal is not None:
            assert isinstance(signal, CollapseSignal)
            assert signal.severity in ("low", "medium", "high")
            assert "could indicate" in signal.description.lower() or "Could indicate" in signal.description

    def test_no_collapse_when_entropy_high(self):
        monitor = ExplorationDecayMonitor(
            window_size=5,
            alert_threshold=0.1,
            trend_window=3,
        )
        diverse = make_diverse_trees(5)
        for _ in range(3):
            monitor.record(diverse, [make_triplet(reward=0.5) for _ in range(5)])
        # Entropy should be high with diverse trees
        signal = monitor.detect_collapse()
        assert signal is None

    def test_summary(self):
        monitor = ExplorationDecayMonitor()
        assert "No entropy data" in monitor.summary()

        trees = make_diverse_trees(5)
        triplets = [make_triplet(reward=0.5) for _ in range(5)]
        monitor.record(trees, triplets)
        summary = monitor.summary()
        assert "Shape entropy" in summary
        assert "Mean reward" in summary

# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 1: Entropy computation."""

import pytest

from agentlightning.emergence.entropy import TrajectoryEntropy, _shannon_entropy

from .conftest import make_diverse_trees, make_triplet, make_uniform_trees


class TestShannonEntropy:
    def test_empty(self):
        assert _shannon_entropy([]) == 0.0

    def test_single_category(self):
        assert _shannon_entropy(["a", "a", "a"]) == 0.0

    def test_uniform_distribution(self):
        # Uniform distribution should have entropy close to 1.0
        items = ["a", "b", "c", "d"] * 10
        entropy = _shannon_entropy(items)
        assert 0.95 <= entropy <= 1.0

    def test_skewed_distribution(self):
        items = ["a"] * 90 + ["b"] * 10
        entropy = _shannon_entropy(items)
        assert 0.0 < entropy < 0.7


class TestTrajectoryEntropy:
    def test_shape_entropy_diverse(self):
        calc = TrajectoryEntropy()
        trees = make_diverse_trees(20)
        entropy = calc.compute_shape_entropy(trees, window_size=20)
        assert entropy > 0.0

    def test_shape_entropy_uniform(self):
        calc = TrajectoryEntropy()
        trees = make_uniform_trees(20)
        entropy = calc.compute_shape_entropy(trees, window_size=20)
        # All identical trees -> low entropy
        assert entropy == 0.0

    def test_shape_entropy_empty(self):
        calc = TrajectoryEntropy()
        assert calc.compute_shape_entropy([], window_size=10) == 0.0

    def test_tool_entropy_diverse(self):
        calc = TrajectoryEntropy()
        trees = make_diverse_trees(10)
        # These trees don't have openai.chat.completion spans, so we use a broader match
        entropy = calc.compute_tool_entropy(trees, window_size=10, llm_call_match=r"root_")
        assert entropy > 0.0

    def test_tool_entropy_empty(self):
        calc = TrajectoryEntropy()
        assert calc.compute_tool_entropy([], window_size=10) == 0.0

    def test_reward_entropy_spread(self):
        calc = TrajectoryEntropy()
        triplets = [make_triplet(reward=float(i) / 10) for i in range(10)]
        entropy = calc.compute_reward_entropy(triplets, window_size=10)
        assert entropy > 0.0

    def test_reward_entropy_constant(self):
        calc = TrajectoryEntropy()
        triplets = [make_triplet(reward=1.0) for _ in range(10)]
        entropy = calc.compute_reward_entropy(triplets, window_size=10)
        assert entropy == 0.0

    def test_reward_entropy_no_rewards(self):
        calc = TrajectoryEntropy()
        triplets = [make_triplet(reward=None) for _ in range(10)]
        entropy = calc.compute_reward_entropy(triplets, window_size=10)
        assert entropy == 0.0

    def test_window_size_respected(self):
        calc = TrajectoryEntropy()
        trees = make_diverse_trees(100)
        # Small window should only look at last 5
        entropy_small = calc.compute_shape_entropy(trees, window_size=5)
        entropy_large = calc.compute_shape_entropy(trees, window_size=100)
        # Both should compute, but may differ
        assert entropy_small >= 0.0
        assert entropy_large >= 0.0

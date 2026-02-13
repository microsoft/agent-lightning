# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 5: Novelty Detection."""

import pytest

from agentlightning.adapter.triplet import TracerTraceToTriplet
from agentlightning.emergence.novelty import (
    NoveltyAwareAdapter,
    NoveltyDetector,
    NoveltyScore,
    TrajectoryShape,
    _shape_similarity,
)

from .conftest import make_diverse_trees, make_span, make_tree, make_uniform_trees


class TestTrajectoryShape:
    def test_from_trace_tree(self):
        child = make_tree("child_a")
        tree = make_tree("root", children=[child])
        shape = TrajectoryShape.from_trace_tree(tree)
        assert len(shape.nodes) == 2
        assert shape.nodes[0][0] == "root"  # name
        assert shape.nodes[0][1] == 0  # depth
        assert shape.nodes[1][0] == "child_a"
        assert shape.nodes[1][1] == 1  # depth

    def test_fingerprint_uniqueness(self):
        tree_a = make_tree("root", children=[make_tree("a")])
        tree_b = make_tree("root", children=[make_tree("b")])
        shape_a = TrajectoryShape.from_trace_tree(tree_a)
        shape_b = TrajectoryShape.from_trace_tree(tree_b)
        assert shape_a.fingerprint != shape_b.fingerprint

    def test_fingerprint_stability(self):
        tree = make_tree("root", children=[make_tree("child")])
        shape1 = TrajectoryShape.from_trace_tree(tree)
        shape2 = TrajectoryShape.from_trace_tree(tree)
        assert shape1.fingerprint == shape2.fingerprint


class TestShapeSimilarity:
    def test_identical_shapes(self):
        tree = make_tree("root", children=[make_tree("a")])
        shape = TrajectoryShape.from_trace_tree(tree)
        assert _shape_similarity(shape, shape) == 1.0

    def test_completely_different(self):
        shape_a = TrajectoryShape.from_trace_tree(
            make_tree("x", children=[make_tree("y", children=[make_tree("z")])])
        )
        shape_b = TrajectoryShape.from_trace_tree(
            make_tree("a", children=[make_tree("b")])
        )
        sim = _shape_similarity(shape_a, shape_b)
        assert 0.0 <= sim < 1.0

    def test_empty_shapes(self):
        a = TrajectoryShape(nodes=(), fingerprint="a")
        b = TrajectoryShape(nodes=(), fingerprint="b")
        assert _shape_similarity(a, b) == 1.0


class TestNoveltyDetector:
    def test_first_trajectory_is_novel(self):
        detector = NoveltyDetector()
        tree = make_tree("root", children=[make_tree("child")])
        score = detector.score_novelty(tree)
        assert score.score == 1.0
        assert score.first_seen is True
        assert score.classification == "novel"

    def test_repeated_trajectory_becomes_routine(self):
        detector = NoveltyDetector(novelty_decay_rate=0.5)
        tree = make_tree("root", children=[make_tree("child")])

        # First observation
        score1 = detector.score_novelty(tree)
        assert score1.classification == "novel"

        # Repeated observations decay novelty
        for _ in range(10):
            score = detector.score_novelty(tree)

        assert score.score < 0.3
        assert score.classification == "routine"

    def test_different_structures_are_novel(self):
        detector = NoveltyDetector()
        tree_a = make_tree("root", children=[make_tree("a")])
        tree_b = make_tree("root", children=[make_tree("b"), make_tree("c")])

        score_a = detector.score_novelty(tree_a)
        score_b = detector.score_novelty(tree_b)
        assert score_a.first_seen is True
        assert score_b.first_seen is True

    def test_similar_shapes_recognized(self):
        detector = NoveltyDetector(shape_similarity_threshold=0.5)
        tree_a = make_tree("root", children=[make_tree("search"), make_tree("respond")])
        detector.score_novelty(tree_a)

        # Very similar tree
        tree_b = make_tree("root", children=[make_tree("search"), make_tree("respond")])
        score = detector.score_novelty(tree_b)
        assert score.first_seen is False

    def test_codebook_eviction(self):
        detector = NoveltyDetector(max_codebook_size=3)
        for i in range(5):
            tree = make_tree(f"root_{i}", children=[make_tree(f"child_{i}")])
            detector.score_novelty(tree)
        assert len(detector._codebook) <= 3

    def test_discovery_rate(self):
        detector = NoveltyDetector()
        trees = make_diverse_trees(10)
        for tree in trees:
            detector.score_novelty(tree)
        rate = detector.get_discovery_rate(window_size=10)
        assert rate > 0.0

    def test_discovery_rate_empty(self):
        detector = NoveltyDetector()
        assert detector.get_discovery_rate() == 0.0

    def test_codebook_summary(self):
        detector = NoveltyDetector()
        assert "empty" in detector.get_codebook_summary()

        for tree in make_diverse_trees(5):
            detector.score_novelty(tree)
        summary = detector.get_codebook_summary()
        assert "known shapes" in summary
        assert "Discovery rate" in summary


class TestNoveltyAwareAdapter:
    def test_enriches_triplets_with_metadata(self):
        base = TracerTraceToTriplet(
            llm_call_match=r"openai\.chat\.completion",
            repair_hierarchy=False,
        )
        detector = NoveltyDetector()
        adapter = NoveltyAwareAdapter(base, detector)

        # Create spans that form a tree with an LLM call
        root = make_span("root", span_id="root-1", start_time=0.0, end_time=2.0)
        llm = make_span(
            "openai.chat.completion",
            parent_id="root-1",
            start_time=0.1,
            end_time=0.5,
            attributes={
                "gen_ai.response.id": "resp-1",
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "hello",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "world",
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
            },
        )
        triplets = adapter.adapt([root, llm])
        if triplets:
            assert "novelty_score" in triplets[0].metadata
            assert "novelty_classification" in triplets[0].metadata
            assert "sampling_weight" in triplets[0].metadata

    def test_handles_empty_source(self):
        base = TracerTraceToTriplet(repair_hierarchy=False)
        detector = NoveltyDetector()
        adapter = NoveltyAwareAdapter(base, detector)
        # Empty source should not crash
        # (TracerTraceToTriplet will raise ValueError on empty spans, but that's expected)
        with pytest.raises(ValueError):
            adapter.adapt([])

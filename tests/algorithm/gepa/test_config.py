# Copyright (c) Microsoft. All rights reserved.

"""Tests for GEPAConfig defaults and overrides."""

from __future__ import annotations

from agentlightning.algorithm.gepa.config import GEPAConfig


def test_defaults():
    config = GEPAConfig()
    assert config.max_metric_calls is None
    assert config.candidate_selection_strategy == "pareto"
    assert config.frontier_type == "instance"
    assert config.reflection_minibatch_size is None
    assert config.module_selector == "round_robin"
    assert config.seed == 0
    assert config.use_merge is False
    assert config.max_merge_invocations == 5
    assert config.skip_perfect_score is True
    assert config.perfect_score == 1.0
    assert config.display_progress_bar is False
    assert config.raise_on_exception is True
    assert config.rollout_batch_timeout == 3600.0
    assert config.rollout_poll_interval == 2.0
    assert config.reflection_model is None
    assert config.extra_kwargs == {}


def test_overrides():
    config = GEPAConfig(
        max_metric_calls=50,
        candidate_selection_strategy="current_best",
        frontier_type="aggregate",
        reflection_minibatch_size=8,
        module_selector="all",
        seed=42,
        use_merge=True,
        rollout_batch_timeout=120.0,
        rollout_poll_interval=0.5,
        reflection_model="gpt-4.1-mini",
    )
    assert config.max_metric_calls == 50
    assert config.candidate_selection_strategy == "current_best"
    assert config.frontier_type == "aggregate"
    assert config.reflection_minibatch_size == 8
    assert config.module_selector == "all"
    assert config.seed == 42
    assert config.use_merge is True
    assert config.rollout_batch_timeout == 120.0
    assert config.rollout_poll_interval == 0.5
    assert config.reflection_model == "gpt-4.1-mini"


def test_extra_kwargs():
    config = GEPAConfig(extra_kwargs={"run_dir": "/tmp/gepa_run", "cache_evaluation": True})
    assert config.extra_kwargs["run_dir"] == "/tmp/gepa_run"
    assert config.extra_kwargs["cache_evaluation"] is True

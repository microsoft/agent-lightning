# Copyright (c) Microsoft. All rights reserved.

"""Tests for LightningGEPACallback — all methods log without raising."""

from __future__ import annotations

from agentlightning.algorithm.gepa.callbacks import LightningGEPACallback


def test_all_callbacks_log_without_raising():
    cb = LightningGEPACallback()

    # Each callback should be callable with **kwargs and not raise
    cb.on_optimization_start()
    cb.on_optimization_end()
    cb.on_iteration_start(iteration=1)
    cb.on_iteration_end(iteration=1)
    cb.on_candidate_selected(candidate={"prompt": "test"})
    cb.on_candidate_accepted(candidate={"prompt": "test"}, score=0.8)
    cb.on_candidate_rejected(candidate={"prompt": "test"}, score=0.1)
    cb.on_evaluation_start()
    cb.on_evaluation_end()
    cb.on_evaluation_skipped(reason="cached")
    cb.on_valset_evaluated(score=0.9)
    cb.on_reflective_dataset_built()
    cb.on_proposal_start()
    cb.on_proposal_end()
    cb.on_merge_attempted()
    cb.on_merge_accepted()
    cb.on_merge_rejected()
    cb.on_pareto_front_updated(frontier_size=3)
    cb.on_state_saved()
    cb.on_budget_updated(remaining=10)
    cb.on_error(error="test error")


def test_callbacks_accept_arbitrary_kwargs():
    cb = LightningGEPACallback()
    # All methods use **kwargs, so any keyword argument should be accepted
    cb.on_optimization_start(extra="value", count=42)
    cb.on_error(error="message", traceback="stack", severity="high")

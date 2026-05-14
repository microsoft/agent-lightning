"""Focused tests for W&B-friendly trainer metric helpers."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("verl")

from agl_lite.verl.trainer import AglLiteRayPPOTrainer, _suffix_metrics, _tracking_backends_with_wandb


def test_tracking_backends_adds_wandb_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    assert _tracking_backends_with_wandb(["console"]) == ["console", "wandb"]
    assert _tracking_backends_with_wandb("console") == ["console", "wandb"]
    assert _tracking_backends_with_wandb(["console", "wandb"]) == ["console", "wandb"]


def test_tracking_backends_respects_disabled_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    assert _tracking_backends_with_wandb(["console", "wandb"]) == ["console"]
    assert _tracking_backends_with_wandb(["wandb"]) == ["console"]


def test_suffix_metrics_keeps_group_prefixes() -> None:
    assert _suffix_metrics({"critic/score/mean": 1.0}, "_after_processing") == {
        "critic/score/mean_after_processing": 1.0
    }


def test_drop_remainder_metric_stays_in_training_group() -> None:
    source = inspect.getsource(AglLiteRayPPOTrainer._train_step)

    assert 'metrics["training/n_triplets_dropped_remainder"]' in source
    assert 'metrics["critic/n_transition_before_dropping"]' in source
    assert 'metrics["critic/n_transition_after_dropping"]' in source
    assert 'metrics["critic/n_triplets_dropped_remainder"]' not in source

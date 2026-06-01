"""Focused tests for W&B-friendly trainer metric helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

pytest.importorskip("verl")

from agl_lite.verl.trainer import (
    AglLiteRayPPOTrainer,
    _count_zero_advantage_triplets,
    _suffix_metrics,
    _tracking_backends_with_wandb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_count_zero_advantage_triplets_ignores_padding() -> None:
    batch = SimpleNamespace(
        batch={
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 7.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.bool,
            ),
        }
    )

    assert _count_zero_advantage_triplets(cast(Any, batch)) == 2


def test_base_config_enables_verl_actor_entropy_metric() -> None:
    config = yaml.safe_load((REPO_ROOT / "agl_lite/verl/config.yaml").read_text())

    assert config["actor_rollout_ref"]["actor"]["calculate_entropy"] is True
    assert "is_shuffle" not in config["actor_rollout_ref"]["actor"]
    assert config["agentlightning"]["is_shuffle"] is False


def test_drop_remainder_metric_stays_in_training_group() -> None:
    source = inspect.getsource(AglLiteRayPPOTrainer._train_step)
    async_source = inspect.getsource(AglLiteRayPPOTrainer._async_train_step)

    assert 'metrics["training/n_triplets_dropped_remainder"]' in source
    assert 'metrics["training/n_advantage_zero"]' in source
    assert 'metrics["training/n_advantage_zero"]' in async_source
    assert 'if self.config.agentlightning.get("is_shuffle", True):' in source
    assert "ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n" in source
    assert 'metrics["critic/n_transition_before_dropping"]' in source
    assert 'metrics["critic/n_transition_after_dropping"]' in source
    assert 'metrics["critic/n_triplets_dropped_remainder"]' not in source


def test_old_log_prob_entropy_stays_out_of_training_batch_union() -> None:
    sync_source = inspect.getsource(AglLiteRayPPOTrainer._train_step)
    async_source = inspect.getsource(AglLiteRayPPOTrainer._async_train_step)

    assert 'old_log_prob.batch.pop("entropys")' in sync_source
    assert 'old_log_prob.batch.pop("entropys")' in async_source

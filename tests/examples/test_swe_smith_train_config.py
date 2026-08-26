# Copyright (c) Microsoft. All rights reserved.

"""SWE-Smith trainer image-filter configuration behavior."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agentlightning.verl import entrypoint as verl_entrypoint
from examples.swe_smith import train_smith_agent
from examples.swe_smith.train_smith_agent import build_config

SWE_SMITH_DIR = Path(__file__).resolve().parents[2] / "examples" / "swe_smith"
sys.path.insert(0, str(SWE_SMITH_DIR))
train_smith_agent_megatron = importlib.import_module("examples.swe_smith.train_smith_agent_megatron")


def test_k8s_image_filter_is_disabled_by_default() -> None:
    config = build_config(model="Qwen/Test-Model")

    assert config.agentlightning.k8s.filter_unavailable_images is False
    assert config.agentlightning.k8s.image_readiness_timeout_seconds == 60


def test_k8s_image_filter_accepts_single_hydra_override() -> None:
    config = build_config(
        model="Qwen/Test-Model",
        config_overrides=["agentlightning.k8s.filter_unavailable_images=true"],
    )

    assert config.agentlightning.k8s.filter_unavailable_images is True


@pytest.mark.parametrize("trainer_module", [train_smith_agent, train_smith_agent_megatron])
def test_train_forwards_validation_cap_after_loading_full_split(
    trainer_module,
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(json.dumps({"instance_id": "train-1", "repo": "repo"}) + "\n")
    val_path.write_text(
        json.dumps({"instance_id": "val-1", "repo": "repo"})
        + "\n"
        + json.dumps({"instance_id": "val-2", "repo": "repo"})
        + "\n"
    )
    captured: dict[str, Any] = {}

    def capture_run_ppo(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(verl_entrypoint, "run_ppo", capture_run_ppo)

    trainer_module.train(
        train_dataset_path=str(train_path),
        val_dataset_path=str(val_path),
        max_val_instances=1,
        agl_key="secret",
    )

    assert len(captured["val_dataset"]) == 2
    assert captured["max_val_instances"] == 1

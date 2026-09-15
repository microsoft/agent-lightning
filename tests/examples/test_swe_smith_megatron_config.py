# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "swe_smith"
sys.path.insert(0, str(EXAMPLE_DIR))

from examples.swe_smith.train_smith_agent_megatron import (  # noqa: E402
    DEFAULT_MODEL,
    build_config,
    verl_megatron_config,
)


def _config() -> dict[str, Any]:
    return verl_megatron_config()


def test_default_config_uses_qwen35_megatron_r3_sync_rollout() -> None:
    config = _config()
    actor_rollout_ref = config["actor_rollout_ref"]
    actor = actor_rollout_ref["actor"]

    assert DEFAULT_MODEL == "Qwen/Qwen3.5-35B-A3B"
    assert actor_rollout_ref["model"]["path"] == DEFAULT_MODEL
    assert actor["strategy"] == "megatron"
    assert actor["model_engine"] == "megatron"
    assert actor["megatron"]["use_mbridge"] is True
    assert actor["megatron"]["router_replay"]["mode"] == "R3"
    assert actor_rollout_ref["rollout"]["enable_rollout_routing_replay"] is True

    assert config["algorithm"]["enable_rollout_level_advantage"] is True
    assert config["algorithm"]["rollout_correction"]["bypass_mode"] is False
    assert config["algorithm"]["rollout_correction"]["rollout_is"] is None
    assert config["algorithm"]["rollout_correction"]["rollout_rs"] is None
    assert actor["policy_loss"]["loss_mode"] == "per_rollout_mean"

    assert config["agentlightning"]["async_rollout"]["enabled"] is False
    assert config["agentlightning"]["async_rollout"]["async_train_batch_size"] is None
    assert actor_rollout_ref["rollout"]["mode"] == "async"
    assert Path(config["agentlightning"]["k8s"]["job_template_path"]).name == "job-template-openai.yaml"


def test_default_config_uses_conservative_four_gpu_sizing() -> None:
    config = _config()
    actor_rollout_ref = config["actor_rollout_ref"]
    rollout = actor_rollout_ref["rollout"]
    actor = actor_rollout_ref["actor"]
    actor_megatron = actor["megatron"]

    assert config["trainer"]["n_gpus_per_node"] == 4
    assert config["data"]["train_batch_size"] == 16
    assert config["data"]["max_prompt_length"] == 65536
    assert config["data"]["max_response_length"] == 65536
    assert rollout["tensor_model_parallel_size"] == 1
    assert rollout["n"] == 8
    assert config["data"]["train_batch_size"] * rollout["n"] == 128
    assert rollout["gpu_memory_utilization"] == 0.8
    assert rollout["max_model_len"] == 81920
    assert rollout["enforce_eager"] is False
    assert rollout["enable_chunked_prefill"] is True
    assert rollout["val_kwargs"] == {"temperature": 0.7, "do_sample": True}
    assert actor["ppo_mini_batch_size"] == 16
    assert config["trainer"]["balance_batch"] is True
    assert config["trainer"]["total_epochs"] == 4
    assert config["agentlightning"]["max_ppo_update_times"] == 2

    assert actor_megatron["pipeline_model_parallel_size"] == 1
    assert actor_megatron["tensor_model_parallel_size"] == 1
    assert actor_megatron["expert_model_parallel_size"] == 4
    assert actor_megatron["expert_tensor_parallel_size"] == 1
    assert actor_megatron["param_offload"] is True
    assert actor_megatron["optimizer_offload"] is True
    assert actor_megatron["grad_offload"] is True
    assert actor_rollout_ref["ref"]["megatron"]["tensor_model_parallel_size"] == 1
    assert actor_rollout_ref["ref"]["megatron"]["param_offload"] is True


def test_ci_config_is_one_step_without_changing_model_topology() -> None:
    pytest.importorskip("verl")

    config = build_config(agl_key="dummy", ci=True)

    assert config.actor_rollout_ref.model.path == DEFAULT_MODEL
    assert config.trainer.n_gpus_per_node == 4
    assert config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size == 1
    assert config.actor_rollout_ref.actor.megatron.expert_model_parallel_size == 4
    assert config.actor_rollout_ref.actor.megatron.router_replay.mode == "R3"
    assert config.trainer.total_epochs == 1
    assert config.trainer.total_training_steps == 1
    assert config.trainer.test_freq == -1
    assert config.trainer.save_freq == -1
    assert config.trainer.logger == ["console"]
    assert config.data.train_batch_size == 1
    assert config.actor_rollout_ref.rollout.n == 2
    assert config.actor_rollout_ref.rollout.max_model_len == 8192
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 1
    assert config.trainer.experiment_name.endswith("_ci")

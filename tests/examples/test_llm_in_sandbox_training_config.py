from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
from typing import Any

TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples/llm-in-sandbox/train_llm_in_sandbox.py"


def _training_config() -> dict:
    tree = ast.parse(TRAIN_SCRIPT_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "RL_TRAINING_CONFIG"
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RL_TRAINING_CONFIG" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("RL_TRAINING_CONFIG not found")


def test_llm_in_sandbox_train_and_validation_temperature_defaults() -> None:
    rollout_config = _training_config()["actor_rollout_ref"]["rollout"]

    assert rollout_config["temperature"] == 1
    assert rollout_config["val_kwargs"]["temperature"] == 0
    assert rollout_config["val_kwargs"]["do_sample"] is False


def _training_module() -> Any:
    spec = importlib.util.spec_from_file_location("llm_in_sandbox_train_config", TRAIN_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "ci_fast": False,
        "experiment_name": "arg-exp",
        "total_steps": 3,
        "rollout_n": 2,
        "train_batch_size": 4,
        "minibsz": 5,
        "max_train_samples": 0,
        "max_test_samples": 0,
        "val_before_train": True,
        "n_gpus_per_node": 6,
        "save_freq": 7,
        "max_prompt_length": 123,
        "max_response_length": 456,
        "temperature": 0.3,
        "gpu_memory_utilization": 0.4,
        "tensor_model_parallel_size": 2,
        "loss_agg_mode": "token-mean",
        "model": "arg-model",
        "logger": "console",
        "project_name": "arg-project",
        "test_freq": 8,
        "total_epochs": 9,
        "val_only": False,
        "rollout_timeout_seconds": 77,
        "trace_level": "transition",
        "trajectory_max_prompt_length": 321,
        "trajectory_max_response_length": 654,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_verl_config_training_config_overrides_args(monkeypatch) -> None:
    module = _training_module()
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["data"], "train_batch_size", 99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["data"], "max_prompt_length", 9991)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["data"], "max_response_length", 9992)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["rollout"], "n", 99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["rollout"], "temperature", 0.99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["rollout"], "gpu_memory_utilization", 0.91)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["rollout"], "tensor_model_parallel_size", 3)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["actor"], "ppo_mini_batch_size", 99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["actor"], "loss_agg_mode", "config-loss")
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["actor_rollout_ref"]["model"], "path", "config-model")
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["agentlightning"], "is_shuffle", False)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["trainer"], "experiment_name", "config-exp")
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["trainer"], "n_gpus_per_node", 99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["trainer"], "test_freq", 99)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["agentlightning"], "timeout_seconds", 999)
    monkeypatch.setitem(module.RL_TRAINING_CONFIG["agentlightning"], "poll_timeout_seconds", 998)
    monkeypatch.setitem(
        module.RL_TRAINING_CONFIG["agentlightning"],
        "trace_aggregator",
        {
            "level": "trajectory",
            "trajectory_max_prompt_length": 9993,
            "trajectory_max_response_length": 9994,
        },
    )
    monkeypatch.setenv("AGL_CLEANUP_AGENT_JOBS", "false")
    monkeypatch.setenv("AGL_NAMESPACE", "arg-namespace")

    config = module.build_verl_config(
        _args(),
        resources_id="arg-resources",
        base_url="http://arg-base",
        agl_key="arg-key",
    )

    assert config.data.train_batch_size == 99
    assert config.data.max_prompt_length == 9991
    assert config.data.max_response_length == 9992
    assert config.actor_rollout_ref.rollout.n == 99
    assert config.actor_rollout_ref.rollout.temperature == 0.99
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.91
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 3
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 99
    assert config.actor_rollout_ref.actor.loss_agg_mode == "config-loss"
    assert config.actor_rollout_ref.model.path == "config-model"
    assert "is_shuffle" not in config.actor_rollout_ref.actor
    assert config.agentlightning.is_shuffle is False
    assert config.trainer.experiment_name == "config-exp"
    assert config.trainer.n_gpus_per_node == 99
    assert config.trainer.test_freq == 99
    assert config.agentlightning.agl_base_url == "http://arg-base"
    assert config.agentlightning.agl_key == "arg-key"
    assert config.agentlightning.resources_id == "arg-resources"
    assert config.agentlightning.timeout_seconds == 999
    assert config.agentlightning.poll_timeout_seconds == 998
    assert config.agentlightning.cleanup_agent_jobs is False
    assert config.agentlightning.cleanup_namespace == "arg-namespace"
    assert config.agentlightning.trace_aggregator.level == "trajectory"
    assert config.agentlightning.trace_aggregator.trajectory_max_prompt_length == 9993
    assert config.agentlightning.trace_aggregator.trajectory_max_response_length == 9994
    assert config.agentlightning.trace_aggregator.debug is False
    assert config.agentlightning.trace_aggregator.mismatch_log_dir == str(module.EXAMPLE_DIR / "mismatch_cases")

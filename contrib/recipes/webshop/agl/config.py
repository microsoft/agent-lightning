# Copyright (c) Microsoft. All rights reserved.

"""Training configurations for the WebShop agent.

This module provides training configurations for the WebShop agent using Agent Lightning.
The configurations support external runner mode where the TypeScript/Vercel AI SDK agent
executes rollouts while this script coordinates the training.

Configurations:
    - 'fast': Lightweight config for CI testing (Qwen2.5-0.5B, 1 epoch)
    - 'qwen': Standard config (Qwen2.5-3B-Instruct, 50 epochs)
    - 'qwen_1_5b': Legacy smaller model (Qwen2.5-1.5B-Instruct)
    - 'qwen_7b': Larger model for best instruction following (Qwen2.5-7B-Instruct)
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Tuple

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 8,  # Must be <= number of tasks (sample tasks = 8)
        "max_prompt_length": 4096,
        "max_response_length": 1024,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 8,  # Must be <= train_batch_size
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 5e-6},  # Increased from 1e-6 for faster convergence
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0.01,  # Added entropy bonus to encourage exploration
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            # Qwen 3B model for better instruction following
            "path": "Qwen/Qwen2.5-3B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "webshop",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 50,
    },
}


def _write_github_output(project_name: str, experiment_name: str) -> None:
    """Write experiment metadata to GitHub Actions output file if running in CI.

    This enables downstream workflow steps to reference the project/experiment names.
    Does nothing if GITHUB_OUTPUT environment variable is not set.
    """
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={project_name}\n")
            f.write(f"run_name={experiment_name}\n")


def _create_config(
    experiment_prefix: str,
    project_name: str = "AgentLightning",
    **overrides: Any,
) -> Tuple[Dict[str, Any], str, str]:
    """Create a training config from the base with timestamp and optional overrides.

    Args:
        experiment_prefix: Prefix for the experiment name (timestamp appended).
        project_name: W&B project name for tracking.
        **overrides: Nested key paths with values to override. Use double underscores
            to denote nesting (e.g., `trainer__total_epochs=1`).

    Returns:
        Tuple of (config dict, experiment_name, project_name).
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"{experiment_prefix}_{timestamp}"

    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["experiment_name"] = experiment_name
    config["trainer"]["project_name"] = project_name

    # Apply overrides using double-underscore notation for nested keys
    for key, value in overrides.items():
        parts = key.split("__")
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    return config, experiment_name, project_name


def config_fast() -> Dict[str, Any]:
    """Fast training configuration for CI testing.

    Uses a smaller model (Qwen2.5-0.5B-Instruct) and minimal epochs
    for quick validation of the training pipeline.
    """
    config, experiment_name, project_name = _create_config(
        experiment_prefix="webshop_fast",
        project_name="AgentLightningCI",
        actor_rollout_ref__model__path="Qwen/Qwen2.5-0.5B-Instruct",
        actor_rollout_ref__rollout__gpu_memory_utilization=0.6,
        trainer__total_epochs=1,
        trainer__total_training_steps=1,
        trainer__test_freq=1,
    )
    _write_github_output(project_name, experiment_name)
    return config


def config_qwen() -> Dict[str, Any]:
    """Standard Qwen training configuration.

    Uses Qwen2.5-3B-Instruct for full training runs with improved hyperparameters.
    """
    config, _, _ = _create_config(experiment_prefix="webshop_qwen")
    return config


def config_qwen_1_5b() -> Dict[str, Any]:
    """Legacy Qwen 1.5B configuration (smaller, faster).

    Use this for quick iterations or when GPU memory is limited.
    """
    config, _, _ = _create_config(
        experiment_prefix="webshop_qwen1.5b",
        actor_rollout_ref__model__path="Qwen/Qwen2.5-1.5B-Instruct",
        actor_rollout_ref__rollout__gpu_memory_utilization=0.8,
    )
    return config


def config_qwen_7b() -> Dict[str, Any]:
    """Qwen 7B configuration for best instruction following.

    Uses the larger Qwen2.5-7B-Instruct model. Requires more GPU memory.
    """
    config, _, _ = _create_config(
        experiment_prefix="webshop_qwen7b",
        actor_rollout_ref__model__path="Qwen/Qwen2.5-7B-Instruct",
        actor_rollout_ref__rollout__gpu_memory_utilization=0.9,
    )
    return config

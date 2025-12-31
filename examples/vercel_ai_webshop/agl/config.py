# Copyright (c) Microsoft. All rights reserved.

"""Training configurations for the WebShop agent.

This module provides training configurations for the WebShop agent using Agent Lightning.
The configurations support external runner mode where the TypeScript/Vercel AI SDK agent
executes rollouts while this script coordinates the training.

Configurations:
    - 'fast': Lightweight config for CI testing (Qwen2.5-0.5B, 1 epoch)
    - 'qwen': Standard config (Qwen2.5-1.5B-Instruct, 2 epochs)
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,
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
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
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
            # Non-coding Qwen instruct model
            "path": "Qwen/Qwen2.5-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console"],
        "project_name": "AgentLightning",
        "experiment_name": "webshop",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


def config_fast() -> Dict[str, Any]:
    """Fast training configuration for CI testing.

    Uses a smaller model (Qwen2.5-0.5B-Instruct) and minimal epochs
    for quick validation of the training pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"webshop_fast_{timestamp}"
    project_name = "AgentLightningCI"

    # Write to GitHub output if running in CI
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={project_name}\n")
            f.write(f"run_name={experiment_name}\n")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-0.5B-Instruct"
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["test_freq"] = 1
    config["trainer"]["experiment_name"] = experiment_name
    config["trainer"]["project_name"] = project_name
    return config


def config_qwen() -> Dict[str, Any]:
    """Standard Qwen training configuration.

    Uses Qwen2.5-1.5B-Instruct for full training runs.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["experiment_name"] = f"webshop_qwen_{timestamp}"
    return config

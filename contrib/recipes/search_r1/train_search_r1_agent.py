# Copyright (c) Microsoft. All rights reserved.


from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from search_r1_agent import SearchR1Agent

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train.parquet",
        "val_files": "data/test.parquet",
        "train_batch_size": 512,
        "max_prompt_length": 6000,
        "max_response_length": 4096,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 5,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.5,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6, "lr_warmup_steps_ratio": 0.95},
            "use_kl_loss": True,
            "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 4,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 8,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "searchr1",
        "nnodes": 1,
        "test_freq": 10,
        "save_freq": 10,
        "total_epochs": 15,
        "total_training_steps": 300,
        "default_local_dir": "checkpoints/searchr1_checkpoints/",
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"searchr1_{timestamp}"
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    config["data"]["val_files"] = "data/test_dev.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-3B-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-3B-Instruct"
    return config


def train(config: Dict[str, Any]) -> None:

    agent = SearchR1Agent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=32, algorithm=algorithm)

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train a Search-R1 agent using different model configurations")

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen-2.5-Coder-1.5B), 'llama' (LLaMA-3.2-3B-Instruct)",
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {"fast": config_train_fast, "qwen": config_train_qwen, "llama": config_train_llama}

    config = config_functions[args.config]()

    print(f"Starting training with '{args.config}' configuration...")

    train(config)


if __name__ == "__main__":
    main()

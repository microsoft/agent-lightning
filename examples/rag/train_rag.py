# Copyright (c) Microsoft. All rights reserved.

"""Train a RAG agent using Agent-lightning.

Usage:
    python train_rag.py fast        # Fast training for CI/testing
    python train_rag.py single_gpu  # Optimized for Single GPU (1.5B/7B models)
"""

from __future__ import annotations

import argparse
import os
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from rag_agent import RAGAgent  # Make sure to import your RAGAgent class

import agentlightning as agl

# Base configuration (default configuration, can be overridden)
RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",  # Use GRPO algorithm
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,  # Default configuration for multi-GPU
        "max_prompt_length": 8192,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,  # Generate 4 responses per sampling
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},  # Ensure using template format matching the model
            "name": "vllm",
            "gpu_memory_utilization": 0.6,  # vLLM GPU memory utilization
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
                "param_offload": True,  # Enable parameter offloading to save GPU memory
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-1.5B-Instruct",  # Default model
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console"],  # Disable wandb for easier local debugging, add back when needed
        "project_name": "AgentLightning",
        "experiment_name": "rag_agent",
        "nnodes": 1,
        "test_freq": 10,
        "total_epochs": 200,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """Fast training configuration for CI/testing"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    EXPERIMENT_NAME = f"rag_fast_{timestamp}_{random_suffix}"

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

    # Keep it tiny/light without adding new knobs
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.8
    config["trainer"]["total_epochs"] = 2
    config["trainer"]["test_freq"] = 5
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["logger"] = ["console", "wandb"]
    return config


def config_train_single_gpu() -> Dict[str, Any]:
    """Single GPU training optimized configuration (optimized for 24GB GPU memory)"""

    config = deepcopy(RL_TRAINING_CONFIG)

    # 1. Reduce vLLM memory usage to leave space for training
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.4

    # 2. Reduce Batch Size to prevent OOM
    config["data"]["train_batch_size"] = 4
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2

    # 3. Ensure Offload is enabled
    config["actor_rollout_ref"]["actor"]["fsdp_config"]["param_offload"] = True
    config["actor_rollout_ref"]["actor"]["fsdp_config"]["optimizer_offload"] = True

    return config


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the RAG agent with the given configuration."""

    # 1. Instantiate your Agent
    agent = RAGAgent()

    # 2. Initialize algorithm (VERL)
    algorithm = agl.VERL(config)

    # 3. Initialize Trainer
    # n_runners=4 means 4 concurrent rollout runners (can be reduced if insufficient memory, or managed internally by VERL)
    trainer = agl.Trainer(n_runners=4, algorithm=algorithm, adapter={"agent_match": active_agent})

    # 4. Load data
    # NOTE: Fill in the path to your previously converted parquet file here
    # For demo purposes, we use the same dataset for training and validation,
    # which should be avoided in production.
    train_df: pd.DataFrame = pd.read_parquet("data/dataset_tiny.parquet")  # type: ignore
    val_df: pd.DataFrame = pd.read_parquet("data/dataset_tiny.parquet")  # type: ignore

    # Keep the rest of the code unchanged
    train_data: List[Dict[str, Any]] = train_df.to_dict(orient="records")  # type: ignore
    val_data: List[Dict[str, Any]] = val_df.to_dict(orient="records")  # type: ignore

    # 5. Start training
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RAG agent using different configurations")

    parser.add_argument(
        "config",
        choices=["fast", "single_gpu"],
        default="single_gpu",
        nargs="?",
        help="Training configuration name",
    )

    parser.add_argument("--active-agent", type=str, help="Override the active agent name")

    args = parser.parse_args()

    config_functions = {
        "fast": config_train_fast,
        "single_gpu": config_train_single_gpu,
    }
    config = config_functions[args.config]()

    # Print key information for confirmation
    print(f"Starting training with '{args.config}' configuration...")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"Batch Size: {config['data']['train_batch_size']}")
    print(f"GPU Mem Util: {config['actor_rollout_ref']['rollout']['gpu_memory_utilization']}")

    train(config, args.active_agent)


if __name__ == "__main__":
    main()

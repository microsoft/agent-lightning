# Copyright (c) Microsoft. All rights reserved.

"""Train a ChartQA agent on the ChartQA dataset using Agent-lightning.

This module provides a training script for ChartQA agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen2-VL-2B-Instruct model
3. 'llama' - Configuration using LLaMA-3.2-11B-Vision-Instruct model

Usage:
    python train_chartqa_agent.py fast    # Fast training for CI/testing
    python train_chartqa_agent.py qwen    # Standard Qwen model training
    python train_chartqa_agent.py llama   # LLaMA vision model training

The script uses reinforcement learning with VERL framework
to train agents on the ChartQA dataset for visual reasoning about charts and graphs.
"""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from chartqa_agent import LitChartQAAgent

import agentlightning as agl

# Calculate absolute path to images directory (portable across environments)
# Use realpath to resolve symlinks and match vLLM's path validation
CHARTQA_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTQA_IMAGES_DIR = os.path.realpath(os.path.join(CHARTQA_EXAMPLES_DIR, "data", "images"))

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train_chartqa.parquet",
        "val_files": "data/test_chartqa.parquet",
        "train_batch_size": 4,  # Smaller due to vision tokens
        "max_prompt_length": 4096,  # Increased for vision tokens
        "max_response_length": 1024,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 2,
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
            "enable_prefix_caching": True,  # Cache vision tokens
            "engine_kwargs": {
                "vllm": {
                    "allowed_local_media_path": CHARTQA_IMAGES_DIR,
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 2,
            "optim": {"lr": 1e-6},  # Lower LR for vision-language models
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
            "log_prob_micro_batch_size_per_gpu": 2,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
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
        "experiment_name": "chartqa",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="chartqa_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"chartqa_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it's set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2-VL-2B-Instruct"
    config["data"]["train_batch_size"] = 2
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen2-VL-2B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-11B-Vision-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    config["data"]["train_batch_size"] = 2  # Reduce batch size for larger model
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
    return config


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the ChartQA agent with the given configuration."""

    # Initialize Ray debugpy on port 5679
    try:
        import debugpy
        import ray

        # Check if we're in a Ray worker
        if ray.is_initialized():
            # Only attach debugger if not already attached
            if not debugpy.is_client_connected():
                debugpy.listen(("0.0.0.0", 5679))
                print(f"[DEBUG] Debugpy listening on port 5679 (Ray worker PID: {os.getpid()})")
                print("[DEBUG] Waiting for debugger to attach...")
                debugpy.wait_for_client()
                print("[DEBUG] Debugger attached!")
        else:
            # For main process, optionally enable debugging
            if os.getenv("ENABLE_DEBUG", "0") == "1":
                debugpy.listen(("0.0.0.0", 5679))
                print(f"[DEBUG] Debugpy listening on port 5679 (Main process PID: {os.getpid()})")
                print("[DEBUG] Waiting for debugger to attach...")
                debugpy.wait_for_client()
                print("[DEBUG] Debugger attached!")
    except ImportError:
        print("[WARNING] debugpy not installed. Skipping debug setup.")
    except Exception as e:
        print(f"[WARNING] Failed to initialize debugpy: {e}")

    # Setup timestamped file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    log_file = os.path.join(log_dir, f"chartqa_training_{timestamp}.log")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging with both console and file output
    agl.setup_logging(level="DEBUG", apply_to=["agentlightning", __name__])

    # Add file handler manually
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s",
        datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning").addHandler(file_handler)
    logging.getLogger(__name__).addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Training started - logs saved to {log_file}")

    agent = LitChartQAAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a ChartQA agent on the ChartQA dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen2-VL-2B), 'llama' (LLaMA-3.2-11B-Vision)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {"fast": config_train_fast, "qwen": config_train_qwen, "llama": config_train_llama}

    config = config_functions[args.config]()

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    train(config, active_agent)


if __name__ == "__main__":
    main()

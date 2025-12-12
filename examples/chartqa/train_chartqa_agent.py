# Copyright (c) Microsoft. All rights reserved.

"""Train ChartQA agent using VERL reinforcement learning."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Optional, cast

import env_var as chartqa_env_var
import pandas as pd
from chartqa_agent import ChartQAAgent

import agentlightning as agl
from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var

RL_CONFIG: Dict[str, Any] = {
    "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
    "data": {
        "image_base_dir": chartqa_env_var.CHARTQA_IMAGES_DIR,
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 1024,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 1,
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": True,
            "engine_kwargs": {"vllm": {"allowed_local_media_path": chartqa_env_var.CHARTQA_IMAGES_DIR}},
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {"param_offload": True, "optimizer_offload": True},
        },
        "ref": {"log_prob_micro_batch_size_per_gpu": 1, "fsdp_config": {"param_offload": True}},
        "model": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "chartqa",
        "nnodes": 1,
    },
}


def config_ci() -> Dict[str, Any]:
    """Config for CI testing."""
    config = deepcopy(RL_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 1
    config["trainer"]["total_training_steps"] = 6
    config["trainer"]["val_before_train"] = True
    config["trainer"]["test_freq"] = 3
    return config


def config_debug() -> Dict[str, Any]:
    """debug config for debugging and testing (several steps)."""
    config = deepcopy(RL_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
    config["trainer"]["total_training_steps"] = 10
    config["trainer"]["test_freq"] = 2
    return config


def config_qwen() -> Dict[str, Any]:
    config = deepcopy(RL_CONFIG)
    config["trainer"]["val_before_train"] = True
    config["trainer"]["n_gpus_per_node"] = 2
    config["trainer"]["total_epochs"] = 2
    config["trainer"]["test_freq"] = 32
    return config


def train(
    config: Dict[str, Any],
    train_data: agl.Dataset[Any],
    val_data: agl.Dataset[Any],
    external_store_address: str,
    n_runners: int,
    debug: bool,
) -> None:
    agl.setup_logging(level="DEBUG" if debug else "INFO", apply_to=["agentlightning", __name__])
    agent = ChartQAAgent()
    algorithm = agl.VERL(config)

    if external_store_address:
        store: Optional[agl.LightningStore] = agl.LightningStoreClient(external_store_address)
    else:
        store = None

    trainer = agl.Trainer(
        n_runners=n_runners,
        algorithm=algorithm,
        store=store,
    )

    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main():
    agl.setup_logging(apply_to=["chartqa_agent"])
    parser = argparse.ArgumentParser(description="Train ChartQA agent")
    parser.add_argument("config", choices=["debug", "qwen", "ci"], help="Training configuration")
    parser.add_argument("--n-runners", type=int, default=10, help="Number of runners for Trainer")
    parser.add_argument(
        "--external-store-address",
        type=str,
        default=None,
        help="Connect to an external store instead of creating a new one in memory (e.g., http://localhost:4747)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.external_store_address:
        print(f"Connecting to external store at: {args.external_store_address}")
        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError(
                "When using an external store, please set the environment variable AGL_MANAGED_STORE=0. "
                "Otherwise the trainer will still try to manage the store lifecycle for you!"
            )

    CONFIGS = {
        "debug": config_debug,
        "qwen": config_qwen,
        "ci": config_ci,
    }

    train_data_path = os.path.join(chartqa_env_var.CHARTQA_DATA_DIR, "train_chartqa.parquet")
    val_data_path = os.path.join(chartqa_env_var.CHARTQA_DATA_DIR, "test_chartqa.parquet")

    train_data = pd.read_parquet(train_data_path).to_dict(orient="records")  # type: ignore

    if args.config in ["debug", "ci"]:
        val_data = pd.read_parquet(val_data_path).sample(n=100, random_state=42).to_dict(orient="records")  # type: ignore
    else:
        val_data = pd.read_parquet(val_data_path).to_dict(orient="records")  # type: ignore

    train(
        config=CONFIGS[args.config](),
        train_data=cast(agl.Dataset[Any], train_data),
        val_data=cast(agl.Dataset[Any], val_data),
        external_store_address=args.external_store_address,
        n_runners=args.n_runners,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

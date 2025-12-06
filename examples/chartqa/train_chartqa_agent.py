"""Train ChartQA agent using VERL reinforcement learning."""

from __future__ import annotations

import argparse
import asyncio
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import nest_asyncio
import pandas as pd
from chartqa_agent import LitChartQAAgent

import agentlightning as agl

nest_asyncio.apply()

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.realpath(os.path.join(EXAMPLES_DIR, "data"))
IMAGES_DIR = os.path.realpath(os.path.join(EXAMPLES_DIR, "data", "images"))

RL_CONFIG: Dict[str, Any] = {
    "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
    "data": {
        "train_files": "data/train_chartqa.parquet",
        "val_files": "data/test_chartqa.parquet",
        "image_base_dir": DATA_DIR,
        "train_batch_size": 1,
        "max_prompt_length": 2048,
        "max_response_length": 512,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 2,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 1,
            "name": "vllm",
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": True,
            "engine_kwargs": {"vllm": {"allowed_local_media_path": IMAGES_DIR}},
        },
        "actor": {
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {"param_offload": True, "optimizer_offload": True},
        },
        "ref": {"log_prob_micro_batch_size_per_gpu": 1, "fsdp_config": {"param_offload": True}},
        "model": {"path": "Qwen/Qwen2-VL-2B-Instruct", "use_remove_padding": True, "enable_gradient_checkpointing": True},
    },
    "trainer": {
        "n_gpus_per_node": 2,
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


def config_fast() -> Dict[str, Any]:
    config = deepcopy(RL_CONFIG)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config["trainer"]["experiment_name"] = f"chartqa_{timestamp}"
    config["trainer"]["project_name"] = "AgentLightningCI"
    config["trainer"]["total_training_steps"] = 100000
    config["trainer"]["test_freq"] = 1
    if github_output := os.getenv("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.write(f"project_name={config['trainer']['project_name']}\n")
            f.write(f"run_name={config['trainer']['experiment_name']}\n")
    return config


def config_qwen() -> Dict[str, Any]:
    return deepcopy(RL_CONFIG)


def train(config: Dict[str, Any]) -> None:
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

    agl.setup_logging(level="DEBUG", apply_to=["agentlightning", __name__])
    agent = LitChartQAAgent()
    algorithm = agl.VERL(config)

    async def run():
        store = agl.InMemoryLightningStore()
        store_server = agl.LightningStoreServer(store, "127.0.0.1", 4747)
        await store_server.start()
        try:
            store_client = agl.LightningStoreClient("http://127.0.0.1:4747")
            trainer = agl.Trainer(
                n_runners=10,
                algorithm=algorithm,
                store=store_client,
                strategy={"name": "cs", "managed_store": False},
            )
            train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")
            val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")
            trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)
        finally:
            await store_server.stop()

    asyncio.run(run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChartQA agent")
    parser.add_argument("config", choices=["fast", "qwen"], help="Training configuration")
    args = parser.parse_args()
    train({"fast": config_fast, "qwen": config_qwen}[args.config]())

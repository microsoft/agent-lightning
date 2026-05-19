"""Training script for Calc-X agent with VERL on agl-lite (async-rollout).

Loads the Calc-X dataset, builds a VERL config, and calls run_ppo().
Assumes agl-lite serve + controller + vLLM are already running
(started by run.sh or manually).

Usage:
  # Full E2E (via run.sh):
  examples/async_calc_x/run.sh

  # Standalone (infra already up):
  python examples/async_calc_x/train_calc_agent.py \\
      --train-file examples/async_calc_x/data/train.parquet \\
      --val-file examples/async_calc_x/data/test.parquet

  # CI smoke test:
  python examples/async_calc_x/train_calc_agent.py --ci-fast \\
      --train-file examples/async_calc_x/data/train.parquet \\
      --val-file examples/async_calc_x/data/test_mini.parquet

Environment variables:
  AGL_BASE_URL   — agl-lite server URL (default: http://localhost:8080)
  AGL_KEY        — auth key for agl-lite
  AGL_ADMIN_KEY  — trainer-only admin key for /admin/gateway/* (required by async)
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for Calc-X training.

    These are merged on top of agl-lite's base config
    (agl_lite/verl/config.yaml → verl/trainer/config/ppo_trainer.yaml).
    """
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
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
                "path": "Qwen/Qwen2.5-1.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agl-lite",
            "experiment_name": "async_calc_x_v1",
            "nnodes": 1,
            "save_freq": 64,
            "test_freq": 32,
            "total_epochs": 2,
        },
        "agentlightning": {
            "timeout_seconds": 1800,
            "async_rollout": {
                "enabled": True,
                "async_train_batch_size": 48,  # train_batch_size=32 × 1.5
                "gateway_retry_after_seconds": 5,
                "gateway_drain_timeout_seconds": 30,
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    ci: bool = False,
    ci_fast: bool = False,
) -> Any:
    """Build the full OmegaConf config by merging base + overrides.

    Uses Hydra compose to load agl-lite's base config (which includes
    verl's ppo_trainer defaults), then merges Calc-X overrides on top.
    """
    import importlib.resources

    # Locate the agl_lite/verl package directory for Hydra.
    verl_pkg = importlib.resources.files("agl_lite.verl")
    config_dir = str(verl_pkg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()

    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model

    if ci or ci_fast:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid.uuid4().hex[:8]

        overrides["trainer"]["project_name"] = "agl-lite-CI"
        overrides["trainer"]["experiment_name"] = f"async_calc_x_{timestamp}_{random_suffix}"
        overrides["trainer"]["total_epochs"] = 1
        overrides["trainer"]["total_training_steps"] = 20
        overrides["trainer"]["test_freq"] = 20
        overrides["trainer"].pop("save_freq", None)
        overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.8

        if ci_fast:
            overrides["trainer"]["total_training_steps"] = 1
            overrides["trainer"]["test_freq"] = 1
            overrides["trainer"]["n_gpus_per_node"] = 1
            overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6

    override_conf = OmegaConf.create(overrides)
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf)
    return config


def train(
    *,
    train_file: str,
    val_file: str,
    model: str | None = None,
    ci: bool = False,
    ci_fast: bool = False,
) -> None:
    """Load datasets, build config, and launch VERL training via agl-lite."""
    from agl_lite.verl.entrypoint import run_ppo

    # Load datasets.
    train_dataset: Sequence[Any] = cast(
        Sequence[Any],
        HuggingFaceDataset.from_parquet(train_file).to_list(),  # type: ignore
    )
    val_dataset: Sequence[Any] = cast(
        Sequence[Any],
        HuggingFaceDataset.from_parquet(val_file).to_list(),  # type: ignore
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")

    config = build_config(model=model, ci=ci, ci_fast=ci_fast)

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Calc-X agent with VERL on agl-lite.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="examples/async_calc_x/data/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="examples/async_calc_x/data/test.parquet",
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (overrides default Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run a minimal CI-style training loop",
    )
    parser.add_argument(
        "--ci-fast",
        action="store_true",
        help="Single PPO step (implies --ci)",
    )
    args = parser.parse_args()

    if args.ci_fast:
        args.ci = True

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        ci=args.ci,
        ci_fast=args.ci_fast,
    )


if __name__ == "__main__":
    main()

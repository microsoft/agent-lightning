"""Train a ScienceWorld agent with VERL on agl-lite (local runner mode).

Generates a dataset on-the-fly from a list of task names crossed with
variation indices, then drives VERL's async-rollout PPO/GRPO trainer.

Assumes ``agl-lite serve`` and ``agl-lite controller`` are already running
on this host (started by ``run.sh``).

Usage::

    examples/science_world/run.sh                  # full training
    examples/science_world/run.sh --ci-fast        # 1 PPO step smoke test

    # Standalone (infra already up):
    python examples/science_world/train_sw_agent.py \\
        --task-names find-non-living-thing,find-living-thing \\
        --variations-per-task 50

Environment variables (read by VERL / the bridge):

  AGL_BASE_URL   agl-lite server URL (default http://localhost:8080)
  AGL_KEY        shared API key
    AGL_ADMIN_KEY  admin key for /proxy/{pause,resume,state} (required by async-rollout)
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime
from typing import Any

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

DATA_SOURCE = "science_world"


def build_dataset(
    task_names: list[str],
    variations_per_task: int,
    simplification: str,
    val_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the train / val splits as lists of dicts.

    For each task, the per-task variation budget is
    ``min(variations_per_task, env.get_max_variations(task_name))`` — some
    ScienceWorld tasks (e.g. ``identify-life-stages-2``) only define 10
    variations, so an unconditional cap of 50 would crash inside
    ``env.load``. The budget is then deterministically split: the last
    ``val_fraction`` of indices go to val so the split is reproducible
    without an RNG seed.
    """
    if variations_per_task <= 0:
        raise ValueError("variations_per_task must be positive")

    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for task_name in task_names:
        budget = min(variations_per_task, env.get_max_variations(task_name))
        n_val = max(1, int(budget * val_fraction))
        n_train = budget - n_val
        if n_train <= 0:
            raise ValueError(
                f"task {task_name!r} has only {budget} variations after "
                f"val_fraction={val_fraction}; no train rows left"
            )
        for v in range(n_train):
            train.append(_row(task_name, v, simplification))
        for v in range(n_train, budget):
            val.append(_row(task_name, v, simplification))
    return train, val


def resolve_task_names(arg: str) -> list[str]:
    """Resolve the --task-names argument.

    ``"all"`` (case-insensitive) expands to every ScienceWorld task name
    via ``env.get_task_names()``. Otherwise parses as a comma-separated
    list.
    """
    if arg.strip().lower() == "all":
        from scienceworld import ScienceWorldEnv

        return ScienceWorldEnv().get_task_names()
    return [t.strip() for t in arg.split(",") if t.strip()]


def _row(task_name: str, variation_idx: int, simplification: str) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "variation_idx": variation_idx,
        "simplification": simplification,
        "data_source": DATA_SOURCE,
    }


def verl_default_config() -> dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 10240,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 2,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "name": "vllm",
                "gpu_memory_utilization": 0.75,
                "checkpoint_engine": {
                    "update_weights_bucket_megabytes": 4096,
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
                "path": os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": int(os.environ.get("AGL_N_GPUS_PER_NODE", "8")),
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agl-lite",
            "experiment_name": "science_world_v1",
            "nnodes": 1,
            "save_freq": 32,
            "test_freq": 32,
            "total_epochs": 4,
        },
        "agentlightning": {
            "timeout_seconds": 1800,
            # Local runner has no K8s Jobs to garbage-collect — the controller
            # tears down each rollout subprocess on its own.
            "cleanup_agent_jobs": False,
            "async_rollout": {
                "enabled": True,
                "async_train_batch_size": 48,
                "gateway_retry_after_seconds": 5,
                "gateway_drain_timeout_seconds": 45,
            },
        },
    }


def build_config(*, model: str | None = None, ci: bool = False, ci_fast: bool = False) -> Any:
    import importlib.resources

    verl_pkg = importlib.resources.files("agl_lite.verl")
    config_dir = str(verl_pkg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()
    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model

    if ci or ci_fast:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        overrides["trainer"]["project_name"] = "agl-lite-CI"
        overrides["trainer"]["experiment_name"] = f"science_world_{timestamp}_{suffix}"
        overrides["trainer"]["total_epochs"] = 1
        overrides["trainer"]["total_training_steps"] = 20
        overrides["trainer"]["test_freq"] = 20
        overrides["trainer"].pop("save_freq", None)
        overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.8

        if ci_fast:
            overrides["trainer"]["total_training_steps"] = 1
            overrides["trainer"]["test_freq"] = 1
            overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
            overrides["data"]["train_batch_size"] = 4
            overrides["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
            overrides["actor_rollout_ref"]["rollout"]["n"] = 2
            overrides["agentlightning"]["async_rollout"]["async_train_batch_size"] = 6

    override_conf = OmegaConf.create(overrides)
    OmegaConf.set_struct(base_cfg, False)
    return OmegaConf.merge(base_cfg, override_conf)


def train(
    *,
    task_names: list[str],
    variations_per_task: int,
    simplification: str,
    model: str | None,
    ci: bool,
    ci_fast: bool,
) -> None:
    from agl_lite.verl.entrypoint import run_ppo

    train_rows, val_rows = build_dataset(task_names, variations_per_task, simplification)
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")
    print(f"Tasks: {task_names}  variations/task: {variations_per_task}  simplification: {simplification}")

    # VERL's bridge expects something it can index by column (HuggingFaceDataset semantics).
    train_dataset = HuggingFaceDataset.from_list(train_rows).to_list()
    val_dataset = HuggingFaceDataset.from_list(val_rows).to_list()

    config = build_config(model=model, ci=ci, ci_fast=ci_fast)

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ScienceWorld agent with VERL on agl-lite.")
    parser.add_argument(
        "--task-names",
        type=str,
        default=os.environ.get("AGL_TASK_NAMES", "all"),
        help="Comma-separated ScienceWorld task names, or 'all' for every task",
    )
    parser.add_argument(
        "--variations-per-task",
        type=int,
        default=int(os.environ.get("AGL_VARIATIONS_PER_TASK", "50")),
        help="Max variation indices per task (auto-capped at env.get_max_variations)",
    )
    parser.add_argument(
        "--simplification",
        type=str,
        default=os.environ.get("AGL_SIMPLIFICATION", "easy"),
        help="ScienceWorld simplification preset (easy / medium / hard)",
    )
    parser.add_argument("--model", type=str, default=None, help="HF model id (overrides AGL_MODEL_NAME)")
    parser.add_argument("--ci", action="store_true", help="Minimal CI-style loop (20 steps)")
    parser.add_argument("--ci-fast", action="store_true", help="Single PPO step (implies --ci)")
    args = parser.parse_args()

    if args.ci_fast:
        args.ci = True

    task_names = resolve_task_names(args.task_names)
    if not task_names:
        raise SystemExit("no task names provided")

    train(
        task_names=task_names,
        variations_per_task=args.variations_per_task,
        simplification=args.simplification,
        model=args.model,
        ci=args.ci,
        ci_fast=args.ci_fast,
    )


if __name__ == "__main__":
    main()

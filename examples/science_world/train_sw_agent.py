# Copyright (c) Microsoft. All rights reserved.

"""Train a ScienceWorld agent with VERL on Agent Lightning (local runner mode).

Generates a dataset on-the-fly from a list of task names crossed with
variation indices, then drives VERL's PPO/GRPO trainer. Each rollout is run
by the local controller as a short-lived ``SWAgent`` subprocess.

Assumes ``agl-server`` and ``agl-controller runner_type=local`` are
already running on this host (started by ``run_local.sh``).

Usage::

    examples/science_world/run_local.sh

    # Standalone (infra already up):
    python examples/science_world/train_sw_agent.py \\
        --task-names find-non-living-thing,find-living-thing \\
        --variations-per-task 50
"""

from __future__ import annotations

import argparse
from typing import Any

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

DATA_SOURCE = "science_world"


def resolve_task_names(arg: str) -> list[str]:
    """Resolve the --task-names argument.

    ``"all"`` (case-insensitive) expands to every ScienceWorld task name
    via ``env.get_task_names()``. Otherwise parses as a comma-separated list.
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


def build_dataset(
    task_names: list[str],
    variations_per_task: int,
    simplification: str,
    val_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the train / val splits as lists of dicts.

    For each task, the per-task variation budget is
    ``min(variations_per_task, env.get_max_variations(task_name))`` — some
    ScienceWorld tasks only define a handful of variations, so an
    unconditional cap would crash inside ``env.load``. The budget is then
    deterministically split: the last ``val_fraction`` of indices go to val.
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


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for ScienceWorld training (local runner).

    Merged on top of Agent Lightning's base config
    (agentlightning/verl/config.yaml → verl/trainer/config/ppo_trainer.yaml).
    """
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "rollout_correction": {
                "bypass_mode": True,
                "loss_type": "ppo_clip",
                "rollout_is": None,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            },
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            # Trajectory level merges all turns into one sequence, so the
            # response tensor must be large enough to hold the merged turns;
            # keep this == trace_aggregator.trajectory_max_response_length.
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 2,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
                "checkpoint_engine": {"update_weights_bucket_megabytes": 4096},
            },
            "actor": {
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 4,
                "ulysses_sequence_parallel_size": 2,
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
                "fsdp_config": {"param_offload": False},
            },
            "model": {
                "path": "Qwen/Qwen2.5-7B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 8,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "science_world",
            "nnodes": 1,
            "save_freq": 32,
            "test_freq": 16,
            "total_epochs": 1,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "rollout_timeout_seconds": 1800,
            "trace_aggregator": {
                # Merge all turns of a rollout into one trajectory sequence
                # (multi-turn credit assignment) instead of per-transition rows.
                "level": "trajectory",
                "trajectory_max_prompt_length": 4096,
                "trajectory_max_response_length": 1024,
            },
            "async_rollout": {
                "enabled": True,
                # Over-sample beyond train_batch_size (32) so group-finish early
                # stopping can cut the long tail; must be strictly greater.
                "async_train_batch_size": 48,
            },
            "local": {
                "agent_class": "examples.science_world.agents.sw_agent:SWAgent",
                "env_map": {
                    "TASK_NAME": "input.task_name",
                    "VARIATION_IDX": "input.variation_idx",
                    "SIMPLIFICATION": "input.simplification",
                },
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
) -> Any:
    """Build the full OmegaConf config by merging base + overrides."""
    import importlib.resources

    verl_pkg = importlib.resources.files("agentlightning.verl")
    config_dir = str(verl_pkg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()

    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key
    if run_name:
        overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'

    override_conf = OmegaConf.create(overrides)
    OmegaConf.set_struct(base_cfg, False)
    return OmegaConf.merge(base_cfg, override_conf)


def train(
    *,
    task_names: list[str],
    variations_per_task: int,
    simplification: str,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
) -> None:
    from agentlightning.verl.entrypoint import run_ppo

    train_rows, val_rows = build_dataset(task_names, variations_per_task, simplification)
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")
    print(f"Tasks: {task_names}  variations/task: {variations_per_task}  simplification: {simplification}")

    # VERL's bridge expects HuggingFaceDataset.to_list() semantics.
    train_dataset = HuggingFaceDataset.from_list(train_rows).to_list()
    val_dataset = HuggingFaceDataset.from_list(val_rows).to_list()

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
    )

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ScienceWorld agent with VERL on Agent Lightning.")
    parser.add_argument(
        "--task-names",
        type=str,
        default="all",
        help="Comma-separated ScienceWorld task names, or 'all' for every task",
    )
    parser.add_argument(
        "--variations-per-task",
        type=int,
        default=50,
        help="Max variation indices per task (auto-capped at env.get_max_variations)",
    )
    parser.add_argument(
        "--simplification",
        type=str,
        default="easy",
        help="ScienceWorld simplification preset (easy / medium / hard)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--agl-base-url",
        type=str,
        default="http://localhost:8080",
        help="Agent Lightning server URL for the trainer",
    )
    parser.add_argument(
        "--agl-key",
        type=str,
        default="",
        help="Agent Lightning API key for the trainer",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Suffix appended to trainer.experiment_name",
    )
    args = parser.parse_args()

    task_names = resolve_task_names(args.task_names)
    if not task_names:
        raise SystemExit("no task names provided")

    train(
        task_names=task_names,
        variations_per_task=args.variations_per_task,
        simplification=args.simplification,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

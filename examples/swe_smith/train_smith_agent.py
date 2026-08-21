# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import importlib.resources
import json
from pathlib import Path
from pprint import pprint
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DATA_SOURCE = "swe_smith"
TRAIN_BACKEND = "fsdp"

INSTANCE_FIELDS = (
    "instance_id",
    "problem_statement",
    "image_name",
    "repo",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def log(message: str) -> None:
    print(message, flush=True)


def _project(instance: dict[str, Any]) -> dict[str, Any]:
    row = {field: instance.get(field) for field in INSTANCE_FIELDS}
    row["data_source"] = DATA_SOURCE
    row["data_id"] = str(instance.get("instance_id", ""))
    return row


def load_split_file(
    path: str,
    *,
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    """Load a pre-split, pre-curated JSONL dataset and project to the VERL schema."""
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    selected = [_project(row) for row in rows]
    if max_instances:
        selected = selected[:max_instances]
    if not selected:
        raise ValueError(f"No instances loaded from {path}")
    return selected


def verl_default_config() -> dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 16,
            "max_prompt_length": 65536,
            "max_response_length": 65536,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "mode": "async",
                "tensor_model_parallel_size": 1,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.8,
                "max_model_len": 81920,
                "max_num_batched_tokens": 8192,
                "enforce_eager": False,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
                        # vLLM 0.20 FlashInfer MoE uses a blocked runtime layout
                        # that is not compatible with bucketed IPC weight refit.
                        "moe_backend": "triton",
                    }
                },
                "temperature": 1,
                "val_kwargs": {"temperature": 0.7, "do_sample": True},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "checkpoint_engine": {"update_weights_bucket_megabytes": 4096},
            },
            "actor": {
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 16384,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                    "entropy_from_logits_with_chunking": True,
                },
                "loss_agg_mode": "token-mean",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "use_fused_kernels": True,
                "fused_kernel_options": {"impl_backend": "torch"},
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "swe_smith",
            "nnodes": 1,
            "nccl_timeout": 1800,
            "test_freq": 16,
            "save_freq": 16,
            "total_epochs": 4,
            "total_training_steps": 1000,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "rollout_timeout_seconds": 5400,
            "reward_fillna_value": 0.0,
            "max_ppo_update_times": 2,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 65536,
                "trajectory_max_response_length": 65536,
            },
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 50,
            },
            "k8s": {
                "job_template_path": str(EXAMPLE_DIR / "job-template-openai.yaml"),
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()

    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key

    rollout_mode = overrides["actor_rollout_ref"]["rollout"]["mode"]
    model_path = overrides["actor_rollout_ref"]["model"]["path"]
    overrides["trainer"]["experiment_name"] = f"swe_smith_{rollout_mode}_{model_path.split('/')[-1]}_{TRAIN_BACKEND}"
    if run_name:
        overrides["trainer"]["experiment_name"] = f"{overrides['trainer']['experiment_name']}_{run_name}"

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    OmegaConf.set_struct(config, False)
    return config


def train(
    *,
    train_dataset_path: str,
    val_dataset_path: str,
    max_val_instances: int | None = None,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> None:
    from agentlightning.verl.entrypoint import run_ppo

    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    train_dataset = load_split_file(train_dataset_path)
    val_dataset = load_split_file(val_dataset_path, max_instances=max_val_instances)
    instances = train_dataset + val_dataset
    distinct_repos = sorted({row["repo"] for row in instances})

    log("=== Preflight ===")
    log(f"  Agent Lightning: {agl_base_url or 'http://localhost:8080'}")
    log(f"  model:        {model or DEFAULT_MODEL}")
    log(f"  train file:   {train_dataset_path}")
    log(f"  val file:     {val_dataset_path}")
    log(f"  instances:    {len(instances)}  (train {len(train_dataset)} / val {len(val_dataset)})")
    log(f"  distinct repos (images to prepare): {len(distinct_repos)}")

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
    )
    log("\n=== VERL config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a SWE-smith agent with VERL/GRPO via Agent Lightning")
    parser.add_argument(
        "--train-dataset-path",
        default=str(EXAMPLE_DIR / "train_dataset_mixed.jsonl"),
        help="Pre-split training JSONL, used as-is.",
    )
    parser.add_argument(
        "--val-dataset-path",
        default=str(EXAMPLE_DIR / "val_dataset_filtered.jsonl"),
        help="Pre-split validation JSONL, used as-is. Pairs with --train-dataset-path.",
    )
    parser.add_argument(
        "--max-val-instances",
        type=int,
        default=None,
        help="Optional cap on validation instances (default: all). Each validation eval "
        "runs ALL val instances at the test_freq cadence, so capping bounds eval time.",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agl-base-url", default="http://localhost:8080")
    parser.add_argument("--agl-key", default="")
    parser.add_argument("--run-name", default=None)
    args, config_overrides = parser.parse_known_args()
    return args, config_overrides


def main() -> None:
    args, config_overrides = parse_args()
    train(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        max_val_instances=args.max_val_instances,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()

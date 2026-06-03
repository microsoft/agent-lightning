#!/usr/bin/env python3
"""Train llm-in-sandbox with VERL through agl-lite."""

from __future__ import annotations

import argparse
import copy
import importlib.resources
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRAIN_DATA_DIR = "examples/llm-in-sandbox/data/llm_sandbox_sampled_pretrain_mini"
DEFAULT_VAL_DATA_DIR = "examples/llm-in-sandbox/data/llm_sandbox_sampled_vali_mini"


def log(message: str) -> None:
    print(message, flush=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def path_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json_dataset(
    path: Path,
    *,
    folder_name: str,
    filename: str,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        raw_items = json.load(file)
    if not isinstance(raw_items, list):
        raise TypeError(f"dataset must be a JSON list: {path}")

    limit = len(raw_items) if max_samples <= 0 else min(max_samples, len(raw_items))
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items[:limit]):
        if not isinstance(raw, dict):
            raise TypeError(f"dataset sample {index} is not an object: {path}")
        sample = copy.deepcopy(raw)
        extra_info = sample.setdefault("extra_info", {})
        if not isinstance(extra_info, dict):
            raise TypeError(f"sample {index} extra_info is not an object: {path}")
        extra_info["data_folder_name"] = folder_name
        extra_info["data_filename"] = filename
        extra_info["data_index"] = index
        extra_info.setdefault("index", index)
        sample["data_id"] = str(extra_info.get("id") or f"{folder_name}:{index}")
        items.append(sample)
    return items


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for llm-in-sandbox training."""
    example_dir = Path(__file__).resolve().parent
    return {
        "algorithm": {
            "adv_estimator": "rloo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 8,
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
                "gpu_memory_utilization": 0.7,
                "max_model_len": 65536,
                "enforce_eager": True,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
                "temperature": 1,
                "val_kwargs": {"temperature": 0, "do_sample": False},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": False,
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_type": "low_var_kl",
                "kl_loss_coef": 0.001,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
                "loss_agg_mode": "seq-mean-token-sum",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "AgentLightning-k8s",
            "experiment_name": "train_llm_in_sandbox_new_code",
            "nnodes": 1,
            "test_freq": 20,
            "save_freq": 20,
            "total_epochs": 15,
            "total_training_steps": 1000,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "is_shuffle": False,
            "timeout_seconds": 1500,
            "poll_timeout_seconds": 1500,
            "reward_fillna_value": 0.0,
            "cleanup_agent_jobs": env_bool("AGL_CLEANUP_AGENT_JOBS", False),
            "cleanup_namespace": os.environ.get("AGL_NAMESPACE", "default"),
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 8000,
                "trajectory_max_response_length": 12000,
                "debug": False,
                "mismatch_log_dir": str(example_dir / "mismatch_cases"),
            },
            "k8s": {
                "job_template_path": str(example_dir / "job-template.yaml"),
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
    ci: bool = False,
) -> DictConfig:
    """Build the full OmegaConf config by merging base + overrides."""
    verl_pkg = importlib.resources.files("agl_lite.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
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

    if ci:
        overrides["trainer"]["project_name"] = "AgentLightning-k8s-CI"
        overrides["trainer"]["experiment_name"] = "train_llm_in_sandbox_ci"
        if run_name:
            overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'
        overrides["trainer"]["total_epochs"] = 1
        overrides["trainer"]["total_training_steps"] = 5
        overrides["trainer"]["test_freq"] = -1
        overrides["trainer"].pop("save_freq", None)
        overrides["trainer"]["n_gpus_per_node"] = 1
        overrides["trainer"]["logger"] = ["console"]
        overrides["data"]["train_batch_size"] = 1
        overrides["data"]["max_prompt_length"] = 2048
        overrides["data"]["max_response_length"] = 2048
        overrides["actor_rollout_ref"]["rollout"]["n"] = 1
        overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
        overrides["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 1
        overrides["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
        overrides["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
        overrides["agentlightning"]["timeout_seconds"] = 300
        overrides["agentlightning"]["poll_timeout_seconds"] = 300
        overrides["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"] = 1024
        overrides["agentlightning"]["trace_aggregator"]["trajectory_max_response_length"] = 1024

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    OmegaConf.set_struct(config, False)
    return config


def train(
    *,
    train_data_dir: str,
    val_data_dir: str,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    ci: bool = False,
) -> None:
    """Load datasets, build config, and launch VERL training via agl-lite."""
    from agl_lite.verl.entrypoint import run_ppo

    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    train_dir = resolve_path(train_data_dir)
    train_dataset = load_json_dataset(
        train_dir / "train_verl.json",
        folder_name=train_dir.name,
        filename="train_verl.json",
        max_samples=max_train_samples,
    )

    val_datasets: list[tuple[str, list[dict[str, Any]]]] = []
    for val_data_dir_arg in path_list(val_data_dir):
        val_dir = resolve_path(val_data_dir_arg)
        val_datasets.append(
            (
                val_dir.name,
                load_json_dataset(
                    val_dir / "test_verl.json",
                    folder_name=val_dir.name,
                    filename="test_verl.json",
                    max_samples=max_val_samples,
                ),
            )
        )
    val_dataset = [sample for _, dataset in val_datasets for sample in dataset]

    log("=== Preflight ===")
    log(f"  agl-lite:     {agl_base_url or 'http://localhost:8080'}")
    log(f"  model:        {model or '(default)'}")
    log(f"  train:        {len(train_dataset)} samples from {train_dir.name}")
    for val_folder_name, dataset in val_datasets:
        log(f"  val/test:     {len(dataset)} samples from {val_folder_name}")
    log(f"  val total:    {len(val_dataset)} samples")

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
        ci=ci,
    )
    log("\n=== VERL config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train llm-in-sandbox with VERL via agl-lite")
    parser.add_argument(
        "--train-data-dir",
        default=os.environ.get("AGL_TRAIN_DATA_DIR", DEFAULT_TRAIN_DATA_DIR),
    )
    parser.add_argument(
        "--val-data-dir",
        "--test-data-dir",
        dest="val_data_dir",
        default=os.environ.get("AGL_TEST_DATA_DIR", DEFAULT_VAL_DATA_DIR),
    )
    parser.add_argument("--model", default=os.environ.get("AGL_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--agl-base-url", default=os.environ.get("AGL_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--agl-key", default=os.environ.get("AGL_KEY", ""))
    parser.add_argument(
        "--run-name",
        default=os.environ.get("AGL_VERL_EXPERIMENT_NAME", None),
        help="Suffix appended to trainer.experiment_name",
    )
    parser.add_argument("--max-train-samples", type=int, default=env_int("AGL_VERL_MAX_TRAIN_SAMPLES", 0))
    parser.add_argument("--max-val-samples", type=int, default=env_int("AGL_VERL_MAX_TEST_SAMPLES", 0))
    parser.add_argument("--ci", action="store_true", help="Run a small smoke training loop")
    args, config_overrides = parser.parse_known_args()
    return args, config_overrides


def main() -> None:
    args, config_overrides = parse_args()
    train(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        ci=args.ci,
    )


if __name__ == "__main__":
    main()

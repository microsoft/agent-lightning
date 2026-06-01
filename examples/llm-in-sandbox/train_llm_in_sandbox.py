#!/usr/bin/env python3
"""Train llm-in-sandbox with VERL through agl-lite."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.resources
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Any, cast

import httpx
import yaml

if TYPE_CHECKING:
    from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_TOKENS_PER_CALL = 20000


RL_TRAINING_CONFIG: dict[str, Any] = {
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
            "max_model_len": 65536,  # max context length for vllm
            "enforce_eager": True,  # check
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
            "ppo_mini_batch_size": 8, # full on-policy mini_batch_size = ppo_mini_batch_size * rollout.n
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            # "use_dynamic_bsz": True,
            # "ppo_max_token_len_per_gpu": 25000,
            # max_token_len_per_gpu for ppo forward and backward when use_dynamic_bsz is True
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
            "path": "Qwen/Qwen3-4B-Instruct-2507",
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
        "experiment_name": "train_llm_in_sandbox_sampled_data_no_shuffle",
        "nnodes": 1,
        "test_freq": 20,
        "save_freq": 20,
        "total_epochs": 15,
        "total_training_steps": 1000,
    },
    "agentlightning": {
        "is_shuffle": False,
        "trace_aggregator": {
            "level": "trajectory",
            "trajectory_max_prompt_length": 8000, # 10000,
            "trajectory_max_response_length": 12000, # 15000
        },
        "timeout_seconds": 1500,
        "poll_timeout_seconds": 1500,
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def configure_rollout_token_env(args: argparse.Namespace) -> None:
    trajectory_max_response_length = str(args.trajectory_max_response_length)
    os.environ["AGL_VERL_TRAJECTORY_MAX_RESPONSE_LENGTH"] = trajectory_max_response_length
    os.environ["MAX_TOKENS_PER_CALL"] = str(args.max_tokens_per_call)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def logger_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


async def register_resources(base_url: str, agl_key: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as health_client:
        response = await health_client.get(f"{base_url.rstrip('/')}/healthz")
        response.raise_for_status()

    from agl_lite.client import AglLiteClient

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)
    try:
        await client.list_models()
        with (EXAMPLE_DIR / "job-template.yaml").open(encoding="utf-8") as file:
            job_template = yaml.safe_load(file)
        resources = await client.add_resources({"job_template": job_template})
        return resources.resources_id
    finally:
        await client.close()


def build_verl_config(args: argparse.Namespace, *, resources_id: str, base_url: str, agl_key: str) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    config = copy.deepcopy(RL_TRAINING_CONFIG)
    experiment_name = args.experiment_name

    if args.ci_fast:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_name = f"llm_sandbox_ci_{timestamp}_{uuid.uuid4().hex[:8]}"
        args.total_steps = 1
        args.rollout_n = 1
        args.train_batch_size = 1
        args.minibsz = 1
        args.max_train_samples = max(args.max_train_samples, 1)
        args.max_test_samples = max(args.max_test_samples, 1)
        args.val_before_train = False
        args.n_gpus_per_node = 1
        args.save_freq = -1

    verl_pkg = importlib.resources.files("agl_lite.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    runtime_config = {
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        },
        "actor_rollout_ref": {
            "rollout": {
                "n": args.rollout_n,
                "temperature": args.temperature,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_model_parallel_size": args.tensor_model_parallel_size,
            },
            "actor": {
                "ppo_mini_batch_size": args.minibsz,
                "loss_agg_mode": args.loss_agg_mode,
            },
            "model": {"path": args.model},
        },
        "trainer": {
            "n_gpus_per_node": args.n_gpus_per_node,
            "val_before_train": args.val_before_train,
            "logger": logger_list(args.logger),
            "project_name": args.project_name,
            "experiment_name": experiment_name,
            "test_freq": args.test_freq,
            "save_freq": args.save_freq,
            "total_epochs": args.total_epochs,
            "total_training_steps": args.total_steps,
        },
        "agentlightning": {
            "agl_base_url": base_url,
            "agl_key": agl_key,
            "resources_id": resources_id,
            "timeout_seconds": args.rollout_timeout_seconds,
            "poll_timeout_seconds": args.rollout_timeout_seconds,
            "cleanup_agent_jobs": env_bool("AGL_CLEANUP_AGENT_JOBS", True),
            "cleanup_namespace": os.environ.get("AGL_NAMESPACE", "default"),
            "trace_aggregator": {
                "level": args.trace_level,
                "trajectory_max_prompt_length": args.trajectory_max_prompt_length,
                "trajectory_max_response_length": args.trajectory_max_response_length,
                "debug": False,
                "mismatch_log_dir": str(EXAMPLE_DIR / "mismatch_cases"),
            },
        },
    }
    if args.val_only:
        runtime_config["trainer"]["val_before_train"] = True
        runtime_config["trainer"]["val_only"] = True

    OmegaConf.set_struct(base_cfg, False)
    merged = cast(DictConfig, OmegaConf.merge(base_cfg, OmegaConf.create(runtime_config), OmegaConf.create(config)))
    OmegaConf.set_struct(merged, False)
    return merged


def parse_args() -> argparse.Namespace:
    data_config = RL_TRAINING_CONFIG["data"]
    trainer_config = RL_TRAINING_CONFIG["trainer"]
    actor_rollout_ref_config = RL_TRAINING_CONFIG["actor_rollout_ref"]
    rollout_config = actor_rollout_ref_config["rollout"]
    actor_config = actor_rollout_ref_config["actor"]
    model_config = actor_rollout_ref_config["model"]
    agentlightning_config = RL_TRAINING_CONFIG["agentlightning"]
    trace_config = agentlightning_config["trace_aggregator"]

    parser = argparse.ArgumentParser(description="Train llm-in-sandbox with VERL via agl-lite")
    parser.add_argument(
        "--train-data-dir",
        default=os.environ.get(
            "AGL_TRAIN_DATA_DIR",
            "examples/llm-in-sandbox/data/llm_sandbox_instruct_pretrain",
        ),
    )
    parser.add_argument(
        "--test-data-dir",
        default=os.environ.get(
            "AGL_TEST_DATA_DIR",
            "examples/llm-in-sandbox/data/llm_sandbox_math_mini,"
            "examples/llm-in-sandbox/data/llm_sandbox_chem_mini",
        ),
    )
    parser.add_argument("--model", default=os.environ.get("AGL_MODEL_NAME", str(model_config["path"])))
    parser.add_argument(
        "--project-name",
        default=os.environ.get("AGL_VERL_PROJECT_NAME", str(trainer_config["project_name"])),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("AGL_VERL_EXPERIMENT_NAME", str(trainer_config["experiment_name"])),
    )
    parser.add_argument(
        "--logger",
        default=os.environ.get("AGL_VERL_LOGGER", ",".join(trainer_config["logger"])),
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=env_int("AGL_VERL_ROLLOUT_N", int(rollout_config["n"])),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=env_int("AGL_VERL_TRAIN_BATCH_SIZE", int(data_config["train_batch_size"])),
    )
    parser.add_argument(
        "--minibsz",
        type=int,
        default=env_int("AGL_VERL_MINIBSZ", int(actor_config["ppo_mini_batch_size"])),
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=env_int("AGL_VERL_TOTAL_STEPS", int(trainer_config["total_training_steps"])),
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=env_int("AGL_VERL_TOTAL_EPOCHS", int(trainer_config["total_epochs"])),
    )
    parser.add_argument(
        "--test-freq",
        type=int,
        default=env_int("AGL_VERL_TEST_FREQ", int(trainer_config["test_freq"])),
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=env_int("AGL_VERL_SAVE_FREQ", int(trainer_config["save_freq"])),
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=env_int("AGL_VERL_N_GPUS_PER_NODE", int(trainer_config["n_gpus_per_node"])),
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=env_int(
            "AGL_VERL_TENSOR_MODEL_PARALLEL_SIZE",
            int(rollout_config["tensor_model_parallel_size"]),
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=env_float(
            "AGL_VERL_GPU_MEMORY_UTILIZATION",
            float(rollout_config["gpu_memory_utilization"]),
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=env_float("AGL_LLM_TEMPERATURE", float(rollout_config["temperature"])),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=env_int("AGL_VERL_MAX_PROMPT_LENGTH", int(data_config["max_prompt_length"])),
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=env_int("AGL_VERL_MAX_RESPONSE_LENGTH", int(data_config["max_response_length"])),
    )
    parser.add_argument(
        "--trajectory-max-prompt-length",
        type=int,
        default=env_int(
            "AGL_VERL_TRAJECTORY_MAX_PROMPT_LENGTH",
            int(trace_config["trajectory_max_prompt_length"]),
        ),
    )
    parser.add_argument(
        "--trajectory-max-response-length",
        type=int,
        default=env_int(
            "AGL_VERL_TRAJECTORY_MAX_RESPONSE_LENGTH",
            int(trace_config["trajectory_max_response_length"]),
        ),
    )
    parser.add_argument(
        "--max-tokens-per-call",
        type=int,
        default=env_int("MAX_TOKENS_PER_CALL", DEFAULT_MAX_TOKENS_PER_CALL),
        help="Maximum completion tokens per LLM API call; independent from trajectory aggregation length.",
    )
    parser.add_argument(
        "--rollout-timeout-seconds",
        type=int,
        default=env_int("AGL_ROLLOUT_TIMEOUT_SECONDS", int(agentlightning_config["timeout_seconds"])),
    )
    parser.add_argument("--max-train-samples", type=int, default=env_int("AGL_VERL_MAX_TRAIN_SAMPLES", 0))
    parser.add_argument(
        "--max-test-samples",
        dest="max_test_samples",
        type=int,
        default=env_int("AGL_VERL_MAX_TEST_SAMPLES", 0),
    )
    parser.add_argument(
        "--loss-agg-mode",
        default=os.environ.get("AGL_VERL_LOSS_AGG_MODE", str(actor_config["loss_agg_mode"])),
    )
    parser.add_argument(
        "--trace-level",
        choices=["transition", "trajectory", "trajectory-force"],
        default=os.environ.get("AGL_VERL_TRACE_LEVEL", str(trace_config["level"])),
    )
    parser.add_argument(
        "--val-before-train",
        action=argparse.BooleanOptionalAction,
        default=env_bool("AGL_VERL_VAL_BEFORE_TRAIN", bool(trainer_config["val_before_train"])),
    )
    parser.add_argument("--val-only", action="store_true", default=env_bool("AGL_VERL_VAL_ONLY", False))
    parser.add_argument("--ci-fast", action="store_true", help="single rollout smoke training path")
    return parser.parse_args()


def main() -> None:
    from omegaconf import OmegaConf

    from agl_lite.verl.entrypoint import run_ppo

    args = parse_args()
    configure_rollout_token_env(args)
    base_url = os.environ.get("AGL_BASE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY", "")
    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    train_data_dir = resolve_path(args.train_data_dir)
    train_dataset = load_json_dataset(
        train_data_dir / "train_verl.json",
        folder_name=train_data_dir.name,
        filename="train_verl.json",
        max_samples=args.max_train_samples,
    )
    test_datasets: list[tuple[str, list[dict[str, Any]]]] = []
    for test_data_dir_arg in path_list(args.test_data_dir):
        test_data_dir = resolve_path(test_data_dir_arg)
        test_folder_name = test_data_dir.name
        test_datasets.append(
            (
                test_folder_name,
                load_json_dataset(
                    test_data_dir / "test_verl.json",
                    folder_name=test_folder_name,
                    filename="test_verl.json",
                    max_samples=args.max_test_samples,
                ),
            )
        )
    test_dataset = [sample for _, dataset in test_datasets for sample in dataset]
    val_dataset = test_dataset

    log("=== Preflight ===")
    log(f"  agl-lite:     {base_url}")
    log(f"  model:        {args.model}")
    log(f"  max tokens:   {os.environ['MAX_TOKENS_PER_CALL']} per LLM call")
    log(f"  train:        {len(train_dataset)} samples from {train_data_dir.name}")
    for test_folder_name, dataset in test_datasets:
        log(f"  test:         {len(dataset)} samples from {test_folder_name}")
    log(f"  test total:   {len(test_dataset)} samples")
    resources_id = asyncio.run(register_resources(base_url, agl_key))
    log(f"  resources:    {resources_id}")

    config = build_verl_config(args, resources_id=resources_id, base_url=base_url, agl_key=agl_key)
    log("\n=== VERL config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()

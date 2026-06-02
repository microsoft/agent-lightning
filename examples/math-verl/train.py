#!/usr/bin/env python3
# ruff: noqa: E402
"""Train math task with VERL via agl-lite (math-verl example).

This script is the training E2E bridge:
- Reuses math-poc dataset + agent pod template
- Registers `job_template` resources in agl-lite
- Builds VERL config and calls `agl_lite.verl.entrypoint.run_ppo(...)`

Prerequisites:
- agl-lite deployed with examples/math-verl/.env.example
- VERL dependencies available in current Python env

Env vars:
- AGL_BASE_URL (default: http://localhost:8080)
- AGL_KEY (required)
- AGL_NAMESPACE (required by VERL cleanup)
- AGL_MODEL_NAME (default: Qwen/Qwen2.5-1.5B-Instruct)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import httpx
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Import from repo root when running as script
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agl_lite.client import AglLiteClient
from agl_lite.schemas import RolloutCreate
from agl_lite.schemas import RolloutState
from agl_lite.verl.entrypoint import run_ppo


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            items.append(json.loads(line))
    if not items:
        raise ValueError(f"Dataset is empty: {path}")
    return items


def split_dataset(items: list[dict[str, Any]], val_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0 or val_size >= len(items):
        raise ValueError(f"val_size must be in [1, {len(items)-1}], got {val_size}")
    train = items[:-val_size]
    val = items[-val_size:]
    return train, val


async def wait_rollout(client: AglLiteClient, rollout_id: str, timeout_s: int = 300) -> RolloutState:
    terminal = {RolloutState.SUCCEEDED, RolloutState.FAILED}
    start = time.time()
    while time.time() - start < timeout_s:
        r = await client.get_rollout(rollout_id)
        if r.status in terminal:
            return r.status
        await asyncio.sleep(3)
    raise TimeoutError(f"Rollout {rollout_id} did not finish within {timeout_s}s")


async def preflight_and_prepare_resources(
    *,
    base_url: str,
    agl_key: str,
    do_smoke_rollout: bool,
) -> str:
    """Run preflight checks and return resources_id for training."""
    # 1) healthz
    async with httpx.AsyncClient(timeout=10.0) as hc:
        r = await hc.get(f"{base_url.rstrip('/')}/healthz")
        r.raise_for_status()

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)
    try:
        # 2) auth check (any authenticated endpoint)
        _ = await client.list_models()

        # 3) add resources with math-poc job template
        template_path = REPO_ROOT / "examples" / "math-poc" / "job-template.yaml"
        with template_path.open() as f:
            job_template = yaml.safe_load(f)
        res = await client.add_resources({"job_template": job_template})
        resources_id = res.resources_id

        # 4) optional smoke rollout check (rollout completion + triplet extraction)
        if do_smoke_rollout:
            dataset = load_dataset(REPO_ROOT / "examples" / "math-poc" / "data" / "gsm8k_sample.jsonl")
            one = dataset[0]
            created = await client.enqueue_rollouts([
                RolloutCreate(resources_id=resources_id, input=one),
            ])
            rid = created[0].rollout_id
            status = await wait_rollout(client, rid)
            if status != RolloutState.SUCCEEDED:
                raise RuntimeError(f"Smoke rollout failed: {rid} status={status}")

            triplets = await client.get_events(rid, format="triplet")
            if not any(e.event_type == "model_request" for e in triplets):
                raise RuntimeError("Smoke rollout has no model_request in triplet events")
            if not any(e.event_type == "reward" for e in triplets):
                raise RuntimeError("Smoke rollout has no reward in triplet events")

        return resources_id
    finally:
        await client.close()


def build_verl_config(
    *,
    model_path: str,
    resources_id: str,
    agl_base_url: str,
    agl_key: str,
    rollout_n: int,
    total_steps: int,
    experiment_name: str,
) -> DictConfig:
    overrides = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "mode": "async",
                "tensor_model_parallel_size": 1,
                "n": rollout_n,
                "log_prob_micro_batch_size_per_gpu": 2,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 2,
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
                "log_prob_micro_batch_size_per_gpu": 2,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "agl-lite",
            "experiment_name": experiment_name,
            "nnodes": 1,
            "test_freq": max(total_steps, 1),
            "total_epochs": 1,
            "total_training_steps": total_steps,
        },
        # Keep user-facing namespace name for compatibility.
        "agentlightning": {
            "agl_base_url": agl_base_url,
            "agl_key": agl_key,
            "resources_id": resources_id,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 2048,
                "trajectory_max_response_length": 8192,
                "debug": False,
                "mismatch_log_dir": "./mismatch_cases",
            },
        },
    }

    verl_pkg = importlib.resources.files("agl_lite.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    OmegaConf.set_struct(base_cfg, False)
    return cast(DictConfig, OmegaConf.merge(base_cfg, OmegaConf.create(overrides)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train math-verl with VERL via agl-lite")
    parser.add_argument("--dataset", default="examples/math-poc/data/gsm8k_sample.jsonl")
    parser.add_argument("--val-size", type=int, default=5)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=1)
    parser.add_argument("--experiment-name", default="math_poc_verl")
    parser.add_argument(
        "--smoke-rollout-check",
        action="store_true",
        help="run one rollout before training to validate completion + triplet extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_url = os.environ.get("AGL_BASE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY", "")
    model_name = os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    dataset_path = (REPO_ROOT / args.dataset).resolve()
    items = load_dataset(dataset_path)
    train_dataset, val_dataset = split_dataset(items, args.val_size)

    log("=== Preflight ===")
    log(f"  agl-lite:   {base_url}")
    log("  mode:       vllm (training only)")
    log(f"  model:      {model_name}")
    log(f"  dataset:    train={len(train_dataset)}, val={len(val_dataset)}")

    resources_id = asyncio.run(
        preflight_and_prepare_resources(
            base_url=base_url,
            agl_key=agl_key,
            do_smoke_rollout=args.smoke_rollout_check,
        )
    )
    log(f"  resources:  {resources_id}")

    config = build_verl_config(
        model_path=model_name,
        resources_id=resources_id,
        agl_base_url=base_url,
        agl_key=agl_key,
        rollout_n=args.rollout_n,
        total_steps=args.total_steps,
        experiment_name=args.experiment_name,
    )

    log("\n=== Start VERL training ===")
    run_ppo(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


if __name__ == "__main__":
    main()

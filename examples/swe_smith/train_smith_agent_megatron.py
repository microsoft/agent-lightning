#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.


from __future__ import annotations

from pprint import pprint
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from train_smith_agent import (  # noqa: E402 — sibling module, run from example dir
    DEFAULT_MODEL,
    EXAMPLE_DIR,
    load_split_file,
    log,
)

import importlib.resources  # noqa: E402

CHAT_TEMPLATE_PATH = str(EXAMPLE_DIR / "swe_smith_chat_template.jinja")
TRAIN_BACKEND = "megatron"

def verl_megatron_config() -> dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 32768,
            "max_response_length": 32768,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "mode": "async",
                "name": "vllm",
                "tensor_model_parallel_size": 2,
                "n": 8,
                "gpu_memory_utilization": 0.7,
                "max_model_len": 32768,
                "enforce_eager": True,

                "enable_rollout_routing_replay": True,

                "calculate_log_probs": True,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_use_dynamic_bsz": False,
                "multi_turn": {"format": "hermes"},
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": CHAT_TEMPLATE_PATH,
                        "moe_backend": "triton",
                    }
                },
                "temperature": 1,
                "val_kwargs": {"temperature": 0, "do_sample": False},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": False,
            },
            "actor": {
                "strategy": "megatron",
                "model_engine": "megatron",
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_dynamic_bsz": False,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "loss_agg_mode": "seq-mean-token-sum",

                "megatron": {
                    "pipeline_model_parallel_size": 1,
                    "tensor_model_parallel_size": 2,
                    "expert_model_parallel_size": 4,
                    "expert_tensor_parallel_size": 1,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "grad_offload": True,
                    "use_mbridge": True,

                    "router_replay": {"mode": "R3"},

                    "override_transformer_config": {
                        "moe_enable_deepep": True,
                        "moe_token_dispatcher_type": "flex",
                        "apply_rope_fusion": True,
                        "bias_activation_fusion": True,
                        "moe_router_dtype": "fp32",
                        "recompute_method": "uniform",
                        "recompute_granularity": "full",
                        "recompute_num_layers": 1,
                        "gradient_accumulation_fusion": True,
                        "moe_permute_fusion": False,
                    },
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_use_dynamic_bsz": False,
                "megatron": {
                    "pipeline_model_parallel_size": 1,
                    "tensor_model_parallel_size": 2,
                    "expert_model_parallel_size": 4,
                    "expert_tensor_parallel_size": 1,
                    "param_offload": True,
                },
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
            "nnodes": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "balance_batch": False,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "swe_smith_megatron_r3",
            "nccl_timeout": 1800,
            "test_freq": 8,
            "save_freq": 32,
            "total_epochs": 2,
            "total_training_steps": 1000,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "rollout_timeout_seconds": 5400,
            "reward_fillna_value": 0.0,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 24000,
                "trajectory_max_response_length": 24000,
            },
            "async_rollout": {
                "enabled": True,
                "async_train_batch_size": 48,
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
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_megatron_config()
    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key

    rollout_mode = overrides["actor_rollout_ref"]["rollout"]["mode"]
    model_path = overrides["actor_rollout_ref"]["model"]["path"]
    overrides["trainer"]["experiment_name"] = (
        f"swe_smith_{rollout_mode}_{model_path.split('/')[-1]}_{TRAIN_BACKEND}"
    )
    if run_name:
        overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'

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

    log("=== Preflight (Megatron + R3) ===")
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

    log("\n=== Start VERL training (Megatron actor, R3 router replay, vLLM rollout) ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a SWE-smith agent with VERL/GRPO via Agent Lightning (Megatron + R3)"
    )
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
    return parser.parse_known_args()

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

#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import importlib.resources
from collections.abc import Sequence
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from agentlightning.verl.per_rollout_loss import CISPO_PER_ROLLOUT_MEAN_LOSS_MODE

try:
    from .train_smith_agent import EXAMPLE_DIR, load_split_file
except ImportError:  # Direct script execution.
    from train_smith_agent import EXAMPLE_DIR, load_split_file

MODEL = "Qwen/Qwen3.5-35B-A3B"


def verl_megatron_cispo_config() -> dict[str, Any]:
    """Return Qwen3.5 MoE defaults for Megatron, CISPO, and R3."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "enable_rollout_level_advantage": True,
            "enable_per_rollout_mean_loss": True,
            "enable_rollout_level_advantage_scale": False,
            "rollout_correction": {
                "bypass_mode": False,
                "loss_type": "ppo_clip",
                "rollout_is": "token",
                "rollout_is_threshold": 2.0,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            },
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
                "name": "vllm",
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "expert_parallel_size": 1,
                "n": 8,
                "gpu_memory_utilization": 0.8,
                "max_model_len": 81920,
                "max_num_batched_tokens": 8192,
                "enforce_eager": False,
                "enable_rollout_routing_replay": True,
                "calculate_log_probs": True,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_use_dynamic_bsz": False,
                "multi_turn": {"format": "hermes"},
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
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
                "strategy": "megatron",
                "policy_loss": {"loss_mode": CISPO_PER_ROLLOUT_MEAN_LOSS_MODE},
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 16384,
                "use_dynamic_bsz": False,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 10.0,
                "clip_ratio_high": 0.2,
                "loss_agg_mode": "token-mean",
                "megatron": {
                    "pipeline_model_parallel_size": 1,
                    "tensor_model_parallel_size": 4,
                    "expert_model_parallel_size": 4,
                    "expert_tensor_parallel_size": 1,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "grad_offload": True,
                    "use_mbridge": True,
                    "use_dist_checkpointing": True,
                    "router_replay": {"mode": "R3"},
                    "override_transformer_config": {
                        "moe_enable_deepep": True,
                        "moe_token_dispatcher_type": "flex",
                        "moe_shared_expert_overlap": False,
                        "apply_rope_fusion": False,
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
                    "tensor_model_parallel_size": 4,
                    "expert_model_parallel_size": 4,
                    "expert_tensor_parallel_size": 1,
                    "param_offload": True,
                    "override_transformer_config": {"apply_rope_fusion": False},
                },
            },
            "model": {
                "path": MODEL,
                "use_remove_padding": False,
                "use_fused_kernels": False,
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
            "experiment_name": "swe_smith_megatron_cispo_r3",
            "nccl_timeout": 1800,
            "test_freq": 32,
            "save_freq": 32,
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
                "trajectory_max_total_length": 81920,
            },
            "async_rollout": {"enabled": True, "async_train_batch_size": 24},
            "k8s": {"job_template_path": str(EXAMPLE_DIR / "job-template-openai.yaml")},
        },
    }


def build_config(
    *,
    model: str = MODEL,
    agl_base_url: str = "http://localhost:8080",
    agl_key: str = "",
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        config = compose(config_name="config", overrides=["model_engine=megatron"])

    overrides = verl_megatron_cispo_config()
    overrides["actor_rollout_ref"]["model"]["path"] = model
    overrides["agentlightning"]["agl_base_url"] = agl_base_url
    overrides["agentlightning"]["agl_key"] = agl_key
    suffix = f"_{run_name}" if run_name else ""
    overrides["trainer"]["experiment_name"] = f"swe_smith_qwen35_35b_a3b_megatron_cispo_r3{suffix}"

    OmegaConf.set_struct(config, False)
    merged = OmegaConf.merge(config, overrides, OmegaConf.from_dotlist(list(config_overrides)))
    OmegaConf.set_struct(merged, False)
    return cast(DictConfig, merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3.5-35B-A3B with Megatron, CISPO, and R3")
    parser.add_argument("--train-dataset-path", default=str(EXAMPLE_DIR / "train_dataset_mixed.jsonl"))
    parser.add_argument("--val-dataset-path", default=str(EXAMPLE_DIR / "val_dataset_filtered.jsonl"))
    parser.add_argument("--max-val-instances", type=int)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--agl-base-url", default="http://localhost:8080")
    parser.add_argument("--agl-key", default="")
    parser.add_argument("--run-name")
    args, overrides = parser.parse_known_args()

    train_dataset = load_split_file(args.train_dataset_path)
    val_dataset = load_split_file(args.val_dataset_path, max_instances=args.max_val_instances)
    config = build_config(
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=overrides,
    )

    from agentlightning.verl.entrypoint import run_ppo

    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()

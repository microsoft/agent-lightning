#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

from omegaconf import DictConfig, OmegaConf
from train_smith_agent import EXAMPLE_DIR, load_split_file
from train_smith_agent import build_config as build_fsdp_config

MODEL = "Qwen/Qwen3.5-35B-A3B"


def build_config(
    *,
    model: str = MODEL,
    agl_base_url: str = "http://localhost:8080",
    agl_key: str = "",
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    config = build_fsdp_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        config_overrides=(),
    )

    config.algorithm.update(
        {
            "enable_rollout_level_advantage": True,
            "enable_per_rollout_mean_loss": True,
            "enable_cispo_loss": False,
            "enable_rollout_level_advantage_scale": False,
        }
    )
    config.actor_rollout_ref.rollout.update(
        {
            "enable_rollout_routing_replay": True,
            "calculate_log_probs": True,
            "log_prob_use_dynamic_bsz": False,
            "max_model_len": 81920,
        }
    )

    actor = config.actor_rollout_ref.actor
    actor.pop("fsdp_config", None)
    actor.update(
        {
            "strategy": "megatron",
            "model_engine": "megatron",
            "use_dynamic_bsz": False,
            "loss_agg_mode": "seq-mean-token-sum",
            "policy_loss": {"loss_mode": "per_rollout_mean"},
            "megatron": {
                "pipeline_model_parallel_size": 1,
                "tensor_model_parallel_size": 1,
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
                    "moe_router_dtype": "fp32",
                    "recompute_method": "uniform",
                    "recompute_granularity": "full",
                    "recompute_num_layers": 1,
                    "moe_permute_fusion": False,
                },
            },
        }
    )

    ref = config.actor_rollout_ref.ref
    ref.pop("fsdp_config", None)
    ref.update(
        {
            "log_prob_use_dynamic_bsz": False,
            "megatron": {
                "pipeline_model_parallel_size": 1,
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": 4,
                "expert_tensor_parallel_size": 1,
                "param_offload": True,
            },
        }
    )

    config.trainer.update(
        {
            "val_before_train": False,
            "balance_batch": True,
            "experiment_name": f"swe_smith_qwen35_35b_a3b_megatron_r3{f'_{run_name}' if run_name else ''}",
        }
    )
    config.agentlightning.trace_aggregator.update(
        {
            "level": "trajectory",
            "trajectory_max_prompt_length": 24576,
            "trajectory_max_response_length": 81920,
            "trajectory_max_total_length": 81920,
        }
    )
    config.agentlightning.async_rollout.update({"enabled": False, "async_train_batch_size": None})
    return cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(list(config_overrides))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3.5-35B-A3B with Megatron and R3")
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

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for Search-R1 training.

    These are merged on top of Agent Lightning's base config
    (agentlightning/verl/config.yaml -> verl/trainer/config/ppo_trainer.yaml).
    """
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 512,
            "max_prompt_length": 6000,
            "max_response_length": 4096,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "llama3_json"},
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
                "max_model_len": 32768,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "llama3_json",
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": 256,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6, "lr_warmup_steps_ratio": 0},
                "use_kl_loss": True,
                "kl_loss_type": "low_var_kl",
                "kl_loss_coef": 0.001,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 8,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "search_r1",
            "nnodes": 1,
            "test_freq": 10,
            "save_freq": 10,
            "total_epochs": 15,
            "total_training_steps": 300,
            "default_local_dir": "checkpoints/search_r1_checkpoints/",
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "search-r1-dev-key",
            "rollout_timeout_seconds": 1800,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 4096,
                "trajectory_max_response_length": 34384,
            },
            "local": {
                "agent_class": "examples.search_r1.agents.search_r1_agent:SearchR1Agent",
                "env_map": {
                    "QUESTION": "input.question",
                    "GOLDEN_ANSWERS": "input.golden_answers",
                },
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    api_type: str = "chat",
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
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
    if api_type not in {"chat", "completion"}:
        raise ValueError(f"Unsupported Search-R1 OpenAI API type: {api_type}")
    if api_type == "completion":
        overrides["agentlightning"]["local"]["agent_class"] = (
            "examples.search_r1.agents.search_r1_agent:SearchR1CompletionAgent"
        )
        overrides["agentlightning"]["local"]["env_map"]["SEARCH_R1_TOKENIZER_MODEL"] = overrides["actor_rollout_ref"][
            "model"
        ]["path"]
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key
    if run_name:
        overrides["trainer"]["experiment_name"] = f"{overrides['trainer']['experiment_name']}_{run_name}"

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    return OmegaConf.merge(base_cfg, override_conf, cli_override_conf)


def train(
    *,
    train_file: str,
    val_file: str,
    model: str | None = None,
    api_type: str = "chat",
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> None:
    """Load datasets, build config, and launch VERL training via Agent Lightning."""
    from agentlightning.verl.entrypoint import run_ppo

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

    config = build_config(
        model=model,
        api_type=api_type,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
    )

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Search-R1 agent with VERL on Agent Lightning.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="examples/search_r1/data/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="examples/search_r1/data/test.parquet",
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"HF model id or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-type",
        choices=("chat", "completion"),
        default="chat",
        help="OpenAI-compatible API used by the local rollout agent",
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
        default="search-r1-dev-key",
        help="Agent Lightning API key for the trainer",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Suffix appended to trainer.experiment_name",
    )
    args, config_overrides = parser.parse_known_args()

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        api_type=args.api_type,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()

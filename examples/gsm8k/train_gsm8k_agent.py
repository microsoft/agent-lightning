# Copyright (c) Microsoft. All rights reserved.

import argparse
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def extract_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+)$", str(answer), re.DOTALL)
    return match.group(1).strip() if match else str(answer).strip()


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for GSM8K local training."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": "Qwen/Qwen2.5-1.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "gsm8k",
            "nnodes": 1,
            "save_freq": -1,
            "test_freq": 10,
            "total_epochs": 2,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8181",
            "agl_key": "gsm8k-dev-key",
            "rollout_timeout_seconds": 300,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 1024,
                "trajectory_max_response_length": 1024,
            },
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 64,
            },
            "local": {
                "agent_class": "examples.gsm8k.gsm8k_agent.ChatAgent",
                "env_map": {
                    "QUESTION": "input.question",
                    "ANSWER": "input.answer",
                },
            },
        },
    }


def build_config(
    *,
    model: str | None = None,
    api: str = "chat",
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
    if api not in {"chat", "completion"}:
        raise ValueError(f"Unsupported GSM8K OpenAI API: {api}")
    if api == "completion":
        overrides["agentlightning"]["local"]["agent_class"] = "examples.gsm8k.gsm8k_agent.CompletionAgent"
        overrides["agentlightning"]["local"]["env_map"]["GSM8K_MODEL"] = overrides["actor_rollout_ref"]["model"]["path"]
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key
    if run_name:
        overrides["trainer"]["experiment_name"] = f"{overrides['trainer']['experiment_name']}_{run_name}"

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    return config


def load_gsm8k_dataset(path: str) -> list[dict[str, str]]:
    dataset = cast(
        Sequence[dict[str, Any]],
        HuggingFaceDataset.from_parquet(path).to_list(),  # type: ignore
    )
    return [
        {
            "question": str(item["question"]),
            "answer": extract_gsm8k_answer(str(item["answer"])),
        }
        for item in dataset
    ]


def train(
    *,
    train_file: str,
    val_file: str,
    val_size: int,
    seed: int,
    model: str | None = None,
    api: str = "chat",
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> None:
    """Load GSM8K datasets, build config, and launch VERL training via Agent Lightning."""
    from agentlightning.verl.entrypoint import run_ppo

    train_dataset = load_gsm8k_dataset(train_file)
    val_dataset = load_gsm8k_dataset(val_file)
    if val_size > 0 and val_size < len(val_dataset):
        rng = random.Random(seed)
        val_dataset = rng.sample(val_dataset, val_size)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")

    config = build_config(
        model=model,
        api=api,
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
    data_dir = Path.home() / "dataset" / "gsm8k" / "main"
    parser = argparse.ArgumentParser(
        description="Train GSM8K agent with VERL on Agent Lightning local mode.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(data_dir / "train-00000-of-00001.parquet"),
        help="Path to GSM8K main training parquet file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=str(data_dir / "test-00000-of-00001.parquet"),
        help="Path to GSM8K main test parquet file",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of random GSM8K test samples to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation sampling",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--api",
        choices=("chat", "completion"),
        default="chat",
        help="OpenAI-compatible API used by the local GSM8K agent",
    )
    parser.add_argument(
        "--agl-base-url",
        type=str,
        default="http://localhost:8181",
        help="Agent Lightning server URL for the trainer",
    )
    parser.add_argument(
        "--agl-key",
        type=str,
        default="gsm8k-dev-key",
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
        val_size=args.val_size,
        seed=args.seed,
        model=args.model,
        api=args.api,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()

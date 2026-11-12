# Copyright (c) Microsoft. All rights reserved.

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def verl_default_config(val_only: bool = False) -> dict[str, Any]:
    """VERL config overrides for Calc-X training.

    These are merged on top of Agent Lightning's base config
    (agentlightning/verl/config.yaml → verl/trainer/config/ppo_trainer.yaml).

    With ``val_only`` the run scores a saved checkpoint instead of training one.
    VERL only takes that exit from the validation pass it runs before the first
    step, so ``val_before_train`` has to be on as well.
    """
    example_dir = Path(__file__).resolve().parent
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 4,
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
            "val_before_train": val_only,
            "val_only": val_only,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agentlightning",
            "experiment_name": "calc_x",
            "nnodes": 1,
            "save_freq": 64,
            "test_freq": 10,
            "total_epochs": 2,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8181",
            "agl_key": "calcx-dev-key",
            "rollout_timeout_seconds": 300,
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 64,
            },
            "local": {
                "agent_class": "examples.calc_x.calc_agent.Agent",
                "env_map": {
                    "QUESTION": "input.question",
                    "RESULT": "input.result",
                },
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
    async_mode: bool = False,
    val_only: bool = False,
    resume_path: str | None = None,
    config_overrides: Sequence[str] = (),
) -> Any:
    """Build the full OmegaConf config by merging base + overrides.

    Uses Hydra compose to load Agent Lightning's base config (which includes
    verl's ppo_trainer defaults), then merges Calc-X overrides on top.
    """
    import importlib.resources

    # Locate the agentlightning/verl package directory for Hydra.
    verl_pkg = importlib.resources.files("agentlightning.verl")
    config_dir = str(verl_pkg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config(val_only=val_only)

    if resume_path:
        # A checkpoint is restored through trainer.resume_from_path, not by pointing
        # actor_rollout_ref.model.path at it. That key is the base model, and the gateway
        # requires its own model name to match it exactly, so aiming it at a checkpoint
        # makes the proxy answer "model not found"
        # (docs/25-api-gateway-configuration.md).
        overrides["trainer"]["resume_mode"] = "resume_path"
        overrides["trainer"]["resume_from_path"] = resume_path

    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key
    name_parts = [overrides["trainer"]["experiment_name"]]
    if async_mode:
        overrides["agentlightning"]["async_rollout"]["enabled"] = True
        overrides["agentlightning"]["async_rollout"]["async_train_batch_size"] = (
            overrides["data"]["train_batch_size"] * 2
        )
        name_parts.append("async")
    if val_only:
        name_parts.append("eval")
    if run_name:
        name_parts.append(run_name)
    overrides["trainer"]["experiment_name"] = "_".join(name_parts)

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)

    # The CLI overrides merge last, so any of them can flip these back. In this mode the
    # held-out split is also what run_ppo sizes its sampler from, so a val_before_train of
    # false starts optimizing on it: the run would train on the data it is supposed to be
    # scored on, with nothing in the logs saying so. Re-assert after the merge rather than
    # trusting the keys to survive it.
    if val_only:
        config.trainer.val_only = True
        config.trainer.val_before_train = True
        config.trainer.resume_mode = "resume_path"
        if resume_path:
            config.trainer.resume_from_path = resume_path

    return config


def load_split(path: str) -> Sequence[Any]:
    """Read one parquet split into a list of rows."""
    return cast(
        Sequence[Any],
        HuggingFaceDataset.from_parquet(path).to_list(),  # type: ignore
    )


def train(
    *,
    train_file: str,
    val_file: str,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    async_mode: bool = False,
    config_overrides: Sequence[str] = (),
) -> None:
    """Load datasets, build config, and launch VERL training via Agent Lightning."""
    from agentlightning.verl.entrypoint import run_ppo

    train_dataset = load_split(train_file)
    val_dataset = load_split(val_file)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        async_mode=async_mode,
        config_overrides=config_overrides,
    )

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def evaluate(
    *,
    checkpoint: str,
    data_file: str,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> None:
    """Score a saved checkpoint on a held-out split, without training it.

    The checkpoint goes through ``trainer.resume_from_path``; ``--model`` stays what it
    already was, the base model both the trainer and the gateway are configured with.

    VERL runs one validation pass and returns. ``run_ppo`` still wants a non-empty train
    split, so the held-out split is handed in as both: in this mode it only sizes the
    sampler and no optimizer step is taken. That also means the split has to be at least
    ``data.train_batch_size`` rows long (32 by default), or the sampler has nothing to
    build a batch from.
    """
    dataset = load_split(data_file)
    print(f"Held-out dataset: {len(dataset)} samples")

    config = build_config(
        model=model,
        resume_path=checkpoint,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        val_only=True,
        config_overrides=config_overrides,
    )

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    from agentlightning.verl.entrypoint import run_ppo

    run_ppo(config, train_dataset=dataset, val_dataset=dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Calc-X agent with VERL on Agent Lightning.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/test.parquet",
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (default: Qwen/Qwen2.5-1.5B-Instruct)",
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
        default="calcx-dev-key",
        help="Agent Lightning API key for the trainer",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Suffix appended to trainer.experiment_name",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Enable async rollout with async batch size set to 2x train_batch_size",
    )
    parser.add_argument(
        "--resume",
        dest="resume_path",
        type=str,
        default=None,
        help="Path to a saved checkpoint (the run's global_step_N folder) to score",
    )
    parser.add_argument(
        "--val-only",
        dest="val_only",
        action="store_true",
        help="Score --resume on --val-file once, without training",
    )
    args, config_overrides = parser.parse_known_args()

    if args.val_only:
        if not args.resume_path:
            parser.error("--val-only needs --resume to point at a saved checkpoint")
        evaluate(
            checkpoint=args.resume_path,
            data_file=args.val_file,
            model=args.model,
            agl_base_url=args.agl_base_url,
            agl_key=args.agl_key,
            run_name=args.run_name,
            config_overrides=config_overrides,
        )
        return

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        async_mode=args.async_mode,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()

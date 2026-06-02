import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for Calc-X training.

    These are merged on top of agl-lite's base config
    (agl_lite/verl/config.yaml → verl/trainer/config/ppo_trainer.yaml).
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
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "agl-lite",
            "experiment_name": "calc_x",
            "nnodes": 1,
            "save_freq": 64,
            "test_freq": 32,
            "total_epochs": 2,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "calcx-dev-key",
            "timeout_seconds": 300,
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
    config_overrides: Sequence[str] = (),
    ci: bool = False,
) -> Any:
    """Build the full OmegaConf config by merging base + overrides.

    Uses Hydra compose to load agl-lite's base config (which includes
    verl's ppo_trainer defaults), then merges Calc-X overrides on top.
    """
    import importlib.resources

    # Locate the agl_lite/verl package directory for Hydra.
    verl_pkg = importlib.resources.files("agl_lite.verl")
    config_dir = str(verl_pkg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
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
        overrides["trainer"]["project_name"] = "agl-lite-CI"
        overrides["trainer"]["experiment_name"] = "calc_x_ci"
        if run_name:
            overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'
        overrides["trainer"]["total_epochs"] = 1
        overrides["trainer"]["total_training_steps"] = 5
        overrides["trainer"]["test_freq"] = -1
        overrides["trainer"].pop("save_freq", None)
        overrides["data"]["train_batch_size"] = 2
        overrides["data"]["max_prompt_length"] = 2048
        overrides["data"]["max_response_length"] = 512
        overrides["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
        overrides["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
        overrides["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
        overrides["actor_rollout_ref"]["rollout"]["n"] = 2
        overrides["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
        overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    return config


def train(
    *,
    train_file: str,
    val_file: str,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    ci: bool = False,
) -> None:
    """Load datasets, build config, and launch VERL training via agl-lite."""
    from agl_lite.verl.entrypoint import run_ppo

    # Load datasets.
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
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
        ci=ci,
    )

    from pprint import pprint

    print("\n=== VERL Config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Calc-X agent with VERL on agl-lite.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="examples/calc_x/data/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="examples/calc_x/data/test.parquet",
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--agl-base-url",
        type=str,
        default="http://localhost:8080",
        help="agl-lite server URL for the trainer",
    )
    parser.add_argument(
        "--agl-key",
        type=str,
        default="calcx-dev-key",
        help="agl-lite API key for the trainer",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Suffix appended to trainer.experiment_name",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run a 5-step CI-style training loop",
    )
    args, config_overrides = parser.parse_known_args()

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
        ci=args.ci,
    )


if __name__ == "__main__":
    main()

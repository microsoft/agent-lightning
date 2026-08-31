# Copyright (c) Microsoft. All rights reserved.

"""Train a VLM on synthetic single-image QA with VERL via Agent Lightning.

The dataset is fully synthetic (no downloads): each sample is a 256x256 image
with 1-5 non-overlapping red circles drawn at random positions, and the
question is "How many red circles are in the image?".
"""

import argparse
import base64
import io
import random
from collections.abc import Sequence
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

QUESTION = "How many red circles are in the image? Answer with a single integer."
IMAGE_SIZE = 256


def render_circle_image(n_circles: int, rng: random.Random) -> str:
    """Draw n_circles non-overlapping red circles and return a data: URL."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)
    centers: list[tuple[int, int, int]] = []
    for _ in range(n_circles):
        for _attempt in range(100):
            radius = rng.randint(15, 30)
            x = rng.randint(radius, IMAGE_SIZE - radius)
            y = rng.randint(radius, IMAGE_SIZE - radius)
            if all((x - cx) ** 2 + (y - cy) ** 2 >= (radius + cr + 4) ** 2 for cx, cy, cr in centers):
                centers.append((x, y, radius))
                break
        else:
            raise RuntimeError("Failed to place a non-overlapping circle; retry with another seed.")
    for x, y, radius in centers:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def generate_dataset(n_samples: int, seed: int) -> list[dict[str, str]]:
    """Generate synthetic (image, question, answer) samples."""
    rng = random.Random(seed)
    dataset = []
    for _ in range(n_samples):
        n_circles = rng.randint(1, 5)
        dataset.append(
            {
                "image": render_circle_image(n_circles, rng),
                "question": QUESTION,
                "answer": str(n_circles),
            }
        )
    return dataset


def verl_default_config() -> dict[str, Any]:
    """VERL config overrides for multimodal QA local training."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 4096,
            "max_response_length": 256,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                # vLLM < 0.22.0: the prefix cache can desync from the multimodal
                # receiver cache across sleep/wake cycles (`AssertionError:
                # Expected a cached item for mm_hash=...`, or a silent rollout
                # hang). Keep it off for multimodal training.
                # See https://github.com/vllm-project/vllm/issues/42995
                "enable_prefix_caching": False,
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
                "path": "Qwen/Qwen3.5-2B",
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
            "experiment_name": "multimodal_qa",
            "nnodes": 1,
            "save_freq": -1,
            "test_freq": 10,
            "total_epochs": 2,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8181",
            "agl_key": "multimodal-qa-dev-key",
            "rollout_timeout_seconds": 300,
            "trace_aggregator": {
                # Multimodal training rows require transition-level aggregation;
                # trajectory-level merging breaks the image-to-token alignment.
                "level": "transition",
            },
            "async_rollout": {
                "enabled": False,
                "async_train_batch_size": 64,
            },
            "local": {
                "agent_class": "examples.multimodal_qa.multimodal_qa_agent.MultimodalQAAgent",
                "env_map": {
                    "IMAGE": "input.image",
                    "QUESTION": "input.question",
                    "ANSWER": "input.answer",
                },
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


def train(
    *,
    train_size: int,
    val_size: int,
    seed: int,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> None:
    """Generate the synthetic dataset, build config, and launch VERL training."""
    from agentlightning.verl.entrypoint import run_ppo

    train_dataset = generate_dataset(train_size, seed)
    val_dataset = generate_dataset(val_size, seed + 1)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")

    config = build_config(
        model=model,
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
        description="Train a VLM on synthetic single-image QA with VERL on Agent Lightning local mode.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=64,
        help="Number of synthetic training samples",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=16,
        help="Number of synthetic validation samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or path (default: Qwen/Qwen3.5-2B)",
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
        default="multimodal-qa-dev-key",
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
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()

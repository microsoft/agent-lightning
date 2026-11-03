# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {
    "grpo": {"centering": False, "is_mode": "none"},
    "grpo_mis": {"centering": False, "is_mode": "mis"},
    "sc": {"centering": True, "is_mode": "none"},
    "sc_mis": {"centering": True, "is_mode": "mis"},
    "sc_tis": {"centering": True, "is_mode": "tis"},
}


def build_config(baseline="grpo", *, smoke=False, eval_only=False, overrides=(), config_file=None):
    with initialize_config_dir(config_dir=str(ROOT / "agentlightning/verl"), version_base=None):
        base = compose(config_name="config")
    OmegaConf.set_struct(base, False)
    config = OmegaConf.merge(base, OmegaConf.load(Path(__file__).with_name("config.yaml")))
    config.agentlightning.score_centering = OmegaConf.merge(config.agentlightning.score_centering, BASELINES[baseline])
    if baseline == "grpo":
        config.actor_rollout_ref.model.use_remove_padding = False
        config.actor_rollout_ref.rollout.quantization = None
        config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype = "auto"
        actor = config.actor_rollout_ref.actor
        actor.policy_loss.loss_mode = "per_rollout_mean"
        actor.clip_ratio_high = 0.3
        actor.kl_loss_coef = 0.0
        actor.checkpoint = base.actor_rollout_ref.actor.checkpoint
        actor.fsdp_config.fsdp_size = -1
        del actor.fsdp_config.mixed_precision
        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 8
        config.algorithm.enable_per_rollout_mean_loss = True
        config.algorithm.rollout_correction = OmegaConf.merge(
            base.algorithm.rollout_correction, {"rollout_is": None, "rollout_rs": None, "bypass_mode": False}
        )
        del config.actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls
        del config.actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs
    config.trainer.experiment_name = f"intellect2_math-{baseline}"
    if config_file is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    if smoke:
        config.data.train_batch_size = 2
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
        config.trainer.total_training_steps = 3
        config.trainer.save_freq = -1
    if eval_only:
        config.actor_rollout_ref.rollout.quantization = None
        config.trainer.val_before_train = True
        config.trainer.val_only = True
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    return config


def main():
    parser = argparse.ArgumentParser(description="Train GRPO and score centering on INTELLECT-2 math")
    parser.add_argument("--baseline", choices=BASELINES, default="grpo")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate SC_MODEL in BF16 on the INTELLECT-2 validation split"
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--train-file", default=str(ROOT / "examples/intellect2_math/data/train.parquet"))
    parser.add_argument("--val-file", default=str(ROOT / "examples/intellect2_math/data/test.parquet"))
    args, overrides = parser.parse_known_args()
    config = build_config(
        args.baseline, smoke=args.smoke, eval_only=args.eval_only, overrides=overrides, config_file=args.config
    )
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    import ray
    from datasets import Dataset

    from agentlightning.verl.entrypoint import run_ppo

    train = Dataset.from_parquet(args.train_file).to_list()
    val = Dataset.from_parquet(args.val_file).to_list()
    try:
        run_ppo(config, train_dataset=train, val_dataset=val)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

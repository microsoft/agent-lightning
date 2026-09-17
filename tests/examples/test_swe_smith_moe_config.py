# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from omegaconf import OmegaConf

from examples.swe_smith.train_smith_agent_moe import build_config


def test_moe_config_selects_megatron_cispo_and_r3() -> None:
    config = build_config()
    actor = config.actor_rollout_ref.actor
    reference = config.actor_rollout_ref.ref
    rollout = config.actor_rollout_ref.rollout

    assert config.model_engine == "megatron"
    for component in (actor, reference):
        assert component.strategy == "megatron"
        assert component._target_ == "verl.workers.config.McoreActorConfig"
        assert component.megatron.tensor_model_parallel_size == 4
    assert actor.optim._target_ == "verl.workers.config.McoreOptimizerConfig"
    assert actor.policy_loss.loss_mode == "cispo_per_rollout_mean"
    assert actor.clip_ratio_low == 10.0
    assert actor.clip_ratio_high == 0.2
    assert actor.loss_agg_mode == "token-mean"
    assert actor.megatron.router_replay.mode == "R3"
    assert rollout.enable_rollout_routing_replay is True
    assert config.algorithm.enable_per_rollout_mean_loss is True
    assert config.algorithm.enable_rollout_level_advantage_scale is False
    assert config.agentlightning.trace_aggregator.level == "trajectory"
    assert config.agentlightning.async_rollout.enabled is True
    OmegaConf.to_container(config, resolve=True)


def test_moe_config_applies_cli_overrides() -> None:
    config = build_config(
        model="local/model",
        run_name="smoke",
        config_overrides=["trainer.total_training_steps=1"],
    )

    assert config.actor_rollout_ref.model.path == "local/model"
    assert config.trainer.total_training_steps == 1
    assert config.trainer.experiment_name.endswith("_megatron_cispo_r3_smoke")

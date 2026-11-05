# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import copy

import pytest

from agentlightning.verl.process_credit import validate_process_config


def supported_config():
    return {
        "agentlightning": {
            "process_credit": {"enabled": True},
            "async_rollout": {"enabled": False},
            "max_ppo_update_times": None,
            "trace_aggregator": {"level": "transition"},
        },
        "algorithm": {
            "adv_estimator": "process_reinforce",
            "gamma": 1,
            "enable_rollout_level_advantage": False,
            "use_kl_in_reward": False,
            "rollout_correction": None,
        },
        "critic": {"enable": False},
        "reward_model": {"enable": False},
        "distillation": {"enabled": False},
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp",
                "ulysses_sequence_parallel_size": 1,
                "ppo_epochs": 1,
                "entropy_coeff": 0,
                "use_kl_loss": False,
                "policy_loss": {"loss_mode": "process_reinforce"},
            },
            "rollout": {"temperature": 1, "top_p": 1, "top_k": -1},
        },
    }


def test_supported_config_is_explicit_and_does_not_mutate():
    config = supported_config()
    before = copy.deepcopy(config)
    assert validate_process_config(config, verl_version="0.8.0").normalization_tokens == 1024
    assert config == before
    with pytest.raises(ValueError, match="0.8.0"):
        validate_process_config(config, verl_version="0.9.0")


@pytest.mark.parametrize(
    "path,value",
    [
        ("critic.enable", True),
        ("algorithm.gamma", 0.5),
        ("algorithm.enable_rollout_level_advantage", True),
        ("algorithm.use_kl_in_reward", True),
        ("algorithm.rollout_correction", {"rollout_is": "token"}),
        ("actor_rollout_ref.actor.strategy", "megatron"),
        ("actor_rollout_ref.actor.ppo_epochs", 2),
        ("actor_rollout_ref.actor.use_kl_loss", True),
        ("actor_rollout_ref.actor.entropy_coeff", 0.01),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.rollout.temperature", 0.7),
        ("actor_rollout_ref.rollout.top_p", 0.9),
        ("agentlightning.async_rollout.enabled", True),
        ("agentlightning.max_ppo_update_times", 2),
        ("reward_model.enable", True),
        ("distillation.enabled", True),
    ],
)
def test_unsupported_execution_rejected_before_workers(path, value):
    config = supported_config()
    target = config
    *parents, key = path.split(".")
    for parent in parents:
        target = target[parent]
    target[key] = value
    with pytest.raises(ValueError, match=path):
        validate_process_config(config, verl_version="0.8.0")


def test_disabled_mode_preserves_outcome_and_rejects_unknown_process_discriminator():
    assert validate_process_config({}, verl_version="uninstalled") is None
    assert (
        validate_process_config({"agentlightning": {"process_credit": {"enabled": False}}}, verl_version="old") is None
    )
    with pytest.raises(ValueError, match="enabled"):
        validate_process_config({"algorithm": {"adv_estimator": "process_reinforce"}}, verl_version="0.8.0")

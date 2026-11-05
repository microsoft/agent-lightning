# Copyright (c) Microsoft. All rights reserved.

"""Opt-in fixed-normalizer process-policy-gradient execution contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentlightning.credit.types import CreditProtocolError, ProcessCreditConfig, validate_sampling_parameters

PROCESS_LOSS_MODE = "process_reinforce"


def config_value(config: Any, path: str, default: Any = None) -> Any:
    """Read native DictConfig or ordinary nested mappings without mutation."""
    current = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def validate_process_config(config: Any, *, verl_version: str) -> ProcessCreditConfig | None:
    """Fail before worker launch on unsupported execution; disabled mode is inert."""
    raw = config_value(config, "agentlightning.process_credit", {})
    enabled = raw.get("enabled", False) if isinstance(raw, Mapping) else False
    estimator = config_value(config, "algorithm.adv_estimator")
    loss = config_value(config, "actor_rollout_ref.actor.policy_loss.loss_mode")
    if not enabled:
        if estimator == PROCESS_LOSS_MODE or loss == PROCESS_LOSS_MODE:
            raise CreditProtocolError("process_reinforce requires agentlightning.process_credit.enabled=true")
        return None
    process = ProcessCreditConfig.model_validate(dict(raw))
    if verl_version != "0.8.0":
        raise CreditProtocolError("process credit requires locked VERL 0.8.0")
    requirements = {
        "algorithm.adv_estimator": PROCESS_LOSS_MODE,
        "actor_rollout_ref.actor.policy_loss.loss_mode": PROCESS_LOSS_MODE,
        "critic.enable": False,
        "algorithm.enable_rollout_level_advantage": False,
        "agentlightning.async_rollout.enabled": False,
        "agentlightning.max_ppo_update_times": None,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "algorithm.use_kl_in_reward": False,
        "reward_model.enable": False,
        "distillation.enabled": False,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
    }
    for path, expected in requirements.items():
        actual = config_value(config, path, expected)
        if actual != expected:
            raise CreditProtocolError(f"process credit requires {path}={expected!r}, got {actual!r}")
    if config_value(config, "critic.enable") is not False:
        raise CreditProtocolError("process credit requires explicit critic.enable=false")
    if config_value(config, "algorithm.gamma", 1.0) != process.gamma:
        raise CreditProtocolError("algorithm.gamma must match agentlightning.process_credit.gamma")
    if config_value(config, "actor_rollout_ref.actor.strategy") not in {"fsdp", "fsdp2"}:
        raise CreditProtocolError("process credit requires actor_rollout_ref.actor.strategy fsdp/fsdp2")
    if config_value(config, "algorithm.rollout_correction"):
        raise CreditProtocolError("process credit forbids algorithm.rollout_correction")
    if config_value(config, "actor_rollout_ref.actor.megatron.router_replay.mode", "disabled") != "disabled":
        raise CreditProtocolError("process credit forbids actor_rollout_ref.actor.megatron.router_replay.mode")
    if config_value(config, "agentlightning.trace_aggregator.level", "transition") not in {"transition", "trajectory"}:
        raise CreditProtocolError("process credit requires transition or trajectory trace aggregation")
    sampling = config_value(config, "actor_rollout_ref.rollout", {})
    validate_sampling_parameters(sampling, prefix="actor_rollout_ref.rollout")
    return process

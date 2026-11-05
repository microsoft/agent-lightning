# Copyright (c) Microsoft. All rights reserved.

"""Opt-in fixed-normalizer process-policy-gradient execution contract."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, model_validator

from agentlightning.credit.types import (
    CreditProtocolError,
    FrozenCreditModel,
    ProcessCreditConfig,
    ResolvedCredit,
    finite_float32,
    nonempty,
    validate_sampling_parameters,
)

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


class ScheduledRollout(FrozenCreditModel):
    """A rollout's group and policy fixed before it is made visible for sampling."""

    rollout_id: str
    group_id: str
    group_size: int = Field(gt=0, strict=True)
    policy_version: int = Field(ge=0, strict=True)


class ScheduledGroups(FrozenCreditModel):
    """Complete collection membership, including zero-action/failure trajectories."""

    rollouts: tuple[ScheduledRollout, ...]

    @model_validator(mode="after")
    def complete_groups(self) -> ScheduledGroups:
        groups: dict[str, list[ScheduledRollout]] = defaultdict(list)
        ids = set()
        versions = set()
        for rollout in self.rollouts:
            nonempty(rollout.rollout_id)
            nonempty(rollout.group_id)
            if rollout.rollout_id in ids:
                raise CreditProtocolError("duplicate scheduled rollout identity")
            ids.add(rollout.rollout_id)
            versions.add(rollout.policy_version)
            groups[rollout.group_id].append(rollout)
        if not ids or len(versions) != 1:
            raise CreditProtocolError("scheduled collection requires one nonempty policy version")
        if any(any(item.group_size != len(group) for item in group) for group in groups.values()):
            raise CreditProtocolError("scheduled task group membership is incomplete")
        return self


class ProcessCoefficientBatch(FrozenCreditModel):
    """Detached original-action coefficients; K never comes from training row counts."""

    scheduled_count: int
    config_digest: str
    coefficients: dict[str, tuple[float, ...]]
    returns: dict[str, tuple[float, ...]]
    baselines: dict[str, float]
    episode_returns: dict[str, float]


def compute_process_coefficients(
    resolved: Sequence[ResolvedCredit],
    scheduled_groups: ScheduledGroups,
    config: ProcessCreditConfig,
) -> ProcessCoefficientBatch:
    """Apply outer discount and fixed K*H scaling to every complete scheduled action.

    Whole-task LOO subtracts other independent rollouts' entire discounted episode
    returns, not turn-matched returns or a baseline conditioned on reaching a turn.
    No observed token count, whitening, standard deviation or critic enters here.
    """
    config = ProcessCreditConfig.model_validate(config.model_dump(mode="json"))
    scheduled_groups = ScheduledGroups.model_validate(scheduled_groups.model_dump(mode="json"))
    expected = {item.rollout_id: item for item in scheduled_groups.rollouts}
    observed = {item.rollout_id: item for item in resolved}
    if len(observed) != len(resolved) or set(observed) != set(expected):
        raise CreditProtocolError("every scheduled rollout must have one complete resolution")
    groups: dict[str, list[ResolvedCredit]] = defaultdict(list)
    for item in resolved:
        scheduled = expected[item.rollout_id]
        if (item.group_id, item.group_size, item.policy_version) != (
            scheduled.group_id,
            scheduled.group_size,
            scheduled.policy_version,
        ):
            raise CreditProtocolError("resolved rollout differs from scheduled group/policy identity")
        if item.config_digest != config.digest or not config.enabled:
            raise CreditProtocolError("process estimator configuration differs from sealed resolution")
        lengths = {len(item.action_event_ids), len(item.rewards), len(item.returns), len(item.discounts)}
        if len(lengths) != 1:
            raise CreditProtocolError("resolved action arrays have inconsistent lengths")
        groups[item.group_id].append(item)
    baselines = {}
    coefficients = {}
    denominator = len(expected) * config.normalization_tokens
    if not math.isfinite(denominator):
        raise CreditProtocolError("scheduled K*H normalization must be finite")
    for rollout_id, item in observed.items():
        peers = [other.episode_return for other in groups[item.group_id] if other.rollout_id != rollout_id]
        baseline = math.fsum(peers) / len(peers) if config.baseline == "task_loo" and peers else 0.0
        baselines[rollout_id] = baseline
        coefficients[rollout_id] = tuple(
            finite_float32(discount * (value - baseline) / denominator, "policy coefficient")
            for value, discount in zip(item.returns, item.discounts, strict=True)
        )
    return ProcessCoefficientBatch(
        scheduled_count=len(expected),
        config_digest=config.digest,
        coefficients=coefficients,
        returns={key: item.returns for key, item in observed.items()},
        baselines=baselines,
        episode_returns={key: item.episode_return for key, item in observed.items()},
    )

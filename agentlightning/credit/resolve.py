# Copyright (c) Microsoft. All rights reserved.

"""Pure return-to-go resolution from complete original sealed action evidence."""

from __future__ import annotations

import math

from agentlightning.credit.state import validate_snapshot
from agentlightning.credit.types import (
    CreditProtocolError,
    CreditSnapshot,
    ProcessCreditConfig,
    ResolvedCredit,
    RewardAttribution,
    canonical_json,
    finite_float32,
)


def resolve_credit(snapshot: CreditSnapshot, config: ProcessCreditConfig) -> ResolvedCredit:
    """Book rewards at information cutoffs, then compute complete-action suffix returns.

    Attribution targets identify what a grader judged, not when its information
    became available. No reference is reassigned to a surviving neighboring action.
    Legitimate zero-action episodes have empty returns and remain visible to the
    separately predeclared collection denominator and whole-task baseline.
    """
    config = ProcessCreditConfig.model_validate(config.model_dump(mode="json"))
    if not config.enabled:
        raise CreditProtocolError("process credit resolution requires enabled configuration")
    snapshot = CreditSnapshot.model_validate(snapshot.model_dump(mode="json"))
    if snapshot.limits != config.admission_limits:
        raise CreditProtocolError("sealed admission limits differ from process configuration")
    actions, reward_events = validate_snapshot(snapshot)
    if len(snapshot.events) > config.max_events_per_rollout or len(actions) > config.max_actions_per_rollout:
        raise CreditProtocolError("sealed credit exceeds configured event/action limits")
    if len(canonical_json(snapshot.model_dump(mode="json"))) > config.max_artifact_bytes:
        raise CreditProtocolError("sealed credit exceeds configured artifact size limit")
    indices = {record.envelope.event_id: index for index, record in enumerate(actions)}
    booked: list[list[float]] = [[] for _ in actions]
    attributions = []
    terminal_outcome = None
    for record in reward_events:
        envelope = record.envelope
        if envelope.kind is None:
            raise CreditProtocolError("reward is missing its declared kind")
        raw = record.data.get("value")
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            raise CreditProtocolError("reward values must be finite numbers")
        weight = config.terminal_weight if envelope.kind == "terminal" else config.process_weight
        weighted = raw * weight
        if not math.isfinite(weighted):
            raise CreditProtocolError("weighted reward accumulation must be finite and representable as float32")
        cutoff = (
            indices.get(envelope.evidence_through_event_id) if envelope.evidence_through_event_id is not None else None
        )
        target = indices.get(envelope.target_event_id) if envelope.target_event_id is not None else None
        if envelope.kind == "terminal":
            terminal_outcome = float(raw)
        if cutoff is not None:
            booked[cutoff].append(weighted)
        attributions.append(
            RewardAttribution(
                reward_id=envelope.event_id,
                kind=envelope.kind,
                target_index=target,
                cutoff_index=cutoff,
                raw_value=raw,
                weighted_value=weighted,
                source=record.data["source"],
            )
        )
    try:
        rewards = tuple(finite_float32(math.fsum(values), "booked reward") for values in booked)
        returns = [0.0] * len(actions)
        carry = 0.0
        for index in range(len(actions) - 1, -1, -1):
            carry = finite_float32(math.fsum((rewards[index], config.gamma * carry)), "return-to-go")
            returns[index] = carry
        discounts = tuple(config.gamma**index for index in range(len(actions)))
        episode_return = finite_float32(
            math.fsum(r * d for r, d in zip(rewards, discounts, strict=True)), "episode return"
        )
    except OverflowError as exc:
        raise CreditProtocolError("credit accumulation overflows finite float32 evidence") from exc
    if terminal_outcome is None:
        raise CreditProtocolError("exactly one finite terminal outcome is required")
    return ResolvedCredit(
        rollout_id=snapshot.rollout_id,
        attempt_id=snapshot.attempt_id,
        policy_version=snapshot.policy_version,
        group_id=snapshot.group_id,
        group_size=snapshot.group_size,
        config_digest=config.digest,
        snapshot_hash=snapshot.snapshot_hash,
        action_event_ids=snapshot.action_event_ids,
        rewards=rewards,
        returns=tuple(returns),
        discounts=discounts,
        attributions=tuple(attributions),
        terminal_outcome=terminal_outcome,
        episode_return=episode_return,
    )

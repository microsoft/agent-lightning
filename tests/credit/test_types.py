# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import math

import pytest

from agentlightning.credit.types import CreditEnvelope, FinalizeCreditRequest, ProcessCreditConfig


def test_defaults_and_finite_constraints():
    config = ProcessCreditConfig()
    assert (config.enabled, config.gamma, config.normalization_tokens, config.baseline) == (False, 1, 1024, "zero")
    assert (
        config.finalization_timeout_seconds,
        config.max_events_per_rollout,
        config.max_actions_per_rollout,
        config.max_artifact_bytes,
    ) == (300, 100000, 1024, 67108864)
    with pytest.raises(ValueError):
        config.gamma = 0.5


@pytest.mark.parametrize(
    "values",
    [
        {"gamma": math.nan},
        {"gamma": math.inf},
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"normalization_tokens": 0},
        {"normalization_tokens": math.inf},
        {"terminal_weight": math.nan},
        {"process_weight": math.inf},
        {"max_actions_per_rollout": 0},
        {"unknown": True},
    ],
)
def test_invalid_config_never_repairs_values(values):
    with pytest.raises(ValueError):
        ProcessCreditConfig(**values)


def test_unknown_protocol_and_invalid_reward_identity_rejected():
    with pytest.raises(ValueError):
        CreditEnvelope.model_validate({"version": 2, "event_id": "", "policy_version": 0})
    with pytest.raises(ValueError):
        CreditEnvelope(event_id="not-a-uuid", policy_version=0, kind="terminal")
    with pytest.raises(ValueError):
        FinalizeCreditRequest(
            action_event_ids=("bad",), reward_event_ids=(), terminal_reward_id="bad", policy_version=0
        )

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


@pytest.mark.parametrize(
    "modifier",
    [
        {"min_tokens": 1},
        {"response_format": {"type": "json_object"}},
        {"guided_json": {}},
        {"guided_regex": "a"},
        {"structured_outputs": {}},
        {"stop_token_ids": [2]},
        {"bad_words": ["x"]},
        {"logits_processors": ["custom"]},
        {"extra_body": {"temperature": 0.1}},
        {"vllm_xargs": {}},
        {"best_of": 2},
        {"use_beam_search": True},
        {"seed": 0},
        {"unknown_extension": 1},
    ],
)
def test_supported_request_fields_reject_unproven_sampling_extensions(modifier):
    from agentlightning.credit.types import validate_sampling_parameters

    with pytest.raises(ValueError):
        validate_sampling_parameters({"messages": [], **modifier})


@pytest.mark.parametrize(
    "bound,value",
    [("max_actions_per_rollout", 1025), ("max_events_per_rollout", 100001), ("max_artifact_bytes", 67108865)],
)
def test_unsupported_admission_bounds_rejected_explicitly(bound, value):
    with pytest.raises(ValueError):
        ProcessCreditConfig(**{bound: value})

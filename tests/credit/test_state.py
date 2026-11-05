# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from agentlightning.credit.state import AttemptCreditState
from agentlightning.credit.types import CreditConflictError, FinalizeCreditRequest, IncompletePolicyTraceError
from agentlightning.schemas import Event


def uid(number):
    return UUID(int=number).hex


def reward(number, target=None, cutoff=None, kind="terminal", value=1):
    return {
        "value": value,
        "source": "rule",
        "_agl_credit": {
            "version": 1,
            "event_id": uid(number),
            "kind": kind,
            "target_event_id": target,
            "evidence_through_event_id": cutoff,
            "policy_version": 7,
        },
    }


def action_event(tokens=(3, 2), status=200):
    return Event(
        event_type="model_request",
        rollout_id="r",
        attempt_id="a",
        timestamp=0,
        data={
            "model_version": 7,
            "request": {"temperature": 1},
            "http_status": status,
            "response": {
                "prompt_token_ids": [1],
                "choices": [
                    {
                        "token_ids": list(tokens),
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1] * len(tokens)},
                    }
                ],
            },
        },
    )


@pytest.mark.asyncio
async def test_reward_idempotency_and_atomic_seal():
    state = AttemptCreditState("r", "a", policy_version=7, group_id="task", group_size=1)
    lease = state.begin_request("request-1", 0, "body-digest")
    duplicate = state.begin_request("request-1", 0, "body-digest")
    assert duplicate.future is lease.future
    with pytest.raises(CreditConflictError):
        state.begin_request("request-2", 1, "other")
    with pytest.raises(CreditConflictError):
        state.begin_request("request-1", 0, "conflicting")
    completion = state.complete_request(lease, event=action_event(), response_body={"text": "same"}, status_code=200)
    assert (await asyncio.shield(duplicate.future)).event_id == completion.event_id
    posted = state.append_reward(reward(1, cutoff=completion.event_id))
    assert state.append_reward(reward(1, cutoff=completion.event_id)) == posted
    with pytest.raises(CreditConflictError):
        state.append_reward(reward(1, cutoff=completion.event_id, value=0))
    request = FinalizeCreditRequest(
        action_event_ids=(completion.event_id,), reward_event_ids=(uid(1),), terminal_reward_id=uid(1), policy_version=7
    )
    receipt = state.finalize(request)
    assert state.finalize(request) == receipt
    assert state.snapshot().action_event_ids == (completion.event_id,)
    snapshot = state.snapshot()
    snapshot.events[0].data["model_version"] = 99
    assert state.snapshot().events[0].data["model_version"] == 7
    with pytest.raises(CreditConflictError):
        state.append_reward(reward(2, cutoff=completion.event_id))


@pytest.mark.asyncio
async def test_missing_terminal_pending_reference_and_uncaptured_action_are_not_zero():
    state = AttemptCreditState("r", "a", policy_version=7, group_id="task", group_size=1)
    with pytest.raises(IncompletePolicyTraceError):
        state.snapshot()
    state.append_reward(reward(1, cutoff=uid(999)))
    with pytest.raises(IncompletePolicyTraceError):
        state.finalize(
            FinalizeCreditRequest(
                action_event_ids=(), reward_event_ids=(uid(1),), terminal_reward_id=uid(1), policy_version=7
            )
        )
    lease = state.begin_request("uncaptured", 0, "digest")
    state.fail_request(lease, "transport lost after possible sampling")
    with pytest.raises(IncompletePolicyTraceError):
        state.finalize(
            FinalizeCreditRequest(
                action_event_ids=(), reward_event_ids=(uid(1),), terminal_reward_id=uid(1), policy_version=7
            )
        )
    assert lease.future.done()


@pytest.mark.asyncio
async def test_zero_action_explicit_terminal_and_failed_ordinals():
    state = AttemptCreditState("r", "a", policy_version=7, group_id="task", group_size=1)
    lease = state.begin_request("failed", 0, "error")
    state.complete_request(lease, event=action_event(tokens=(), status=400), response_body={}, status_code=400)
    state.append_reward(reward(1, value=0))
    receipt = state.finalize(
        FinalizeCreditRequest(
            action_event_ids=(), reward_event_ids=(uid(1),), terminal_reward_id=uid(1), policy_version=7
        )
    )
    assert receipt.action_count == 0

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from uuid import UUID

import pytest

from agentlightning.credit.state import AttemptCreditState
from agentlightning.credit.types import FinalizeCreditRequest
from agentlightning.schemas import Event


@pytest.fixture
def sealed_snapshot():
    async def build(
        rewards=(0.2, -0.1, 1),
        cutoff=None,
        target=None,
        action_count=3,
        rollout_id="r",
        group_id="task",
        group_size=1,
        same_prompt=False,
        failed_first=False,
    ):
        state = AttemptCreditState(rollout_id, "a", policy_version=7, group_id=group_id, group_size=group_size)
        actions = []
        if failed_first:
            failed = state.begin_request("failed", 0, "failed-body")
            event = Event(
                event_type="model_request",
                rollout_id=rollout_id,
                attempt_id="a",
                timestamp=0,
                data={"model_version": 7, "http_status": 400, "response": {"error": "no sample"}},
            )
            state.complete_request(failed, event=event, response_body={"error": "no sample"}, status_code=400)
        for index in range(action_count):
            lease = state.begin_request(f"request-{index}", index + int(failed_first), f"body-{index}")
            event = Event(
                event_type="model_request",
                rollout_id=rollout_id,
                attempt_id="a",
                timestamp=123,
                data={
                    "model_version": 7,
                    "request": {"temperature": 1},
                    "http_status": 200,
                    "response": {
                        "prompt_token_ids": [1] + [3, 2, 9] * (0 if same_prompt else index),
                        "choices": [
                            {"token_ids": [3, 2], "finish_reason": "stop", "logprobs": {"token_logprobs": [-0.1, -0.2]}}
                        ],
                    },
                },
            )
            completion = state.complete_request(
                lease, event=event, response_body=event.data["response"], status_code=200
            )
            actions.append(completion.event_id)
        reward_ids = []
        for index, value in enumerate(rewards):
            terminal = index == len(rewards) - 1
            evidence_index = index if cutoff is None else cutoff
            target_index = index if target is None else target
            reward_id = UUID(int=index + 1).hex
            envelope = {
                "event_id": reward_id,
                "policy_version": 7,
                "kind": "terminal" if terminal else "process",
                "evidence_through_event_id": actions[-1]
                if terminal and actions
                else actions[evidence_index]
                if actions
                else None,
            }
            if not terminal:
                envelope["target_event_id"] = actions[target_index]
            state.append_reward({"value": value, "source": "rule", "_agl_credit": envelope})
            reward_ids.append(reward_id)
        state.finalize(
            FinalizeCreditRequest(
                action_event_ids=tuple(actions),
                reward_event_ids=tuple(reward_ids),
                terminal_reward_id=reward_ids[-1],
                policy_version=7,
            )
        )
        return state.snapshot()

    return build

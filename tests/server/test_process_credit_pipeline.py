# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import cast
from uuid import UUID

import httpx
import pytest
import torch
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

from agentlightning.credit.artifacts import read_credit_artifact, replay_credit, write_credit_artifact
from agentlightning.credit.client import ProcessCreditClient
from agentlightning.credit.types import ProcessCreditConfig
from agentlightning.schemas import Rollout
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase, EnqueuedRollout
from agentlightning.verl.process_credit import ScheduledGroups, ScheduledRollout
from agentlightning.verl.process_loss import build_process_actor_payload
from agentlightning.verl.rollout_adapter import RolloutAdapter


def test_native_http_seal_manager_projection_replay_and_actor_loss(
    client, app, auth_headers, cpu_native_padding, tmp_path
):
    calls = []

    async def upstream(request):
        index = len(calls)
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1, 10 + index],
                "usage": {"completion_tokens": 2},
                "choices": [
                    {
                        "message": {"content": str(index)},
                        "token_ids": [3 + index, 2],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ],
            },
        )

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client.headers.update(auth_headers)
    assert (
        client.post(
            "/api/models", json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}]
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/rollouts",
            json=[
                {
                    "input": {},
                    "rollout_id": "r",
                    "metadata": {
                        "process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}
                    },
                }
            ],
        ).status_code
        == 201
    )
    credit = ProcessCreditClient(client, "r", "a", policy_version=7)
    actions = []
    for index in range(3):
        body = {"messages": [{"role": "user", "content": f"local tool step {index}"}]}
        completion = credit.complete(body, request_id=f"logical-{index}")
        assert credit.complete(body, request_id=f"logical-{index}") == completion
        actions.append(completion.event_id)
    assert len(calls) == 3
    credit.emit_process_reward(
        0.2,
        reward_id=UUID(int=101).hex,
        target_event_id=actions[0],
        evidence_through_event_id=actions[0],
        source="prefix-check",
    )
    # This label judges action 0 using action 1 evidence: it belongs at cutoff 1.
    credit.emit_process_reward(
        -0.1,
        reward_id=UUID(int=102).hex,
        target_event_id=actions[0],
        evidence_through_event_id=actions[1],
        source="tool-check",
    )
    credit.emit_terminal_reward(1, reward_id=UUID(int=103).hex, source="final-check")
    receipt = credit.finalize()
    assert credit.finalize() == receipt
    assert (
        client.patch("/api/rollouts/r", json={"status": {"state": "running", "last_attempt_id": "a"}}).status_code
        == 200
    )
    assert (
        client.patch("/api/rollouts/r", json={"status": {"state": "succeeded", "last_attempt_id": "a"}}).status_code
        == 200
    )
    rollout = Rollout.model_validate(client.get("/api/rollouts/r").json()["rollout"])
    config = ProcessCreditConfig(enabled=True, gamma=0.5, normalization_tokens=10)
    manager = AglRolloutManagerBase(
        agl_base_url="http://offline", agl_key="", model="test-model", step=7, process_credit_config=config
    )
    manager.client.close()
    manager.client = client
    completed = manager._build_completed_rollout(
        EnqueuedRollout(rollout_id="r", data_id="task", step=7, sample_idx_in_step=0, enqueue_time=0), rollout
    )
    scheduled = ScheduledGroups(
        rollouts=(ScheduledRollout(rollout_id="r", group_id="task", group_size=1, policy_version=7),)
    )
    adapter = RolloutAdapter(
        max_prompt_length=20,
        max_response_length=20,
        device=torch.device("cpu"),
        pad_token_id=0,
        tokenizer=SimpleNamespace(eos_token_id=2),
        trace_aggregator_level="transition",
        process_credit_config=config,
        process_scheduled_groups=scheduled,
    )
    batch, metrics = adapter.get_train_data_batch([completed])
    assert completed.process_credit is not None
    assert metrics["training/reward"] == 1
    assert completed.process_credit.resolved.rewards == pytest.approx((0.2, -0.1, 1))
    assert completed.process_credit.resolved.returns == pytest.approx((0.4, 0.4, 1))
    torch.testing.assert_close(
        batch.batch["advantages"][:, :2], torch.tensor([[0.04, 0.04], [0.02, 0.02], [0.025, 0.025]])
    )
    path = tmp_path / "audit.json"
    write_credit_artifact(path, completed.process_credit, max_bytes=config.max_artifact_bytes)
    assert (
        replay_credit(read_credit_artifact(path, max_bytes=config.max_artifact_bytes))
        == completed.process_credit.resolved
    )
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
    payload = build_process_actor_payload(batch, seed=3)
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(payload, dp_size=1, global_batch_size=3, batch_num_tokens=None)
    parameter = torch.tensor(-0.2, requires_grad=True)
    actor = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="process_reinforce"),
    )
    loss, _ = ppo_loss(
        actor, {"log_probs": parameter.expand(cast(torch.Tensor, payload["input_ids"]).values().numel())}, payload
    )
    loss.backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(-0.17)


@pytest.mark.asyncio
async def test_documented_arithmetic_agent_seals_the_native_attempt(client, app, auth_headers, monkeypatch):
    from examples.process_credit.agent import Agent

    answers = iter((3, 9, 8))

    async def upstream(request):
        value = next(answers)
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1],
                "choices": [
                    {
                        "message": {"content": str(value)},
                        "token_ids": [value + 3, 2],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ],
            },
        )

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client.headers.update(auth_headers)
    client.post("/api/models", json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}])
    client.post(
        "/api/rollouts",
        json=[
            {
                "input": {},
                "rollout_id": "r",
                "metadata": {
                    "process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}
                },
            }
        ],
    )
    for name, value in {
        "A": "1",
        "B": "2",
        "C": "3",
        "D": "1",
        "AGL_KEY": "test-secret-key",
        "AGL_EVENT_URL": "http://testserver/api/rollouts/r/attempt/a/events",
    }.items():
        monkeypatch.setenv(name, value)
    from contextlib import nullcontext

    monkeypatch.setattr("examples.process_credit.agent.httpx.Client", lambda **kwargs: nullcontext(client))
    await Agent().run()
    response = client.get("/api/rollouts/r/attempt/a/credit")
    assert response.status_code == 200
    from agentlightning.credit.artifacts import make_credit_artifact
    from agentlightning.credit.types import CreditSnapshot

    resolved = make_credit_artifact(
        CreditSnapshot.model_validate(response.json()), ProcessCreditConfig(enabled=True)
    ).resolved
    assert resolved.rewards == pytest.approx((0.2, 0.1, 1))
    assert resolved.terminal_outcome == 1

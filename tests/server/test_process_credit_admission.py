# Copyright (c) Microsoft. All rights reserved.

from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from agentlightning.credit.artifacts import make_credit_artifact
from agentlightning.credit.client import ProcessCreditClient
from agentlightning.credit.types import CreditSnapshot, ProcessCreditConfig
from agentlightning.server.app import create_app
from agentlightning.server.store import _credit_states, _rollouts
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase


@pytest.fixture
def native_admission(client, app, auth_headers):
    calls = []

    async def upstream(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1],
                "choices": [
                    {
                        "token_ids": [3, 2],
                        "message": {"content": "x" * 2000},
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ],
            },
        )

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client.headers.update(auth_headers)
    client.post(
        "/api/models", json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}]
    ).raise_for_status()

    def schedule(config):
        manager = AglRolloutManagerBase(
            agl_base_url="http://offline", agl_key="", model="test-model", step=7, process_credit_config=config
        )
        manager.client.close()
        client.post_with_retry = client.post
        manager.client = client
        enqueued = manager._create_rollouts({"task": ["constant"]}, is_train=True)[0]
        return enqueued, ProcessCreditClient(client, enqueued.rollout_id, "a", policy_version=7)

    return client, calls, schedule


@pytest.mark.parametrize(
    "modifier",
    [
        {"min_tokens": 1},
        {"response_format": {"type": "json_object"}},
        {"guided_json": {}},
        {"extra_body": {"temperature": 0.2}},
        {"logits_processors": ["arbitrary"]},
        {"unknown_extension": 1},
    ],
)
def test_native_sampling_extensions_fail_before_any_forward(native_admission, modifier):
    _client, calls, schedule = native_admission
    _, credit = schedule(ProcessCreditConfig(enabled=True))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        credit.complete({"messages": [], **modifier}, request_id="draw")
    assert exc.value.response.status_code == 400
    assert calls == []


@pytest.mark.parametrize("bound", ["max_actions_per_rollout", "max_events_per_rollout"])
def test_native_enqueue_binds_limits_before_second_generation(native_admission, bound):
    client, calls, schedule = native_admission
    config = ProcessCreditConfig.model_validate({"enabled": True, bound: 1})
    enqueued, credit = schedule(config)
    credit.complete({"messages": []}, request_id="first")
    with pytest.raises(httpx.HTTPStatusError) as exc:
        credit.complete({"messages": []}, request_id="second")
    assert exc.value.response.status_code == 400
    state = _credit_states[(enqueued.rollout_id, "a")]
    assert state.config.admission_limits == config.admission_limits
    assert len(calls) == 1
    marker = client.get(f"/api/rollouts/{enqueued.rollout_id}").json()["rollout"]["metadata"]["process_credit"]
    assert marker["limits"] == config.admission_limits.model_dump(mode="json")


def test_native_byte_limit_aborts_capture_of_an_unknown_large_reply(native_admission):
    client, calls, schedule = native_admission
    enqueued, credit = schedule(ProcessCreditConfig(enabled=True, max_artifact_bytes=512))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        credit.complete({"messages": []}, request_id="oversized-reply")
    assert exc.value.response.status_code == 409
    state = _credit_states[(enqueued.rollout_id, "a")]
    assert state.config.max_artifact_bytes == 512
    assert state.event_count == 0
    assert len(calls) == 1  # Provider reply size is unknown before forwarding.
    assert client.get(f"/api/rollouts/{enqueued.rollout_id}/attempt/a/credit").status_code == 409


def test_admission_binding_is_immutable_and_replayed_with_the_same_limits(native_admission):
    client, calls, schedule = native_admission
    config = ProcessCreditConfig(enabled=True, max_actions_per_rollout=1)
    enqueued, credit = schedule(config)
    credit.complete({"messages": []}, request_id="first")
    credit.emit_terminal_reward(1, reward_id=uuid4().hex, source="constant")
    credit.finalize()
    snapshot = CreditSnapshot.model_validate(client.get(f"/api/rollouts/{enqueued.rollout_id}/attempt/a/credit").json())
    assert snapshot.limits == config.admission_limits
    make_credit_artifact(snapshot, config)
    with pytest.raises(ValueError, match="admission limits"):
        make_credit_artifact(snapshot, ProcessCreditConfig(enabled=True))
    marker = client.get(f"/api/rollouts/{enqueued.rollout_id}").json()["rollout"]["metadata"]["process_credit"]
    marker["limits"]["max_actions_per_rollout"] = 2
    response = client.post(
        "/api/rollouts", json=[{"input": {}, "rollout_id": enqueued.rollout_id, "metadata": {"process_credit": marker}}]
    )
    assert response.status_code == 409
    assert len(calls) == 1


def test_operator_caps_reject_unsupported_admission_before_enqueue_visibility(server_config, auth_headers):
    capped_app = create_app({**server_config, "process_credit_limits": {"max_actions_per_rollout": 1}})
    with TestClient(capped_app, headers=auth_headers) as client:
        response = client.post(
            "/api/rollouts",
            json=[
                {
                    "input": {},
                    "rollout_id": "too-large",
                    "metadata": {
                        "process_credit": {
                            "version": 1,
                            "policy_version": 7,
                            "group_id": "task",
                            "group_size": 1,
                            "limits": {"max_actions_per_rollout": 2},
                        }
                    },
                }
            ],
        )
        assert response.status_code == 400
        assert "operator cap" in response.text
        assert "too-large" not in _rollouts


def test_invalid_operator_caps_fail_at_server_startup(server_config):
    with pytest.raises(ValueError):
        create_app({**server_config, "process_credit_limits": {"max_actions_per_rollout": 1025}})


@pytest.mark.parametrize("limits", [(1, 1024), (1024, 1)])
def test_conflicting_new_batch_ids_abort_before_any_visibility(native_admission, limits):
    client, calls, _schedule = native_admission
    markers = [
        {
            "version": 1,
            "policy_version": 7,
            "group_id": "task",
            "group_size": 1,
            "limits": {"max_actions_per_rollout": bound},
        }
        for bound in limits
    ]
    response = client.post(
        "/api/rollouts",
        json=[{"input": {}, "rollout_id": "duplicate", "metadata": {"process_credit": marker}} for marker in markers],
    )
    assert response.status_code == 409
    assert "duplicate" not in _rollouts
    assert calls == []


def test_later_invalid_binding_does_not_publish_an_earlier_valid_episode(native_admission):
    client, calls, _schedule = native_admission
    marker = {
        "version": 1,
        "policy_version": 7,
        "group_id": "task",
        "group_size": 1,
        "limits": {"max_actions_per_rollout": 1},
    }
    invalid = {**marker, "limits": {"max_actions_per_rollout": 0}}
    response = client.post(
        "/api/rollouts",
        json=[
            {"input": {}, "rollout_id": name, "metadata": {"process_credit": context}}
            for name, context in (("valid", marker), ("invalid", invalid))
        ],
    )
    assert response.status_code == 400
    assert not {"valid", "invalid"}.intersection(_rollouts)
    assert calls == []


@pytest.mark.parametrize("process", [True, False])
def test_identical_new_batch_ids_remain_idempotent(native_admission, process):
    client, calls, _schedule = native_admission
    item = {"input": {}, "rollout_id": "identical"}
    if process:
        item["metadata"] = {
            "process_credit": {
                "version": 1,
                "policy_version": 7,
                "group_id": "task",
                "group_size": 1,
                "limits": {"max_actions_per_rollout": 1},
            }
        }
    response = client.post("/api/rollouts", json=[item, item])
    assert response.status_code == 201
    assert response.json()[0] == response.json()[1]
    assert list(_rollouts) == ["identical"]
    assert calls == []

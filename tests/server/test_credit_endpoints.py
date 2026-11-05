# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from uuid import UUID

from agentlightning.server.store import _credit_states, _events


def test_native_zero_action_seal_idempotency_and_write_boundary(client, auth_headers):
    metadata = {"process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}}
    created = client.post(
        "/api/rollouts", headers=auth_headers, json=[{"input": {}, "rollout_id": "credit-r", "metadata": metadata}]
    )
    assert created.status_code == 201
    reward_id = UUID(int=1).hex
    reward = {
        "event_type": "reward",
        "data": {
            "value": 0,
            "source": "rule",
            "_agl_credit": {"version": 1, "event_id": reward_id, "kind": "terminal", "policy_version": 7},
        },
    }
    path = "/api/rollouts/credit-r/attempt/a"
    assert client.get(path + "/credit", headers=auth_headers).status_code == 409
    posted = client.post(path + "/events", headers=auth_headers, json=reward)
    assert posted.status_code == 200
    assert "_agl_credit" in posted.json()["data"]
    assert client.post(path + "/events", headers=auth_headers, json=reward).json() == posted.json()
    assert len(_events["credit-r"]["a"]) == 1
    manifest = {
        "version": 1,
        "policy_version": 7,
        "action_event_ids": [],
        "reward_event_ids": [reward_id],
        "terminal_reward_id": reward_id,
    }
    receipt = client.post(path + "/credit/finalize", headers=auth_headers, json=manifest)
    assert receipt.status_code == 200
    assert client.post(path + "/credit/finalize", headers=auth_headers, json=manifest).json() == receipt.json()
    assert client.get(path + "/credit", headers=auth_headers).json()["action_event_ids"] == []
    reward["data"]["_agl_credit"]["event_id"] = UUID(int=2).hex
    assert client.post(path + "/events", headers=auth_headers, json=reward).status_code == 409
    assert (
        client.post(
            path + "/events", headers=auth_headers, json={"event_type": "diagnostic", "data": {"ok": True}}
        ).status_code
        == 200
    )
    assert len(client.get(path + "/credit", headers=auth_headers).json()["events"]) == 1
    assert client.delete("/api/rollouts/credit-r", headers=auth_headers).status_code == 204
    assert client.get(path + "/credit", headers=auth_headers).status_code == 404
    assert not any(key[0] == "credit-r" for key in _credit_states)


def test_outcome_reward_wire_shape_is_unchanged(client, auth_headers):
    client.post("/api/rollouts", headers=auth_headers, json=[{"input": {}, "rollout_id": "outcome-r"}])
    response = client.post(
        "/api/rollouts/outcome-r/attempt/a/events",
        headers=auth_headers,
        json={"event_type": "reward", "data": {"value": 0.5, "source": "rule"}},
    )
    assert response.status_code == 200
    assert response.json()["data"] == {"value": 0.5, "source": "rule"}

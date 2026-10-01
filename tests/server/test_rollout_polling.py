# Copyright (c) Microsoft. All rights reserved.

"""Compact rollout lifecycle status API."""

from fastapi.testclient import TestClient


def test_status_batch_returns_only_requested_lifecycle_statuses(client: TestClient, auth_headers: dict[str, str]):
    created = client.post(
        "/api/rollouts",
        headers=auth_headers,
        json=[{"input": {"prompt": "x" * 4096}} for _ in range(4)],
    ).json()
    ids = [item["rollout_id"] for item in created]
    for rollout_id in ids[1:]:
        client.patch(
            f"/api/rollouts/{rollout_id}",
            headers=auth_headers,
            json={"status": {"state": "running", "last_attempt_id": "attempt-1", "k8s_job_name": "job-1"}},
        ).raise_for_status()
    for rollout_id, state in zip(ids[2:], ["succeeded", "failed"], strict=True):
        client.patch(
            f"/api/rollouts/{rollout_id}",
            headers=auth_headers,
            json={"status": {"state": state, "error_message": "diagnostic"}},
        ).raise_for_status()

    requested = [ids[3], ids[0], ids[2], ids[1], ids[3]]
    response = client.post("/api/rollouts/status", headers=auth_headers, json=requested)
    assert response.status_code == 200
    assert response.json() == {
        rid: client.get(f"/api/rollouts/{rid}", headers=auth_headers).json()["rollout"]["status"] for rid in requested
    }
    assert client.post("/api/rollouts/status", headers=auth_headers, json=[ids[1]]).json() == {
        ids[1]: response.json()[ids[1]]
    }
    assert client.post("/api/rollouts/status", headers=auth_headers, json=[]).json() == {}
    assert client.post("/api/rollouts/status", headers=auth_headers, json=[ids[0]] * 256).status_code == 200
    assert client.post("/api/rollouts/status", headers=auth_headers, json=[ids[0]] * 257).status_code == 422
    assert client.post("/api/rollouts/status", json=ids).status_code == 401
    client.delete(f"/api/rollouts/{ids[1]}", headers=auth_headers).raise_for_status()
    for missing in [ids[1], "unknown"]:
        response = client.post("/api/rollouts/status", headers=auth_headers, json=[ids[0], missing])
        assert response.status_code == 404
        assert missing in response.json()["detail"]

"""Concise coverage for the current server endpoints."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from tests.server.conftest import MODEL_NAME


def _rollout(client: TestClient, headers: dict[str, str]) -> dict:
    response = client.post(
        "/api/rollouts",
        json=[{"input": {"prompt": "hi"}, "metadata": {"batch_idx": 1}}],
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()[0]


def test_healthz(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_auth_required(client: TestClient):
    assert client.post("/api/rollouts", json=[]).status_code == 401


def test_rollout_endpoints(client: TestClient, auth_headers: dict[str, str]):
    rollout = _rollout(client, auth_headers)
    assert rollout["input"] == {"prompt": "hi"}
    assert rollout["metadata"]["batch_idx"] == 1
    assert rollout["status"]["state"] == "queuing"

    detail = client.get(f"/api/rollouts/{rollout['rollout_id']}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["attempts"] == []

    patched = client.patch(
        f"/api/rollouts/{rollout['rollout_id']}",
        json={"status": {"state": "running", "k8s_job_name": "job-1"}},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["status"]["state"] == "running"
    assert patched.json()["status"]["k8s_job_name"] == "job-1"

    invalid = client.patch(
        f"/api/rollouts/{rollout['rollout_id']}",
        json={"status": {"state": "queuing"}},
        headers=auth_headers,
    )
    assert invalid.status_code == 409

    cancelled = client.patch(
        f"/api/rollouts/{rollout['rollout_id']}",
        json={"status": {"state": "cancelled"}},
        headers=auth_headers,
    )
    assert cancelled.status_code == 422

    cancel_requested = client.patch(
        f"/api/rollouts/{rollout['rollout_id']}",
        json={"status": {"cancel_requested": True}},
        headers=auth_headers,
    )
    assert cancel_requested.status_code == 422


def test_list_rollouts_filters_by_state_in(client: TestClient, auth_headers: dict[str, str]):
    queuing_rollout = _rollout(client, auth_headers)
    running_rollout = _rollout(client, auth_headers)

    patched = client.patch(
        f"/api/rollouts/{running_rollout['rollout_id']}",
        json={"status": {"state": "running"}},
        headers=auth_headers,
    )
    assert patched.status_code == 200

    response = client.get(
        "/api/rollouts",
        params=[("state_in", "queuing"), ("state_in", "running")],
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert {item["rollout_id"] for item in response.json()} == {
        queuing_rollout["rollout_id"],
        running_rollout["rollout_id"],
    }

    response = client.get("/api/rollouts", params={"state_in": "running"}, headers=auth_headers)
    assert response.status_code == 200
    assert [item["rollout_id"] for item in response.json()] == [running_rollout["rollout_id"]]


def test_event_endpoints(client: TestClient, auth_headers: dict[str, str]):
    rollout = _rollout(client, auth_headers)
    rollout_id = rollout["rollout_id"]

    posted = client.post(
        f"/api/rollouts/{rollout_id}/attempt/0/events",
        json={"event_type": "reward", "data": {"value": 0.7, "extra": "drop"}},
        headers=auth_headers,
    )
    assert posted.status_code == 200
    assert posted.json()["event_type"] == "reward"

    queried = client.get(
        f"/api/rollouts/{rollout_id}/events",
        params={"event_type": "reward", "format": "triplet"},
        headers=auth_headers,
    )
    assert queried.status_code == 200
    assert queried.json()[0]["data"] == {"value": 0.7}

    detail = client.get(f"/api/rollouts/{rollout_id}", headers=auth_headers)
    assert detail.json()["attempts"] == ["0"]


def test_model_endpoints(client: TestClient, auth_headers: dict[str, str]):
    created = client.post(
        "/api/models",
        json=[{"model": MODEL_NAME, "endpoint": "http://model.test/v1", "version": 3}],
        headers=auth_headers,
    )
    assert created.status_code == 201
    assert created.json()[0]["model"] == MODEL_NAME

    deleted = client.delete("/api/models", headers=auth_headers)
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "ok"}


def test_proxy_completion_endpoint(client: TestClient, auth_headers: dict[str, str], monkeypatch):
    async def fake_upstream(*, client: httpx.AsyncClient, url: str, body: dict) -> httpx.Response:
        assert url == "http://model.test/v1/chat/completions"
        assert body["model"] == MODEL_NAME
        assert body["temperature"] == 1.0
        assert body["return_token_ids"] is True
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "token_ids": [2]}],
                "prompt_token_ids": [1],
            },
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr("agl_lite.server.proxy._send_upstream_with_retries", fake_upstream)
    rollout = _rollout(client, auth_headers)
    client.post(
        "/api/models",
        json=[{"model": MODEL_NAME, "endpoint": "http://model.test/v1", "version": 3}],
        headers=auth_headers,
    )

    proxied = client.post(
        f"/proxy/rollout/{rollout['rollout_id']}/attempt/0/mode/train/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=auth_headers,
    )
    assert proxied.status_code == 200
    assert proxied.json()["choices"][0]["message"]["content"] == "ok"

    events = client.get(
        f"/api/rollouts/{rollout['rollout_id']}/events",
        params={"event_type": "model_request", "format": "triplet"},
        headers=auth_headers,
    ).json()
    assert events[0]["data"]["prompt_token_ids"] == [1]
    assert events[0]["data"]["response_token_ids"] == [2]


def test_proxy_error_triplet_preserves_status(client: TestClient, auth_headers: dict[str, str], monkeypatch):
    async def fake_upstream(*, client: httpx.AsyncClient, url: str, body: dict) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "maximum context length is 32768"}},
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr("agl_lite.server.proxy._send_upstream_with_retries", fake_upstream)
    rollout = _rollout(client, auth_headers)
    client.post(
        "/api/models",
        json=[{"model": MODEL_NAME, "endpoint": "http://model.test/v1", "version": 3}],
        headers=auth_headers,
    )

    proxied = client.post(
        f"/proxy/rollout/{rollout['rollout_id']}/attempt/0/mode/train/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=auth_headers,
    )
    assert proxied.status_code == 400

    events = client.get(
        f"/api/rollouts/{rollout['rollout_id']}/events",
        params={"event_type": "model_request", "format": "triplet"},
        headers=auth_headers,
    ).json()
    assert events[0]["data"]["prompt_token_ids"] == []
    assert events[0]["data"]["response_token_ids"] == []
    assert events[0]["data"]["http_status"] == 400
    assert events[0]["data"]["status"] == "error"
    assert events[0]["data"]["error"] == {"message": "maximum context length is 32768"}


def test_proxy_admin_endpoints(client: TestClient, auth_headers: dict[str, str]):
    paused = client.post(
        "/proxy/pause",
        json={"retry_after_seconds": 9, "reason": "scale-down"},
        headers=auth_headers,
    )
    assert paused.status_code == 200
    assert paused.json()["paused"] is True
    assert paused.json()["retry_after_seconds"] == 9

    state = client.get("/proxy/state", headers=auth_headers)
    assert state.status_code == 200
    assert state.json()["reason"] == "scale-down"

    resumed = client.post("/proxy/resume", headers=auth_headers)
    assert resumed.status_code == 200
    assert resumed.json()["paused"] is False

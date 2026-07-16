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


def test_delete_rollout(client: TestClient, auth_headers: dict[str, str]):
    rollout = _rollout(client, auth_headers)
    rollout_id = rollout["rollout_id"]
    client.post(
        f"/api/rollouts/{rollout_id}/attempt/0/events",
        json={"event_type": "reward", "data": {"value": 0.7}},
        headers=auth_headers,
    )

    deleted = client.delete(f"/api/rollouts/{rollout_id}", headers=auth_headers)
    assert deleted.status_code == 204

    assert client.get(f"/api/rollouts/{rollout_id}", headers=auth_headers).status_code == 404
    assert client.get(f"/api/rollouts/{rollout_id}/events", headers=auth_headers).status_code == 404

    # Idempotent: deleting a missing rollout is a no-op, not a 404.
    assert client.delete(f"/api/rollouts/{rollout_id}", headers=auth_headers).status_code == 204


def test_enqueue_with_client_rollout_id_is_idempotent(client: TestClient, auth_headers: dict[str, str]):
    rid = "fixed-rollout-id-123"
    created = client.post(
        "/api/rollouts",
        json=[{"rollout_id": rid, "input": {"prompt": "a"}}],
        headers=auth_headers,
    )
    assert created.status_code == 201
    assert created.json()[0]["rollout_id"] == rid

    client.post(
        f"/api/rollouts/{rid}/attempt/0/events",
        json={"event_type": "reward", "data": {"value": 0.5}},
        headers=auth_headers,
    )

    # Re-enqueue with the same id: returns the existing rollout, different input
    # ignored, and the previously recorded event is left intact.
    again = client.post(
        "/api/rollouts",
        json=[{"rollout_id": rid, "input": {"prompt": "DIFFERENT"}}],
        headers=auth_headers,
    )
    assert again.status_code == 201
    assert again.json()[0]["rollout_id"] == rid
    assert again.json()[0]["input"] == {"prompt": "a"}

    detail = client.get(f"/api/rollouts/{rid}", headers=auth_headers)
    assert detail.json()["attempts"] == ["0"]
    events = client.get(f"/api/rollouts/{rid}/events", headers=auth_headers)
    assert len(events.json()) == 1


def test_re_enqueue_does_not_clobber_a_running_rollout(client: TestClient, auth_headers: dict[str, str]):
    # Race: POST succeeds server-side and the task starts, but the client misses
    # the response and retries. The retry must return the already-running rollout
    # untouched, not reset its state/events back to a fresh QUEUING rollout.
    rid = "racing-rollout-id"
    client.post("/api/rollouts", json=[{"rollout_id": rid, "input": {"x": 1}}], headers=auth_headers)

    client.patch(f"/api/rollouts/{rid}", json={"status": {"state": "running"}}, headers=auth_headers)
    client.post(
        f"/api/rollouts/{rid}/attempt/0/events",
        json={"event_type": "reward", "data": {"value": 1.0}},
        headers=auth_headers,
    )

    again = client.post("/api/rollouts", json=[{"rollout_id": rid, "input": {"x": 1}}], headers=auth_headers)
    assert again.status_code == 201
    # State stayed RUNNING (not reset to queuing) and the event survived.
    assert again.json()[0]["status"]["state"] == "running"
    detail = client.get(f"/api/rollouts/{rid}", headers=auth_headers)
    assert detail.json()["rollout"]["status"]["state"] == "running"
    assert detail.json()["attempts"] == ["0"]


def test_triplet_events_keep_last_model_request_for_duplicate_prompt(client: TestClient, auth_headers: dict[str, str]):
    rollout = _rollout(client, auth_headers)
    rollout_id = rollout["rollout_id"]

    def post_model_request(prompt_token_ids: list[int], response_token_ids: list[int]) -> None:
        response = client.post(
            f"/api/rollouts/{rollout_id}/attempt/0/events",
            json={
                "event_type": "model_request",
                "data": {
                    "response": {
                        "prompt_token_ids": prompt_token_ids,
                        "choices": [{"token_ids": response_token_ids}],
                    },
                    "server": {"model": MODEL_NAME, "version": 3},
                },
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    post_model_request([1, 2], [10])
    client.post(
        f"/api/rollouts/{rollout_id}/attempt/0/events",
        json={"event_type": "reward", "data": {"value": 0.5}},
        headers=auth_headers,
    )
    post_model_request([3, 4], [20])
    post_model_request([1, 2], [30])

    raw_events = client.get(
        f"/api/rollouts/{rollout_id}/events",
        params={"event_type": "model_request"},
        headers=auth_headers,
    ).json()
    assert [event["data"]["response"]["choices"][0]["token_ids"] for event in raw_events] == [[10], [20], [30]]

    triplet_events = client.get(
        f"/api/rollouts/{rollout_id}/events",
        params={"format": "triplet"},
        headers=auth_headers,
    ).json()
    assert [event["event_type"] for event in triplet_events] == ["reward", "model_request", "model_request"]
    assert triplet_events[0]["data"] == {"value": 0.5}
    assert [event["data"]["prompt_token_ids"] for event in triplet_events[1:]] == [[3, 4], [1, 2]]
    assert [event["data"]["response_token_ids"] for event in triplet_events[1:]] == [[20], [30]]


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


def _terminal(client: TestClient, headers: dict[str, str], data_id: str, is_train: bool) -> str:
    """Create a rollout (with a data_id) and drive it queuing -> running -> succeeded."""
    created = client.post(
        "/api/rollouts",
        json=[{"input": {"data_id": data_id}, "is_train": is_train}],
        headers=headers,
    )
    assert created.status_code == 201
    rid = created.json()[0]["rollout_id"]
    assert client.patch(f"/api/rollouts/{rid}", json={"status": {"state": "running"}}, headers=headers).status_code == 200
    assert client.patch(f"/api/rollouts/{rid}", json={"status": {"state": "succeeded"}}, headers=headers).status_code == 200
    return rid


def test_terminal_rollouts_cursor_pagination(client: TestClient, auth_headers: dict[str, str]):
    # A rollout that never reaches a terminal state must NOT appear in the log.
    pending = client.post(
        "/api/rollouts", json=[{"input": {"data_id": "pending"}}], headers=auth_headers
    ).json()[0]["rollout_id"]

    # Complete three rollouts; the log is ordered by COMPLETION (append-on-terminal).
    rid_a = _terminal(client, auth_headers, "a", is_train=True)
    rid_b = _terminal(client, auth_headers, "b", is_train=False)
    rid_c = _terminal(client, auth_headers, "c", is_train=True)

    page1 = client.get("/api/rollouts/terminal", params={"after": 0, "limit": 2}, headers=auth_headers)
    assert page1.status_code == 200
    body1 = page1.json()
    assert body1["total_terminal"] == 3
    assert body1["next_after"] == 2
    assert [it["rollout_id"] for it in body1["items"]] == [rid_a, rid_b]
    assert body1["items"][0] == {"rollout_id": rid_a, "state": "succeeded", "data_id": "a", "is_train": True}
    assert body1["items"][1]["is_train"] is False  # projection carries is_train

    page2 = client.get("/api/rollouts/terminal", params={"after": 2, "limit": 2}, headers=auth_headers)
    body2 = page2.json()
    assert [it["rollout_id"] for it in body2["items"]] == [rid_c]
    assert body2["next_after"] == 3

    # Cursor caught up: no new items, cursor and total unchanged; pending never appears.
    page3 = client.get("/api/rollouts/terminal", params={"after": 3}, headers=auth_headers)
    assert page3.json() == {"items": [], "next_after": 3, "total_terminal": 3}
    assert pending not in {it["rollout_id"] for it in body1["items"] + body2["items"]}

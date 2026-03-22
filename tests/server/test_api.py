"""Tests for model server, resource, event, and archive API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _enqueue(client: TestClient, headers: dict, **kwargs) -> dict:
    kwargs.setdefault("input", {"prompt": "hello"})
    kwargs.setdefault("config", {"image": "agent:v1"})
    body = {"rollouts": [kwargs]}
    resp = client.post("/api/rollouts", json=body, headers=headers)
    assert resp.status_code == 201
    return resp.json()[0]


# --- Models ---


class TestModels:
    def test_register_and_list(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/models",
            json=[{"model": "qwen-7b", "endpoint": "http://vllm-0:8000/v1", "version": 3}],
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert len(resp.json()) == 1
        assert resp.json()[0]["model"] == "qwen-7b"

        resp = client.get("/api/models", headers=auth_headers)
        assert len(resp.json()) == 1

    def test_delete_model(self, client: TestClient, auth_headers: dict):
        client.post(
            "/api/models",
            json=[
                {"model": "qwen-7b", "endpoint": "http://vllm-0:8000/v1"},
                {"model": "qwen-7b", "endpoint": "http://vllm-1:8000/v1"},
            ],
            headers=auth_headers,
        )
        resp = client.delete("/api/models/qwen-7b", headers=auth_headers)
        assert resp.status_code == 200
        assert client.get("/api/models", headers=auth_headers).json() == []

    def test_delete_specific_endpoints(self, client: TestClient, auth_headers: dict):
        client.post(
            "/api/models",
            json=[
                {"model": "qwen-7b", "endpoint": "http://vllm-0:8000/v1"},
                {"model": "qwen-7b", "endpoint": "http://vllm-1:8000/v1"},
            ],
            headers=auth_headers,
        )
        resp = client.request(
            "DELETE",
            "/api/models/qwen-7b",
            json={"endpoints": ["http://vllm-0:8000/v1"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        models = client.get("/api/models", headers=auth_headers).json()
        assert len(models) == 1
        assert models[0]["endpoint"] == "http://vllm-1:8000/v1"

    def test_delete_all(self, client: TestClient, auth_headers: dict):
        client.post(
            "/api/models",
            json=[{"model": "qwen-7b", "endpoint": "http://vllm-0:8000/v1"}],
            headers=auth_headers,
        )
        resp = client.delete("/api/models", headers=auth_headers)
        assert resp.status_code == 200
        assert client.get("/api/models", headers=auth_headers).json() == []

    def test_delete_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.delete("/api/models/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


# --- Resources ---


class TestResources:
    def test_add_and_get(self, client: TestClient, auth_headers: dict):
        resp = client.post("/api/resources", json={"system_prompt": "Be helpful"}, headers=auth_headers)
        assert resp.status_code == 201
        res_id = resp.json()["resources_id"]

        resp = client.get(f"/api/resources/{res_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["resources"]["system_prompt"] == "Be helpful"

    def test_get_latest(self, client: TestClient, auth_headers: dict):
        client.post("/api/resources", json={"v": 1}, headers=auth_headers)
        client.post("/api/resources", json={"v": 2}, headers=auth_headers)
        resp = client.get("/api/resources/latest", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["resources"]["v"] == 2

    def test_get_latest_empty(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/resources/latest", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() is None

    def test_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/resources/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


# --- Events ---


class TestEvents:
    def test_query(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]

        # Post an event via gateway endpoint.
        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={"event_type": "reward", "data": {"value": 0.85}},
            headers=auth_headers,
        )
        assert resp.status_code == 200

        # Query events.
        resp = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth_headers)
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 1
        assert events[0]["event_type"] == "reward"

    def test_smart_resolution(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]
        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={"event_type": "model_request", "data": {"model": "gpt-4"}},
            headers=auth_headers,
        )
        # No attempt_id → smart resolution (latest).
        resp = client.get(f"/api/events?rollout_id={rid}", headers=auth_headers)
        assert len(resp.json()) == 1

    def test_rollout_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/events?rollout_id=nonexistent", headers=auth_headers)
        assert resp.status_code == 404

    def test_triplet_format_streaming(self, client: TestClient, auth_headers: dict):
        """format=triplet trims model_request to prompt/response token_ids."""
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]

        # Post a model_request event with streaming response (list of SSE chunks)
        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={
                "event_type": "model_request",
                "data": {
                    "request": {"model": "m", "messages": [{"role": "user", "content": "hi"}], "return_token_ids": True},
                    "response": [
                        {"choices": [{"delta": {"content": "Hello"}, "token_ids": [10, 20]}], "prompt_token_ids": [1, 2, 3]},
                        {"choices": [{"delta": {"content": " world"}, "token_ids": [30]}]},
                    ],
                    "server": {"model": "m", "version": 1, "endpoint": "http://vllm:8000/v1"},
                },
            },
            headers=auth_headers,
        )
        # Post a reward event
        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={"event_type": "reward", "data": {"value": 0.85, "message": "correct"}},
            headers=auth_headers,
        )

        # Query with format=triplet
        resp = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1&format=triplet", headers=auth_headers)
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 2

        mr = events[0]
        assert mr["event_type"] == "model_request"
        assert mr["data"]["prompt_token_ids"] == [1, 2, 3]
        assert mr["data"]["response_token_ids"] == [10, 20, 30]
        assert mr["data"]["server"] == {"model": "m", "version": 1}
        # Full request/response should be stripped
        assert "request" not in mr["data"]
        assert "response" not in mr["data"]
        assert "endpoint" not in mr["data"].get("server", {})

        rw = events[1]
        assert rw["event_type"] == "reward"
        assert rw["data"] == {"value": 0.85}
        # Message stripped
        assert "message" not in rw["data"]

    def test_triplet_format_non_streaming(self, client: TestClient, auth_headers: dict):
        """format=triplet works for non-streaming (dict) response."""
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]

        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={
                "event_type": "model_request",
                "data": {
                    "request": {"model": "m", "messages": []},
                    "response": {
                        "choices": [{"message": {"content": "hi"}, "token_ids": [100, 200]}],
                        "prompt_token_ids": [5, 6],
                    },
                    "server": {"model": "m", "version": 2, "endpoint": "http://vllm:8000/v1"},
                },
            },
            headers=auth_headers,
        )

        resp = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1&format=triplet", headers=auth_headers)
        events = resp.json()
        assert len(events) == 1
        assert events[0]["data"]["prompt_token_ids"] == [5, 6]
        assert events[0]["data"]["response_token_ids"] == [100, 200]

    def test_triplet_format_no_token_ids(self, client: TestClient, auth_headers: dict):
        """format=triplet returns empty lists when token_ids are absent."""
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]

        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={
                "event_type": "model_request",
                "data": {
                    "request": {"model": "m", "messages": []},
                    "response": [{"choices": [{"delta": {"content": "hello"}}]}],
                    "server": {"model": "m", "version": 1, "endpoint": "http://x"},
                },
            },
            headers=auth_headers,
        )

        resp = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1&format=triplet", headers=auth_headers)
        events = resp.json()
        assert events[0]["data"]["prompt_token_ids"] == []
        assert events[0]["data"]["response_token_ids"] == []

    def test_no_format_returns_full_events(self, client: TestClient, auth_headers: dict):
        """Without format=triplet, events are returned in full."""
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]

        client.post(
            f"/rollout/{rid}/attempt/pod-1/events",
            json={
                "event_type": "model_request",
                "data": {
                    "request": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    "response": [{"choices": [{"delta": {"content": "x"}}]}],
                    "server": {"model": "m", "version": 1, "endpoint": "http://x"},
                },
            },
            headers=auth_headers,
        )

        resp = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth_headers)
        events = resp.json()
        # Full data preserved
        assert "request" in events[0]["data"]
        assert "response" in events[0]["data"]
        assert events[0]["data"]["request"]["messages"] == [{"role": "user", "content": "hi"}]


# --- Archive ---


class TestArchive:
    def test_archive_purge(self, client: TestClient, auth_headers: dict, tmp_path):
        r = _enqueue(client, auth_headers)
        rid = r["rollout_id"]
        client.patch(f"/api/rollouts/{rid}", json={"status": "terminal_failed"}, headers=auth_headers)

        resp = client.post("/api/rollouts/archive", json={"rollout_ids": [rid]}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["archived"] == 1
        assert resp.json()["purged"] == 1

        # Purged — should be gone.
        resp = client.get(f"/api/rollouts/{rid}", headers=auth_headers)
        assert resp.status_code == 404

    def test_archive_non_terminal(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        resp = client.post("/api/rollouts/archive", json={"rollout_ids": [r["rollout_id"]]}, headers=auth_headers)
        assert resp.status_code == 409

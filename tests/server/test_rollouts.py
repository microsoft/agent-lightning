"""Tests for rollout API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _enqueue(client: TestClient, headers: dict, **kwargs) -> dict:
    """Helper: enqueue a single rollout."""
    kwargs.setdefault("input", {"prompt": "hello"})
    kwargs.setdefault("config", {"image": "agent:v1"})
    body = {"rollouts": [kwargs]}
    resp = client.post("/api/rollouts", json=body, headers=headers)
    assert resp.status_code == 201
    return resp.json()[0]


class TestEnqueueRollouts:
    def test_single(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/rollouts",
            json={"rollouts": [{"input": {"prompt": "hi"}, "config": {}}]},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "queuing"
        assert data[0]["input"]["prompt"] == "hi"

    def test_batch(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/rollouts",
            json={
                "rollouts": [
                    {"input": {"i": 0}, "config": {}},
                    {"input": {"i": 1}, "config": {}},
                    {"input": {"i": 2}, "config": {}},
                ]
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert len(resp.json()) == 3


class TestQueryRollouts:
    def test_empty(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/rollouts", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_ids(self, client: TestClient, auth_headers: dict):
        r1 = _enqueue(client, auth_headers)
        _enqueue(client, auth_headers)
        resp = client.get(f"/api/rollouts?ids={r1['rollout_id']}", headers=auth_headers)
        assert len(resp.json()) == 1

    def test_filter_by_status(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        client.patch(f"/api/rollouts/{r['rollout_id']}", json={"status": "running"}, headers=auth_headers)
        resp = client.get("/api/rollouts?status=running", headers=auth_headers)
        assert len(resp.json()) == 1

    def test_pagination(self, client: TestClient, auth_headers: dict):
        for _ in range(5):
            _enqueue(client, auth_headers)
        resp = client.get("/api/rollouts?limit=2&offset=0", headers=auth_headers)
        assert len(resp.json()) == 2


class TestGetRollout:
    def test_found(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        resp = client.get(f"/api/rollouts/{r['rollout_id']}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["rollout"]["rollout_id"] == r["rollout_id"]
        assert data["attempts"] == []

    def test_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/rollouts/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


class TestPatchRollout:
    def test_transition(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        resp = client.patch(f"/api/rollouts/{r['rollout_id']}", json={"status": "running"}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_invalid_transition(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        resp = client.patch(f"/api/rollouts/{r['rollout_id']}", json={"status": "succeeded"}, headers=auth_headers)
        assert resp.status_code == 409

    def test_not_found(self, client: TestClient, auth_headers: dict):
        resp = client.patch("/api/rollouts/nonexistent", json={"status": "running"}, headers=auth_headers)
        assert resp.status_code == 404


class TestCancelRollout:
    def test_cancel(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        resp = client.post(f"/api/rollouts/{r['rollout_id']}/cancel", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["cancel_requested"] is True

    def test_cancel_terminal(self, client: TestClient, auth_headers: dict):
        r = _enqueue(client, auth_headers)
        client.patch(f"/api/rollouts/{r['rollout_id']}", json={"status": "terminal_failed"}, headers=auth_headers)
        resp = client.post(f"/api/rollouts/{r['rollout_id']}/cancel", headers=auth_headers)
        assert resp.status_code == 409

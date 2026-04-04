"""Tests for auth and health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings


class TestHealthz:
    def test_no_auth_required(self, client: TestClient):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuth:
    def test_valid_bearer(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/rollouts", headers=auth_headers)
        assert resp.status_code == 200

    def test_valid_x_api_key(self, client: TestClient):
        resp = client.get("/api/rollouts", headers={"x-api-key": "test-secret-key"})
        assert resp.status_code == 200

    def test_missing_key(self, client: TestClient):
        resp = client.get("/api/rollouts")
        assert resp.status_code == 401

    def test_wrong_key(self, client: TestClient):
        resp = client.get("/api/rollouts", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_auth_disabled(self):
        """When AGL_KEY is empty, all requests pass."""
        settings = ServerSettings(key="")
        app = create_app(settings)
        with TestClient(app) as c:
            resp = c.get("/api/rollouts")
            assert resp.status_code == 200

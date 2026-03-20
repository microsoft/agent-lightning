"""Tests for gateway LLM proxy — end-to-end with mock model server."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings

AGL_KEY = "test-key"


@pytest.fixture
def settings(tmp_path) -> ServerSettings:
    # Write a gateway config with a route.
    config_file = tmp_path / "gateway.yaml"
    config_file.write_text("""
routes:
  gpt-4:
    model: qwen-7b
    params:
      add:
        temperature: 0.7
      drop:
        - frequency_penalty
""")
    return ServerSettings(agl_key=AGL_KEY, gateway_config=str(config_file))


@pytest.fixture
def app(settings: ServerSettings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGL_KEY}"}


def _enqueue(client: TestClient, auth: dict) -> str:
    """Enqueue a rollout and return its ID."""
    resp = client.post(
        "/api/rollouts",
        json={"rollouts": [{"input": {"prompt": "hello"}, "config": {"image": "agent:v1"}}]},
        headers=auth,
    )
    return resp.json()[0]["rollout_id"]


def _register_model(client: TestClient, auth: dict, model: str = "qwen-7b", endpoint: str = "http://mock:8000/v1"):
    """Register a model server."""
    client.post("/api/models", json=[{"model": model, "endpoint": endpoint}], headers=auth)


class TestProxyValidation:
    def test_rollout_not_found(self, client: TestClient, auth: dict):
        resp = client.post(
            "/rollout/nonexistent/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers=auth,
        )
        assert resp.status_code == 404

    def test_missing_model_field(self, client: TestClient, auth: dict):
        rid = _enqueue(client, auth)
        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"messages": []},
            headers=auth,
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"]

    def test_no_servers(self, client: TestClient, auth: dict):
        rid = _enqueue(client, auth)
        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers=auth,
        )
        assert resp.status_code == 503


class TestProxyNonStreaming:
    def test_forward_and_capture_event(self, client: TestClient, auth: dict, httpx_mock):
        """Non-streaming: forward request, capture event, return response."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        # Mock the model server response.
        mock_response = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "model": "qwen-7b",
        }
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            json=mock_response,
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers=auth,
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Hello!"

        # Verify event was captured.
        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "model_request"
        assert event["data"]["request"]["model"] == "gpt-4"  # original model
        assert event["data"]["server"]["model"] == "qwen-7b"
        assert event["data"]["server"]["endpoint"] == "http://mock:8000/v1"

    def test_route_rewrite_and_params(self, client: TestClient, auth: dict, httpx_mock):
        """Verify model rewrite and param adjustment in forwarded request."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        httpx_mock.add_response(url="http://mock:8000/v1/chat/completions", json={"ok": True})

        client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [], "frequency_penalty": 0.5},
            headers=auth,
        )

        # Check what was sent to the model server.
        request = httpx_mock.get_request()
        sent_body = json.loads(request.content)
        assert sent_body["model"] == "qwen-7b"  # rewritten
        assert sent_body["temperature"] == 0.7  # added
        assert "frequency_penalty" not in sent_body  # dropped

    def test_passthrough_no_route(self, client: TestClient, auth: dict, httpx_mock):
        """Model with no route config passes through unchanged."""
        rid = _enqueue(client, auth)
        _register_model(client, auth, model="llama-70b", endpoint="http://llama:8000/v1")

        httpx_mock.add_response(url="http://llama:8000/v1/chat/completions", json={"ok": True})

        client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "llama-70b", "messages": [], "top_p": 0.9},
            headers=auth,
        )

        request = httpx_mock.get_request()
        sent_body = json.loads(request.content)
        assert sent_body["model"] == "llama-70b"  # unchanged
        assert sent_body["top_p"] == 0.9  # unchanged

    def test_auth_required(self, client: TestClient):
        resp = client.post(
            "/rollout/xxx/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert resp.status_code == 401

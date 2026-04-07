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
  - model_in: gpt-4
    model_out: qwen-7b
    params:
      add:
        temperature: 0.7
      drop:
        - frequency_penalty
""")
    return ServerSettings(key=AGL_KEY, gateway_config=str(config_file))


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
        assert event["data"]["request"]["model"] == "qwen-7b"  # prepared (rewritten) model
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


class TestProxyStreaming:
    def _make_sse(self, chunks: list[dict]) -> bytes:
        """Build SSE byte stream from a list of JSON-serializable chunk dicts."""
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}\n\n")
        lines.append("data: [DONE]\n\n")
        return "".join(lines).encode()

    def test_streaming_forward_and_capture(self, client: TestClient, auth: dict, httpx_mock):
        """Streaming: tee chunks to client, buffer, capture assembled event."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_chunks = [
            {"id": "chatcmpl-1", "choices": [{"delta": {"role": "assistant"}}]},
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "!"}}]},
        ]
        sse_bytes = self._make_sse(sse_chunks)

        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200

        # Verify the SSE data was forwarded to the client.
        assert b"data: " in resp.content
        assert b"[DONE]" in resp.content

        # Verify event was captured with parsed SSE chunks.
        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "model_request"
        assert event["data"]["request"]["stream"] is True
        assert event["data"]["request"]["model"] == "qwen-7b"  # prepared (rewritten) model
        assert event["data"]["server"]["model"] == "qwen-7b"

        # Response is now assembled into a ChatCompletion-shaped dict (same shape as non-streaming).
        response_data = event["data"]["response"]
        assert isinstance(response_data, dict)
        assert response_data["id"] == "chatcmpl-1"
        assert response_data["object"] == "chat.completion"
        assert len(response_data["choices"]) == 1
        choice = response_data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello!"  # delta chunks concatenated

    def test_streaming_route_rewrite(self, client: TestClient, auth: dict, httpx_mock):
        """Verify model rewrite works for streaming requests too."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_bytes = self._make_sse([{"id": "1", "choices": []}])
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [], "stream": True},
            headers=auth,
        )

        # Check the forwarded request had model rewritten.
        request = httpx_mock.get_request()
        sent_body = json.loads(request.content)
        assert sent_body["model"] == "qwen-7b"

    def test_streaming_empty_response(self, client: TestClient, auth: dict, httpx_mock):
        """Edge case: stream with only [DONE] and no data chunks."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [], "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200

        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        assert events[0]["data"]["response"] == {}  # no chunks -> assembled but empty

    def test_streaming_non_chat_path_stores_raw_chunks(self, client: TestClient, auth: dict, httpx_mock):
        """Non-chat paths store raw parsed chunks, not an assembled ChatCompletion."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        # Legacy /v1/completions chunk: uses 'text', not 'delta'
        legacy_chunk = {"id": "cmpl-1", "choices": [{"text": " world", "finish_reason": None}]}
        sse_bytes = self._make_sse([legacy_chunk])
        httpx_mock.add_response(
            url="http://mock:8000/v1/completions",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/completions",
            json={"model": "gpt-4", "prompt": "hello", "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200

        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        response_data = events[0]["data"]["response"]
        # Raw chunks stored under 'chunks' key — not assembled
        assert "chunks" in response_data
        assert isinstance(response_data["chunks"], list)
        assert len(response_data["chunks"]) == 1
        assert response_data["chunks"][0]["choices"][0]["text"] == " world"

    def test_streaming_preserves_token_ids(self, client: TestClient, auth: dict, httpx_mock):
        """Streaming assembly preserves prompt_token_ids and per-choice token_ids."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {"role": "assistant", "content": ""}}],
                "prompt_token_ids": [10, 20, 30],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {"content": "Hi"}, "token_ids": [100, 200]}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {"content": "!"}, "token_ids": [300], "finish_reason": "stop"}],
            },
        ]
        sse_bytes = self._make_sse(sse_chunks)
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hey"}], "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200

        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        response_data = events[0]["data"]["response"]

        # prompt_token_ids from first chunk preserved at top level
        assert response_data["prompt_token_ids"] == [10, 20, 30]
        # token_ids concatenated across chunks into the choice
        assert response_data["choices"][0]["token_ids"] == [100, 200, 300]
        # text still assembled correctly
        assert response_data["choices"][0]["message"]["content"] == "Hi!"

    def test_streaming_token_ids_triplet_roundtrip(self, client: TestClient, auth: dict, httpx_mock):
        """Token IDs survive gateway assembly → triplet extraction."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_chunks = [
            {"id": "c1", "choices": [{"delta": {"role": "assistant", "content": ""}}],
             "prompt_token_ids": [5, 6, 7]},
            {"id": "c1", "choices": [{"delta": {"content": "ok"}, "token_ids": [50]}]},
        ]
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=self._make_sse(sse_chunks),
            headers={"content-type": "text/event-stream"},
        )

        client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
            json={"model": "gpt-4", "messages": [], "stream": True},
            headers=auth,
        )

        # Query with format=triplet
        resp = client.get(
            f"/api/events?rollout_id={rid}&attempt_id=pod-1&format=triplet",
            headers=auth,
        )
        events = resp.json()
        assert len(events) == 1
        data = events[0]["data"]
        assert data["prompt_token_ids"] == [5, 6, 7]
        assert data["response_token_ids"] == [50]

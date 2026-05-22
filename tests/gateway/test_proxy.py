"""Tests for gateway LLM proxy — end-to-end with mock model server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from agl_lite.gateway.proxy import GatewayPauseState, forward_request
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.model_server import ModelServer
from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings
from agl_lite.store.memory import InMemoryStore

AGL_KEY = "test-key"
ADMIN_KEY = "test-admin-key"


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
    return ServerSettings(key=AGL_KEY, admin_key=ADMIN_KEY, gateway_config=str(config_file))


@pytest.fixture
def app(settings: ServerSettings):
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {AGL_KEY}"}


@pytest.fixture
def admin_auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


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


class _AsyncIteratorByteStream(httpx.AsyncByteStream):
    def __init__(self, iterator):
        self._iterator = iterator

    async def __aiter__(self):
        async for chunk in self._iterator:
            yield chunk


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
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "model": "qwen-7b",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
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
        assert event["data"]["model"] == "qwen-7b"
        assert event["data"]["status"] == "ok"
        assert event["data"]["http_status"] == 200
        assert event["data"]["retry_count"] == 0
        assert event["data"]["latency_ms"] >= 0
        assert event["data"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        assert event["data"]["finish_reason"] == "stop"

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


class TestProxyRetry:
    def test_non_streaming_retries_retryable_status(self, client: TestClient, auth: dict, httpx_mock):
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        httpx_mock.add_response(url="http://mock:8000/v1/chat/completions", status_code=503, json={"error": "busy"})
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

        with patch("agl_lite.gateway.proxy._retry_delay_seconds", return_value=0):
            resp = client.post(
                f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
                headers=auth,
            )

        assert resp.status_code == 200
        assert len(httpx_mock.get_requests()) == 2
        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        assert events[0]["data"]["response"]["choices"][0]["message"]["content"] == "ok"
        assert events[0]["data"]["retry_count"] == 1
        assert events[0]["data"]["status"] == "ok"

    def test_non_streaming_does_not_retry_bad_request(self, client: TestClient, auth: dict, httpx_mock):
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        httpx_mock.add_response(url="http://mock:8000/v1/chat/completions", status_code=400, json={"error": "bad"})

        with patch("agl_lite.gateway.proxy._retry_delay_seconds", return_value=0):
            resp = client.post(
                f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
                headers=auth,
            )

        assert resp.status_code == 400
        assert len(httpx_mock.get_requests()) == 1

    def test_non_streaming_final_retryable_response_is_returned_once(self, client: TestClient, auth: dict, httpx_mock):
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        for _ in range(6):
            httpx_mock.add_response(
                url="http://mock:8000/v1/chat/completions",
                status_code=503,
                json={"error": "busy"},
            )

        with patch("agl_lite.gateway.proxy._retry_delay_seconds", return_value=0):
            resp = client.post(
                f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
                headers=auth,
            )

        assert resp.status_code == 503
        assert len(httpx_mock.get_requests()) == 6
        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        assert events[0]["data"]["response"] == {"error": "busy"}
        assert events[0]["data"]["retry_count"] == 5
        assert events[0]["data"]["status"] == "error"
        assert events[0]["data"]["http_status"] == 503

    def test_streaming_retries_initial_retryable_status(self, client: TestClient, auth: dict, httpx_mock):
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        httpx_mock.add_response(url="http://mock:8000/v1/chat/completions", status_code=503, json={"error": "busy"})
        sse_bytes = "".join(
            [
                'data: {"id":"c1","choices":[{"delta":{"role":"assistant","content":"ok"}}]}\n\n',
                "data: [DONE]\n\n",
            ]
        ).encode()
        httpx_mock.add_response(
            url="http://mock:8000/v1/chat/completions",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        with patch("agl_lite.gateway.proxy._retry_delay_seconds", return_value=0):
            resp = client.post(
                f"/rollout/{rid}/attempt/pod-1/v1/chat/completions",
                json={"model": "gpt-4", "messages": [], "stream": True},
                headers=auth,
            )

        assert resp.status_code == 200
        assert len(httpx_mock.get_requests()) == 2
        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        assert events[0]["data"]["response"]["choices"][0]["message"]["content"] == "ok"


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
        assert event["data"]["status"] == "ok"
        assert event["data"]["http_status"] == 200
        assert event["data"]["latency_ms"] >= 0

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

    def test_streaming_legacy_completions_assembled(self, client: TestClient, auth: dict, httpx_mock):
        """Legacy /v1/completions streaming is assembled into a Completion-shaped dict."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_chunks = [
            {
                "id": "cmpl-1",
                "created": 1700000000,
                "model": "qwen-7b",
                "choices": [{"text": "Hello", "index": 0, "finish_reason": None}],
            },
            {
                "id": "cmpl-1",
                "created": 1700000000,
                "model": "qwen-7b",
                "choices": [{"text": " world", "index": 0, "finish_reason": "stop"}],
            },
        ]
        sse_bytes = self._make_sse(sse_chunks)
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
        # Now assembled into a Completion-shaped dict
        assert response_data["id"] == "cmpl-1"
        assert response_data["object"] == "text_completion"
        assert len(response_data["choices"]) == 1
        assert response_data["choices"][0]["text"] == "Hello world"
        assert response_data["choices"][0]["finish_reason"] == "stop"

    def test_streaming_unknown_path_stores_raw_chunks(self, client: TestClient, auth: dict, httpx_mock):
        """Unknown paths store raw parsed chunks — no assembly attempted."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        raw_chunk = {"id": "x-1", "data": "something"}
        sse_bytes = self._make_sse([raw_chunk])
        httpx_mock.add_response(
            url="http://mock:8000/v1/some/custom/endpoint",
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        resp = client.post(
            f"/rollout/{rid}/attempt/pod-1/v1/some/custom/endpoint",
            json={"model": "gpt-4", "prompt": "hello", "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200

        events = client.get(f"/api/events?rollout_id={rid}&attempt_id=pod-1", headers=auth).json()
        assert len(events) == 1
        response_data = events[0]["data"]["response"]
        assert "chunks" in response_data
        assert len(response_data["chunks"]) == 1
        assert response_data["chunks"][0]["id"] == "x-1"

    def test_streaming_preserves_token_ids(self, client: TestClient, auth: dict, httpx_mock):
        """Streaming assembly preserves prompt_token_ids and per-choice token_ids.

        Mirrors real vLLM output: chunk[0] has prompt_token_ids at top level
        AND choices with role+content+token_ids; subsequent chunks have delta
        content + token_ids.
        """
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        # Realistic vLLM shape: chunk[0] carries prompt_token_ids and may
        # include content + token_ids in choices (e.g. when echo or first
        # token is generated alongside the role init).
        sse_chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {"role": "assistant", "content": "Hi"}, "token_ids": [100]}],
                "prompt_token_ids": [10, 20, 30],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {"content": " there"}, "token_ids": [200, 201]}],
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
        # token_ids from ALL chunks (including chunk[0]) concatenated into the choice
        assert response_data["choices"][0]["token_ids"] == [100, 200, 201, 300]
        # text from ALL chunks (including chunk[0]) assembled correctly
        assert response_data["choices"][0]["message"]["content"] == "Hi there!"

    def test_streaming_token_ids_triplet_roundtrip(self, client: TestClient, auth: dict, httpx_mock):
        """Token IDs survive gateway assembly → triplet extraction."""
        rid = _enqueue(client, auth)
        _register_model(client, auth)

        sse_chunks = [
            {"id": "c1", "choices": [{"delta": {"role": "assistant", "content": ""}}], "prompt_token_ids": [5, 6, 7]},
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


class TestGatewayPauseInflight:
    async def test_pause_counts_only_already_started_requests_and_blocks_new_ones(self):
        """Concurrent forward + pause invariant required by docs §3.1."""
        pause_state = GatewayPauseState()
        release_upstream = asyncio.Event()
        upstream_started = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_started
            upstream_started += 1
            await release_upstream.wait()
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            store = InMemoryStore()
            rollouts = store.enqueue_rollouts([
                EnqueueRolloutRequest(input={"i": i}) for i in range(4)
            ])
            server = ModelServer(
                model="qwen-7b",
                endpoint="http://mock:8000/v1",
                version=0,
                created_at=0.0,
            )

            tasks = [
                asyncio.create_task(
                    forward_request(
                        client=http_client,
                        server=server,
                        path="chat/completions",
                        body={"model": "qwen-7b", "messages": []},
                        store=store,
                        rollout_id=rollouts[i].rollout_id,
                        attempt_id="pod-1",
                        original_body={"model": "qwen-7b", "messages": []},
                        pause_state=pause_state,
                    )
                )
                for i in range(3)
            ]

            for _ in range(100):
                async with pause_state.lock:
                    if pause_state.inflight == 3:
                        break
                await asyncio.sleep(0.01)

            async with pause_state.lock:
                assert pause_state.inflight == 3
                pause_state.paused = True
                pause_state.retry_after_seconds = 7
                pause_state.reason = "unit-test"
                inflight_at_pause = pause_state.inflight

            blocked = await forward_request(
                client=http_client,
                server=server,
                path="chat/completions",
                body={"model": "qwen-7b", "messages": []},
                store=store,
                rollout_id=rollouts[3].rollout_id,
                attempt_id="pod-1",
                original_body={"model": "qwen-7b", "messages": []},
                pause_state=pause_state,
            )

            assert inflight_at_pause == 3
            assert blocked.status_code == 429
            assert blocked.headers["Retry-After"] == "7"
            assert blocked.headers["X-Agl-Paused"] == "true"
            assert upstream_started == 3
            async with pause_state.lock:
                assert pause_state.inflight == 3

            release_upstream.set()
            responses = await asyncio.gather(*tasks)
            assert [r.status_code for r in responses] == [200, 200, 200]
            async with pause_state.lock:
                assert pause_state.inflight == 0
                assert pause_state.drained.is_set()

    async def test_streaming_inflight_lives_until_stream_is_consumed(self):
        pause_state = GatewayPauseState()
        release_second_chunk = asyncio.Event()

        async def stream_body():
            yield b'data: {"id":"c1","choices":[{"delta":{"content":"a"}}]}\n\n'
            await release_second_chunk.wait()
            yield b'data: {"id":"c1","choices":[{"delta":{"content":"b"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_AsyncIteratorByteStream(stream_body()))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            store = InMemoryStore()
            [rollout] = store.enqueue_rollouts([EnqueueRolloutRequest(input={"stream": True})])
            server = ModelServer(
                model="qwen-7b",
                endpoint="http://mock:8000/v1",
                version=0,
                created_at=0.0,
            )
            response = await forward_request(
                client=http_client,
                server=server,
                path="chat/completions",
                body={"model": "qwen-7b", "messages": [], "stream": True},
                store=store,
                rollout_id=rollout.rollout_id,
                attempt_id="pod-1",
                original_body={"model": "qwen-7b", "messages": [], "stream": True},
                pause_state=pause_state,
            )

            async with pause_state.lock:
                assert pause_state.inflight == 1

            chunks = []
            streaming_response = cast(Any, response)
            iterator = streaming_response.body_iterator.__aiter__()
            chunks.append(await iterator.__anext__())
            async with pause_state.lock:
                assert pause_state.inflight == 1

            release_second_chunk.set()
            async for chunk in iterator:
                chunks.append(chunk)

            assert b"".join(chunks).endswith(b"data: [DONE]\n\n")
            async with pause_state.lock:
                assert pause_state.inflight == 0
                assert pause_state.drained.is_set()

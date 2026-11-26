# Copyright (c) Microsoft. All rights reserved.

"""Invalid upstream responses must remain observable gateway failures."""

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from .conftest import MODEL_NAME


@pytest.mark.parametrize(
    ("upstream_status", "content_type", "content", "expected_status"),
    [
        (200, "application/json", "{", 502),
        (200, "application/json", "[]", 502),
        (200, "application/json", "null", 502),
        (200, "application/json", '"unexpected"', 502),
        (200, "application/json", "42", 502),
        (200, "text/html", "<html>backend unavailable</html>", 502),
        (400, "application/json", "{", 400),
        (503, "text/html", "<html>backend unavailable</html>", 503),
    ],
)
def test_invalid_upstream_response_records_error(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    upstream_status: int,
    content_type: str,
    content: str,
    expected_status: int,
) -> None:
    async def fake_upstream(**kwargs: Any) -> httpx.Response:
        response = httpx.Response(upstream_status, text=content, headers={"content-type": content_type})
        response.extensions["agl_retry_count"] = 2
        return response

    monkeypatch.setattr("agentlightning.server.proxy._send_upstream_with_retries", fake_upstream)
    rollout_id = client.post("/api/rollouts", json=[{"input": "hello"}], headers=auth_headers).json()[0]["rollout_id"]
    client.post(
        "/api/models",
        json=[{"model": MODEL_NAME, "endpoint": "http://model.test/v1", "version": 3}],
        headers=auth_headers,
    )
    response = client.post(
        f"/proxy/rollout/{rollout_id}/attempt/0/mode/train/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=auth_headers,
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["type"] == "upstream_response_error"
    assert "<html>" not in response.text
    events = client.get(
        f"/api/rollouts/{rollout_id}/events", params={"event_type": "model_request"}, headers=auth_headers
    ).json()
    assert len(events) == 1
    data = events[0]["data"]
    assert data["http_status"] == expected_status
    assert data["status"] == "error"
    assert data["response"] == response.json()
    assert data["retry_count"] == 2
    assert client.get("/proxy/state", headers=auth_headers).json()["inflight"] == 0

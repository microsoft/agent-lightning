# Copyright (c) Microsoft. All rights reserved.

"""Retry policy coverage for AgentLightningSyncClient.post_with_retry."""

import httpx
import pytest

from agentlightning.client import AgentLightningSyncClient


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid sleeping through the retry backoff in unit tests."""
    monkeypatch.setattr("agentlightning.client.time.sleep", lambda _seconds: None)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 422])
def test_post_with_retry_does_not_retry_client_errors(status_code: int, no_retry_sleep: None) -> None:
    """Deterministic 4xx answers are final: retrying only delays the failure."""
    requests: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        requests.append(status_code)
        return httpx.Response(status_code, json={"detail": "boom"})

    with (
        AgentLightningSyncClient(key="test-key", max_retries=3, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.post_with_retry("https://example.test/api/rollouts", json=[])

    assert requests == [status_code]


def test_post_with_retry_retries_server_errors(no_retry_sleep: None) -> None:
    requests: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        requests.append(1)
        if len(requests) == 1:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(201, json=[])

    with AgentLightningSyncClient(key="test-key", max_retries=3, transport=httpx.MockTransport(handler)) as client:
        response = client.post_with_retry("https://example.test/api/rollouts", json=[])

    assert response.status_code == 201
    assert len(requests) == 2


def test_post_with_retry_retries_rate_limits(no_retry_sleep: None) -> None:
    requests: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        requests.append(1)
        if len(requests) == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(201, json=[])

    with AgentLightningSyncClient(key="test-key", max_retries=3, transport=httpx.MockTransport(handler)) as client:
        response = client.post_with_retry("https://example.test/api/rollouts", json=[])

    assert response.status_code == 201
    assert len(requests) == 2

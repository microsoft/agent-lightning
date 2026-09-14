# Copyright (c) Microsoft. All rights reserved.

"""Regression coverage for the clients' optional bearer key."""

import httpx
import pytest

from agentlightning.client import AgentLightningAsyncClient, AgentLightningSyncClient


@pytest.fixture(
    params=[
        {"Authorization": "Bearer old-test-key", "X-Custom": "preserved"},
        {"authorization": "Bearer old-test-key", "X-Custom": "preserved"},
        {"aUtHoRiZaTiOn": "Bearer old-test-key", "X-Custom": "preserved"},
        httpx.Headers({"Authorization": "Bearer old-test-key", "X-Custom": "preserved"}),
    ]
)
def headers(request: pytest.FixtureRequest) -> httpx.Headers | dict[str, str]:
    return request.param.copy()


def _assert_bearer(request: httpx.Request) -> httpx.Response:
    assert request.headers.get_list("Authorization") == ["Bearer new-test-key"]
    assert request.headers["X-Custom"] == "preserved"
    return httpx.Response(200)


def test_sync_key_replaces_authorization_case_insensitively(headers: httpx.Headers | dict[str, str]) -> None:
    original = dict(headers)
    with AgentLightningSyncClient(
        headers=headers, key="new-test-key", max_retries=0, transport=httpx.MockTransport(_assert_bearer)
    ) as client:
        assert client.get("https://example.test").status_code == 200
    assert dict(headers) == original


@pytest.mark.asyncio
async def test_async_key_replaces_authorization_case_insensitively(headers: httpx.Headers | dict[str, str]) -> None:
    original = dict(headers)
    async with AgentLightningAsyncClient(
        headers=headers, key="new-test-key", transport=httpx.MockTransport(_assert_bearer)
    ) as client:
        assert (await client.get("https://example.test")).status_code == 200
    assert dict(headers) == original


@pytest.mark.parametrize("key", [None, ""])
def test_sync_without_key_preserves_authorization(key: str | None) -> None:
    with AgentLightningSyncClient(headers={"authorization": "Bearer old-test-key"}, key=key) as client:
        assert client.headers.get_list("Authorization") == ["Bearer old-test-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, ""])
async def test_async_without_key_preserves_authorization(key: str | None) -> None:
    async with AgentLightningAsyncClient(headers={"authorization": "Bearer old-test-key"}, key=key) as client:
        assert client.headers.get_list("Authorization") == ["Bearer old-test-key"]

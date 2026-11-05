# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import httpx
import pytest

from agentlightning.server.store import _events


def test_process_proxy_coalesces_retries_and_retains_equal_prompt_actions(client, app, auth_headers):
    calls = []

    async def upstream(request):
        calls.append(request)
        assert "x-agl-request-id" not in request.headers
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1],
                "choices": [
                    {"token_ids": [3, 2], "finish_reason": "stop", "logprobs": {"token_logprobs": [-0.1, -0.2]}}
                ],
            },
        )

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client.post(
        "/api/models",
        headers=auth_headers,
        json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}],
    )
    client.post(
        "/api/rollouts",
        headers=auth_headers,
        json=[
            {
                "input": {},
                "rollout_id": "r",
                "metadata": {
                    "process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}
                },
            }
        ],
    )
    path = "/proxy/rollout/r/attempt/a/mode/train/openai/v1/chat/completions"
    headers = {**auth_headers, "X-AGL-Request-ID": "request-1", "X-AGL-Action-Index": "0"}
    first = client.post(path, headers=headers, json={"messages": [{"role": "user", "content": "same"}]})
    assert first.status_code == 200
    event_id = first.headers["X-AGL-Event-ID"]
    retry = client.post(path, headers=headers, json={"messages": [{"role": "user", "content": "same"}]})
    assert retry.json() == first.json()
    assert retry.headers["X-AGL-Event-ID"] == event_id
    assert len(calls) == 1
    assert client.post(path, headers=headers, json={"messages": []}).status_code == 409
    headers.update({"X-AGL-Request-ID": "request-2", "X-AGL-Action-Index": "1"})
    second = client.post(path, headers=headers, json={"messages": [{"role": "user", "content": "same"}]})
    assert second.status_code == 200
    assert second.headers["X-AGL-Event-ID"] != event_id
    assert len(calls) == 2
    assert len(_events["r"]["a"]) == 2
    client.patch("/api/rollouts/r", headers=auth_headers, json={"status": {"state": "running", "last_attempt_id": "a"}})
    view = client.get("/api/rollouts/r/events?format=triplet", headers=auth_headers).json()
    assert len(view) == 2
    assert all("_agl_credit" in event["data"] for event in view)


@pytest.mark.asyncio
async def test_duplicate_waiter_disconnect_keeps_original_capture(app, auth_headers):
    import asyncio

    from .test_proxy_disconnect import _serve

    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1],
                "choices": [
                    {"token_ids": [3, 2], "finish_reason": "stop", "logprobs": {"token_logprobs": [-0.1, -0.2]}}
                ],
            },
        )

    async with _serve(app) as url, httpx.AsyncClient(base_url=url, headers=auth_headers) as caller:
        app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        await caller.post(
            "/api/models", json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}]
        )
        await caller.post(
            "/api/rollouts",
            json=[
                {
                    "input": {},
                    "rollout_id": "r",
                    "metadata": {
                        "process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}
                    },
                }
            ],
        )
        path = "/proxy/rollout/r/attempt/a/mode/train/openai/v1/chat/completions"
        headers = {"X-AGL-Request-ID": "shared", "X-AGL-Action-Index": "0"}
        abandoned = asyncio.create_task(caller.post(path, json={"messages": []}, headers=headers))
        await asyncio.wait_for(started.wait(), 2)
        healthy = asyncio.create_task(caller.post(path, json={"messages": []}, headers=headers))
        await asyncio.sleep(0.05)
        abandoned.cancel()
        await asyncio.gather(abandoned, return_exceptions=True)
        release.set()
        response = await asyncio.wait_for(healthy, 2)
        assert response.status_code == 200
        assert response.headers["X-AGL-Event-ID"]
        assert calls == 1
        assert len(_events["r"]["a"]) == 1


def test_uncaptured_transport_failure_prevents_finalization(client, app, auth_headers):
    from uuid import UUID

    async def upstream(request):
        raise httpx.ReadError("reply lost after possible sampling")

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client.post(
        "/api/models",
        headers=auth_headers,
        json=[{"model": "test-model", "endpoint": "http://offline/v1", "version": 7}],
    )
    client.post(
        "/api/rollouts",
        headers=auth_headers,
        json=[
            {
                "input": {},
                "rollout_id": "r",
                "metadata": {
                    "process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}
                },
            }
        ],
    )
    response = client.post(
        "/proxy/rollout/r/attempt/a/mode/train/openai/v1/chat/completions",
        headers={**auth_headers, "X-AGL-Request-ID": "lost", "X-AGL-Action-Index": "0"},
        json={"messages": []},
    )
    assert response.status_code == 409
    assert "uncaptured" in response.text
    reward_id = UUID(int=2).hex
    client.post(
        "/api/rollouts/r/attempt/a/events",
        headers=auth_headers,
        json={
            "event_type": "reward",
            "data": {
                "value": 0,
                "source": "rule",
                "_agl_credit": {"event_id": reward_id, "kind": "terminal", "policy_version": 7},
            },
        },
    )
    assert (
        client.post(
            "/api/rollouts/r/attempt/a/credit/finalize",
            headers=auth_headers,
            json={
                "action_event_ids": [],
                "reward_event_ids": [reward_id],
                "terminal_reward_id": reward_id,
                "policy_version": 7,
            },
        ).status_code
        == 409
    )

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from agentlightning.credit.client import AsyncProcessCreditClient, ProcessCreditClient


def transport_calls():
    calls = []

    def handle(request):
        import json

        body = json.loads(request.content)
        calls.append((str(request.url.path), body, dict(request.headers)))
        if "/proxy/" in request.url.path:
            return httpx.Response(
                200,
                json={"choices": [{"token_ids": [3, 2]}]},
                headers={"X-AGL-Event-ID": UUID(int=20).hex, "X-AGL-Event-Sequence": "0"},
            )
        if request.url.path.endswith("/events"):
            data = body["data"]
            return httpx.Response(
                200, json={"event_type": "reward", "rollout_id": "r", "attempt_id": "a", "timestamp": 0, "data": data}
            )
        return httpx.Response(
            200,
            json={
                "version": 1,
                "rollout_id": "r",
                "attempt_id": "a",
                "policy_version": 7,
                "snapshot_hash": "snapshot",
                "watermark": 2,
                "action_count": 1,
                "reward_count": 2,
            },
        )

    return calls, httpx.MockTransport(handle)


def test_sync_helper_retains_complete_manifest_and_stable_retry_id():
    calls, transport = transport_calls()
    with httpx.Client(transport=transport, base_url="http://offline") as native:
        client = ProcessCreditClient(native, "r", "a", policy_version=7)
        first = client.complete({"messages": []}, request_id="stable")
        assert client.complete({"messages": []}, request_id="stable").event_id == first.event_id
        client.emit_process_reward(
            -0.2,
            reward_id=UUID(int=1).hex,
            target_event_id=first.event_id,
            evidence_through_event_id=first.event_id,
            source="rule",
        )
        client.emit_terminal_reward(1, reward_id=UUID(int=2).hex, source="terminal")
        receipt = client.finalize()
    assert receipt.action_count == 1
    manifest = calls[-1][1]
    assert manifest["action_event_ids"] == [UUID(int=20).hex]
    assert manifest["reward_event_ids"] == [UUID(int=1).hex, UUID(int=2).hex]
    assert calls[0][2]["x-agl-action-index"] == calls[1][2]["x-agl-action-index"] == "0"


@pytest.mark.asyncio
async def test_async_helper_uses_identical_protocol():
    calls, transport = transport_calls()
    async with httpx.AsyncClient(transport=transport, base_url="http://offline") as native:
        client = AsyncProcessCreditClient(native, "r", "a", policy_version=7)
        completion = await client.acomplete({"messages": []}, request_id="stable")
        await client.aemit_process_reward(
            -0.2,
            reward_id=UUID(int=1).hex,
            target_event_id=completion.event_id,
            evidence_through_event_id=completion.event_id,
            source="rule",
        )
        await client.aemit_terminal_reward(1, reward_id=UUID(int=2).hex, source="terminal")
        await client.afinalize()
    assert calls[-1][1]["action_event_ids"] == [UUID(int=20).hex]
    assert calls[-1][1]["policy_version"] == 7

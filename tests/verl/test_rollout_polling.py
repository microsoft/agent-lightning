# Copyright (c) Microsoft. All rights reserved.

"""Trainer batch polling through the server API."""

from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest
from fastapi.testclient import TestClient

from agentlightning.client import AgentLightningSyncClient
from agentlightning.hooks import RolloutHooks, TraceWriter
from agentlightning.schemas import Rollout, RolloutCreate
from agentlightning.verl.agl_rollout_manager import AglAsyncRolloutManager, AglRolloutManager, AglRolloutManagerBase
from tests.server.conftest import (
    AGL_KEY,
)
from tests.server.conftest import (
    app as app,
)
from tests.server.conftest import (
    auth_headers as auth_headers,
)
from tests.server.conftest import (
    clean_store as clean_store,
)
from tests.server.conftest import (
    client as client,
)
from tests.server.conftest import (
    server_config as server_config,
)


def test_status_reader_chunks_and_retries(client: TestClient, auth_headers: dict[str, str], monkeypatch):
    created = client.post("/api/rollouts", headers=auth_headers, json=[{"input": i} for i in range(257)]).json()
    ids = [item["rollout_id"] for item in created]
    requests: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/api/rollouts/status"
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(503)
        return client.post(request.url.path, headers=dict(request.headers), content=request.content)

    monkeypatch.setattr("agentlightning.client.time.sleep", lambda _: None)
    manager = AglRolloutManagerBase(agl_base_url="http://testserver", agl_key=AGL_KEY, model="test", step=0)
    manager.client.close()
    with AgentLightningSyncClient(
        base_url="http://testserver", key=AGL_KEY, max_retries=1, transport=httpx.MockTransport(handle)
    ) as manager.client:
        assert manager._get_rollout_statuses([]) == {}
        statuses = manager._get_rollout_statuses(ids)
        assert requests == [ids[:256], ids[:256], ids[256:]]
        assert {rid: status.model_dump(mode="json") for rid, status in statuses.items()} == {
            item["rollout_id"]: item["status"] for item in created
        }


@pytest.mark.parametrize("status_code", [401, 404, 405])
def test_status_reader_does_not_retry_permanent_errors(status_code: int, monkeypatch):
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code)

    monkeypatch.setattr("agentlightning.client.time.sleep", sleeps.append)
    manager = AglRolloutManagerBase(agl_base_url="http://testserver", agl_key=AGL_KEY, model="test", step=0)
    manager.client.close()
    with AgentLightningSyncClient(
        base_url="http://testserver", key=AGL_KEY, transport=httpx.MockTransport(handle)
    ) as manager.client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            manager._get_rollout_statuses(["missing"])
        assert error.value.response.status_code == status_code
        assert len(requests) == 1
        assert sleeps == []


class _RewardHooks(RolloutHooks):
    def __init__(self):
        self.called: list[str] = []

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        return request.model_copy(update={"metadata": {"marker": "from-hook"}})

    def on_succeeded(self, rollout: Rollout, events: dict, store: TraceWriter) -> None:
        self.called.append(rollout.rollout_id)
        assert rollout.input["prompt"] in {"first", "second"}
        store.add_event(rollout.rollout_id, "0", "reward", {"value": 1.0})

    def on_failed(self, rollout: Rollout, store: TraceWriter) -> None:
        self.called.append(rollout.rollout_id)
        store.add_event(rollout.rollout_id, "0", "reward", {"value": -1.0})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_polling_preserves_completion_hooks_timestamps_and_groups(
    client: TestClient,
    auth_headers: dict[str, str],
    asynchronous: bool,
):
    ids: list[str] = []
    batches: list[list[str]] = []
    details: list[str] = []
    running_at: dict[str, float] = {}
    finished_at: dict[str, float] = {}

    def transition(rid: str, state: str) -> None:
        response = client.patch(
            f"/api/rollouts/{rid}",
            headers=auth_headers,
            json={"status": {"state": state, "last_attempt_id": "0"}},
        )
        response.raise_for_status()
        timestamps = running_at if state == "running" else finished_at
        timestamps[rid] = response.json()["status"]["updated_at"]

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/rollouts/status":
            batches.append(json.loads(request.content))
            if len(batches) == 1:
                for rid in ids:
                    transition(rid, "running")
            elif len(batches) == 2:
                for rid, state in [(ids[0], "succeeded"), (ids[2], "succeeded"), (ids[3], "failed")]:
                    transition(rid, state)
            elif len(batches) == 3:
                transition(ids[1], "succeeded")
            else:
                pytest.fail("Unexpected extra polling sweep")
        elif request.method == "GET" and path.count("/") == 3:
            rid = path.rsplit("/", 1)[-1]
            assert rid in finished_at, "Polling fetched full details for an unfinished rollout"
            details.append(rid)
        response = client.request(
            request.method, str(request.url), headers=dict(request.headers), content=request.content
        )
        if path == "/api/rollouts" and request.method == "POST":
            ids.extend(item["rollout_id"] for item in response.json())
        return response

    hooks = _RewardHooks()
    manager_type = AglAsyncRolloutManager if asynchronous else AglRolloutManager
    manager = manager_type(
        agl_base_url="http://testserver",
        agl_key=AGL_KEY,
        model="test",
        step=7,
        train_rollout_n=2,
        poll_interval_seconds=0,
        hooks=hooks,
    )
    manager.client.close()
    with AgentLightningSyncClient(
        base_url="http://testserver", key=AGL_KEY, max_retries=0, transport=httpx.MockTransport(handle)
    ) as manager.client:
        data = {"prompt": ["first", "second"]}
        if isinstance(manager, AglAsyncRolloutManager):
            completed, carry_over = manager.enqueue_and_wait_until_group_completed(
                data,
                [],
                is_train=True,
                target_finished_group_num=1,
            )
            assert [item.rollout_id for item in completed] == ids[2:]
            assert [item.rollout_id for item in carry_over] == ids[:2]
            assert [item.running_at for item in carry_over] == [running_at[rid] for rid in ids[:2]]
            assert carry_over[0].finished_at == finished_at[ids[0]]
            assert carry_over[1].finished_at is None
            assert batches == [ids, ids]
            assert details == [ids[0], ids[2], ids[3]]
        else:
            completed = manager.enqueue_and_wait_until_completed(data, is_train=True)
            assert [item.rollout_id for item in completed] == [ids[0], ids[2], ids[3], ids[1]]
            assert batches == [ids, ids, [ids[1]]]
            assert details == [item.rollout_id for item in completed]
        assert Counter(hooks.called) == Counter(details)
        for item in completed:
            assert item.running_at == running_at[item.rollout_id]
            assert item.finished_at == finished_at[item.rollout_id]
            assert item.metadata["marker"] == "from-hook"
            assert item.final_reward == (-1.0 if item.rollout_id == ids[3] else 1.0)
            assert item.step == 7
            assert client.get(f"/api/rollouts/{item.rollout_id}", headers=auth_headers).status_code == 404
        if isinstance(manager, AglAsyncRolloutManager):
            for rid in ids[:2]:
                assert client.get(f"/api/rollouts/{rid}", headers=auth_headers).status_code == 200

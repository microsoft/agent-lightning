# Copyright (c) Microsoft. All rights reserved.

"""Completion hooks must not replay when unfinished groups cross training steps."""

from __future__ import annotations

from collections import Counter

import httpx
import pytest
from fastapi.testclient import TestClient

from agentlightning.client import AgentLightningSyncClient
from agentlightning.hooks import RolloutHooks, TraceWriter
from agentlightning.schemas import Rollout, RolloutCreate
from agentlightning.verl.agl_rollout_manager import AglAsyncRolloutManager, EnqueuedRollout
from tests.server.conftest import AGL_KEY
from tests.server.conftest import app as app
from tests.server.conftest import auth_headers as auth_headers
from tests.server.conftest import clean_store as clean_store
from tests.server.conftest import client as client
from tests.server.conftest import server_config as server_config


class _RewardHooks(RolloutHooks):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: Counter[str] = Counter()

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        return request.model_copy(update={"metadata": {"hook_marker": "kept"}})

    def _reward(self, rollout: Rollout, store: TraceWriter, value: float) -> None:
        self.calls[rollout.rollout_id] += 1
        assert rollout.input["prompt"].startswith(("slow", "fast"))
        assert rollout.metadata.model_dump()["hook_marker"] == "kept"
        if self.mode == "raise":
            raise RuntimeError("grading failed")
        store.add_event(rollout.rollout_id, "0", "reward", {"value": value})

    def on_succeeded(self, rollout: Rollout, events: dict, store: TraceWriter) -> None:
        self._reward(rollout, store, 1.0)

    def on_failed(self, rollout: Rollout, store: TraceWriter) -> None:
        self._reward(rollout, store, -1.0)


@pytest.mark.parametrize("early_state", ["succeeded", "failed"])
@pytest.mark.parametrize("hook_mode", ["reward", "raise", "post_failure", "lost_response", "disabled"])
def test_completion_hooks_do_not_replay_across_training_steps(
    client: TestClient,
    auth_headers: dict[str, str],
    early_state: str,
    hook_mode: str,
) -> None:
    ids: list[str] = []
    event_posts: Counter[str] = Counter()
    hooks = _RewardHooks(hook_mode)
    finished_at: dict[str, float] = {}

    def transition(rid: str, state: str) -> None:
        response = client.patch(
            f"/api/rollouts/{rid}",
            headers=auth_headers,
            json={"status": {"state": state, "last_attempt_id": "0"}},
        )
        response.raise_for_status()
        if state != "running":
            finished_at[rid] = response.json()["status"]["updated_at"]

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/events"):
            event_posts[path.split("/")[3]] += 1
            if hook_mode == "post_failure":
                return httpx.Response(503)
        response = client.request(
            request.method, str(request.url), headers=dict(request.headers), content=request.content
        )
        if request.method == "POST" and path.endswith("/events") and hook_mode == "lost_response":
            raise httpx.ReadError("Reward was stored but its response was lost", request=request)
        if request.method == "POST" and path == "/api/rollouts":
            new_ids = [item["rollout_id"] for item in response.json()]
            ids.extend(new_ids)
            for rid in new_ids:
                transition(rid, "running")
            # The first group's second sibling remains running for two steps.
            for rid in new_ids:
                if rid != ids[1]:
                    transition(rid, early_state if rid == ids[0] else "succeeded")
        return response

    def collect(
        step: int,
        data: dict[str, list[str]],
        carry_over: list[EnqueuedRollout],
    ):
        # The trainer constructs a new manager for each step. Round-trip the
        # carry-over metadata too, so no manager-local or object-identity cache helps.
        carried = [EnqueuedRollout.model_validate_json(item.model_dump_json()) for item in carry_over]
        manager = AglAsyncRolloutManager(
            agl_base_url="http://testserver",
            agl_key=AGL_KEY,
            model="test",
            step=step,
            train_rollout_n=2,
            poll_interval_seconds=0,
            hooks=None if hook_mode == "disabled" else hooks,
        )
        manager.client.close()
        with AgentLightningSyncClient(
            base_url="http://testserver", key=AGL_KEY, max_retries=0, transport=httpx.MockTransport(handle)
        ) as manager.client:
            return manager.enqueue_and_wait_until_group_completed(
                data, carried, is_train=True, target_finished_group_num=1
            )

    first, carry_over = collect(0, {"prompt": ["slow", "fast-1"]}, [])
    assert [item.rollout_id for item in first] == ids[2:4]
    assert [item.rollout_id for item in carry_over] == ids[:2]
    assert carry_over[0].finished_at == finished_at[ids[0]]
    assert carry_over[1].finished_at is None
    if hook_mode != "disabled":
        assert hooks.calls == Counter({rid: 1 for rid in [ids[0], *ids[2:4]]})

    second, carry_over = collect(1, {"prompt": ["fast-2"]}, carry_over)
    assert [item.rollout_id for item in second] == ids[4:6]
    assert [item.rollout_id for item in carry_over] == ids[:2]
    if hook_mode != "disabled":
        assert hooks.calls == Counter({rid: 1 for rid in [ids[0], *ids[2:]]})

    transition(ids[1], "succeeded")
    third, carry_over = collect(2, {}, carry_over)
    assert [item.rollout_id for item in third] == ids[:2]
    assert carry_over == []
    assert hooks.calls == (Counter() if hook_mode == "disabled" else Counter({rid: 1 for rid in ids}))
    assert event_posts == (
        Counter({rid: 1 for rid in ids}) if hook_mode in {"reward", "post_failure", "lost_response"} else Counter()
    )

    for completed in [*first, *second, *third]:
        expected_reward = -1.0 if completed.rollout_id == ids[0] and early_state == "failed" else 1.0
        has_reward = hook_mode in {"reward", "lost_response"}
        assert completed.final_reward == (expected_reward if has_reward else None)
        assert len(completed.events) == (1 if has_reward else 0)
        assert completed.finished_at == finished_at[completed.rollout_id]
        assert completed.step == (1 if completed.rollout_id in ids[4:] else 0)
        assert client.get(f"/api/rollouts/{completed.rollout_id}", headers=auth_headers).status_code == 404


def test_event_read_failure_before_hook_does_not_mark_it_processed(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    hooks = _RewardHooks("reward")
    fail_read = True

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal fail_read
        if fail_read and request.method == "GET" and request.url.path.endswith("/events"):
            fail_read = False
            return httpx.Response(503)
        return client.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)

    def manager_for_step(step: int) -> AglAsyncRolloutManager:
        manager = AglAsyncRolloutManager(
            agl_base_url="http://testserver", agl_key=AGL_KEY, model="test", step=step, train_rollout_n=2, hooks=hooks
        )
        manager.client.close()
        manager.client = AgentLightningSyncClient(
            base_url="http://testserver", key=AGL_KEY, max_retries=0, transport=httpx.MockTransport(handle)
        )
        return manager

    first_manager = manager_for_step(0)
    with first_manager.client:
        carry_over = first_manager._create_rollouts({"prompt": ["slow"]}, is_train=True)
        for item in carry_over:
            for state in ("running", "succeeded"):
                client.patch(
                    f"/api/rollouts/{item.rollout_id}",
                    headers=auth_headers,
                    json={"status": {"state": state, "last_attempt_id": "0"}},
                ).raise_for_status()
        with pytest.raises(httpx.HTTPStatusError):
            first_manager.enqueue_and_wait_until_group_completed(
                {}, carry_over, is_train=True, target_finished_group_num=1
            )
        assert hooks.calls == Counter()
        assert carry_over[0].finished_at is not None
        assert not carry_over[0].completion_hook_processed

    next_manager = manager_for_step(1)
    with next_manager.client:
        completed, remaining = next_manager.enqueue_and_wait_until_group_completed(
            {}, carry_over, is_train=True, target_finished_group_num=1
        )
    assert remaining == []
    assert hooks.calls == Counter({item.rollout_id: 1 for item in carry_over})
    assert [item.final_reward for item in completed] == [1.0, 1.0]
    assert [len(item.events) for item in completed] == [1, 1]

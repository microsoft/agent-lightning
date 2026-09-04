# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for local subprocess reconciliation."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller.local_reconciler import LocalReconciler, Proc
from agentlightning.schemas import Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutState


def _response(json: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=json,
        request=httpx.Request("GET", "http://server/api/rollouts"),
    )


def _reconciler(
    *, state: RolloutState = RolloutState.QUEUING, timeout_seconds: int = 3600
) -> tuple[LocalReconciler, AsyncMock]:
    rollout = Rollout(
        rollout_id="rollout-1",
        input={"question": "1 + 1"},
        config=RolloutConfig(timeout_seconds=timeout_seconds),
        status=RolloutLifecycleStatus(state=state, created_at=1.0, updated_at=1.0),
    )
    api = AsyncMock(spec=AgentLightningAsyncClient)
    api.get.return_value = _response([rollout.model_dump(mode="json")])
    api.patch.return_value = _response({})
    config = OmegaConf.create(
        {
            "runner_type": "local",
            "local_runner": {"maximum_size": 1, "poll_interval": 0.01},
        }
    )
    return LocalReconciler(api, config), api


@pytest.mark.asyncio
async def test_shutdown_does_not_spawn_queued_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, api = _reconciler()
    spawn_for = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_spawn_for", spawn_for)

    await reconciler._shutdown()

    spawn_for.assert_not_awaited()
    api.patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_reconcile_still_spawns_queued_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, _ = _reconciler()
    spawn_for = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_spawn_for", spawn_for)

    await reconciler._reconcile_once()

    spawn_for.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_does_not_spawn_after_stop_requested_during_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, api = _reconciler()
    spawn_for = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_spawn_for", spawn_for)
    get_started = asyncio.Event()
    release_get = asyncio.Event()
    response = api.get.return_value

    async def delayed_get(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        get_started.set()
        await release_get.wait()
        return response

    api.get.side_effect = delayed_get
    reconcile = asyncio.create_task(reconciler._reconcile_once())
    await get_started.wait()

    reconciler.stop()
    release_get.set()
    await reconcile

    spawn_for.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_still_fails_running_rollout_without_local_process() -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)

    await reconciler._shutdown()

    assert api.patch.await_args.kwargs["json"]["status"] == {
        "state": "failed",
        "error_message": "local subprocess is not running",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "expected_status"),
    [
        (0, {"state": "succeeded", "last_attempt_id": "attempt-1"}),
        (1, {"state": "failed", "error_message": "subprocess exited with code 1"}),
        (-9, {"state": "failed", "error_message": "subprocess exited with code -9"}),
    ],
)
async def test_reconcile_removes_completed_process_after_terminal_patch(
    returncode: int, expected_status: dict[str, str]
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = returncode
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic())

    await reconciler._reconcile_once()

    assert "rollout-1" not in reconciler._rid_to_proc
    assert api.patch.await_args.kwargs["json"]["status"] == expected_status


@pytest.mark.asyncio
async def test_reconcile_keeps_completed_process_when_terminal_patch_fails_then_retries() -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = 0
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic())
    api.patch.side_effect = [
        httpx.Response(500, request=httpx.Request("PATCH", "http://server/api/rollouts/rollout-1")),
        _response({}),
    ]

    await reconciler._reconcile_once()

    assert "rollout-1" in reconciler._rid_to_proc

    await reconciler._reconcile_once()

    assert "rollout-1" not in reconciler._rid_to_proc
    assert api.patch.await_count == 2


@pytest.mark.asyncio
async def test_reconcile_keeps_running_process() -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic())

    await reconciler._reconcile_once()

    assert reconciler._rid_to_proc["rollout-1"].proc is proc
    api.patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_removes_timed_out_process_after_failure_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING, timeout_seconds=1)
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic() - 2)
    kill_process_group = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_kill_process_group", kill_process_group)

    await reconciler._reconcile_once()

    assert "rollout-1" not in reconciler._rid_to_proc
    kill_process_group.assert_awaited_once()
    assert api.patch.await_args.kwargs["json"]["status"] == {
        "state": "failed",
        "error_message": "local subprocess timed out",
    }


@pytest.mark.asyncio
async def test_reconcile_keeps_timed_out_process_when_failure_patch_fails_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING, timeout_seconds=1)
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic() - 2)
    async def kill_process_group(_: str, item: Proc) -> bool:
        item.killed = True
        proc.returncode = -9
        return True

    monkeypatch.setattr(reconciler, "_kill_process_group", kill_process_group)
    api.patch.side_effect = [
        httpx.Response(500, request=httpx.Request("PATCH", "http://server/api/rollouts/rollout-1")),
        _response({}),
    ]

    await reconciler._reconcile_once()

    assert "rollout-1" in reconciler._rid_to_proc

    await reconciler._reconcile_once()

    assert "rollout-1" not in reconciler._rid_to_proc
    assert [call.kwargs["json"]["status"] for call in api.patch.await_args_list] == [
        {"state": "failed", "error_message": "local subprocess timed out"},
        {"state": "failed", "error_message": "local subprocess timed out"},
    ]


@pytest.mark.asyncio
async def test_reconcile_finishes_queued_completed_process_after_running_transition() -> None:
    reconciler, api = _reconciler()
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = 0
    reconciler._rid_to_proc["rollout-1"] = Proc("attempt-1", proc, spawned_at=time.monotonic())

    await reconciler._reconcile_once()

    assert "rollout-1" not in reconciler._rid_to_proc
    assert [call.kwargs["json"]["status"] for call in api.patch.await_args_list] == [
        {"state": "running", "last_attempt_id": "attempt-1"},
        {"state": "succeeded", "last_attempt_id": "attempt-1"},
    ]

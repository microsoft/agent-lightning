# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for local subprocess reconciliation."""

import asyncio
import signal
import time
from typing import cast
from unittest.mock import AsyncMock, Mock

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
    *,
    state: RolloutState = RolloutState.QUEUING,
    timeout_seconds: int = 3600,
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


class _Process:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.pid = 1234
        self.wait = AsyncMock()


def _proc(*, returncode: int | None, killed: bool = False, attempt_id: str = "attempt-1") -> tuple[Proc, _Process]:
    process = _Process(returncode)
    return (
        Proc(
            attempt_id=attempt_id,
            proc=cast(asyncio.subprocess.Process, process),
            spawned_at=time.monotonic(),
            killed=killed,
        ),
        process,
    )


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
@pytest.mark.parametrize("failed_terminal_patches", [0, 1, 2])
@pytest.mark.parametrize("returncode", [-9, 0])
async def test_timeout_reconciliation_retries_and_retains_process_record(
    monkeypatch: pytest.MonkeyPatch,
    failed_terminal_patches: int,
    returncode: int,
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING, timeout_seconds=1)
    rollout = Rollout.model_validate(api.get.return_value.json()[0])
    item, process = _proc(returncode=None)
    reconciler._rid_to_proc[rollout.rollout_id] = item
    item.spawned_at = 0.0
    api.patch.side_effect = [
        *[httpx.ConnectError("server unavailable") for _ in range(failed_terminal_patches)],
        _response({}),
    ]
    killpg = Mock()
    monkeypatch.setattr("agentlightning.controller.local_reconciler.os.killpg", killpg)

    async def complete_wait() -> None:
        process.returncode = returncode

    process.wait.side_effect = complete_wait
    spawn_for = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_spawn_for", spawn_for)

    for _ in range(failed_terminal_patches + 1):
        await reconciler._reconcile_once()

    assert item.killed
    assert reconciler._rid_to_proc[rollout.rollout_id] is item
    assert item.attempt_id == "attempt-1"
    process.wait.assert_awaited_once_with()
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)
    spawn_for.assert_not_awaited()
    assert [call.kwargs["json"]["status"] for call in api.patch.await_args_list] == [
        {"state": "failed", "error_message": "local subprocess timed out"}
    ] * (failed_terminal_patches + 1)

    api.get.return_value = _response([])
    await reconciler._reconcile_once()
    assert reconciler._rid_to_proc[rollout.rollout_id] is item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (0, {"state": "succeeded", "last_attempt_id": "attempt-1"}),
        (1, {"state": "failed", "error_message": "subprocess exited with code 1"}),
        (-9, {"state": "failed", "error_message": "subprocess exited with code -9"}),
    ],
)
@pytest.mark.parametrize("failed_terminal_patches", [0, 1])
async def test_unmarked_terminal_process_retries_and_retains_record(
    returncode: int,
    expected: dict[str, str],
    failed_terminal_patches: int,
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)
    rollout = Rollout.model_validate(api.get.return_value.json()[0])
    item, _ = _proc(returncode=returncode)
    reconciler._rid_to_proc[rollout.rollout_id] = item
    api.patch.side_effect = [
        *[httpx.ConnectError("server unavailable") for _ in range(failed_terminal_patches)],
        _response({}),
    ]

    for _ in range(failed_terminal_patches + 1):
        await reconciler._reconcile_once()

    assert reconciler._rid_to_proc[rollout.rollout_id] is item
    assert [call.kwargs["json"]["status"] for call in api.patch.await_args_list] == [expected] * (
        failed_terminal_patches + 1
    )
    api.get.return_value = _response([])
    await reconciler._reconcile_once()
    assert reconciler._rid_to_proc[rollout.rollout_id] is item


@pytest.mark.asyncio
@pytest.mark.parametrize("patch_fails", [False, True])
async def test_run_stops_then_shutdown_kills_after_final_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    patch_fails: bool,
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING)
    item, process = _proc(returncode=None)
    reconciler._rid_to_proc["rollout-1"] = item
    events: list[str] = []

    async def get(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        events.append("reconcile")
        reconciler.stop()
        return _response([Rollout.model_validate(api.get.return_value.json()[0]).model_dump(mode="json")])

    async def complete_wait() -> None:
        process.returncode = -9

    process.wait.side_effect = complete_wait
    killpg = Mock(side_effect=lambda *args: events.append("kill"))
    monkeypatch.setattr("agentlightning.controller.local_reconciler.os.killpg", killpg)
    api.get.side_effect = get
    if patch_fails:
        api.patch.side_effect = httpx.ConnectError("server unavailable")

    await reconciler.run()

    assert events == ["reconcile", "reconcile", "kill"]
    assert item.killed
    assert reconciler._rid_to_proc["rollout-1"] is item
    assert api.patch.await_args.kwargs["json"]["status"] == {
        "state": "failed",
        "error_message": "local controller shutdown",
    }


@pytest.mark.asyncio
async def test_timeout_during_run_retries_original_reason_during_shutdown_without_rekill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler, api = _reconciler(state=RolloutState.RUNNING, timeout_seconds=1)
    rollout = Rollout.model_validate(api.get.return_value.json()[0])
    item, process = _proc(returncode=None)
    item.spawned_at = 0.0
    reconciler._rid_to_proc[rollout.rollout_id] = item
    events: list[str] = []

    async def get(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        events.append("reconcile")
        reconciler.stop()
        return _response([rollout.model_dump(mode="json")])

    async def complete_wait() -> None:
        process.returncode = -9

    process.wait.side_effect = complete_wait
    killpg = Mock(side_effect=lambda *args: events.append("kill"))
    monkeypatch.setattr("agentlightning.controller.local_reconciler.os.killpg", killpg)
    api.get.side_effect = get
    api.patch.side_effect = [httpx.ConnectError("server unavailable"), _response({})]

    await reconciler.run()

    assert events == ["reconcile", "kill", "reconcile"]
    assert reconciler._rid_to_proc[rollout.rollout_id] is item
    process.wait.assert_awaited_once_with()
    assert [call.kwargs["json"]["status"] for call in api.patch.await_args_list] == [
        {"state": "failed", "error_message": "local subprocess timed out"},
        {"state": "failed", "error_message": "local subprocess timed out"},
    ]

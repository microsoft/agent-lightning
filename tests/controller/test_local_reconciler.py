# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for local subprocess reconciliation."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller.local_reconciler import LocalReconciler
from agentlightning.schemas import Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutState


def _response(json: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=json,
        request=httpx.Request("GET", "http://server/api/rollouts"),
    )


def _reconciler(*, state: RolloutState = RolloutState.QUEUING) -> tuple[LocalReconciler, AsyncMock]:
    rollout = Rollout(
        rollout_id="rollout-1",
        input={"question": "1 + 1"},
        config=RolloutConfig(),
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

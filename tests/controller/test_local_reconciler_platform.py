# Copyright (c) Microsoft. All rights reserved.

"""Platform support tests for the local reconciler."""

from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller import local_reconciler
from agentlightning.controller.local_reconciler import LocalReconciler


@pytest.mark.asyncio
async def test_run_fails_before_reconciling_or_spawning_on_native_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = AsyncMock(spec=AgentLightningAsyncClient)
    config = OmegaConf.create(
        {
            "runner_type": "local",
            "local_runner": {"maximum_size": 1, "poll_interval": 0.01},
        }
    )
    reconciler = LocalReconciler(api, config)
    reconcile_loop = AsyncMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(reconciler, "_reconcile_loop", reconcile_loop)
    monkeypatch.setattr(reconciler, "_shutdown", shutdown)
    monkeypatch.setattr(local_reconciler, "_is_native_windows", lambda: True)

    with pytest.raises(RuntimeError, match="runner_type=local is not supported on native Windows"):
        await reconciler.run()

    reconcile_loop.assert_not_awaited()
    shutdown.assert_not_awaited()
    api.get.assert_not_awaited()
    api.patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reconciles_and_shuts_down_on_supported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    api = AsyncMock(spec=AgentLightningAsyncClient)
    config = OmegaConf.create(
        {
            "runner_type": "local",
            "local_runner": {"maximum_size": 1, "poll_interval": 0.01},
        }
    )
    reconciler = LocalReconciler(api, config)
    reconcile_loop = AsyncMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(reconciler, "_reconcile_loop", reconcile_loop)
    monkeypatch.setattr(reconciler, "_shutdown", shutdown)
    monkeypatch.setattr(local_reconciler, "_is_native_windows", lambda: False)

    await reconciler.run()

    reconcile_loop.assert_awaited_once_with()
    shutdown.assert_awaited_once_with()

# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for local reconciler worker environment variables."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller.local_reconciler import LocalReconciler
from agentlightning.schemas import Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutLocalConfig


@pytest.mark.parametrize(
    ("agent_url", "expected_base_url"),
    [
        ("http://agent-gateway:8080/", "http://agent-gateway:8080"),
        (None, "http://controller:8080"),
    ],
)
@pytest.mark.asyncio
async def test_spawn_uses_agent_url_for_worker_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    agent_url: str | None,
    expected_base_url: str,
) -> None:
    api = AsyncMock(spec=AgentLightningAsyncClient)
    config = OmegaConf.create(
        {
            "runner_type": "local",
            "agl_server": {
                "url": "http://controller:8080",
                "agent_url": agent_url,
                "key": "secret",
            },
            "local_runner": {"maximum_size": 1, "poll_interval": 0.01},
        }
    )
    rollout = Rollout(
        rollout_id="rollout-1",
        input={"question": "1 + 1"},
        config=RolloutConfig(local=RolloutLocalConfig(agent_class="example.Agent")),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )
    reconciler = LocalReconciler(api, config)
    patch = AsyncMock(return_value=True)
    monkeypatch.setattr(reconciler, "_patch", patch)
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.pid = 123
    proc.returncode = None
    create_subprocess = AsyncMock(return_value=proc)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    assert await reconciler._spawn_for(rollout)

    spawn_call = create_subprocess.await_args
    assert spawn_call is not None
    env = spawn_call.kwargs["env"]
    assert env["AGL_KEY"] == "secret"
    assert env["AGL_OPENAI_BASE_URL"] == (f"{expected_base_url}/proxy/rollout/rollout-1/attempt/0/mode/train/openai/v1")
    assert env["AGL_EVENT_URL"] == f"{expected_base_url}/api/rollouts/rollout-1/attempt/0/events"
    patch.assert_awaited_once()

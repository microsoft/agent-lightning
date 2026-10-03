# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Python entrypoint that launches the TypeScript agent."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from examples.ai_sdk import ai_sdk_agent


@pytest.mark.asyncio
async def test_agent_requires_compiled_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ai_sdk_agent, "_ENTRYPOINT", tmp_path / "missing.js")

    with pytest.raises(FileNotFoundError, match="npm ci && npm run build"):
        await ai_sdk_agent.AISDKAgent().run()


@pytest.mark.asyncio
async def test_agent_propagates_node_failure_without_a_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entrypoint = tmp_path / "agent.js"
    entrypoint.write_text("", encoding="utf-8")
    monkeypatch.setattr(ai_sdk_agent, "_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(ai_sdk_agent.shutil, "which", lambda executable: "/test/node")
    monkeypatch.setenv("WRAPPER_TEST_MARKER", "present")
    process = AsyncMock()
    process.wait.return_value = 7
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(ai_sdk_agent.asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(RuntimeError, match="status 7"):
        await ai_sdk_agent.AISDKAgent().run()

    args = create_process.await_args
    assert args is not None
    assert args.args == ("/test/node", str(entrypoint))
    assert args.kwargs["env"]["WRAPPER_TEST_MARKER"] == "present"


class _CancellableProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.wait_started = asyncio.Event()
        self.exited = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        self.wait_started.set()
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.exited.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.exited.set()


@pytest.mark.asyncio
async def test_agent_terminates_node_process_when_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entrypoint = tmp_path / "agent.js"
    entrypoint.write_text("", encoding="utf-8")
    monkeypatch.setattr(ai_sdk_agent, "_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(ai_sdk_agent.shutil, "which", lambda executable: "/test/node")
    process = _CancellableProcess()
    monkeypatch.setattr(ai_sdk_agent.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    task = asyncio.create_task(ai_sdk_agent.AISDKAgent().run())
    await process.wait_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated
    assert not process.killed

# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the local reconciler worker entry point."""

import asyncio
import importlib
from types import ModuleType

import pytest

from agentlightning.controller.local_reconciler import _run_local_reconciler_worker


def _run(monkeypatch: pytest.MonkeyPatch, agent_class: type) -> int:
    module = ModuleType("_test_agent_module")
    module.Agent = agent_class
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    return _run_local_reconciler_worker("_test_agent_module:Agent")


def test_sync_result_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        def run(self) -> str:
            return "sync-done"

    assert _run(monkeypatch, Agent) == 0


def test_coroutine_result_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        async def run(self) -> str:
            return "coro-done"

    assert _run(monkeypatch, Agent) == 0


def test_awaitable_result_is_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    completed: list[bool] = []

    class Deferred:
        def __await__(self):
            async def complete() -> None:
                await asyncio.sleep(0)
                completed.append(True)

            return complete().__await__()

    class Agent:
        def run(self) -> Deferred:
            return Deferred()

    assert _run(monkeypatch, Agent) == 0
    assert completed == [True]


def test_raising_agent_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        def run(self) -> str:
            raise RuntimeError("boom")

    assert _run(monkeypatch, Agent) == 1

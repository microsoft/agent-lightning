# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for AgentModeDaemon completed-rollout bookkeeping (#355)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock

import pytest

from agentlightning.types import RolloutLegacy, Triplet


def _load_daemon_module():
    """Load daemon.py without importing agentlightning.verl package __init__."""
    if "verl" not in sys.modules:
        verl_stub = types.ModuleType("verl")

        class DataProto:
            pass

        verl_stub.DataProto = DataProto
        sys.modules["verl"] = verl_stub

    daemon_path = Path(__file__).resolve().parents[2] / "agentlightning" / "verl" / "daemon.py"
    spec = importlib.util.spec_from_file_location("agentlightning_verl_daemon_under_test", daemon_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_daemon_module = _load_daemon_module()
AgentModeDaemon = _daemon_module.AgentModeDaemon


def _bare_daemon(*, mode: str = "v1") -> AgentModeDaemon:
    """Build a daemon instance without starting servers or proxy threads."""
    daemon = object.__new__(AgentModeDaemon)
    daemon.mode = mode
    daemon._total_tasks_queued = 0
    daemon._completed_rollouts_v0 = {}
    daemon._task_id_to_original_sample = {}
    daemon.is_train = True
    return daemon


def _failed(rollout_id: str) -> RolloutLegacy:
    return RolloutLegacy(rollout_id=rollout_id, final_reward=None, triplets=None)


def _success(rollout_id: str, reward: float = 1.0) -> RolloutLegacy:
    return RolloutLegacy(
        rollout_id=rollout_id,
        final_reward=reward,
        triplets=[Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2]})],
    )


def test_is_failed_rollout_detects_empty_backend_failures() -> None:
    assert AgentModeDaemon._is_failed_rollout(_failed("r1")) is True
    assert AgentModeDaemon._is_failed_rollout(RolloutLegacy(rollout_id="r1", final_reward=None, triplets=[])) is True
    assert AgentModeDaemon._is_failed_rollout(_success("r1")) is False
    assert (
        AgentModeDaemon._is_failed_rollout(RolloutLegacy(rollout_id="r1", final_reward=0.0, triplets=None)) is False
    )


def test_drop_stale_completed_rollouts_removes_unregistered_ids() -> None:
    daemon = _bare_daemon()
    daemon._task_id_to_original_sample = {"r1": {"data_id": "d1"}}
    daemon._total_tasks_queued = 1
    daemon._completed_rollouts_v0 = {
        "r1": _success("r1"),
        "stale-from-previous-batch": _success("stale-from-previous-batch"),
    }

    dropped = daemon._drop_stale_completed_rollouts()

    assert dropped == 1
    assert list(daemon._completed_rollouts_v0.keys()) == ["r1"]


def test_ensure_completed_matches_queued_drops_extras_without_asserting() -> None:
    daemon = _bare_daemon()
    daemon._task_id_to_original_sample = {"r1": {"data_id": "d1"}, "r2": {"data_id": "d2"}}
    daemon._total_tasks_queued = 2
    daemon._completed_rollouts_v0 = {
        "r1": _success("r1"),
        "r2": _success("r2"),
        "extra": _success("extra"),
    }

    daemon._ensure_completed_matches_queued()

    assert set(daemon._completed_rollouts_v0) == {"r1", "r2"}


@pytest.mark.asyncio
async def test_async_run_until_finished_rejects_unknown_and_overwrites_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _bare_daemon(mode="v1")
    daemon._task_id_to_original_sample = {"r1": {"data_id": "d1"}, "r2": {"data_id": "d2"}}
    daemon._total_tasks_queued = 2

    polls: List[List[RolloutLegacy]] = [
        # First poll: one failure, one unknown stale id from a previous batch.
        [_failed("r1"), _success("stale-old")],
        # Second poll: successful retry for r1 and a success for r2.
        [_success("r1", reward=0.5), _success("r2")],
    ]

    async def fake_wait_for_rollouts(*, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutLegacy]:
        del timeout
        assert set(rollout_ids).issubset({"r1", "r2"})
        if not polls:
            return []
        return polls.pop(0)

    daemon.store = AsyncMock()
    daemon.store.wait_for_rollouts = fake_wait_for_rollouts
    daemon._validate_data = lambda rollout: None  # type: ignore[method-assign]

    async def immediate_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)

    await daemon._async_run_until_finished(verbose=False)

    assert set(daemon._completed_rollouts_v0) == {"r1", "r2"}
    assert daemon._completed_rollouts_v0["r1"].final_reward == 0.5
    assert AgentModeDaemon._is_failed_rollout(daemon._completed_rollouts_v0["r1"]) is False


@pytest.mark.asyncio
async def test_async_run_until_finished_grace_polls_when_only_failure_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every queued ID has a result but it failed, keep polling for retries.

    Unlike the mixed missing+failed case, this hits the grace-init branch with
    ``missing == []`` so a successful retry on the next poll can replace it.
    """
    daemon = _bare_daemon(mode="v1")
    daemon._task_id_to_original_sample = {"r1": {"data_id": "d1"}}
    daemon._total_tasks_queued = 1

    polls: List[List[RolloutLegacy]] = [
        [_failed("r1")],
        [_success("r1", reward=0.7)],
    ]

    async def fake_wait_for_rollouts(*, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutLegacy]:
        del timeout
        assert set(rollout_ids) == {"r1"}
        if not polls:
            return []
        return polls.pop(0)

    daemon.store = AsyncMock()
    daemon.store.wait_for_rollouts = fake_wait_for_rollouts
    daemon._validate_data = lambda rollout: None  # type: ignore[method-assign]

    async def immediate_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)

    await daemon._async_run_until_finished(verbose=False)

    assert set(daemon._completed_rollouts_v0) == {"r1"}
    assert daemon._completed_rollouts_v0["r1"].final_reward == 0.7
    assert AgentModeDaemon._is_failed_rollout(daemon._completed_rollouts_v0["r1"]) is False
    assert polls == []

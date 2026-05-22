"""Tests for LocalReconciler — fake subprocess factory + mocked AglLiteClient."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agl_lite.client import AglLiteClient
from agl_lite.controller.config import ControllerSettings, RunnerType
from agl_lite.controller.local_reconciler import LocalReconciler, RunningProc
from agl_lite.schemas.api import PatchRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutStatus


def _settings(**kwargs: Any) -> ControllerSettings:
    defaults: dict[str, Any] = {
        "base_url": "http://agl-lite:8000",
        "key": "test",
        "namespace": "default",
        "runner_type": RunnerType.LOCAL,
        "local_pool_size": 2,
        "local_agent_class": "tests.fixtures.local_agent:FakeAgent",
        "local_tick_interval": 0.01,
        "max_queue_time": 3600,
    }
    defaults.update(kwargs)
    return ControllerSettings(**defaults)


def _rollout(
    rollout_id: str = "r1",
    status: RolloutStatus = RolloutStatus.QUEUING,
    cancel_requested: bool = False,
    job_name: str | None = None,
    timeout: int | None = None,
    created_at: float | None = None,
    rollout_input: Any = None,
) -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        status=status,
        cancel_requested=cancel_requested,
        input=rollout_input if rollout_input is not None else {"task": "test"},
        config=RolloutConfig(timeout=timeout),
        job_name=job_name,
        created_at=created_at if created_at is not None else time.time(),
        updated_at=created_at if created_at is not None else time.time(),
    )


@dataclass
class FakeProc:
    """Stand-in for ``asyncio.subprocess.Process`` driven by the test."""

    pid: int = 4242
    returncode: int | None = None

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


@dataclass
class FakeSpawner:
    """Records spawn calls and returns scripted ``FakeProc`` instances.

    ``next_returncodes`` is consumed in order; each spawn creates a FakeProc
    whose returncode starts as None. ``finish(idx, rc)`` simulates the worker
    exiting with that returncode.
    """

    raise_on_spawn: Exception | None = None
    procs: list[FakeProc] = field(default_factory=list)
    envs: list[dict[str, str]] = field(default_factory=list)
    next_pid: int = 1000

    async def __call__(self, *args: Any, env: dict[str, str], **kwargs: Any) -> FakeProc:
        if self.raise_on_spawn is not None:
            raise self.raise_on_spawn
        self.next_pid += 1
        proc = FakeProc(pid=self.next_pid)
        self.procs.append(proc)
        self.envs.append(env)
        return proc

    def finish(self, idx: int, rc: int) -> None:
        self.procs[idx].returncode = rc


@pytest.fixture
def spawner() -> FakeSpawner:
    return FakeSpawner()


@pytest.fixture
def mock_api() -> AsyncMock:
    api = AsyncMock(spec=AglLiteClient)
    api.query_rollouts = AsyncMock(return_value=[])
    api.patch_rollout = AsyncMock()
    api.get_rollout = AsyncMock()
    return api


def _make_reconciler(
    api: AsyncMock,
    spawner: FakeSpawner,
    settings: ControllerSettings | None = None,
) -> LocalReconciler:
    rec = LocalReconciler(
        api=api,
        settings=settings or _settings(),
        base_env={"PYTHONPATH": "/x"},
    )
    return rec


async def _run_tick(rec: LocalReconciler, spawner: FakeSpawner) -> None:
    with patch(
        "agl_lite.controller.local_reconciler.asyncio.create_subprocess_exec",
        new=spawner,
    ):
        await rec._reconcile_once()


# ---------------------------------------------------------------------------
# basic timing: spawn → running, exit rc=0 → succeeded
# ---------------------------------------------------------------------------


class TestSpawnAndReap:
    async def test_spawn_patches_running_and_adds_to_list(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        assert len(spawner.procs) == 1
        assert len(rec._running) == 1
        assert rec._running[0].rollout_id == "r1"
        assert rec._running[0].attempt_id == "local-r1"

        mock_api.patch_rollout.assert_called_once()
        rollout_id, patch_arg = mock_api.patch_rollout.call_args[0]
        assert rollout_id == "r1"
        assert patch_arg.status == RolloutStatus.RUNNING
        assert patch_arg.job_name == "local-r1"

    async def test_exit_rc0_patches_succeeded(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        mock_api.get_rollout = AsyncMock(return_value=_rollout("r1", status=RolloutStatus.RUNNING))

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)
        # Now simulate the worker exiting successfully.
        spawner.finish(0, rc=0)
        # No new QUEUING rollouts for the second tick.
        mock_api.query_rollouts = AsyncMock(return_value=[])
        await _run_tick(rec, spawner)

        assert rec._running == []
        # Find the SUCCEEDED patch.
        succeeded_calls = [
            c for c in mock_api.patch_rollout.call_args_list if c[0][1].status == RolloutStatus.SUCCEEDED
        ]
        assert len(succeeded_calls) == 1
        patch_arg = succeeded_calls[0][0][1]
        assert patch_arg.succeeded_attempt_id == "local-r1"
        assert patch_arg.job_name == "local-r1"

    async def test_exit_nonzero_patches_terminal_failed(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        mock_api.get_rollout = AsyncMock(return_value=_rollout("r1", status=RolloutStatus.RUNNING))

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)
        spawner.finish(0, rc=2)
        mock_api.query_rollouts = AsyncMock(return_value=[])
        await _run_tick(rec, spawner)

        terminal_calls = [
            c for c in mock_api.patch_rollout.call_args_list if c[0][1].status == RolloutStatus.TERMINAL_FAILED
        ]
        assert len(terminal_calls) == 1
        patch_arg = terminal_calls[0][0][1]
        assert "exited with code 2" in (patch_arg.error_message or "")
        assert patch_arg.job_name == "local-r1"


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------


class TestCapacity:
    async def test_respects_pool_size(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        rs = [_rollout(f"r{i}") for i in range(5)]
        mock_api.query_rollouts = AsyncMock(return_value=rs)

        rec = _make_reconciler(mock_api, spawner, settings=_settings(local_pool_size=2))
        await _run_tick(rec, spawner)

        assert len(spawner.procs) == 2
        assert {item.rollout_id for item in rec._running} == {"r0", "r1"}

    async def test_reap_frees_slot_within_same_tick(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        """First tick fills the pool; second tick reaps one and admits a new one."""
        rs = [_rollout(f"r{i}") for i in range(3)]
        mock_api.query_rollouts = AsyncMock(return_value=rs)
        mock_api.get_rollout = AsyncMock(side_effect=lambda rid: _rollout(rid, status=RolloutStatus.RUNNING))

        rec = _make_reconciler(mock_api, spawner, settings=_settings(local_pool_size=2))
        await _run_tick(rec, spawner)
        assert len(rec._running) == 2

        # Finish r0; next tick should reap it AND admit r2 in the same tick.
        spawner.finish(0, rc=0)
        # Re-supply the QUEUING list so admit sees r2 (other slots are tracked
        # internally; in practice the store no longer lists r0/r1 as QUEUING).
        mock_api.query_rollouts = AsyncMock(return_value=[rs[2]])
        await _run_tick(rec, spawner)

        assert len(spawner.procs) == 3
        assert {item.rollout_id for item in rec._running} == {"r1", "r2"}


# ---------------------------------------------------------------------------
# spawn failure
# ---------------------------------------------------------------------------


class TestSpawnFailure:
    async def test_spawn_exception_patches_terminal_failed(self, mock_api: AsyncMock) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        spawner = FakeSpawner(raise_on_spawn=OSError("no exec"))

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        assert rec._running == []
        assert spawner.procs == []
        mock_api.patch_rollout.assert_called_once()
        patch_arg = mock_api.patch_rollout.call_args[0][1]
        assert patch_arg.status == RolloutStatus.TERMINAL_FAILED
        assert "spawn failed" in (patch_arg.error_message or "")
        assert patch_arg.job_name == "local-r1"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


class TestCancel:
    async def test_cancel_before_spawn_marks_cancelled(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1", cancel_requested=True)
        mock_api.query_rollouts = AsyncMock(return_value=[r])

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        assert spawner.procs == []
        mock_api.patch_rollout.assert_called_once()
        patch_arg = mock_api.patch_rollout.call_args[0][1]
        assert patch_arg.status == RolloutStatus.CANCELLED
        assert patch_arg.job_name == "local-r1"

    async def test_running_with_cancel_requested_triggers_sigkill(
        self, mock_api: AsyncMock, spawner: FakeSpawner
    ) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        # Spawn first.
        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)
        assert len(rec._running) == 1

        # Now flip cancel_requested True on the next get_rollout return.
        mock_api.get_rollout = AsyncMock(
            return_value=_rollout("r1", status=RolloutStatus.RUNNING, cancel_requested=True)
        )
        mock_api.query_rollouts = AsyncMock(return_value=[])

        with patch("agl_lite.controller.local_reconciler.os.killpg") as kpg:
            await _run_tick(rec, spawner)
            kpg.assert_called_once()
            args = kpg.call_args[0]
            assert args[0] == spawner.procs[0].pid

        # The proc is now killed in the test sense — simulate rc=-9.
        spawner.finish(0, rc=-9)
        await _run_tick(rec, spawner)
        cancelled = [c for c in mock_api.patch_rollout.call_args_list if c[0][1].status == RolloutStatus.CANCELLED]
        assert len(cancelled) == 1
        assert cancelled[0][0][1].job_name == "local-r1"

    async def test_success_wins_over_cancel(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        # cancel_requested + rc=0 → SUCCEEDED.
        mock_api.get_rollout = AsyncMock(
            return_value=_rollout("r1", status=RolloutStatus.RUNNING, cancel_requested=True)
        )
        mock_api.query_rollouts = AsyncMock(return_value=[])
        spawner.finish(0, rc=0)
        await _run_tick(rec, spawner)

        succeeded = [c for c in mock_api.patch_rollout.call_args_list if c[0][1].status == RolloutStatus.SUCCEEDED]
        assert len(succeeded) == 1


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_triggers_sigkill_and_terminal_failed(
        self, mock_api: AsyncMock, spawner: FakeSpawner
    ) -> None:
        r = _rollout("r1", timeout=1)
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        # Backdate spawned_at so the next tick sees timeout exceeded.
        rec._running[0].spawned_at = time.monotonic() - 1000

        mock_api.get_rollout = AsyncMock(return_value=_rollout("r1", status=RolloutStatus.RUNNING, timeout=1))
        mock_api.query_rollouts = AsyncMock(return_value=[])

        with patch("agl_lite.controller.local_reconciler.os.killpg") as kpg:
            await _run_tick(rec, spawner)
            kpg.assert_called_once()

        spawner.finish(0, rc=-9)
        await _run_tick(rec, spawner)
        terminal = [c for c in mock_api.patch_rollout.call_args_list if c[0][1].status == RolloutStatus.TERMINAL_FAILED]
        assert len(terminal) == 1

    async def test_timeout_none_does_not_kill(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1", timeout=None)
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        rec._running[0].spawned_at = time.monotonic() - 1_000_000
        mock_api.get_rollout = AsyncMock(return_value=_rollout("r1", status=RolloutStatus.RUNNING, timeout=None))
        mock_api.query_rollouts = AsyncMock(return_value=[])

        with patch("agl_lite.controller.local_reconciler.os.killpg") as kpg:
            await _run_tick(rec, spawner)
            kpg.assert_not_called()


# ---------------------------------------------------------------------------
# queue timeout
# ---------------------------------------------------------------------------


class TestQueueTimeout:
    async def test_max_queue_time_exceeded(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1", created_at=time.time() - 7200)
        mock_api.query_rollouts = AsyncMock(return_value=[r])

        rec = _make_reconciler(mock_api, spawner, settings=_settings(max_queue_time=3600))
        await _run_tick(rec, spawner)

        assert spawner.procs == []
        mock_api.patch_rollout.assert_called_once()
        patch_arg = mock_api.patch_rollout.call_args[0][1]
        assert patch_arg.status == RolloutStatus.TERMINAL_FAILED
        assert "max queue time" in (patch_arg.error_message or "")
        assert patch_arg.job_name == "local-r1"


# ---------------------------------------------------------------------------
# startup cleanup
# ---------------------------------------------------------------------------


class TestStartupCleanup:
    async def test_leftover_local_running_marked_terminal_failed(
        self, mock_api: AsyncMock, spawner: FakeSpawner
    ) -> None:
        leftover = _rollout("r1", status=RolloutStatus.RUNNING, job_name="local-r1")
        non_local = _rollout("r2", status=RolloutStatus.RUNNING, job_name="agl-rollout-r2")
        mock_api.query_rollouts = AsyncMock(return_value=[leftover, non_local])

        rec = _make_reconciler(mock_api, spawner)
        await rec._startup_cleanup()

        mock_api.patch_rollout.assert_called_once()
        rollout_id, patch_arg = mock_api.patch_rollout.call_args[0]
        assert rollout_id == "r1"
        assert patch_arg.status == RolloutStatus.TERMINAL_FAILED

    async def test_leftover_local_running_with_cancel_marked_cancelled(
        self, mock_api: AsyncMock, spawner: FakeSpawner
    ) -> None:
        leftover = _rollout(
            "r1",
            status=RolloutStatus.RUNNING,
            cancel_requested=True,
            job_name="local-r1",
        )
        mock_api.query_rollouts = AsyncMock(return_value=[leftover])

        rec = _make_reconciler(mock_api, spawner)
        await rec._startup_cleanup()

        patch_arg = mock_api.patch_rollout.call_args[0][1]
        assert patch_arg.status == RolloutStatus.CANCELLED


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_shutdown_kills_and_marks_cancelled(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(return_value=[r])
        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        # During shutdown, wait() must return; flip rc so wait() exits quickly.
        spawner.procs[0].returncode = -9

        with patch("agl_lite.controller.local_reconciler.os.killpg"):
            await rec._shutdown()

        cancelled = [
            c
            for c in mock_api.patch_rollout.call_args_list
            if c[0][1].status == RolloutStatus.CANCELLED and c[0][1].error_message == "local controller shutdown"
        ]
        assert len(cancelled) == 1
        assert rec._running == []


# ---------------------------------------------------------------------------
# env injection
# ---------------------------------------------------------------------------


class TestEnvBuild:
    async def test_env_contains_gateway_urls_and_local_extras(self, mock_api: AsyncMock, spawner: FakeSpawner) -> None:
        r = _rollout("r1", rollout_input={"task": "compute"})
        mock_api.query_rollouts = AsyncMock(return_value=[r])

        rec = _make_reconciler(mock_api, spawner)
        await _run_tick(rec, spawner)

        env = spawner.envs[0]
        assert env["AGL_ROLLOUT_ID"] == "r1"
        assert env["AGL_ATTEMPT_ID"] == "local-r1"
        assert env["OPENAI_BASE_URL"].endswith("/rollout/r1/attempt/local-r1/v1")
        assert env["ANTHROPIC_BASE_URL"].endswith("/rollout/r1/attempt/local-r1")
        assert env["AGL_EVENT_URL"].endswith("/rollout/r1/attempt/local-r1/events")
        assert env["AGL_LOG_DIR"].endswith("/local-r1")
        assert env["AGL_TASK_INPUT"] == '{"task": "compute"}'
        assert env["AGL_LOCAL_AGENT_CLASS"] == "tests.fixtures.local_agent:FakeAgent"
        # base_env passthrough.
        assert env["PYTHONPATH"] == "/x"


# ---------------------------------------------------------------------------
# job_name prefix invariant on every patch path
# ---------------------------------------------------------------------------


class TestJobNameInvariant:
    @pytest.mark.parametrize(
        "scenario",
        ["spawn_running", "cancel_queuing", "queue_timeout", "spawn_failed"],
    )
    async def test_patch_carries_local_prefix(self, mock_api: AsyncMock, scenario: str) -> None:
        spawner = FakeSpawner()
        if scenario == "spawn_running":
            r = _rollout("r1")
        elif scenario == "cancel_queuing":
            r = _rollout("r1", cancel_requested=True)
        elif scenario == "queue_timeout":
            r = _rollout("r1", created_at=time.time() - 7200)
        else:  # spawn_failed
            r = _rollout("r1")
            spawner = FakeSpawner(raise_on_spawn=RuntimeError("boom"))

        mock_api.query_rollouts = AsyncMock(return_value=[r])
        rec = _make_reconciler(mock_api, spawner, settings=_settings(max_queue_time=3600))
        await _run_tick(rec, spawner)

        assert mock_api.patch_rollout.called
        for call in mock_api.patch_rollout.call_args_list:
            patch_arg: PatchRolloutRequest = call[0][1]
            assert patch_arg.job_name == "local-r1"


# ---------------------------------------------------------------------------
# attempt_id derivation
# ---------------------------------------------------------------------------


def test_attempt_id_for() -> None:
    rec = LocalReconciler(
        api=AsyncMock(spec=AglLiteClient),
        settings=_settings(),
        base_env={},
    )
    assert rec._attempt_id_for("abc") == "local-abc"


def test_running_proc_dataclass_defaults() -> None:
    item = RunningProc(
        rollout_id="r1",
        attempt_id="local-r1",
        proc=FakeProc(),  # type: ignore[arg-type]
        spawned_at=time.monotonic(),
    )
    assert item.killed is False

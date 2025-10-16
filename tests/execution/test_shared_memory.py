# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import pytest
from _pytest.logging import LogCaptureFixture

from agentlightning.execution.events import ExecutionEvent, ThreadingEvent
from agentlightning.execution.shared_memory import SharedMemoryExecutionStrategy
from agentlightning.store.base import LightningStore

from ..store.dummy_store import DummyLightningStore, minimal_dummy_store


@pytest.fixture
def store() -> DummyLightningStore:
    return minimal_dummy_store()


def test_env_managed_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_MANAGED_STORE", "false")

    strat = SharedMemoryExecutionStrategy()

    assert strat.managed_store is False


def test_explicit_managed_store_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGL_MANAGED_STORE", "0")

    strat = SharedMemoryExecutionStrategy(managed_store=True)

    assert strat.managed_store is True


def test_execute_uses_unmanaged_store_directly(store: DummyLightningStore) -> None:
    strat = SharedMemoryExecutionStrategy(managed_store=False)
    used: Dict[str, LightningStore] = {}

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        used["algo"] = store
        event.set()

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        used["runner"] = store
        while not event.is_set():
            await asyncio.sleep(0.01)

    strat.execute(algo, runner, store)

    assert used["algo"] is store
    assert used["runner"] is store


def tiny_sleep(seconds: float) -> float:
    """Sleep a tiny bit and return the elapsed time (monotonic)."""
    start = time.monotonic()
    time.sleep(seconds)
    return time.monotonic() - start


# Helper bundles for tests


def make_cooperative_algorithm(
    started: List[str],
    finished: List[str],
    poll_delay: float = 0.005,
):
    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        started.append("algo")
        # cooperatively exit when asked
        while not event.is_set():
            await asyncio.sleep(poll_delay)
        finished.append("algo")

    return algo


def make_cooperative_runner(
    started: List[int],
    finished: List[int],
    poll_delay: float = 0.005,
):
    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        started.append(worker_id)
        while not event.is_set():
            await asyncio.sleep(poll_delay)
        finished.append(worker_id)

    return runner


def make_sleepy_coro(delay: float, result: Any):
    async def _coro():
        await asyncio.sleep(delay)
        return result

    return _coro()


def make_never_polls_coro(total_time: float):
    """A coro that just sleeps in chunks; cancellation lands at await points only."""

    async def _coro():
        end = time.monotonic() + total_time
        while time.monotonic() < end:
            await asyncio.sleep(0.05)
        return "done"

    return _coro()


def make_slow_cleanup_on_cancel_coro(cleanup_time: float):
    """On cancellation, do slow cleanup before exiting (tests second-chance timeout)."""

    async def _coro():
        try:
            # Idle until canceled
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(cleanup_time)
            raise

    return _coro()


# Unit tests: _run_until_completed_or_canceled


def test_run_until_completes_naturally(caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=0.05, join_timeout=0.2)

    evt = ThreadingEvent()

    result = asyncio.run(
        strat._run_until_completed_or_canceled(make_sleepy_coro(0.02, "ok"), evt)  # pyright: ignore[reportPrivateUsage]
    )
    assert result == "ok"


def test_run_until_stops_gracefully_with_event(caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=0.05, join_timeout=0.2)

    evt = ThreadingEvent()

    async def cooperative(event: ExecutionEvent):
        # Stop promptly after event flips
        while not event.is_set():
            await asyncio.sleep(0.005)
        return "bye"

    # Set the stop event before we start to exercise the "finish during grace" path
    evt.set()
    result = asyncio.run(
        strat._run_until_completed_or_canceled(cooperative(evt), evt)  # pyright: ignore[reportPrivateUsage]
    )
    assert result == "bye"
    # Should have logged that the bundle finished during grace period
    assert any("finished by itself during grace period" in rec.message for rec in caplog.records)


def test_run_until_cancels_after_grace_if_not_quick_to_stop(caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    # Grace: 30ms; the task won't stop within that, so cancel fires; then second-chance wait succeeds.
    graceful = 0.03
    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=graceful, join_timeout=0.2)

    evt = ThreadingEvent()

    async def slow_to_notice():
        # Doesn't poll stop_evt; cancellation must interrupt sleeps
        await asyncio.sleep(1.0)

    async def runner():
        # flip stop after a short time
        asyncio.get_running_loop().call_later(0.01, evt.set)
        return await strat._run_until_completed_or_canceled(  # pyright: ignore[reportPrivateUsage]
            slow_to_notice(), evt
        )

    t0 = time.monotonic()
    result = asyncio.run(runner())
    elapsed = time.monotonic() - t0
    # We don't get a "result" because the task was canceled
    assert result is None
    # Should be roughly >= grace window, but way less than full 1.0s sleep
    assert elapsed >= graceful * 0.9
    assert elapsed < 0.5
    assert any("Graceful delay elapsed; canceling bundle task..." in rec.message for rec in caplog.records)


def test_run_until_second_chance_timeout_logs_and_returns(caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    # The task takes longer to clean up than graceful_delay; we hit the second-chance timeout branch.
    graceful = 0.03
    cleanup = 0.08
    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=graceful, join_timeout=0.2)

    evt = ThreadingEvent()

    async def runner():
        # Schedule stop, then run a task that sleeps extra on cancel
        asyncio.get_running_loop().call_later(0.005, evt.set)
        return await strat._run_until_completed_or_canceled(  # pyright: ignore[reportPrivateUsage]
            make_slow_cleanup_on_cancel_coro(cleanup), evt
        )

    t0 = time.monotonic()
    result = asyncio.run(runner())
    elapsed = time.monotonic() - t0

    # No result expected because the task gets canceled; we abandon await after timeout.
    assert result is None
    assert elapsed >= graceful  # we at least burned the grace + a bit
    assert any("did not stop after cancellation; abandoning task" in rec.message for rec in caplog.records)


# Unit tests: _run_algorithm and _run_runner


def test_run_algorithm_sets_stop_on_exception(store: DummyLightningStore):
    strat = SharedMemoryExecutionStrategy()

    evt = ThreadingEvent()

    async def boom(store: LightningStore, event: ExecutionEvent) -> None:
        await asyncio.sleep(0.005)
        raise ValueError("algo crash")

    with pytest.raises(ValueError):
        strat._run_algorithm(boom, store, evt, None)  # pyright: ignore[reportPrivateUsage]

    assert evt.is_set(), "stop_evt must be set when algorithm raises"


def test_run_runner_sets_stop_on_exception(store: DummyLightningStore):
    strat = SharedMemoryExecutionStrategy()

    evt = ThreadingEvent()

    async def boom(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        await asyncio.sleep(0.005)
        raise RuntimeError("runner crash")

    with pytest.raises(RuntimeError):
        strat._run_runner(boom, store, 0, evt, None)  # pyright: ignore[reportPrivateUsage]

    assert evt.is_set(), "stop_evt must be set when a runner raises"


# Integration tests: execute(...)


def test_execute_main_algorithm_normal_stop_sets_event(store: DummyLightningStore):
    started_r: List[int] = []
    finished_r: List[int] = []
    started_a: List[str] = []
    finished_a: List[str] = []

    strat = SharedMemoryExecutionStrategy(n_runners=2, main_thread="algorithm", graceful_delay=0.02, join_timeout=0.2)

    runner = make_cooperative_runner(started_r, finished_r, poll_delay=0.005)

    # Algorithm finishes quickly; then execute() should set stop_evt for runners.
    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        started_a.append("algo")
        await asyncio.sleep(0.02)
        finished_a.append("algo")

    strat.execute(algo, runner, store)

    # Runners should have started and finished due to stop_evt set by execute()
    assert sorted(started_r) == [0, 1]
    assert sorted(finished_r) == [0, 1]
    assert started_a == ["algo"]
    assert finished_a == ["algo"]


def test_execute_main_runner_waits_for_algorithm_natural_finish(store: DummyLightningStore):
    # Policy: when main_thread='runner' and runner finishes, do NOT set stop; wait for algo to finish.
    started_r: List[int] = []
    finished_r: List[int] = []
    started_a: List[str] = []
    finished_a: List[str] = []
    captured_evt: List[Optional[Any]] = [None]

    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=0.02, join_timeout=0.2)

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        captured_evt[0] = event
        started_a.append("algo")
        await asyncio.sleep(0.05)  # natural finish
        finished_a.append("algo")

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        started_r.append(worker_id)
        await asyncio.sleep(0.01)  # finish quickly
        finished_r.append(worker_id)

    strat.execute(algo, runner, store)

    assert started_r == [0]
    assert finished_r == [0]
    assert started_a == ["algo"]
    assert finished_a == ["algo"]
    # Verify execute() did not set stop_evt just because runner ended.
    assert captured_evt[0] is not None and captured_evt[0].is_set() is False


def test_execute_runner_crash_propagates_and_stops_algorithm(store: DummyLightningStore):
    # Runner raises; algorithm should see stop_evt and exit.
    started_a: List[str] = []
    finished_a: List[str] = []
    saw_stop: List[bool] = []

    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="algorithm", graceful_delay=0.02, join_timeout=0.3)

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        started_a.append("algo")
        # wait until stop evt set (by runner crash)
        while not event.is_set():
            await asyncio.sleep(0.005)
        saw_stop.append(True)
        finished_a.append("algo")

    async def bad_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        await asyncio.sleep(0.02)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        strat.execute(algo, bad_runner, store)

    assert started_a == ["algo"]
    assert saw_stop == [True]
    assert finished_a == ["algo"]


def test_execute_ctrl_c_on_algorithm_stops_runners(store: DummyLightningStore, caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    # Simulate Ctrl+C by having the algorithm raise KeyboardInterrupt on the main thread.
    runner_started: List[int] = []
    runner_finished: List[int] = []

    strat = SharedMemoryExecutionStrategy(n_runners=2, main_thread="algorithm", graceful_delay=0.02, join_timeout=0.3)

    async def algo_kbi(store: LightningStore, event: ExecutionEvent) -> None:
        await asyncio.sleep(0.01)
        raise KeyboardInterrupt()

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        runner_started.append(worker_id)
        while not event.is_set():
            await asyncio.sleep(0.005)
        runner_finished.append(worker_id)

    strat.execute(algo_kbi, runner, store)

    # Runners should have started and then exited due to stop signal
    assert sorted(runner_started) == [0, 1]
    assert sorted(runner_finished) == [0, 1]
    assert any("KeyboardInterrupt received on main thread" in rec.message for rec in caplog.records)


def test_execute_ctrl_c_on_runner_stops_algorithm(store: DummyLightningStore, caplog: LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    # Simulate Ctrl+C on the main thread when main_thread='runner'
    algo_started: List[str] = []
    algo_finished: List[str] = []

    strat = SharedMemoryExecutionStrategy(n_runners=1, main_thread="runner", graceful_delay=0.02, join_timeout=0.3)

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        algo_started.append("a")
        # Await stop_evt after KBI from runner
        while not event.is_set():
            await asyncio.sleep(0.005)
        algo_finished.append("a")

    async def runner_kbi(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        await asyncio.sleep(0.01)
        raise KeyboardInterrupt()

    strat.execute(algo, runner_kbi, store)

    assert algo_started == ["a"]
    assert algo_finished == ["a"]  # algorithm should finish after stop_evt set by KBI handler

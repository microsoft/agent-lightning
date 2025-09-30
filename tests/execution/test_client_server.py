# test_client_server_execution_strategy.py
from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import time
from contextlib import closing
from multiprocessing import Event as MpEvent
from multiprocessing import Process, get_context
from typing import Any, Callable, Dict, List, Optional

import pytest

from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient

from ..store.dummy_store import DummyLightningStore, minimal_dummy_store

# =========================
# Helpers & Fixtures
# =========================


def _free_port() -> int:
    """Return an available TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class DummyEvt:
    """Simple in-process Event-like object required by the strategy."""

    def __init__(self) -> None:
        self._flag: bool = False

    def set(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag

    def clear(self) -> None:
        self._flag = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._flag


@pytest.fixture
def store() -> DummyLightningStore:
    return minimal_dummy_store()


# =========================
# Async bundles for tests
# =========================


async def _noop_algorithm(_store: Any, stop_evt: Any) -> None:
    await asyncio.sleep(0)
    assert not stop_evt.is_set()


async def _algo_calls_store_enqueue(store_obj: Any, stop_evt: Any) -> None:
    # Calls a delegated method on the server wrapper; real server is running.
    await store_obj.enqueue_rollout(input={"x": 1})
    await asyncio.sleep(0)
    assert not stop_evt.is_set()


async def _algo_sets_stop_delayed(_store: Any, stop_evt: Any, delay: float = 0.05) -> None:
    await asyncio.sleep(delay)
    stop_evt.set()


async def _algo_ignores_stop_forever(_store: Any, _stop_evt: Any) -> None:
    # Ignore signals to force escalation phases in shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        await asyncio.sleep(0.1)


async def _raise_in_algorithm(_store: Any, stop_evt: Any) -> None:
    stop_evt.set()
    raise RuntimeError("algo boom")


async def _kbint_in_algorithm(_store: Any, stop_evt: Any) -> None:
    stop_evt.set()
    raise KeyboardInterrupt()


async def _noop_runner(_store: Any, _worker_id: int, stop_evt: Any) -> None:
    await asyncio.sleep(0)
    assert not stop_evt.is_set()


async def _runner_wait_for_stop(_store: Any, _worker_id: int, stop_evt: Any, timeout: float = 0.5) -> None:
    t0: float = time.monotonic()
    while not stop_evt.is_set() and time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.005)


async def _runner_ignores_stop_forever(_store: Any, _worker_id: int, _stop_evt: Any) -> None:
    # Ignore signals to force escalation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        await asyncio.sleep(0.1)


async def _raise_in_runner(_store: Any, _worker_id: int, stop_evt: Any) -> None:
    stop_evt.set()
    raise RuntimeError("runner boom")


async def _kbint_in_runner(_store: Any, _worker_id: int, stop_evt: Any) -> None:
    stop_evt.set()
    raise KeyboardInterrupt()


async def _timeout_error_in_runner(client: LightningStoreClient, _worker_id: int, stop_evt: Any) -> None:
    # Provoke client's validation (pre-request), then raise TimeoutError.
    with pytest.raises(ValueError):
        await client.wait_for_rollouts(rollout_ids=["r1"], timeout=0.2)
    stop_evt.set()
    raise TimeoutError("runner timeout")


# =========================
# Private helper tests
# =========================


def test_join_until_deadline_includes_alive_process() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    ctx = get_context()
    p: Process = ctx.Process(target=time.sleep, args=(0.5,), name="alive")
    p.start()
    try:
        alive: List[Process] = strat._join_until_deadline([p], timeout=0.005)
        assert alive == [p]
        assert p.is_alive()
    finally:
        p.terminate()
        p.join()


def test_join_until_deadline_excludes_finished_process() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    ctx = get_context()
    p: Process = ctx.Process(target=lambda: None, name="done")
    p.start()
    p.join()
    alive: List[Process] = strat._join_until_deadline([p], timeout=0.05)
    assert alive == []


def test_join_until_deadline_zero_timeout_path() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    ctx = get_context()
    p: Process = ctx.Process(target=time.sleep, args=(0.2,), name="zero-join")
    p.start()
    try:
        alive: List[Process] = strat._join_until_deadline([p], timeout=0.0)
        assert alive == [p]
    finally:
        p.terminate()
        p.join()


def test_signal_processes_invokes_action_and_suppresses_exceptions() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    ctx = get_context()
    p: Process = ctx.Process(target=time.sleep, args=(0.2,), name="foo")
    p.start()
    seen: List[int] = []

    def action(proc: Process) -> None:
        seen.append(proc.pid)  # record
        if len(seen) == 1:
            raise RuntimeError("deliberate")  # ensure suppression

    try:
        strat._signal_processes([p], action)
        assert seen == [p.pid]
    finally:
        p.terminate()
        p.join()


def test_shutdown_processes_empty_list_noop() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    strat._shutdown_processes([], DummyEvt())  # should not raise


def test_shutdown_processes_phase1_cooperative() -> None:
    """
    Process exits during the cooperative (graceful) wait; no signals required.
    """
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    ctx = get_context()
    p: Process = ctx.Process(target=time.sleep, args=(0.01,), name="coop")
    p.start()
    try:
        strat._shutdown_processes([p], DummyEvt())
        assert not p.is_alive() and p.exitcode == 0
    finally:
        if p.is_alive():
            p.kill()
        p.join()


def test_shutdown_processes_phase2_sigint() -> None:
    """
    Process survives cooperative window, exits cleanly on SIGINT.
    """
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=1.0,
        terminate_timeout=1.0,
    )
    ctx = get_context()

    def target() -> None:
        def on_sigint(_sig: int, _frm: Any) -> None:
            sys.exit(0)

        signal.signal(signal.SIGINT, on_sigint)
        while True:
            time.sleep(0.1)

    p: Process = ctx.Process(target=target, name="sigint-exit")
    p.start()
    try:
        strat._shutdown_processes([p], DummyEvt())
        assert not p.is_alive() and p.exitcode == 0
    finally:
        if p.is_alive():
            p.terminate()
        p.join()


def test_shutdown_processes_phase2_try_catch() -> None:
    """
    Process survives cooperative window, exits cleanly on SIGINT.
    """
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=1.0,
        terminate_timeout=1.0,
    )
    ctx = get_context()

    def target() -> None:
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("keyboard interrupt")
            return

    p: Process = ctx.Process(target=target, name="sigint-exit")
    p.start()
    try:
        strat._shutdown_processes([p], DummyEvt())
        assert not p.is_alive() and p.exitcode == 0
    finally:
        if p.is_alive():
            p.terminate()
        p.join()


def test_shutdown_processes_phase3_terminate_when_sigint_ignored() -> None:
    """
    Process ignores SIGINT but exits on terminate() / SIGTERM.
    """
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=0.02,
        terminate_timeout=0.05,
    )
    ctx = get_context()

    def target() -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            time.sleep(0.1)

    p: Process = ctx.Process(target=target, name="term-on-sigterm")
    p.start()
    try:
        strat._shutdown_processes([p], DummyEvt())
        assert not p.is_alive()
        # Non-zero expected for terminated processes.
        assert p.exitcode is not None and p.exitcode != 0
    finally:
        if p.is_alive():
            p.kill()
        p.join()


def test_shutdown_processes_phase4_kill_when_term_ignored() -> None:
    """
    Process ignores both SIGINT and SIGTERM; kill() should be used.
    """
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=0.02,
        terminate_timeout=0.05,
    )
    ctx = get_context()

    def target() -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(0.1)

    p: Process = ctx.Process(target=target, name="kill-required")
    p.start()
    try:
        strat._shutdown_processes([p], DummyEvt())
        assert not p.is_alive()
        assert p.exitcode is not None and p.exitcode != 0
    finally:
        if p.is_alive():
            p.kill()
        p.join()


def test_shutdown_processes_when_stop_already_set() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_port=_free_port(),
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    ctx = get_context()
    # Short-lived process; stop_evt already set should be a no-op on the flag.
    p: Process = ctx.Process(target=time.sleep, args=(0.01,), name="pre-set")
    p.start()
    evt: DummyEvt = DummyEvt()
    evt.set()
    try:
        strat._shutdown_processes([p], evt)
        assert not p.is_alive()
    finally:
        if p.is_alive():
            p.terminate()
        p.join()


# =========================
# Private coroutine tests using REAL server
# =========================


def test_execute_algorithm_success_invokes_store(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="algorithm",
        server_host="127.0.0.1",
        server_port=port,
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    # Should run and stop the real HTTP server, while delegating to underlying store.
    asyncio.run(strat._execute_algorithm(_algo_calls_store_enqueue, store, DummyEvt()))
    # The DummyLightningStore should have recorded the delegated call.
    recorded: List[tuple[str, tuple[Any, ...], Dict[str, Any]]] = store.calls
    assert any(name == "enqueue_rollout" for name, _, _ in recorded)


def test_execute_algorithm_sets_stop_on_exception_and_propagates(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="algorithm",
        server_host="127.0.0.1",
        server_port=port,
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    evt: DummyEvt = DummyEvt()
    with pytest.raises(RuntimeError, match="algo boom"):
        asyncio.run(strat._execute_algorithm(_raise_in_algorithm, store, evt))
    assert evt.is_set()


def test_execute_algorithm_keyboardinterrupt_sets_stop_and_propagates(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="algorithm",
        server_host="127.0.0.1",
        server_port=port,
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    evt: DummyEvt = DummyEvt()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(strat._execute_algorithm(_kbint_in_algorithm, store, evt))
    assert evt.is_set()


def test_execute_runner_success_closes_client() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    closed: List[bool] = []

    async def patched_close(self: LightningStoreClient) -> None:
        closed.append(True)

    orig_close: Callable[[LightningStoreClient], Any] = LightningStoreClient.close  # type: ignore[attr-defined]
    try:
        LightningStoreClient.close = patched_close  # type: ignore[assignment]
        asyncio.run(strat._execute_runner(_noop_runner, worker_id=0, stop_evt=DummyEvt()))
    finally:
        LightningStoreClient.close = orig_close  # type: ignore[assignment]

    assert closed == [True]


def test_execute_runner_exception_sets_stop_and_closes_client() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    closed: List[bool] = []

    async def patched_close(self: LightningStoreClient) -> None:
        closed.append(True)

    orig_close: Callable[[LightningStoreClient], Any] = LightningStoreClient.close  # type: ignore[attr-defined]
    evt: DummyEvt = DummyEvt()
    try:
        LightningStoreClient.close = patched_close  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="runner boom"):
            asyncio.run(strat._execute_runner(_raise_in_runner, worker_id=7, stop_evt=evt))
    finally:
        LightningStoreClient.close = orig_close  # type: ignore[assignment]

    assert evt.is_set()
    assert closed == [True]


def test_execute_runner_keyboardinterrupt_sets_stop_and_propagates() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    evt: DummyEvt = DummyEvt()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(strat._execute_runner(_kbint_in_runner, worker_id=0, stop_evt=evt))
    assert evt.is_set()


def test_execute_runner_distinguishes_timeout_error() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    evt: DummyEvt = DummyEvt()
    with pytest.raises(TimeoutError, match="runner timeout"):
        asyncio.run(strat._execute_runner(_timeout_error_in_runner, worker_id=0, stop_evt=evt))
    assert evt.is_set()


def test_spawn_runners_creates_processes_and_they_exit_on_event() -> None:
    strat = ClientServerExecutionStrategy(
        role="runner",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=2,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    ctx = get_context()
    stop_evt: MpEvent = MpEvent()

    def runner_sync() -> None:
        asyncio.run(strat._execute_runner(_runner_wait_for_stop, worker_id=0, stop_evt=stop_evt))

    procs: List[Process] = []
    for i in range(2):
        p: Process = ctx.Process(target=runner_sync, name=f"runner-{i}")
        p.start()
        procs.append(p)

    try:
        assert all(p.is_alive() for p in procs)
        stop_evt.set()
        for p in procs:
            p.join(timeout=2.0)
        assert all(not p.is_alive() and p.exitcode == 0 for p in procs)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join()


def test_spawn_algorithm_process_creates_and_runs(store: LightningStore) -> None:
    strat = ClientServerExecutionStrategy(
        role="both",
        server_host="127.0.0.1",
        server_port=_free_port(),
        n_runners=1,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    ctx = get_context()
    stop_evt: MpEvent = MpEvent()

    p: Process = strat._spawn_algorithm_process(_noop_algorithm, store, stop_evt, ctx=ctx)
    try:
        p.join(timeout=4.0)
        assert not p.is_alive()
        assert p.exitcode == 0
    finally:
        if p.is_alive():
            p.terminate()
        p.join()


# =========================
# Integration tests: execute()
# =========================


def test_execute_role_algorithm_success(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="algorithm",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    strat.execute(algorithm=_noop_algorithm, runner=_noop_runner, store=store)


def test_execute_role_runner_single_success(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="runner",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    strat.execute(algorithm=_noop_algorithm, runner=_noop_runner, store=store)


def test_execute_role_runner_multi_success(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="runner",
        n_runners=2,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    strat.execute(algorithm=_noop_algorithm, runner=_noop_runner, store=store)


def test_execute_role_runner_multi_raises_on_child_failure(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="runner",
        n_runners=2,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    with pytest.raises(RuntimeError):
        strat.execute(algorithm=_noop_algorithm, runner=_raise_in_runner, store=store)


def test_execute_both_main_algorithm_cooperative_shutdown(store: LightningStore) -> None:
    """
    Spawn runners, run algorithm in the main process.
    Algorithm sets stop_evt after a short delay to unwind the runners.
    """
    port: int = _free_port()

    async def algo(store_obj: Any, stop_evt: Any) -> None:
        await _algo_sets_stop_delayed(store_obj, stop_evt, delay=0.05)

    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="algorithm",
        n_runners=2,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    strat.execute(algorithm=algo, runner=_runner_wait_for_stop, store=store)


def test_execute_both_main_runner_debug_cooperative_shutdown(store: LightningStore) -> None:
    """
    main_process='runner' requires n_runners == 1.
    Runner runs in main process, algorithm runs in a child process hosting the server.
    Runner sets stop_evt to ask algorithm to exit.
    """
    port: int = _free_port()

    async def runner(_client: Any, _wid: int, stop_evt: Any) -> None:
        await asyncio.sleep(0.05)
        stop_evt.set()

    async def algo(_store: Any, stop_evt: Any) -> None:
        t0: float = time.monotonic()
        while not stop_evt.is_set() and time.monotonic() - t0 < 1.0:
            await asyncio.sleep(0.005)

    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="runner",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        # Allow a generous timeout to ensure the server starts
        graceful_timeout=5.0,
        terminate_timeout=5.0,
    )
    strat.execute(algorithm=algo, runner=runner, store=store)


def test_execute_algorithm_exception_bubbles_and_shuts_down(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="algorithm",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    with pytest.raises(RuntimeError, match="algo boom"):
        strat.execute(algorithm=_raise_in_algorithm, runner=_runner_wait_for_stop, store=store)


def test_execute_algorithm_keyboard_interrupt_propagates(store: LightningStore) -> None:
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="algorithm",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    with pytest.raises(KeyboardInterrupt):
        strat.execute(algorithm=_kbint_in_algorithm, runner=_runner_wait_for_stop, store=store)


def test_execute_runner_single_keyboard_interrupt_is_caught(store: LightningStore) -> None:
    """
    For role='runner' with n_runners==1, a KeyboardInterrupt in runner is caught
    by execute() and should NOT propagate as an exception.
    """
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="runner",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    strat.execute(algorithm=_noop_algorithm, runner=_kbint_in_runner, store=store)


def test_execute_runner_single_timeout_error_bubbles(store: LightningStore) -> None:
    """
    Ensure TimeoutError in single-runner mode is not confused with KeyboardInterrupt.
    """
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="runner",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )
    with pytest.raises(TimeoutError, match="runner timeout"):
        strat.execute(algorithm=_noop_algorithm, runner=_timeout_error_in_runner, store=store)


def test_execute_unknown_role_raises() -> None:
    # Mutate after construction to simulate an invalid state.
    strat = ClientServerExecutionStrategy(role="algorithm", n_runners=1)
    strat.role = "wat"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        strat.execute(algorithm=_noop_algorithm, runner=_noop_runner, store=DummyLightningStore({}))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        ClientServerExecutionStrategy(role="runner", main_process="runner")  # invalid combo
    with pytest.raises(ValueError):
        ClientServerExecutionStrategy(role="both", main_process="runner", n_runners=2)

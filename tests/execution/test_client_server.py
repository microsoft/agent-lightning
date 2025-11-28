# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import multiprocessing
import signal
import socket
import sys
import tempfile
import time
from contextlib import closing
from multiprocessing import Event as MpEvent
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.execution.events import ExecutionEvent
from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient
from agentlightning.store.memory import InMemoryLightningStore

from ..store.dummy_store import DummyLightningStore, minimal_dummy_store

# =========================
# Helpers & Fixtures
# =========================


def _free_port() -> int:
    """Return an available TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


RecordedCall = Tuple[str, Tuple[Any, ...], Dict[str, Any]]


class DummyEvt(ExecutionEvent):
    """Simple in-process ExecutionEvent-like object required by the strategy."""

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
# Environment configuration
# =========================


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGL_CURRENT_ROLE", raising=False)
    monkeypatch.delenv("AGL_SERVER_HOST", raising=False)
    monkeypatch.delenv("AGL_SERVER_PORT", raising=False)
    monkeypatch.delenv("AGL_MANAGED_STORE", raising=False)


def test_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    port = _free_port()
    monkeypatch.setenv("AGL_CURRENT_ROLE", "both")
    monkeypatch.setenv("AGL_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("AGL_SERVER_PORT", str(port))

    strat = ClientServerExecutionStrategy(main_process="runner")

    assert strat.role == "both"
    assert strat.server_host == "0.0.0.0"
    assert strat.server_port == port
    assert strat.main_process == "runner"


def test_env_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGL_CURRENT_ROLE", "algorithm")

    strat = ClientServerExecutionStrategy()

    assert strat.server_host == "localhost"
    assert strat.server_port == 4747
    assert strat.main_process == "algorithm"
    assert strat.managed_store is True


def test_env_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGL_CURRENT_ROLE", "algorithm")
    monkeypatch.setenv("AGL_SERVER_PORT", "not-an-int")

    with pytest.raises(ValueError, match="AGL_SERVER_PORT must be an integer"):
        ClientServerExecutionStrategy()


def test_env_missing_role(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    # When role is None and env var is not set, it defaults to "both"
    # So this test no longer applies - it should succeed
    strat = ClientServerExecutionStrategy()
    assert strat.role == "both"


def test_env_managed_store(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGL_CURRENT_ROLE", "algorithm")
    monkeypatch.setenv("AGL_MANAGED_STORE", "0")

    strat = ClientServerExecutionStrategy()

    assert strat.managed_store is False


def test_explicit_managed_store_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGL_MANAGED_STORE", "0")

    strat = ClientServerExecutionStrategy(role="algorithm", managed_store=True)

    assert strat.managed_store is True


def test_env_managed_store_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGL_CURRENT_ROLE", "algorithm")
    monkeypatch.setenv("AGL_MANAGED_STORE", "maybe")

    with pytest.raises(ValueError, match="AGL_MANAGED_STORE must be one of"):
        ClientServerExecutionStrategy()


# =========================
# Managed store behaviour
# =========================


def test_execute_algorithm_managed_store_starts_server(
    monkeypatch: pytest.MonkeyPatch, store: DummyLightningStore
) -> None:
    created: list["RecordingServer"] = []

    class RecordingServer(LightningStore):  # type: ignore[misc]
        def __init__(self, wrapped: LightningStore, host: str, port: int) -> None:
            self.wrapped = wrapped
            self.host = host
            self.port = port
            self.endpoint = f"http://{host}:{port}"
            self.started = False
            self.stopped = False
            created.append(self)

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("agentlightning.execution.client_server.LightningStoreServer", RecordingServer)

    strat = ClientServerExecutionStrategy(
        role="algorithm",
        managed_store=True,
        server_host="127.0.0.1",
        server_port=8123,
    )

    seen_store: list[LightningStore] = []

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        seen_store.append(store)
        event.set()

    asyncio.run(strat._execute_algorithm(algo, store, DummyEvt()))  # pyright: ignore[reportPrivateUsage]

    assert len(created) == 1
    server = created[0]
    assert server.wrapped is store
    assert server.started is True
    assert server.stopped is True
    assert seen_store and isinstance(seen_store[0], RecordingServer)


def test_execute_algorithm_unmanaged_uses_provided_store(
    monkeypatch: pytest.MonkeyPatch, store: DummyLightningStore
) -> None:
    class ShouldNotBeCalled:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Server wrapper should not be constructed when unmanaged")

    monkeypatch.setattr("agentlightning.execution.client_server.LightningStoreServer", ShouldNotBeCalled)

    strat = ClientServerExecutionStrategy(role="algorithm", managed_store=False)

    seen: dict[str, LightningStore] = {}

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        seen["store"] = store
        event.set()

    asyncio.run(strat._execute_algorithm(algo, store, DummyEvt()))  # pyright: ignore[reportPrivateUsage]

    assert seen["store"] is store


def test_execute_runner_managed_creates_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingClient(LightningStore):  # type: ignore[misc]
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("agentlightning.execution.client_server.LightningStoreClient", RecordingClient)

    strat = ClientServerExecutionStrategy(
        role="runner",
        managed_store=True,
        server_host="localhost",
        server_port=9001,
    )

    seen: dict[str, LightningStore] = {}

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        seen["store"] = store
        event.set()

    dummy_store = DummyLightningStore({})
    asyncio.run(strat._execute_runner(runner, 0, dummy_store, DummyEvt()))  # pyright: ignore[reportPrivateUsage]

    client = seen["store"]
    assert isinstance(client, RecordingClient)
    assert client.url == "http://localhost:9001"
    assert client.closed is True


def test_execute_runner_unmanaged_requires_store() -> None:
    strat = ClientServerExecutionStrategy(role="runner", managed_store=False)

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        _ = (store, worker_id, event)

    # When managed_store=False, a store must be provided via the store parameter
    provided_store = DummyLightningStore({})
    asyncio.run(strat._execute_runner(runner, 0, provided_store, DummyEvt()))  # pyright: ignore[reportPrivateUsage]


def test_execute_runner_unmanaged_uses_provided_store() -> None:
    strat = ClientServerExecutionStrategy(role="runner", managed_store=False)

    class ProvidedStore(LightningStore):
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    provided = ProvidedStore()
    seen: dict[str, LightningStore] = {}

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        seen["store"] = store
        event.set()

    asyncio.run(strat._execute_runner(runner, 1, provided, DummyEvt()))  # pyright: ignore[reportPrivateUsage]

    assert seen["store"] is provided
    assert provided.close_calls == 0


# =========================
# Async bundles for tests
# =========================


async def _noop_algorithm(store: LightningStore, event: ExecutionEvent) -> None:
    _ = store  # explicitly acknowledge unused parameter
    await asyncio.sleep(0)
    assert not event.is_set()


async def _algo_calls_store_enqueue(store: LightningStore, event: ExecutionEvent) -> None:
    # Calls a delegated method on the server wrapper; real server is running.
    await store.enqueue_rollout(input={"x": 1})
    await asyncio.sleep(0)
    assert not event.is_set()


async def _algo_sets_stop_delayed(store: LightningStore, event: ExecutionEvent, delay: float = 0.05) -> None:
    _ = store
    await asyncio.sleep(delay)
    event.set()


async def _raise_in_algorithm(store: LightningStore, event: ExecutionEvent) -> None:
    _ = store
    event.set()
    raise RuntimeError("algo boom")


async def _kbint_in_algorithm(store: LightningStore, event: ExecutionEvent) -> None:
    _ = store
    event.set()
    raise KeyboardInterrupt()


def _subprocess_algorithm_write_util(store: LightningStore) -> None:
    """Subprocess that writes to the store."""

    async def do_work() -> None:
        # This should auto-delegate to HTTP client in subprocess
        await store.enqueue_rollout(input={"origin": "algo-subprocess"})

    asyncio.run(do_work())


async def _subprocess_algorithm(store: LightningStore, event: ExecutionEvent) -> None:
    """Algorithm that spawns a subprocess to write to the store."""
    # Spawn subprocess to write
    ctx = multiprocessing.get_context()
    process = ctx.Process(target=_subprocess_algorithm_write_util, args=(store,))
    process.start()
    await asyncio.to_thread(process.join, timeout=5.0)

    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Subprocess hung")

    assert process.exitcode == 0

    event.set()
    # Wait for client to see the signal
    await asyncio.sleep(0.5)

    # Algorithm should see the write
    algo_rollouts = await store.query_rollouts()
    assert len(algo_rollouts) == 1, "Algorithm should see the write"


async def _noop_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    _ = (store, worker_id)
    await asyncio.sleep(0)
    assert not event.is_set()


async def _runner_wait_for_stop(
    store: LightningStore, worker_id: int, event: ExecutionEvent, timeout: float = 0.5
) -> None:
    _ = (store, worker_id)
    t0: float = time.monotonic()
    while not event.is_set() and time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.005)


async def _runner_ignores_stop_forever(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    # Ignore signals to force escalation.
    _ = (store, worker_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        await asyncio.sleep(0.1)


async def _raise_in_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    _ = (store, worker_id)
    event.set()
    raise RuntimeError("runner boom")


async def _kbint_in_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    _ = (store, worker_id)
    event.set()
    raise KeyboardInterrupt()


async def _timeout_error_in_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    # Provoke client's validation (pre-request), then raise TimeoutError.
    with pytest.raises(ValueError):
        await store.wait_for_rollouts(rollout_ids=["r1"], timeout=0.2)
    _ = worker_id
    event.set()
    raise TimeoutError("runner timeout")


async def _waiting_for_rollout_runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
    """Runner that waits for algorithm signal then checks store."""
    # Wait for algorithm to finish writing
    t0 = time.monotonic()
    while not event.is_set() and time.monotonic() - t0 < 5.0:
        await asyncio.sleep(0.05)

    # Runner should see the write via HTTP
    runner_rollouts = await store.query_rollouts()
    assert len(runner_rollouts) == 1, "Runner should see the write"


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
        alive: List[Process] = strat._join_until_deadline([p], timeout=0.005)  # pyright: ignore[reportPrivateUsage]
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
    alive: List[Process] = strat._join_until_deadline([p], timeout=0.05)  # pyright: ignore[reportPrivateUsage]
    assert alive == []


def test_join_until_deadline_zero_timeout_path() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    ctx = get_context()
    p: Process = ctx.Process(target=time.sleep, args=(0.2,), name="zero-join")
    p.start()
    try:
        alive: List[Process] = strat._join_until_deadline([p], timeout=0.0)  # pyright: ignore[reportPrivateUsage]
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
        pid = proc.pid
        if pid is not None:
            seen.append(pid)  # record
        if len(seen) == 1:
            raise RuntimeError("deliberate")  # ensure suppression

    try:
        strat._signal_processes([p], action)  # pyright: ignore[reportPrivateUsage]
        assert seen == [p.pid]
    finally:
        p.terminate()
        p.join()


def test_shutdown_processes_empty_list_noop() -> None:
    strat = ClientServerExecutionStrategy(role="runner", server_port=_free_port())
    strat._shutdown_processes([], DummyEvt())  # pyright: ignore[reportPrivateUsage]


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
        strat._shutdown_processes([p], DummyEvt())  # pyright: ignore[reportPrivateUsage]
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
        strat._shutdown_processes([p], DummyEvt())  # pyright: ignore[reportPrivateUsage]
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
        strat._shutdown_processes([p], DummyEvt())  # pyright: ignore[reportPrivateUsage]
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
        strat._shutdown_processes([p], DummyEvt())  # pyright: ignore[reportPrivateUsage]
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
        strat._shutdown_processes([p], DummyEvt())  # pyright: ignore[reportPrivateUsage]
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
        strat._shutdown_processes([p], evt)  # pyright: ignore[reportPrivateUsage]
        assert not p.is_alive()
    finally:
        if p.is_alive():
            p.terminate()
        p.join()


# =========================
# Private coroutine tests using REAL server
# =========================


def test_execute_algorithm_success_invokes_store(store: DummyLightningStore) -> None:
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
    asyncio.run(
        strat._execute_algorithm(_algo_calls_store_enqueue, store, DummyEvt())  # pyright: ignore[reportPrivateUsage]
    )
    # The DummyLightningStore should have recorded the delegated call.
    recorded: List[RecordedCall] = store.calls
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
        asyncio.run(strat._execute_algorithm(_raise_in_algorithm, store, evt))  # pyright: ignore[reportPrivateUsage]
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
        asyncio.run(strat._execute_algorithm(_kbint_in_algorithm, store, evt))  # pyright: ignore[reportPrivateUsage]
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
        dummy_store = DummyLightningStore({})
        asyncio.run(
            strat._execute_runner(  # pyright: ignore[reportPrivateUsage]
                _noop_runner, worker_id=0, store=dummy_store, stop_evt=DummyEvt()
            )
        )
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
        dummy_store = DummyLightningStore({})
        with pytest.raises(RuntimeError, match="runner boom"):
            asyncio.run(
                strat._execute_runner(  # pyright: ignore[reportPrivateUsage]
                    _raise_in_runner, worker_id=7, store=dummy_store, stop_evt=evt
                )
            )
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
    dummy_store = DummyLightningStore({})
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            strat._execute_runner(  # pyright: ignore[reportPrivateUsage]
                _kbint_in_runner, worker_id=0, store=dummy_store, stop_evt=evt
            )
        )
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
    dummy_store = DummyLightningStore({})
    with pytest.raises(TimeoutError, match="runner timeout"):
        asyncio.run(
            strat._execute_runner(  # pyright: ignore[reportPrivateUsage]
                _timeout_error_in_runner, worker_id=0, store=dummy_store, stop_evt=evt
            )
        )
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
    stop_evt: ExecutionEvent = MpEvent()

    def runner_sync() -> None:
        dummy_store = DummyLightningStore({})
        asyncio.run(
            strat._execute_runner(  # pyright: ignore[reportPrivateUsage]
                _runner_wait_for_stop, worker_id=0, store=dummy_store, stop_evt=stop_evt
            )
        )

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
    stop_evt: ExecutionEvent = MpEvent()

    p: Process = strat._spawn_algorithm_process(  # pyright: ignore[reportPrivateUsage]
        _noop_algorithm, store, stop_evt, ctx=ctx
    )
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

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        await _algo_sets_stop_delayed(store, event, delay=0.05)

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

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        _ = (store, worker_id)
        await asyncio.sleep(0.05)
        event.set()

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        _ = store
        t0: float = time.monotonic()
        while not event.is_set() and time.monotonic() - t0 < 1.0:
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
    # ``execute`` handles KeyboardInterrupt raised by the algorithm by
    # initiating a cooperative shutdown without propagating the exception to the
    # caller. The method should therefore return cleanly.
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


def test_execute_main_runner_rejects_multiple_runners(store: LightningStore) -> None:
    """
    main_process='runner' should validate that n_runners=1 and raise ValueError otherwise.
    """
    port: int = _free_port()
    with pytest.raises(ValueError, match="main_process='runner' requires n_runners to be 1"):
        strat = ClientServerExecutionStrategy(
            role="both",
            main_process="runner",
            n_runners=2,  # Invalid: must be 1
            server_host="127.0.0.1",
            server_port=port,
        )
        strat.execute(algorithm=_noop_algorithm, runner=_noop_runner, store=store)


def test_execute_main_runner_waits_for_algorithm_completion(store: LightningStore) -> None:
    """
    When main_process='runner', the algorithm process should complete fully
    even after the runner finishes. This test verifies the algorithm process
    is joined properly instead of being interrupted prematurely.
    """
    port: int = _free_port()
    temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    marker_file = Path(temp_file.name)
    temp_file.close()

    try:

        async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
            _ = (store, worker_id)
            # Runner completes quickly
            await asyncio.sleep(0.01)

        async def algo(store: LightningStore, event: ExecutionEvent) -> None:
            _ = store
            # Algorithm takes longer and should complete fully
            await asyncio.sleep(0.5)
            marker_file.write_text("done")
            event.set()

        strat = ClientServerExecutionStrategy(
            role="both",
            main_process="runner",
            n_runners=1,
            server_host="127.0.0.1",
            server_port=port,
            graceful_timeout=0.05,
            terminate_timeout=0.05,
        )
        strat.execute(algorithm=algo, runner=runner, store=store)

        # Verify the algorithm completed by checking the marker file exists
        assert marker_file.exists()
        assert marker_file.read_text() == "done"
    finally:
        if marker_file.exists():
            marker_file.unlink()


def test_execute_both_main_algo_runner_ignores_stop(store: LightningStore) -> None:
    """
    When main_process='algorithm' and runners ignore stop, the algorithm
    setting stop_evt should trigger shutdown that forcefully terminates runners.
    """
    port: int = _free_port()
    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="algorithm",
        n_runners=2,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=0.05,
        terminate_timeout=0.05,
    )

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        _ = store
        await asyncio.sleep(0.1)

    # Should complete without hanging despite runners ignoring signals
    with pytest.raises(RuntimeError, match="Subprocesses failed"):
        strat.execute(algorithm=algo, runner=_runner_ignores_stop_forever, store=store)


@pytest.mark.parametrize("main_process", ["algorithm", "runner"])
def test_subprocess_spawned_in_algorithm_visible_to_all(main_process: str) -> None:
    """
    Test that when the algorithm spawns a subprocess that writes to the store,
    those writes are visible to:
    - The algorithm itself (via LightningStoreServer auto-delegation)
    - The runner processes (via HTTP client)
    - The main pytest process (only when main_process='algorithm')

    When main_process='runner', the algorithm runs in a subprocess, so the main
    pytest process won't see the changes (they're isolated to that subprocess).
    """
    port: int = _free_port()
    store = InMemoryLightningStore()

    strat = ClientServerExecutionStrategy(
        role="both",
        main_process=main_process,  # type: ignore
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=2.0,
        terminate_timeout=2.0,
    )

    strat.execute(algorithm=_subprocess_algorithm, runner=_waiting_for_rollout_runner, store=store)

    # Main process check depends on which process the algorithm ran in
    if main_process == "algorithm":
        # Algorithm ran in main process, so main process should see the write
        main_rollouts = asyncio.run(store.query_rollouts())
        assert len(main_rollouts) == 1, "Main process should see write when algorithm runs in main process"
    else:
        # Algorithm ran in subprocess, main process won't see the write
        # (it's isolated to the subprocess's copy of the store)
        main_rollouts = asyncio.run(store.query_rollouts())
        assert len(main_rollouts) == 0, "Main process should NOT see write when algorithm runs in subprocess"


def test_execute_main_runner_store_state_isolated_in_subprocess(store: DummyLightningStore) -> None:
    """
    When main_process='runner', the algorithm runs in a subprocess with the store.
    Any state modifications made during execution remain in that subprocess and
    are NOT reflected in the original store object passed to execute().
    """
    port: int = _free_port()

    async def runner(store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        _ = (store, worker_id)
        # Runner completes quickly
        await asyncio.sleep(0.05)
        event.set()

    async def algo(store: LightningStore, event: ExecutionEvent) -> None:
        # Algorithm modifies the store
        await store.enqueue_rollout(input={"x": 42})
        # Wait for runner to signal completion
        t0: float = time.monotonic()
        while not event.is_set() and time.monotonic() - t0 < 1.0:
            await asyncio.sleep(0.005)

    # Record initial state
    initial_call_count = len(store.calls)

    strat = ClientServerExecutionStrategy(
        role="both",
        main_process="runner",
        n_runners=1,
        server_host="127.0.0.1",
        server_port=port,
        graceful_timeout=5.0,
        terminate_timeout=5.0,
    )
    strat.execute(algorithm=algo, runner=runner, store=store)

    # Verify the original store object was NOT modified
    # (the enqueue_rollout call happened in the subprocess)
    assert (
        len(store.calls) == initial_call_count
    ), "Store state should not be modified in main process when main_process='runner'"

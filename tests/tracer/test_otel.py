# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import multiprocessing
import pickle
import threading
import time
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, Request
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from portpicker import pick_unused_port

from agentlightning.reward import emit_reward, find_reward_spans, get_reward_value, reward
from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import LightningSpanProcessor
from agentlightning.tracer.otel import OtelTracer
from agentlightning.utils import otlp

from ..common.tracer import clear_agentops_init, clear_tracer_provider


def create_span(name: str, sampled: bool = True, with_context: bool = True) -> MagicMock:
    """Helper to create mock spans with different properties."""
    span = MagicMock(spec=ReadableSpan)
    span.name = name
    if with_context:
        span.context = SpanContext(
            trace_id=hash(name) % (2**64),
            span_id=hash(name) % (2**64),
            is_remote=False,
            trace_flags=TraceFlags(0x01 if sampled else 0x00),
        )
    else:
        span.context = None
    return span


def create_mock_store(otlp_supported: bool = False) -> MagicMock:
    """Helper to create a mock LightningStore."""
    store = MagicMock(spec=LightningStore)
    store.add_otel_span = AsyncMock(return_value=None)
    store.capabilities = {"otlp_traces": otlp_supported}
    store.otlp_traces_endpoint.return_value = "http://store/v1/traces"
    return store


@pytest.fixture(params=[False, True], ids=["store-no-otlp", "store-otlp"])
def store_supports_otlp(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture
def otlp_server(store_supports_otlp: bool):
    if not store_supports_otlp:
        yield None
        return

    app = FastAPI()
    received: List[ExportTraceServiceRequest] = []

    @app.post("/v1/traces")
    async def _export_traces(request: Request):  # type: ignore
        async def capture(message: ExportTraceServiceRequest) -> None:
            received.append(message)

        return await otlp.handle_otlp_export(
            request,
            ExportTraceServiceRequest,
            ExportTraceServiceResponse,
            capture,
            signal_name="traces",
        )

    port = pick_unused_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    timeout = time.time() + 5
    while not getattr(server, "started", False):
        if time.time() > timeout:
            raise RuntimeError("OTLP test server failed to start")
        if not thread.is_alive():
            raise RuntimeError("OTLP test server thread exited before startup")
        time.sleep(0.01)

    try:
        yield {"url": f"http://127.0.0.1:{port}/v1/traces", "received": received}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def store(store_supports_otlp: bool, otlp_server: Optional[Dict[str, Any]]) -> MagicMock:
    mock_store = create_mock_store(store_supports_otlp)
    if store_supports_otlp:
        assert otlp_server is not None
        mock_store.otlp_traces_endpoint.return_value = otlp_server["url"]
    return mock_store


def test_initialization_and_shutdown():
    """Test processor lifecycle: initialization, loop thread, and shutdown."""
    processor = LightningSpanProcessor()

    # Verify initialization
    assert processor._spans == []
    assert processor._store is None
    assert processor._rollout_id is None
    assert processor._attempt_id is None

    assert processor._loop is None
    assert processor._loop_thread is None

    # Start the loop
    processor._ensure_loop()

    # Verify loop thread is running correctly
    assert processor._loop is not None
    assert processor._loop.is_running()
    assert processor._loop_thread is not None
    assert processor._loop_thread.is_alive()
    assert processor._loop_thread.daemon is True
    assert processor._loop_thread.name == "otel-loop"

    # Verify shutdown stops everything
    thread = processor._loop_thread
    processor.shutdown()

    time.sleep(0.1)  # Give thread time to stop
    assert not thread.is_alive()
    assert processor._loop is None

    # Verify double shutdown is safe (idempotent)
    processor.shutdown()  # Should not raise
    assert processor._loop is None


def test_span_collection_with_filtering():
    """Test that spans are collected with proper filtering."""
    processor = LightningSpanProcessor()

    sampled_span = create_span("sampled", sampled=True)
    unsampled_span = create_span("unsampled", sampled=False)
    no_context_span = create_span("no_context", with_context=False)

    # Process different types of spans
    processor.on_end(sampled_span)
    processor.on_end(unsampled_span)
    processor.on_end(no_context_span)

    # Only sampled span with context should be collected
    collected = processor.spans()
    assert len(collected) == 1
    assert collected[0] == sampled_span

    processor.shutdown()


def test_context_managers_clear_state(store: MagicMock):
    """Test that both context managers properly manage state."""
    processor = LightningSpanProcessor()

    # Add a span first
    span1 = create_span("span1")
    processor.on_end(span1)
    assert len(processor.spans()) == 1

    # Test basic context manager: __enter__ clears spans
    with processor:
        assert len(processor.spans()) == 0
        span2 = create_span("span2")
        processor.on_end(span2)
        assert len(processor.spans()) == 1

    # After exit, store context should be cleared
    assert processor._store is None
    assert processor._rollout_id is None
    assert processor._attempt_id is None

    # Test with_context: sets and clears store context
    with processor.with_context(store=store, rollout_id="r1", attempt_id="a1") as proc:
        assert proc is processor
        assert processor._store is store
        assert processor._rollout_id == "r1"
        assert processor._attempt_id == "a1"

    # After exit, context should be cleared
    assert processor._store is None
    assert processor._rollout_id is None
    assert processor._attempt_id is None

    processor.shutdown()


def test_store_integration_complete(store: MagicMock):
    """Test all store integration scenarios: writes, errors, timeout, thread verification."""
    processor = LightningSpanProcessor()

    # Test 1: Successful store writes
    with processor.with_context(store=store, rollout_id="r1", attempt_id="a1"):
        span1 = create_span("span1")
        span2 = create_span("span2")
        processor.on_end(span1)
        processor.on_end(span2)

    # Verify both spans written to store
    assert store.add_otel_span.call_count == 2
    assert len(processor.spans()) == 2

    # Verify call arguments
    calls = store.add_otel_span.call_args_list
    assert calls[0][0] == ("r1", "a1", span1)
    assert calls[1][0] == ("r1", "a1", span2)

    # Test 2: Store write errors are caught and don't crash
    store.add_otel_span.reset_mock()
    store.add_otel_span.side_effect = RuntimeError("Store failure")

    with processor.with_context(store=store, rollout_id="r2", attempt_id="a2"):
        span3 = create_span("span3")
        processor.on_end(span3)  # Should not raise

    # Span still collected despite store error
    assert len(processor.spans()) == 1
    store.add_otel_span.side_effect = None

    # Test 3: Verify writes happen in loop thread
    execution_thread = None

    async def track_thread(*args: Any, **kwargs: Any):
        nonlocal execution_thread
        execution_thread = threading.current_thread()

    store.add_otel_span = AsyncMock(side_effect=track_thread)

    with processor.with_context(store=store, rollout_id="r3", attempt_id="a3"):
        span4 = create_span("span4")
        processor.on_end(span4)

    assert execution_thread is not None
    assert execution_thread.name == "otel-loop"

    # Test 4: Unsampled and no-context spans don't trigger store writes
    store.add_otel_span.reset_mock()
    store.add_otel_span = AsyncMock()

    with processor.with_context(store=store, rollout_id="r4", attempt_id="a4"):
        unsampled = create_span("unsampled", sampled=False)
        no_ctx = create_span("no_ctx", with_context=False)
        processor.on_end(unsampled)
        processor.on_end(no_ctx)

    # Store should not be called for filtered spans
    store.add_otel_span.assert_not_called()

    processor.shutdown()


def test_event_loop_operations():
    """Test _await_in_loop: execution, exceptions, and timeout."""
    processor = LightningSpanProcessor()

    # Test 1: Successful coroutine execution
    async def successful_coro():
        await asyncio.sleep(0.01)
        return "success"

    result = processor._await_in_loop(successful_coro())
    assert result == "success"

    # Test 2: Exception propagation
    async def failing_coro():
        raise ValueError("Expected error")

    with pytest.raises(ValueError, match="Expected error"):
        processor._await_in_loop(failing_coro())

    # Test 3: Timeout handling
    async def slow_coro():
        await asyncio.sleep(10)
        return "too_slow"

    with pytest.raises(Exception):  # Should timeout
        processor._await_in_loop(slow_coro(), timeout=0.1)

    processor.shutdown()


def test_concurrent_access(store: MagicMock):
    """Test thread-safe concurrent span processing."""
    processor = LightningSpanProcessor()

    num_threads = 10
    spans_per_thread = 5
    barrier = threading.Barrier(num_threads)

    def process_spans(thread_id: int) -> None:
        barrier.wait()  # Synchronize all threads to start together
        for i in range(spans_per_thread):
            span = create_span(f"thread{thread_id}_span{i}")
            processor.on_end(span)

    with processor.with_context(store=store, rollout_id="concurrent", attempt_id="test"):
        threads = [threading.Thread(target=process_spans, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # All spans should be collected and written
    assert len(processor.spans()) == num_threads * spans_per_thread
    assert store.add_otel_span.call_count == num_threads * spans_per_thread

    processor.shutdown()


def test_multiprocessing_behavior(store_supports_otlp: bool):
    """Test processor behavior across process boundaries."""

    # Test 1: Creating new processor in subprocess works
    def subprocess_task(result_queue: "multiprocessing.Queue[tuple[str, Any]]", supports_otlp: bool) -> None:
        try:
            processor = LightningSpanProcessor()

            span = create_span("subprocess_span")
            processor.on_end(span)

            # Loop is not used
            assert processor._loop is None
            assert processor._loop_thread is None

            with processor.with_context(store=create_mock_store(supports_otlp), rollout_id="r1", attempt_id="a1"):
                processor.on_end(span)

            # Verify processor works in new process
            assert processor._loop is not None
            assert processor._loop_thread is not None
            assert processor._loop_thread.is_alive()

            result_queue.put(("success", len(processor.spans())))
            processor.shutdown()
        except Exception as e:
            result_queue.put(("error", str(e)))

    result_queue: multiprocessing.Queue[tuple[str, Any]] = multiprocessing.Queue()
    process = multiprocessing.Process(target=subprocess_task, args=(result_queue, store_supports_otlp))
    process.start()
    process.join(timeout=5)

    assert not process.is_alive()
    status, value = result_queue.get(timeout=1)
    assert status == "success", f"Subprocess failed: {status}, {value}"
    assert value == 1

    # Test 2: Processor cannot be pickled (threads aren't picklable)
    processor = LightningSpanProcessor()

    with pytest.raises((TypeError, AttributeError)):
        pickle.dumps(processor)

    processor.shutdown()


def test_store_write_timeout(store: MagicMock):
    """Test that slow store writes respect timeout."""
    processor = LightningSpanProcessor()

    # Create a slow async function that exceeds timeout
    async def slow_write(*args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(10)

    store.add_otel_span = AsyncMock(side_effect=slow_write)

    with processor.with_context(store=store, rollout_id="r1", attempt_id="a1"):
        span = create_span("test_span")
        # Should not raise - timeout is caught in on_end
        processor.on_end(span)

    # Span still collected despite timeout
    # Note: The timeout in on_end is 5.0 seconds, but the error is caught
    assert len(processor.spans()) == 1

    processor.shutdown()


def test_multiple_processors_in_same_process():
    """Test that multiple processors can coexist in the same process."""
    processor1 = LightningSpanProcessor()
    processor2 = LightningSpanProcessor()

    processor1._ensure_loop()
    processor2._ensure_loop()

    # Both should have independent loops and threads
    assert processor1._loop is not processor2._loop
    assert processor1._loop_thread is not processor2._loop_thread
    assert processor1._loop is not None and processor1._loop.is_running()
    assert processor2._loop is not None and processor2._loop.is_running()

    # Both should work independently
    span1 = create_span("p1_span")
    span2 = create_span("p2_span")

    processor1.on_end(span1)
    processor2.on_end(span2)

    assert len(processor1.spans()) == 1
    assert len(processor2.spans()) == 1
    assert processor1.spans()[0] == span1
    assert processor2.spans()[0] == span2

    processor1.shutdown()
    processor2.shutdown()


def test_context_manager_reusability(store: MagicMock):
    """Test that context managers can be entered and exited multiple times."""
    processor = LightningSpanProcessor()

    # First usage
    with processor.with_context(store=store, rollout_id="r1", attempt_id="a1"):
        span1 = create_span("span1")
        processor.on_end(span1)

    assert len(processor.spans()) == 1
    assert processor._store is None

    # Second usage - should work fine
    with processor.with_context(store=store, rollout_id="r2", attempt_id="a2"):
        span2 = create_span("span2")
        processor.on_end(span2)

    # Spans should be cleared from first context, only span2 present
    assert len(processor.spans()) == 1
    assert processor.spans()[0].name == "span2"
    assert processor._store is None

    processor.shutdown()


def _otel_reward_subprocess(mode: str, conn: Connection[tuple[str, Any]]) -> None:
    asyncio.run(_otel_reward_subprocess_async(mode, conn))


async def _otel_reward_subprocess_async(mode: str, conn: Connection[tuple[str, Any]]) -> None:
    tracer: OtelTracer | None = None
    try:
        try:
            clear_agentops_init()
        except Exception:
            # Some environments ship a minimal agentops stub without tracer helpers.
            pass
        clear_tracer_provider()

        tracer = OtelTracer()
        tracer.init_worker(0)

        if mode == "decorator":
            expected_reward = 2.5

            @reward
            def compute_reward() -> float:
                return expected_reward

            async with tracer.trace_context(name="reward-decorator"):
                returned = compute_reward()
                if returned != expected_reward:
                    raise AssertionError(f"Expected reward {expected_reward}, got {returned}")

        elif mode == "emit":
            expected_reward = 4.5
            async with tracer.trace_context(name="reward-emit"):
                emit_reward(expected_reward)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        spans = tracer.get_last_trace()
        total_spans = len(spans)
        reward_spans = find_reward_spans(spans)
        if not reward_spans:
            raise AssertionError("No reward spans captured by OtelTracer.")

        actual_reward = get_reward_value(reward_spans[-1])
        if actual_reward is None:
            raise AssertionError("Captured reward span does not contain a reward value.")

        conn.send(
            (
                "success",
                {
                    "mode": mode,
                    "reward": actual_reward,
                    "expected_reward": expected_reward,
                    "total_spans": total_spans,
                    "reward_spans": len(reward_spans),
                },
            )
        )
    except Exception as exc:  # pragma: no cover - defensive path
        try:
            conn.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        if tracer is not None:
            tracer.teardown_worker(0)
        conn.close()


def _run_otel_reward_test(mode: str) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_otel_reward_subprocess, args=(mode, child_conn))
    proc.start()
    proc.join(15.0)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        pytest.fail(f"Subprocess for mode {mode!r} hung.")

    try:
        if not parent_conn.poll(1.0):
            pytest.fail(f"No result from subprocess for mode {mode!r}: timed out")
        status, payload = parent_conn.recv()
    finally:
        parent_conn.close()

    if status != "success":
        pytest.fail(f"Subprocess for mode {mode!r} failed: {payload}")

    assert proc.exitcode == 0, f"Subprocess for mode {mode!r} exited with code {proc.exitcode}"
    return payload


def test_otel_tracer_captures_reward_from_decorator():
    payload = _run_otel_reward_test("decorator")
    assert payload["reward_spans"] == 1
    assert payload["total_spans"] == 1
    assert payload["reward"] == pytest.approx(payload["expected_reward"])  # pyright: ignore[reportUnknownMemberType]
    assert payload["expected_reward"] == pytest.approx(2.5)  # pyright: ignore[reportUnknownMemberType]


def test_otel_tracer_captures_reward_from_emit_reward():
    payload = _run_otel_reward_test("emit")
    assert payload["reward_spans"] == 1
    assert payload["total_spans"] == 1
    assert payload["reward"] == pytest.approx(payload["expected_reward"])  # pyright: ignore[reportUnknownMemberType]
    assert payload["expected_reward"] == pytest.approx(4.5)  # pyright: ignore[reportUnknownMemberType]

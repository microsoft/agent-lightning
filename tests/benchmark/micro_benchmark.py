# Copyright (c) Microsoft. All rights reserved.

"""Micro benchmarks for the store."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

import agentlightning as agl
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus
from agentlightning.utils.system_snapshot import system_snapshot

console = Console()


def _close_store_client(store: agl.LightningStoreClient) -> None:
    try:
        asyncio.run(store.close())
    except Exception:
        pass


def _make_span(rollout_id: str, attempt_id: str, sequence_id: int, name: str) -> Span:
    trace_hex = f"{sequence_id:032x}"
    span_hex = f"{sequence_id:016x}"
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id=trace_hex,
        span_id=span_hex,
        parent_id=None,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes={},
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


@dataclass
class BenchmarkSummary:
    mode: str
    total_tasks: int
    successes: int
    duration: float

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successes / self.total_tasks

    @property
    def throughput(self) -> float:
        if self.duration <= 0:
            return 0.0
        return self.successes / self.duration


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro benchmarks for the store.")
    parser.add_argument("--store-url", default="http://localhost:4747", help="Lightning Store endpoint base URL.")
    parser.add_argument("--summary-file", help="File to append final benchmark summary.")
    parser.add_argument(
        "mode",
        choices=("worker", "dequeue-empty", "rollout"),
        help="Mode to exercise different operations.",
    )
    args = parser.parse_args(argv)
    return args


def _update_worker_task(args: tuple[str, str, str]) -> bool:
    store_url, worker_id, task_id = args
    console.print(f"Updating worker {worker_id} for task {task_id}")
    store = agl.LightningStoreClient(store_url)
    try:
        asyncio.run(store.update_worker(worker_id, system_snapshot()))
        return True
    except Exception as e:
        console.print(f"Error updating worker {worker_id} for task {task_id}: {e}")
        return False
    finally:
        _close_store_client(store)


def simulate_many_update_workers(store_url: str) -> BenchmarkSummary:
    """Simulate many update workers."""

    start_time = time.time()

    # Use a multiprocessing pool to update workers.
    worker_ids = [(f"Worker-{i % 1024}", f"Task-{j}") for i in range(1024) for j in range(10)]
    with multiprocessing.get_context("fork").Pool(processes=1024) as pool:
        successful_tasks = pool.map(_update_worker_task, [(store_url, *worker_id) for worker_id in worker_ids])

    end_time = time.time()
    successes = sum(successful_tasks)
    duration = end_time - start_time
    throughput = successes / duration if duration > 0 else 0.0
    console.print(f"Success rate: {successes / len(worker_ids):.3f}")
    console.print(f"Time taken: {duration:.3f} seconds")
    console.print(f"Throughput: {throughput:.3f} workers/second")
    return BenchmarkSummary(mode="worker", total_tasks=len(worker_ids), successes=successes, duration=duration)


def _dequeue_empty_and_update_workers_task(args: tuple[str, str, str]) -> bool:
    store_url, worker_id, task_id = args
    console.print(f"Dequeueing empty and updating worker {worker_id} for task {task_id}")
    store = agl.LightningStoreClient(store_url)

    async def _async_task() -> None:
        await store.dequeue_rollout(worker_id=worker_id)
        await store.update_worker(worker_id, system_snapshot())

    try:
        asyncio.run(_async_task())
        return True
    except Exception as e:
        console.print(f"Error dequeueing empty and updating worker {worker_id} for task {task_id}: {e}")
        return False
    finally:
        _close_store_client(store)


def simulate_dequeue_empty_and_update_workers(store_url: str) -> BenchmarkSummary:
    """Simulate dequeue empty and update workers."""
    start_time = time.time()

    worker_ids = [(f"Worker-{i % 1024}", f"Task-{j}") for i in range(1024) for j in range(10)]
    with multiprocessing.get_context("fork").Pool(processes=1024) as pool:
        successful_tasks = pool.map(
            _dequeue_empty_and_update_workers_task, [(store_url, *worker_id) for worker_id in worker_ids]
        )

    end_time = time.time()
    successes = sum(successful_tasks)
    duration = end_time - start_time
    throughput = successes / duration if duration > 0 else 0.0
    console.print(f"Success rate: {successes / len(worker_ids):.3f}")
    console.print(f"Time taken: {duration:.3f} seconds")
    console.print(f"Throughput: {throughput:.3f} workers/second")
    return BenchmarkSummary(mode="dequeue-empty", total_tasks=len(worker_ids), successes=successes, duration=duration)


def _rollout_flow_task(args: tuple[str, int, int]) -> bool:
    store_url, task_id, spans_per_attempt = args
    store = agl.LightningStoreClient(store_url)

    async def _async_task() -> None:
        console.print(f"Starting rollout for task {task_id} with {spans_per_attempt} spans")
        attempted = await store.start_rollout(input={"task": task_id})
        rollout_id = attempted.rollout_id
        attempt_id = attempted.attempt.attempt_id
        for seq in range(1, spans_per_attempt + 1):
            console.print(f"Adding span {seq} for task {task_id} with {spans_per_attempt} spans")
            span = _make_span(
                rollout_id,
                attempt_id,
                task_id * spans_per_attempt + seq,
                f"micro-span-{seq}",
            )
            await store.add_span(span)
        console.print(f"Updating attempt {attempt_id} for task {task_id} with {spans_per_attempt} spans")
        await store.update_attempt(rollout_id, attempt_id, status="succeeded")

    try:
        asyncio.run(_async_task())
        return True
    except Exception as e:
        console.print(f"Error running rollout task {task_id}: {e}")
        return False
    finally:
        _close_store_client(store)


def simulate_rollout_with_spans(store_url: str, spans_per_attempt: int = 4) -> BenchmarkSummary:
    """Simulate full rollout lifecycle with spans."""
    start_time = time.time()
    task_ids = list(range(1024 * 4))
    with multiprocessing.get_context("fork").Pool(processes=256) as pool:
        successful_tasks = pool.map(
            _rollout_flow_task, [(store_url, task_id, spans_per_attempt) for task_id in task_ids]
        )

    end_time = time.time()
    successes = sum(successful_tasks)
    duration = end_time - start_time
    throughput = successes / duration if duration > 0 else 0.0
    console.print(f"Rollout success rate: {successes / len(task_ids):.3f}")
    console.print(f"Time taken: {duration:.3f} seconds")
    console.print(f"Throughput: {throughput:.3f} rollouts/second")
    return BenchmarkSummary(mode="rollout", total_tasks=len(task_ids), successes=successes, duration=duration)


def record_summary(summary: BenchmarkSummary, summary_file: Optional[str]) -> None:
    message = (
        f"[summary] mode={summary.mode} success_rate={summary.success_rate:.3f} "
        f"throughput={summary.throughput:.3f} ops/s duration={summary.duration:.3f}s "
        f"success={summary.successes}/{summary.total_tasks}"
    )
    console.print(message)
    if summary_file:
        path = Path(summary_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == "worker":
        summary = simulate_many_update_workers(args.store_url)
    elif args.mode == "dequeue-empty":
        summary = simulate_dequeue_empty_and_update_workers(args.store_url)
    elif args.mode == "rollout":
        summary = simulate_rollout_with_spans(args.store_url)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    record_summary(summary, args.summary_file)


if __name__ == "__main__":
    main()

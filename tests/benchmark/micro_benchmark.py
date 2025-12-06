# Copyright (c) Microsoft. All rights reserved.

"""Micro benchmarks for the store."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console

import agentlightning as agl
from agentlightning.types import EnqueueRolloutRequest, OtelResource, Span, SpanContext, TraceStatus
from agentlightning.utils.system_snapshot import system_snapshot

from .utils import flatten_dict, random_dict

console = Console()


async def _enqueue_rollouts_for_benchmark(store_url: str, *, total_rollouts: int, task_prefix: str) -> None:
    """Utility that enqueues a fixed number of rollouts for a benchmark."""
    store = agl.LightningStoreClient(store_url)
    console.print(f"Enqueuing {total_rollouts} rollouts for {task_prefix} benchmark")
    try:
        await store.enqueue_many_rollouts(
            [EnqueueRolloutRequest(input={"task": f"{task_prefix}-Task-{i}"}) for i in range(total_rollouts)]
        )
    finally:
        await store.close()


def _close_store_client(store: agl.LightningStoreClient) -> None:
    try:
        asyncio.run(store.close())
    except Exception:
        pass


def _make_span(rollout_id: str, attempt_id: str, sequence_id: int, name: str, attribute_size: int) -> Span:
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
        attributes=flatten_dict(
            random_dict(
                depth=1,
                breadth=attribute_size,
                key_length=(3, 20),
                value_length=(5, 300),
            )
        ),
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
        choices=("worker", "dequeue-empty", "dequeue-only", "rollout", "dequeue-update-attempt"),
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
                attribute_size=1,
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


def _dequeue_only_task(args: tuple[str, str, str]) -> bool:
    store_url, worker_id, task_id = args
    console.print(f"[Dequeue-Only Task {task_id}] Dequeueing rollout for worker {worker_id}")
    store = agl.LightningStoreClient(store_url)

    async def _async_task() -> bool:
        attempted = await store.dequeue_rollout()  # no worker_id
        if attempted is None:
            console.print(f"[Dequeue-Only Task {task_id}] No rollout available to dequeue")
            return False
        return True

    try:
        return asyncio.run(_async_task())
    except Exception as e:
        console.print(f"Error dequeueing only worker {worker_id} for task {task_id}: {e}")
        return False
    finally:
        _close_store_client(store)


def dequeue_rollouts(store_url: str) -> BenchmarkSummary:
    """Benchmark simple dequeues without any additional mutations."""
    start_time = time.time()
    total_workers = 512
    attempts_per_worker = 16
    total_rollouts = total_workers * attempts_per_worker

    asyncio.run(_enqueue_rollouts_for_benchmark(store_url, total_rollouts=total_rollouts, task_prefix="DequeueOnly"))

    worker_jobs = [
        (f"Worker-{worker_idx}-Attempt-{attempt_idx}", f"Task-{attempt_idx * total_workers + worker_idx}")
        for worker_idx in range(total_workers)
        for attempt_idx in range(attempts_per_worker)
    ]
    with multiprocessing.get_context("fork").Pool(processes=total_workers) as pool:
        successful_tasks = pool.map(
            _dequeue_only_task, [(store_url, worker_id, task_id) for worker_id, task_id in worker_jobs]
        )

    async def _query_remaining_rollouts() -> List[str]:
        store = agl.LightningStoreClient(store_url)
        remaining_rollouts = await store.query_rollouts(status_in=["queuing"])
        return [item.rollout_id for item in remaining_rollouts]

    end_time = time.time()
    remaining_rollouts = asyncio.run(_query_remaining_rollouts())
    successes = sum(successful_tasks)
    duration = end_time - start_time
    throughput = successes / duration if duration > 0 else 0.0
    console.print(f"Remaining rollouts: {remaining_rollouts}")
    console.print(f"Remaining rollouts count: {len(remaining_rollouts)}")
    console.print(f"Dequeue-only success rate: {successes / len(worker_jobs):.3f}")
    console.print(f"Time taken: {duration:.3f} seconds")
    console.print(f"Throughput: {throughput:.3f} rollouts/second")
    return BenchmarkSummary(mode="dequeue-only", total_tasks=len(worker_jobs), successes=successes, duration=duration)


def _dequeue_and_update_attempt_task(args: tuple[str, str, str, int]) -> bool:
    store_url, worker_id, task_id, spans_per_attempt = args
    console.print(f"Dequeueing and update attempt with worker {worker_id} for task {task_id}")
    store = agl.LightningStoreClient(store_url)

    async def _async_task() -> bool:
        console.print(f"[Task {task_id}] Dequeueing rollout")
        attempted = await store.dequeue_rollout(worker_id=worker_id)
        if attempted is None:
            console.print(f"[Task {task_id}] No rollout available to dequeue")
            return False
        console.print(f"[Task {task_id}] Retrieving span sequence IDs")
        sequence_ids = await store.get_many_span_sequence_ids(
            [(attempted.rollout_id, attempted.attempt.attempt_id) for _ in range(spans_per_attempt)]
        )
        if len(sequence_ids) != spans_per_attempt:
            console.print(
                f"[Task {task_id}] Unable to retrieve enough span sequence IDs: "
                f"expected={spans_per_attempt} got={len(sequence_ids)}"
            )
            return False
        console.print(f"[Task {task_id}] Adding {spans_per_attempt} spans")
        spans = [
            _make_span(
                attempted.rollout_id,
                attempted.attempt.attempt_id,
                sequence_id,
                f"micro-span-{sequence_id}",
                attribute_size=32,
            )
            for sequence_id in sequence_ids
        ]
        stored_spans = await store.add_many_spans(spans)
        if len(stored_spans) != len(spans):
            console.print(
                f"[Task {task_id}] Only stored {len(stored_spans)}/{len(spans)} spans for "
                f"rollout_id={attempted.rollout_id} attempt_id={attempted.attempt.attempt_id}"
            )
            return False
        console.print(
            f"[Task {task_id}] Updating attempt to succeeded: rollout_id={attempted.rollout_id} "
            f"attempt_id={attempted.attempt.attempt_id}"
        )
        await store.update_attempt(attempted.rollout_id, attempted.attempt.attempt_id, status="succeeded")
        return True

    try:
        return asyncio.run(_async_task())
    except Exception as e:
        console.print(f"Error dequeueing and updating worker {worker_id} for task {task_id}: {e}")
        return False
    finally:
        _close_store_client(store)


def dequeue_and_update_attempts(store_url: str, spans_per_attempt: int = 4) -> BenchmarkSummary:
    """Simulate dequeueing rollouts and updating attempts with spans."""
    start_time = time.time()
    total_workers = 512
    attempts_per_worker = 16
    total_rollouts = total_workers * attempts_per_worker

    asyncio.run(_enqueue_rollouts_for_benchmark(store_url, total_rollouts=total_rollouts, task_prefix="Dequeue"))

    worker_jobs = [
        (f"Worker-{worker_idx}-Attempt-{attempt_idx}", f"Task-{attempt_idx * total_workers + worker_idx}")
        for worker_idx in range(total_workers)
        for attempt_idx in range(attempts_per_worker)
    ]
    with multiprocessing.get_context("fork").Pool(processes=total_workers) as pool:
        successful_tasks = pool.map(
            _dequeue_and_update_attempt_task,
            [(store_url, worker_id, task_id, spans_per_attempt) for worker_id, task_id in worker_jobs],
        )

    end_time = time.time()
    successes = sum(successful_tasks)
    duration = end_time - start_time
    throughput = successes / duration if duration > 0 else 0.0
    console.print(f"Dequeue and update attempt success rate: {successes / len(worker_jobs):.3f}")
    console.print(f"Time taken: {duration:.3f} seconds")
    console.print(f"Throughput: {throughput:.3f} rollouts/second")
    return BenchmarkSummary(
        mode="dequeue-update-attempt", total_tasks=len(worker_jobs), successes=successes, duration=duration
    )


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
    elif args.mode == "dequeue-only":
        summary = dequeue_rollouts(args.store_url)
    elif args.mode == "rollout":
        summary = simulate_rollout_with_spans(args.store_url)
    elif args.mode == "dequeue-update-attempt":
        summary = dequeue_and_update_attempts(args.store_url)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    record_summary(summary, args.summary_file)
    if summary.success_rate < 1.0:
        raise ValueError(f"Benchmark failed with success rate {summary.success_rate:.3f}")


if __name__ == "__main__":
    main()

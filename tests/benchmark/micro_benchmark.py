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
from agentlightning.utils.system_snapshot import system_snapshot

console = Console()


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
        choices=("worker", "dequeue-empty"),
        help="Mode to exercise different operations.",
    )
    args = parser.parse_args(argv)
    return args


def _update_worker_task(args: tuple[str, str, str]) -> bool:
    store_url, worker_id, task_id = args
    store = agl.LightningStoreClient(store_url)
    try:
        asyncio.run(store.update_worker(worker_id, system_snapshot()))
        return True
    except Exception as e:
        console.print(f"Error updating worker {worker_id} for task {task_id}: {e}")
        return False
    finally:
        try:
            asyncio.run(store.close())
        except Exception:
            pass


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
        try:
            asyncio.run(store.close())
        except Exception:
            pass


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
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    record_summary(summary, args.summary_file)


if __name__ == "__main__":
    main()

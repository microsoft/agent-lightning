# Copyright (c) Microsoft. All rights reserved.

"""Micro benchmarks for the store."""

import argparse
import asyncio
import multiprocessing
import time
from typing import Optional, Sequence

from rich.console import Console

import agentlightning as agl
from agentlightning.utils.system_snapshot import system_snapshot

console = Console()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro benchmarks for the store.")
    parser.add_argument("--store-url", default="http://localhost:4747", help="Lightning Store endpoint base URL.")
    parser.add_argument(
        "mode",
        choices=("worker", "dequeue-empty"),
        help="Mode to exercise different operations.",
    )
    args = parser.parse_args(argv)
    return args


def _update_worker_task(args: tuple[agl.LightningStore, str, str]) -> bool:
    store, worker_id, task_id = args
    console.print(f"Updating worker {worker_id} for task {task_id}")
    try:
        asyncio.run(store.update_worker(worker_id, system_snapshot()))
        return True
    except Exception as e:
        console.print(f"Error updating worker {worker_id} for task {task_id}: {e}")
        return False


def simulate_many_update_workers(store: agl.LightningStore) -> None:
    """Simulate many update workers."""

    start_time = time.time()

    # Use a multiprocessing pool to update workers.
    worker_ids = [(f"Worker-{i % 1024}", f"Task-{j}") for i in range(1024) for j in range(10)]
    with multiprocessing.get_context("fork").Pool(processes=1024) as pool:
        successful_tasks = pool.map(_update_worker_task, [(store, *worker_id) for worker_id in worker_ids])

    end_time = time.time()
    console.print(f"Success rate: {sum(successful_tasks) / len(worker_ids):.3f}")
    console.print(f"Time taken: {end_time - start_time:.3f} seconds")
    console.print(f"Throughput: {sum(successful_tasks) / (end_time - start_time):.3f} workers/second")


def _dequeue_empty_and_update_workers_task(args: tuple[agl.LightningStore, str, str]) -> bool:
    store, worker_id, task_id = args
    console.print(f"Dequeueing empty and updating worker {worker_id} for task {task_id}")

    async def _async_task() -> None:
        await store.dequeue_rollout(worker_id=worker_id)
        await store.update_worker(worker_id, system_snapshot())

    try:
        asyncio.run(_async_task())
        return True
    except Exception as e:
        console.print(f"Error dequeueing empty and updating worker {worker_id} for task {task_id}: {e}")
        return False


def simulate_dequeue_empty_and_update_workers(store: agl.LightningStore) -> None:
    """Simulate dequeue empty and update workers."""
    start_time = time.time()

    worker_ids = [(f"Worker-{i % 1024}", f"Task-{j}") for i in range(1024) for j in range(10)]
    with multiprocessing.get_context("fork").Pool(processes=1024) as pool:
        successful_tasks = pool.map(
            _dequeue_empty_and_update_workers_task, [(store, *worker_id) for worker_id in worker_ids]
        )

    end_time = time.time()
    console.print(f"Success rate: {sum(successful_tasks) / len(worker_ids):.3f}")
    console.print(f"Time taken: {end_time - start_time:.3f} seconds")
    console.print(f"Throughput: {sum(successful_tasks) / (end_time - start_time):.3f} workers/second")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    store = agl.LightningStoreClient(args.store_url)
    if args.mode == "worker":
        simulate_many_update_workers(store)
    elif args.mode == "dequeue-empty":
        simulate_dequeue_empty_and_update_workers(store)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

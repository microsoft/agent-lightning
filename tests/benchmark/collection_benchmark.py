# Copyright (c) Microsoft. All rights reserved.

"""Collection-level contention benchmarks for Agent Lightning."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty, Queue
from typing import Any, AsyncContextManager, Callable, Dict, List, Mapping, Sequence

from pymongo import AsyncMongoClient
from rich.console import Console
from rich.table import Table

from agentlightning.store.collection.base import LightningCollections
from agentlightning.store.collection.memory import InMemoryLightningCollections
from agentlightning.store.collection.mongo import MongoClientPool, MongoLightningCollections
from agentlightning.types import Rollout, RolloutConfig

console = Console()

DEFAULT_TOTAL_TASKS = 100_000
DEFAULT_CONCURRENCY = 1_024
DEFAULT_TASK_PREFIX = "collection-bench"
MONGO_DEFAULT_DB = "agentlightning_collection_bench"


@dataclass
class WorkerResult:
    durations: List[float]
    failures: int


@dataclass
class BenchmarkResult:
    backend: str
    name: str
    total_tasks: int
    concurrency: int
    successes: int
    failures: int
    duration: float
    throughput: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    success_rate: float
    ops_per_worker: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LightningStore collections without the store server.")
    parser.add_argument("benchmark", choices=("insert", "dequeue"), help="Benchmarks to run.")
    parser.add_argument("--backend", choices=("memory", "mongo"), default="memory", help="Collection backend to test.")
    parser.add_argument("--total-tasks", type=int, default=DEFAULT_TOTAL_TASKS, help="Total operations to run.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent workers.")
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX, help="Base prefix for generated workload IDs.")
    parser.add_argument("--summary-file", help="Optional newline-delimited JSON summary output.")
    parser.add_argument(
        "--mongo-uri", default="mongodb://localhost:27017/?replicaSet=rs0", help="Mongo connection URI."
    )
    parser.add_argument("--mongo-database", default=MONGO_DEFAULT_DB, help="Mongo database for benchmark artifacts.")
    return parser.parse_args(argv)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[int(rank)]
    return values[lower] * (upper - rank) + values[upper] * (rank - lower)


def _aggregate_results(
    *,
    backend: str,
    name: str,
    results: Sequence[WorkerResult],
    concurrency: int,
    total_tasks: int,
    duration: float,
) -> BenchmarkResult:
    successes = sum(len(result.durations) for result in results)
    failures = sum(result.failures for result in results)
    latencies = [lat for result in results for lat in result.durations]
    throughput = successes / duration if duration > 0 else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    return BenchmarkResult(
        backend=backend,
        name=name,
        total_tasks=total_tasks,
        concurrency=concurrency,
        successes=successes,
        failures=failures,
        duration=duration,
        throughput=throughput,
        avg_latency=avg_latency,
        p50_latency=_percentile(sorted_latencies, 0.50),
        p95_latency=_percentile(sorted_latencies, 0.95),
        p99_latency=_percentile(sorted_latencies, 0.99),
        min_latency=sorted_latencies[0] if sorted_latencies else 0.0,
        max_latency=sorted_latencies[-1] if sorted_latencies else 0.0,
        success_rate=(successes / (successes + failures)) if (successes + failures) else 0.0,
        ops_per_worker=(successes / concurrency) if concurrency else 0.0,
    )


def _render_results(results: Sequence[BenchmarkResult]) -> None:
    if not results:
        console.print("[yellow]No benchmark results to display.[/yellow]")
        return
    table = Table(title="Collection Benchmarks", show_lines=False)
    table.add_column("Backend")
    table.add_column("Benchmark")
    table.add_column("Successes", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Throughput (req/s)", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Success Rate", justify="right")
    for result in results:
        table.add_row(
            result.backend,
            result.name,
            f"{result.successes:,}",
            f"{result.failures:,}",
            f"{result.throughput:,.2f}",
            f"{result.avg_latency * 1e3:,.2f}",
            f"{result.p95_latency * 1e3:,.2f}",
            f"{result.p99_latency * 1e3:,.2f}",
            f"{result.success_rate * 100:,.2f}%",
        )
    console.print(table)


def _write_summary(results: Sequence[BenchmarkResult], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict()) + "\n")


def _make_rollout(worker_index: int, sequence: int, task_prefix: str) -> Rollout:
    rollout_id = f"{task_prefix}-ro-{worker_index}-{sequence}-{uuid.uuid4().hex}"
    current_time = time.time()
    return Rollout(
        rollout_id=rollout_id,
        input={"task": rollout_id},
        start_time=current_time,
        end_time=None,
        mode="train",
        resources_id=None,
        status="queuing",
        config=RolloutConfig(),
        metadata={},
    )


async def _preload_queue(collections: LightningCollections, total_tasks: int, task_prefix: str) -> None:
    batch: List[str] = []
    for idx in range(total_tasks):
        batch.append(f"{task_prefix}-queue-{idx}")
        if len(batch) >= 512:
            async with collections.atomic(mode="rw", labels=["rollout_queue"]) as collections_atomic:
                await collections_atomic.rollout_queue.enqueue(batch)
            batch.clear()
    if batch:
        async with collections.atomic(mode="rw", labels=["rollout_queue"]) as collections_atomic:
            await collections_atomic.rollout_queue.enqueue(batch)


async def _reset_mongo_database(uri: str, database: str) -> None:
    client = AsyncMongoClient[Mapping[str, Any]](uri)
    try:
        await client.drop_database(database)
    finally:
        await client.close()


class BaseBenchmark:
    """Shared control flow for collection benchmarks across backends."""

    def __init__(
        self, *, backend: str, total_tasks: int, concurrency: int, task_prefix: str, name: str, kind: str
    ) -> None:
        self.backend = backend
        self.total_tasks = total_tasks
        self.concurrency = concurrency
        self.task_prefix = task_prefix
        self.name = name
        self.kind = kind

    def run(self) -> BenchmarkResult:
        asyncio.run(self.setup())
        start = time.perf_counter()

        results = self.spawn_workers(worker_fn=self.worker_entrypoint)
        duration = time.perf_counter() - start
        return _aggregate_results(
            backend=self.backend,
            name=self.name,
            results=results,
            concurrency=self.concurrency,
            total_tasks=self.total_tasks,
            duration=duration,
        )

    def spawn_workers(
        self,
        worker_fn: Callable[[int, Any, Any], WorkerResult],
    ) -> List[WorkerResult]:
        raise NotImplementedError()

    def worker_entrypoint(self, worker_index: int, task_queue: Any, start_barrier: Any) -> WorkerResult:
        start_barrier.wait()
        console.print(f"Worker {worker_index} starting")

        async def _runner() -> WorkerResult:
            async with self.worker_context() as collections:
                if self.kind == "insert":
                    return await insert_worker_async(
                        collections,
                        worker_index=worker_index,
                        task_queue=task_queue,
                        task_prefix=self.task_prefix,
                    )
                if self.kind == "dequeue":
                    return await dequeue_worker_async(
                        collections,
                        worker_index=worker_index,
                        task_queue=task_queue,
                    )
                raise ValueError(f"Unknown benchmark kind: {self.kind}")

        return asyncio.run(_runner())

    def worker_context(self, *args: Any, **kwargs: Any) -> AsyncContextManager[LightningCollections]:
        """Provide the execution context for the benchmark workers."""
        raise NotImplementedError()

    async def setup(self) -> None:
        """Prepare backend-specific state before running workers."""
        if self.kind == "dequeue":
            async with self.worker_context() as collections:
                await _preload_queue(collections, self.total_tasks, self.task_prefix)


class MemoryBenchmark(BaseBenchmark):

    def __init__(
        self,
        *,
        total_tasks: int,
        concurrency: int,
        task_prefix: str,
        kind: str,
    ) -> None:
        super().__init__(
            total_tasks=total_tasks,
            concurrency=concurrency,
            task_prefix=task_prefix,
            name=f"collection-{kind}",
            backend="memory",
            kind=kind,
        )
        self.collections = InMemoryLightningCollections(lock_type="thread")

    def spawn_workers(
        self,
        worker_fn: Callable[[int, Any, Any], WorkerResult],
    ) -> List[WorkerResult]:
        task_queue: Queue[int] = Queue()
        for task_id in range(self.total_tasks):
            task_queue.put(task_id)
        start_barrier = threading.Barrier(self.concurrency)
        results: List[WorkerResult | None] = [None] * self.concurrency

        def _thread_target(worker_index: int) -> None:
            results[worker_index] = worker_fn(worker_index, task_queue, start_barrier)

        threads: List[threading.Thread] = []
        for worker_index in range(self.concurrency):
            thread = threading.Thread(target=_thread_target, args=(worker_index,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        return [result for result in results if result is not None]

    @asynccontextmanager
    async def worker_context(self, *args: Any, **kwargs: Any):
        yield self.collections


class MongoBenchmark(BaseBenchmark):
    def __init__(
        self,
        *,
        total_tasks: int,
        concurrency: int,
        task_prefix: str,
        kind: str,
        mongo_uri: str,
        mongo_database: str,
    ) -> None:
        super().__init__(
            total_tasks=total_tasks,
            concurrency=concurrency,
            task_prefix=task_prefix,
            name=f"collection-{kind}",
            backend="mongo",
            kind=kind,
        )
        self.mongo_uri = mongo_uri
        self.mongo_database = mongo_database
        self.partition_id = f"partition-{uuid.uuid4().hex}"

    async def setup(self) -> None:
        await _reset_mongo_database(self.mongo_uri, self.mongo_database)
        return await super().setup()

    @asynccontextmanager
    async def worker_context(self):
        pool = MongoClientPool[Mapping[str, Any]](mongo_uri=self.mongo_uri)
        collections = MongoLightningCollections(
            client_pool=pool,
            database_name=self.mongo_database,
            partition_id=self.partition_id,
            tracker=None,
        )

        try:
            yield collections
        finally:
            await pool.close()

    def spawn_workers(
        self,
        worker_fn: Callable[[int, Any, Any], WorkerResult],
    ) -> List[WorkerResult]:
        ctx = mp.get_context("fork")
        task_queue = ctx.Queue()
        for task_id in range(self.total_tasks):
            task_queue.put(task_id)
        start_barrier = ctx.Barrier(self.concurrency)
        result_queue = ctx.Queue()

        processes: List[BaseProcess] = []
        for worker_index in range(self.concurrency):
            process = ctx.Process(
                target=_process_worker_target,
                args=(self, worker_index, task_queue, start_barrier, result_queue),
            )
            process.start()
            processes.append(process)

        collected: List[WorkerResult] = []
        errors: List[Exception] = []
        for _ in range(self.concurrency):
            item = result_queue.get()
            if isinstance(item, Exception):
                errors.append(item)
            else:
                collected.append(item)

        for process in processes:
            process.join()

        if errors:
            raise RuntimeError("One or more worker processes failed") from errors[0]

        return collected


def _process_worker_target(
    benchmark: BaseBenchmark,
    worker_index: int,
    task_queue: Any,
    start_barrier: Any,
    result_queue: Any,
) -> None:
    try:
        result = benchmark.worker_entrypoint(worker_index, task_queue, start_barrier)
    except Exception as exc:
        result_queue.put(exc)
        raise
    else:
        result_queue.put(result)


async def insert_worker_async(
    collections: LightningCollections,
    *,
    worker_index: int,
    task_queue: Any,
    task_prefix: str,
) -> WorkerResult:
    durations: List[float] = []
    failures = 0
    while True:
        try:
            sequence = task_queue.get_nowait()
        except Empty:
            break
        rollout = _make_rollout(worker_index, sequence, task_prefix)
        req_start = time.perf_counter()
        try:
            async with collections.atomic(mode="rw", labels=["rollouts"]) as collections_atomic:
                if random.uniform(0, 1) < 0.01:
                    console.print("Inserting rollout:", rollout.rollout_id)
                await collections_atomic.rollouts.insert([rollout])
            durations.append(time.perf_counter() - req_start)
        except Exception:
            failures += 1
    return WorkerResult(durations=durations, failures=failures)


async def dequeue_worker_async(
    collections: LightningCollections,
    *,
    worker_index: int,
    task_queue: Any,
) -> WorkerResult:
    del worker_index  # unused but kept for symmetry
    durations: List[float] = []
    failures = 0
    while True:
        try:
            task_queue.get_nowait()
        except Empty:
            break
        req_start = time.perf_counter()
        try:
            async with collections.atomic(mode="rw", labels=["rollout_queue"]) as collections_atomic:
                items = await collections_atomic.rollout_queue.dequeue(limit=1)
                if items and random.uniform(0, 1) < 0.01:
                    console.print("Dequeued items:", items[0])
        except Exception:
            failures += 1
            continue
        if not items:
            break
        durations.append(time.perf_counter() - req_start)
    return WorkerResult(durations=durations, failures=failures)


def run_benchmark(args: argparse.Namespace, benchmark_kind: str) -> BenchmarkResult:
    params = {
        "total_tasks": args.total_tasks,
        "concurrency": args.concurrency,
        "task_prefix": args.task_prefix,
    }
    if args.backend == "memory":
        return MemoryBenchmark(kind=benchmark_kind, **params).run()

    mongo_params = {
        **params,
        "mongo_uri": args.mongo_uri,
        "mongo_database": args.mongo_database,
    }
    return MongoBenchmark(kind=benchmark_kind, **mongo_params).run()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.total_tasks <= 0:
        raise ValueError("total-tasks must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")

    results: List[BenchmarkResult] = []
    results.append(run_benchmark(args, args.benchmark))

    _render_results(results)

    if args.summary_file:
        _write_summary(results, Path(args.summary_file))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

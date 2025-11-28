# Copyright (c) Microsoft. All rights reserved.

"""Benchmarking store performance by writing and querying spans from the store."""

import argparse
import asyncio
import os
import random
import sys
import threading
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, cast

from rich.console import Console

import agentlightning as agl
from agentlightning.utils.otel import get_tracer

from .utils import flatten_dict, random_dict

console = Console()

MAX_RUNTIME_SECONDS = 45 * 60


def _abort_due_to_timeout() -> None:
    sys.stderr.write(f"[benchmark] Exiting after exceeding the {MAX_RUNTIME_SECONDS // 60} minute timeout.\n")
    sys.stderr.flush()
    os._exit(1)


def _start_timeout_guard(timeout_seconds: float) -> threading.Timer:
    timer = threading.Timer(timeout_seconds, _abort_due_to_timeout)
    timer.daemon = True
    timer.start()
    return timer


def generate_attributes() -> Dict[str, Any]:
    return flatten_dict(
        random_dict(
            depth=(1, 3),
            breadth=(2, 6),
            key_length=(3, 20),
            value_length=(5, 300),
        )
    )


def make_agent(max_rounds: int, sleep_seconds: float) -> agl.LitAgent[str]:
    @agl.rollout
    async def agent(task: str, llm: agl.LLM):
        tracer = get_tracer()
        rounds = random.randint(1, max_rounds)
        selected_round = random.randint(0, rounds - 1)

        for i in range(rounds):
            with tracer.start_as_current_span(f"agent{i}") as span:
                # Nested Span
                with tracer.start_as_current_span(f"round{i}_1") as span:
                    await asyncio.sleep(random.uniform(0.0, sleep_seconds))
                    span.set_attributes(generate_attributes())
                    if i == selected_round:
                        span.set_attribute("task", task)

                # Nested Span
                with tracer.start_as_current_span(f"round{i}_2") as span:
                    await asyncio.sleep(random.uniform(0.0, sleep_seconds))
                    span.set_attributes(generate_attributes())

            if random.uniform(0, 1) < 0.5:
                agl.emit_reward(random.uniform(0.0, 1.0))

        # Final Span
        with tracer.start_as_current_span("final") as span:
            await asyncio.sleep(random.uniform(0.0, sleep_seconds))
            span.set_attributes(generate_attributes())

        agl.emit_reward(random.uniform(1.0, 2.0))

    return agent


def check_spans(spans: Sequence[agl.Span], task: str) -> None:
    """Check if the spans contain the task."""
    found_task = any(span.attributes.get("task") == task for span in spans)

    final_reward = agl.find_final_reward(spans)
    if final_reward is None:
        raise ValueError("Final reward is not found")
    if not (final_reward >= 1 and final_reward <= 2):
        raise ValueError(f"Final reward {final_reward} is not in the range of 1 to 2")
    if not found_task:
        raise ValueError(f"Task {task} is not found in the spans")


class AlgorithmBatch(agl.Algorithm):
    def __init__(
        self,
        mode: Literal["batch", "batch_partial", "single"],
        total_tasks: int,
        batch_size: Optional[int] = None,
        remaining_tasks: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        self.mode = mode
        self.total_tasks = total_tasks
        self.batch_size = batch_size
        self.remaining_tasks = remaining_tasks
        self.concurrency = concurrency

    async def run(
        self, train_dataset: Optional[agl.Dataset[Any]] = None, val_dataset: Optional[agl.Dataset[Any]] = None
    ):
        if self.mode == "batch":
            assert self.batch_size is not None
            await self.algorithm_batch(self.total_tasks, self.batch_size)
        elif self.mode == "batch_partial":
            assert self.batch_size is not None
            assert self.remaining_tasks is not None
            await self.algorithm_batch_with_completion_threshold(
                self.total_tasks, self.batch_size, self.remaining_tasks
            )
        elif self.mode == "single":
            assert self.concurrency is not None
            await self.algorithm_batch_single(self.total_tasks, self.concurrency)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    async def algorithm_batch(self, total_tasks: int, batch_size: int):
        """
        At each time, the algorithm will enqueue a batch of rollouts of size `batch_size`.
        The algorithm will use wait_for_rollouts to wait for all rollouts to complete.
        It then checks whether all rollouts are successful and check the spans to ensure the task is found
        and the last reward is in the range of 1 to 2.
        After that, the algorithm will enqueue a new batch of new tasks, until the total number of tasks is reached.
        """
        store = self.get_store()
        submitted = 0

        while submitted < total_tasks:
            print(f"Submitting batch {submitted} of {total_tasks}")
            batch_count = min(batch_size, total_tasks - submitted)
            batch_rollouts: List[Tuple[str, str]] = []
            await store.add_resources(
                {
                    "llm": agl.LLM(
                        endpoint=f"http://localhost:{submitted}/v1",
                        model=f"test-model-{submitted}",
                    )
                }
            )
            for _ in range(batch_count):
                task_name = f"task-{submitted}-generated"
                rollout = await store.enqueue_rollout(input=task_name, mode="train")
                batch_rollouts.append((rollout.rollout_id, task_name))
                submitted += 1

            pending = {rollout_id: task_name for rollout_id, task_name in batch_rollouts}
            completed_ids: Set[str] = set()
            while len(completed_ids) < len(batch_rollouts):
                finished_rollouts = await store.wait_for_rollouts(
                    rollout_ids=[rollout_id for rollout_id, _ in batch_rollouts],
                    timeout=0.0,
                )
                for rollout in finished_rollouts:
                    rollout_id = rollout.rollout_id
                    if rollout_id in completed_ids:
                        continue
                    if rollout.status != "succeeded":
                        raise RuntimeError(f"Rollout {rollout_id} finished with status {rollout.status}")
                    spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                    check_spans(spans, pending[rollout_id])
                    completed_ids.add(rollout_id)
                await asyncio.sleep(5.0)

    async def algorithm_batch_with_completion_threshold(self, total_tasks: int, batch_size: int, remaining_tasks: int):
        """Different from `algorithm_batch`, this algorithm will use query_rollouts to get rollouts' status.
        It will enqueue a new batch of new tasks when the number of running rollouts is less than the remaining tasks threshold.
        """
        store = self.get_store()
        submitted = 0
        completed = 0
        active_rollouts: Dict[str, str] = {}

        while completed < total_tasks:
            console.print(f"Completed {completed} of {total_tasks} rollouts")
            if submitted < total_tasks and len(active_rollouts) < remaining_tasks:
                batch_count = min(batch_size, total_tasks - submitted)
                await store.add_resources(
                    {
                        "llm": agl.LLM(
                            endpoint=f"http://localhost:{submitted}/v1",
                            model=f"test-model-{submitted}",
                        )
                    }
                )
                for _ in range(batch_count):
                    task_name = f"task-{submitted}"
                    rollout = await store.enqueue_rollout(input=task_name, mode="train")
                    active_rollouts[rollout.rollout_id] = task_name
                    submitted += 1
                continue

            if not active_rollouts:
                await asyncio.sleep(0.01)
                continue

            rollouts = await store.query_rollouts(rollout_id_in=list(active_rollouts.keys()))
            newly_completed = 0
            for rollout in rollouts:
                rollout_id = rollout.rollout_id
                if rollout_id not in active_rollouts:
                    continue
                if rollout.status in ("queuing", "preparing", "running", "requeuing"):
                    continue
                if rollout.status != "succeeded":
                    raise RuntimeError(f"Rollout {rollout_id} finished with status {rollout.status}")
                spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                check_spans(spans, active_rollouts.pop(rollout_id))
                completed += 1
                newly_completed += 1

            if newly_completed == 0:
                await asyncio.sleep(5.0)

    async def algorithm_batch_single(self, total_tasks: int, concurrency: int):
        """Different from `algorithm_batch`, this algorithm will use one async function to enqueue one rollout at a time.
        The function only cares about the rollout it's currently processing.
        It waits for the rollouts with `get_rollout_by_id` and check the spans to ensure the rollout is successful.
        The concurrency is managed via a asyncio semaphore.
        """
        store = self.get_store()
        semaphore = asyncio.Semaphore(concurrency)

        async def handle_single(task_index: int) -> None:
            task_name = f"task-{task_index}"
            async with semaphore:
                console.print(f"Submitting task {task_index} of {total_tasks}")
                await store.add_resources(
                    {
                        "llm": agl.LLM(
                            endpoint=f"http://localhost:{task_index}/v1",
                            model=f"test-model-{task_index}",
                        )
                    }
                )
                rollout = await store.enqueue_rollout(input=task_name, mode="train")
                rollout_id = rollout.rollout_id
                while True:
                    current = await store.get_rollout_by_id(rollout_id)
                    if current is not None and current.status in ("failed", "succeeded", "cancelled"):
                        if current.status != "succeeded":
                            raise RuntimeError(f"Rollout {rollout_id} finished with status {current.status}")
                        break
                    await asyncio.sleep(5.0)
                spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                check_spans(spans, task_name)

        all_tasks = [handle_single(i) for i in range(total_tasks)]
        await asyncio.gather(*all_tasks)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LightningStore implementations with synthetic rollouts.")
    parser.add_argument("--store-url", default="http://localhost:4747", help="Lightning Store endpoint base URL.")
    parser.add_argument(
        "--mode",
        choices=("batch", "batch_partial", "single"),
        default="batch",
        help="Algorithm mode to exercise different submission patterns.",
    )
    parser.add_argument("--total-tasks", type=int, default=128 * 128, help="Total number of rollouts to submit.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for batch-style modes.")
    parser.add_argument(
        "--remaining-tasks",
        type=int,
        default=512,
        help="Target number of in-flight rollouts before submitting more (batch_partial mode).",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum concurrent rollouts for single mode.")
    parser.add_argument("--n-runners", type=int, default=32, help="Number of runner processes to launch.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds for each rollout.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep seconds for each rollout.")
    args = parser.parse_args(argv)

    if args.total_tasks <= 0:
        parser.error("--total-tasks must be positive")
    if args.n_runners <= 0:
        parser.error("--n-runners must be positive")
    if args.mode in {"batch", "batch_partial"} and (args.batch_size is None or args.batch_size <= 0):
        parser.error("--batch-size must be positive for batch modes")
    if args.mode == "batch_partial" and (args.remaining_tasks is None or args.remaining_tasks <= 0):
        parser.error("--remaining-tasks must be positive for batch_partial mode")
    if args.mode == "single" and (args.concurrency is None or args.concurrency <= 0):
        parser.error("--concurrency must be positive for single mode")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.sleep_seconds <= 0:
        parser.error("--sleep-seconds must be positive")

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    store = agl.LightningStoreClient(args.store_url)
    timeout_guard = _start_timeout_guard(MAX_RUNTIME_SECONDS)
    try:
        trainer = agl.Trainer(
            store=store,
            algorithm=AlgorithmBatch(
                mode=cast(Literal["batch", "batch_partial", "single"], args.mode),
                total_tasks=args.total_tasks,
                batch_size=args.batch_size,
                remaining_tasks=args.remaining_tasks,
                concurrency=args.concurrency,
            ),
            n_runners=args.n_runners,
            strategy={
                "type": "cs",
                "managed_store": False,
            },
        )
        trainer.fit(make_agent(max_rounds=args.max_rounds, sleep_seconds=args.sleep_seconds))
    finally:
        timeout_guard.cancel()
        asyncio.run(store.close())


if __name__ == "__main__":
    main()

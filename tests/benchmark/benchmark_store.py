# Copyright (c) Microsoft. All rights reserved.

"""Benchmarking store performance by writing and querying spans from the store."""

import asyncio
import random
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import agentlightning as agl
from agentlightning.emitter.utils import get_tracer

from .utils import flatten_dict, random_dict


def generate_attributes() -> Dict[str, Any]:
    return flatten_dict(
        random_dict(
            depth=(1, 3),
            breadth=(2, 6),
            key_length=(3, 20),
            value_length=(5, 300),
        )
    )


@agl.rollout
async def agent(task: str, llm: agl.LLM):
    tracer = get_tracer()
    rounds = random.randint(1, 10)
    selected_round = random.randint(0, rounds - 1)

    for i in range(rounds):
        with tracer.start_as_current_span(f"agent{i}") as span:
            # Nested Span
            with tracer.start_as_current_span(f"round{i}_1") as span:
                await asyncio.sleep(random.uniform(0.0, 1.0))
                span.set_attributes(generate_attributes())
                if i == selected_round:
                    span.set_attribute("task", task)

            # Nested Span
            with tracer.start_as_current_span(f"round{i}_2") as span:
                await asyncio.sleep(random.uniform(0.0, 1.0))
                span.set_attributes(generate_attributes())

        if random.uniform(0, 1) < 0.5:
            agl.emit_reward(random.uniform(0.0, 1.0))

    # Final Span
    with tracer.start_as_current_span("final") as span:
        await asyncio.sleep(random.uniform(0.0, 1.0))
        span.set_attributes(generate_attributes())

    agl.emit_reward(random.uniform(1.0, 2.0))


def check_spans(spans: Sequence[agl.Span], task: str) -> None:
    """Check if the spans contain the task."""
    found_task = False
    last_reward_in_12 = None
    for span in spans:
        if span.attributes.get("task") == task:
            found_task = True
        if span.name == agl.SpanNames.REWARD.value:
            if span.attributes.get("reward") is None:
                raise ValueError("Reward is not set for a reward span")
            rew = float(span.attributes.get("reward"))  # type: ignore
            if rew >= 1 and rew <= 2:
                last_reward_in_12 = True
            else:
                last_reward_in_12 = False
    if not found_task:
        raise ValueError(f"Task {task} is not found in the spans")
    if last_reward_in_12 is None:
        raise ValueError("Last reward is not found")
    elif not last_reward_in_12:
        raise ValueError("Last reward is not in the range of 1 to 2")


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


def main() -> None:
    store = agl.LightningStoreClient("http://localhost:4747")
    try:
        trainer = agl.Trainer(
            store=store,
            algorithm=AlgorithmBatch(mode="batch", total_tasks=1024, batch_size=128),
            n_runners=32,
            strategy={
                "type": "cs",
                "managed_store": False,
            },
        )
        trainer.fit(agent)
    finally:
        asyncio.run(store.close())


if __name__ == "__main__":
    main()

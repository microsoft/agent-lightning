# Copyright (c) Microsoft. All rights reserved.

"""Benchmarking store performance by writing and querying spans from the store."""

import asyncio
import random
from typing import Any, Dict, Sequence

import agentlightning as agl
from agentlightning.emitter.utils import get_tracer
from agentlightning.store import LightningStore

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
            if rew > 1 and rew < 2:
                last_reward_in_12 = True
            else:
                last_reward_in_12 = False
    if not found_task:
        raise ValueError(f"Task {task} is not found in the spans")
    if last_reward_in_12 is None:
        raise ValueError("Last reward is not found")
    elif not last_reward_in_12:
        raise ValueError("Last reward is not in the range of 1 to 2")


async def algorithm_batch(store: LightningStore, total_tasks: int, batch_size: int):
    """
    At each time, the algorithm will enqueue a batch of rollouts of size `batch_size`.
    The algorithm will use wait_for_rollouts to wait for all rollouts to complete.
    It then checks whether all rollouts are successful and check the spans to ensure the task is found
    and the last reward is in the range of 1 to 2.
    After that, the algorithm will enqueue a new batch of new tasks, until the total number of tasks is reached.
    """


async def algorithm_batch_with_completion_threshold(
    store: LightningStore, total_tasks: int, batch_size: int, remaining_tasks: int
):
    """Different from `algorithm_batch`, this algorithm will use query_rollouts to get rollouts status.
    It will enqueue a new batch of new tasks when the number of running rollouts is less than the remaining tasks threshold.
    """


async def algorithm_batch_single(store: LightningStore, total_tasks: int, concurrency: int):
    """Different from `algorithm_batch`, this algorithm will use one async function to enqueue one rollout at a time.
    The function only cares about the rollout it's currently processing.
    It waits for it with `get_rollout_by_id` and check the spans to ensure the rollout is successful.
    The concurrency is managed via a asyncio semaphore.
    """

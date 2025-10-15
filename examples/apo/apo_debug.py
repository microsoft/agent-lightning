# Copyright (c) Microsoft. All rights reserved.

"""This example code illustrates several approaches to debugging an agent in agent-lightning."""

import argparse
import asyncio
from typing import cast

from apo import apo_rollout

from agentlightning import Trainer, configure_logger
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer import OtelTracer
from agentlightning.types import Dataset, PromptTemplate


async def debug_with_runner():
    """This appraoch requires no dataset, no trainer, and no algorithm.

    It only needs a runner and you can run get full control of the runner.
    However, you need to manually create other components like tracer and store,
    because trainer does not exist and it will not create for you.
    """
    # You need to manually create a tracer here because the runner will not create for you currently.
    # Tracer is used to record the events (spans) in background during the agent's execution.
    # If you don't need any tracing functionality yet, you can use a dummy OtelTracer.
    tracer = OtelTracer()
    runner = LitAgentRunner[str](tracer)

    # You also need a store here to store the data collected.
    store = InMemoryLightningStore()

    # This is what needs to be tuned (i.e., prompt template)
    resource = PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")

    # The agent here must be the same agent that will be used in the real run.
    with runner.run_context(agent=apo_rollout, store=store):
        await runner.step(
            "Explain why the sky appears blue using principles of light scattering in 100 words.",
            resources={"main_prompt": resource},
        )


def debug_with_trainer():
    """This appraoch integrates the trainer and is very similar to the real `fit()` loop.

    The trainer will create a mock algorithm which will communicates with the runner.
    Do this for end-to-end testing and debugging purposes.
    """
    # To debug with trainer, we need a dataset
    dataset = cast(
        Dataset[str],
        [
            "Explain why the sky appears blue using principles of light scattering in 100 words.",
            "What's the capital of France?",
        ],
    )

    # We also need a resource that is to be tuned (i.e., prompt template)
    resource = PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")
    trainer = Trainer(
        n_workers=1,
        # This is very critical. It will be the only prompt template that will be passed to the agent.
        initial_resources={"main_prompt": resource},
    )
    trainer.dev(apo_rollout, dataset)


if __name__ == "__main__":
    configure_logger()

    parser = argparse.ArgumentParser(description="Debug APO with runner or trainer approach.")
    parser.add_argument(
        "--mode",
        choices=["runner", "trainer"],
        default="runner",
        help="Choose which debugging approach to use: 'runner' (default) or 'trainer'.",
    )

    args = parser.parse_args()

    if args.mode == "runner":
        asyncio.run(debug_with_runner())
    elif args.mode == "trainer":
        # Don't want two mode consecutively in one process,
        # unless you are sure the tracer won't conflict.
        debug_with_trainer()

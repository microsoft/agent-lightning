# Copyright (c) Microsoft. All rights reserved.

"""This example code illustrates several approaches to debugging an agent in agent-lightning."""

import argparse
import asyncio
from typing import cast

from apo_custom_algorithm import apo_rollout

from agentlightning import Trainer, setup_logging
from agentlightning.litagent import LitAgent
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer import AgentOpsTracer, OtelTracer
from agentlightning.types import Dataset, Hook, PromptTemplate, Rollout


async def debug_with_runner():
    """This approach requires no dataset, no trainer, and no algorithm.

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


async def debug_with_hooks():
    """This approach also uses Runner, but allows you to hook into the runner's lifecycle events.

    We use an AgentOpsTracer here so that the tracing is non-empty.
    """
    tracer = AgentOpsTracer()
    # The rest part are the same as debug_with_runner
    runner = LitAgentRunner[str](tracer)
    store = InMemoryLightningStore()
    resource = PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")

    class DebugHook(Hook):

        async def on_trace_end(  # type: ignore
            self, *, agent: LitAgent[str], runner: LitAgentRunner[str], tracer: AgentOpsTracer, rollout: Rollout
        ) -> None:
            """We use `tracer.get_last_trace()` to get all raw OpenTelemetry spans from the Rollout.
            The last reward span is not available yet.
            """
            trace = tracer.get_last_trace()
            print("Trace spans collected during the rollout:")
            for span in trace:
                print(f"- {span.name} (status: {span.status}):\n  {span.attributes}")

    with runner.run_context(
        agent=apo_rollout,
        store=store,
        # Send the hooks into `run_context`
        hooks=[DebugHook()],
    ):
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
    setup_logging()

    parser = argparse.ArgumentParser(description="Debug APO with runner or trainer approach.")
    parser.add_argument(
        "--mode",
        choices=["runner", "hook", "trainer"],
        default="runner",
        help="Choose which debugging approach to use: 'runner' (default), 'hook', or 'trainer'.",
    )

    args = parser.parse_args()

    if args.mode == "runner":
        asyncio.run(debug_with_runner())
    elif args.mode == "hook":
        asyncio.run(debug_with_hooks())
    elif args.mode == "trainer":
        # Don't want two mode consecutively in one process,
        # unless you are sure the tracer won't conflict.
        debug_with_trainer()

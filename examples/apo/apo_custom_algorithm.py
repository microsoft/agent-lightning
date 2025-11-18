# Copyright (c) Microsoft. All rights reserved.

"""This sample code shows how to run a custom algorithm and rollout runner separately.

You can run this in two modes:

1. Algorithm mode - runs the optimization algorithm:
```bash
python apo_custom_algorithm.py algo
```

2. Runner mode - runs the rollout runner:
```bash
python apo_custom_algorithm.py runner
```

To use both together, you need to run them in parallel along with the store:
```bash
agl store
python apo_custom_algorithm.py algo
python apo_custom_algorithm.py runner
```

Or use the integrated version in `apo_custom_algorithm_trainer.py`:
```bash
python apo_custom_algorithm_trainer.py
```
"""

import argparse
import asyncio
from typing import Optional, Sequence

from openai import AsyncOpenAI
from rich.console import Console

import agentlightning as agl

console = Console()


async def apo_algorithm(*, store: agl.LightningStore):
    """
    An example of how a prompt optimization works.
    """
    prompt_candidates = [
        "You are a helpful assistant. {any_question}",
        "You are a knowledgeable AI. {any_question}",
        "You are a friendly chatbot. {any_question}",
    ]

    prompt_and_rewards: list[tuple[str, float]] = []

    algo_marker = "[bold red][Algo][/bold red]"

    for prompt in prompt_candidates:
        # 1. The optimization algorithm updates the prompt template
        console.print(f"\n{algo_marker} Updating prompt template to: '{prompt}'")
        resources: agl.NamedResources = {
            # The "main_prompt" can be replaced with any name you like
            # As long as the PromptTemplate type is used, the rollout function will recognize it
            "main_prompt": agl.PromptTemplate(template=prompt, engine="f-string")
        }
        # How the resource is used fully depends on the client implementation.
        await store.add_resources(resources)

        # 2. The algorithm queues up a task from a dataset
        console.print(f"{algo_marker} Queuing task for clients...")
        rollout = await store.enqueue_rollout(
            input="Explain why the sky appears blue using principles of light scattering in 100 words.", mode="train"
        )
        console.print(f"{algo_marker} Task '{rollout.rollout_id}' is now available for clients.")

        # 3. The algorithm waits for clients to process the task
        for _ in range(30):  # Wait for at most 30 seconds
            rollouts = await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=0.01)
            if rollouts:
                break
            await asyncio.sleep(1.0)
        else:
            raise RuntimeError("Expected a completed rollout from the client, but got none.")

        console.print(f"{algo_marker} Received Result: {rollouts[0]}")
        if rollouts[0].status != "succeeded":
            raise RuntimeError(f"Rollout {rollout.rollout_id} did not succeed. Status: {rollouts[0].status}")
        spans = await store.query_spans(rollout.rollout_id)

        # Logs LLM spans for debugging and inspection here
        await log_llm_span(spans)

        # 4. The algorithm records the final reward for sorting
        final_reward = agl.find_final_reward(spans)
        assert final_reward is not None, "Expected a final reward from the client."
        console.print(f"{algo_marker} Final reward: {final_reward}")
        prompt_and_rewards.append((prompt, final_reward))

    console.print(f"\n[bold red][Algo][/bold red] All prompts and their rewards: {prompt_and_rewards}")
    best_prompt = max(prompt_and_rewards, key=lambda x: x[1])
    console.print(f"[bold red][Algo][/bold red] Best prompt found: '{best_prompt[0]}' with reward {best_prompt[1]}")


@agl.rollout
async def apo_rollout(task: str, prompt_template: agl.PromptTemplate) -> float:
    # This relies on a public OpenAI service
    client = AsyncOpenAI()

    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": prompt_template.format(any_question=task)},
        ],
    )

    text = result.choices[0].message.content
    console.print(f"[bold yellow][Rollout][/bold yellow] LLM returned: {text}")

    return await llm_judge(task, text)


async def log_llm_span(spans: Sequence[agl.Span]) -> None:
    """Logs the LLM related spans that records prompts and responses."""
    for span in spans:
        if "chat.completion" in span.name:
            console.print(f"[bold green][LLM][/bold green] Span {span.span_id} ({span.name}): {span.attributes}")


async def llm_judge(task: str, output: Optional[str]) -> float:
    client = AsyncOpenAI()
    judge_prompt = f"""Evaluate how well the output fulfills the task.
Task: {task}
Output: {output}
You must be very critical and strict in your evaluation.
Return only a number between 0 and 1. No text, punctuation, or explanation."""
    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
    )
    try:
        content = result.choices[0].message.content
        if content is None:
            console.print(f"[bold blue][Judge][/bold blue] Judge returned no content: {result}")
            return 0.0
        score = float(content)
        console.print(f"[bold blue][Judge][/bold blue] Judge returned score: {score}")
        return score
    except ValueError:
        console.print(f"[bold blue][Judge][/bold blue] Error evaluating output: {result}")
        return 0.0


async def apo_runner(*, store: agl.LightningStore):
    """
    A runner that iteratively receives new rollout tasks from the store and executes them.
    """

    runner = agl.LitAgentRunner[str](tracer=agl.AgentOpsTracer())
    with runner.run_context(agent=apo_rollout, store=store):
        await runner.iter()


async def main():
    store = agl.LightningStoreClient("http://localhost:4747")
    parser = argparse.ArgumentParser(description="Run APO custom algorithm in different modes")
    parser.add_argument(
        "mode", choices=["algo", "runner"], help="Mode to run: 'algo' for algorithm or 'runner' for rollout runner"
    )
    args = parser.parse_args()

    try:
        if args.mode == "algo":
            # Run the algorithm mode
            await apo_algorithm(store=store)
        elif args.mode == "runner":
            # Run the runner mode
            await apo_runner(store=store)
    finally:
        await store.close()


if __name__ == "__main__":
    agl.setup_logging()
    asyncio.run(main())

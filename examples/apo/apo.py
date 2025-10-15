# Copyright (c) Microsoft. All rights reserved.

"""This is the APO sample with both rollout and algo in one file."""

from typing import List, Optional

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import Trainer, configure_logger
from agentlightning.algorithm import algo
from agentlightning.litagent.decorator import rollout
from agentlightning.reward import find_final_reward
from agentlightning.store import LightningStore
from agentlightning.types import NamedResources, PromptTemplate, Span

console = Console()


@algo
async def apo_algorithm(*, store: LightningStore):
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
        resources: NamedResources = {
            # The "main_prompt" can be replaced with any name you like
            # As long as the PromptTemplate type is used, the rollout function will recognize it
            "main_prompt": PromptTemplate(template=prompt, engine="f-string")
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
        rollouts = await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=30)
        assert rollouts, "Expected a completed rollout from the client."
        console.print(f"{algo_marker} Received Result: {rollouts[0]}")
        if rollouts[0].status != "succeeded":
            raise RuntimeError(f"Rollout {rollout.rollout_id} did not succeed. Status: {rollouts[0].status}")
        spans = await store.query_spans(rollout.rollout_id)

        # Logs LLM spans for debugging and inspection here
        await log_llm_span(spans)

        # 4. The algorithm records the final reward for sorting
        final_reward = find_final_reward(spans)
        assert final_reward is not None, "Expected a final reward from the client."
        console.print(f"{algo_marker} Final reward: {final_reward}")
        prompt_and_rewards.append((prompt, final_reward))

    console.print(f"\n[bold red][Algo][/bold red] All prompts and their rewards: {prompt_and_rewards}")
    best_prompt = max(prompt_and_rewards, key=lambda x: x[1])
    console.print(f"[bold red][Algo][/bold red] Best prompt found: '{best_prompt[0]}' with reward {best_prompt[1]}")


@rollout
async def apo_rollout(task: str, prompt_template: PromptTemplate) -> float:
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


async def log_llm_span(spans: List[Span]) -> None:
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
            console.print(f"[bold blue][Judge][/bold blue] Judge retured no content: {result}")
            return 0.0
        score = float(content)
        console.print(f"[bold blue][Judge][/bold blue] Judge returned score: {score}")
        return score
    except ValueError:
        console.print(f"[bold blue][Judge][/bold blue] Error evaluating output: {result}")
        return 0.0


if __name__ == "__main__":
    configure_logger()
    trainer = Trainer(n_workers=1, algorithm=apo_algorithm)
    trainer.fit(apo_rollout)

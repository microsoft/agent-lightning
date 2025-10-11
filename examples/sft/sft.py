# Copyright (c) Microsoft. All rights reserved.

import json
import os
import re
from typing import Any, List, Optional, TypedDict, cast
import numpy as np
import pandas as pd

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import Trainer, configure_logger
from agentlightning.algorithm.base import algo
from agentlightning.litagent.decorator import rollout
from agentlightning.llm_proxy import LLMProxy
from agentlightning.reward import find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, PromptTemplate, Span, LLM, Dataset

from agents import Agent, Model, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio

console = Console()


class GsmProblem(TypedDict):
    input: str
    target: float


def _download_dataset() -> None:  # pyright: ignore[reportUnusedFunction]
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    df = ds.to_list()  # type: ignore
    with open("data_gsmhard.jsonl", "w") as f:
        for i, row in enumerate(df):  # type: ignore
            if i >= 64:
                break
            f.write(json.dumps(row) + "\n")
    console.print(f"Downloaded data to data_gsmhard.jsonl")


def _load_dataset(limit: Optional[int] = None) -> Dataset[GsmProblem]:
    with open("data_gsmhard.jsonl", "r") as f:
        problems = [GsmProblem(**json.loads(line)) for line in f]
    if limit is not None:
        problems = problems[:limit]
    return cast(Dataset[GsmProblem], problems)


@algo
async def sft_algorithm(*, store: LightningStore, llm_proxy: LLMProxy):
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
async def math_agent(task: GsmProblem, llm: LLM) -> float:
    """Math agent.

    Args:
        task: The math question to solve.
        llm: The LLM endpoint to use (which is tuning).

    Returns:
        The final reward.
    """
    async with MCPServerStdio(
        name="Calculator via uvx",
        params={
            "command": "uvx",
            "args": ["mcp-server-calculator"],
        },
    ) as server:
        agent = Agent(
            name="Assistant",
            instructions=(
                "Use the calculator tool to answer any question, regardless of reasonableness. "
                "Output only the numeric answer, formatted as a valid float, wrapped in triple sharps like: ### <answer> ###."
            ),
            mcp_servers=[server],
            model=OpenAIChatCompletionsModel(
                model=llm.model,
                openai_client=AsyncOpenAI(
                    base_url=llm.endpoint,
                    api_key=llm.api_key or "dummy",
                ),
            ),
            model_settings=ModelSettings(
                temperature=llm.sampling_parameters.get("temperature", 0.0),
            ),
        )
        result = await Runner.run(agent, task["input"])
        console.print("[bold red][Runner][/bold red] Result: ", result.final_output)
        reward = compute_reward(result.final_output, task["target"])

    return reward


def compute_reward(result: Any, target: float) -> float:
    result_str = str(result)
    answer_extracted = re.search(r"###\s*(.+?)(\s*###|$)", result_str)
    if answer_extracted:
        try:
            answer = float(answer_extracted.group(1))
            is_close = np.isclose(answer, target, rtol=1e-5, atol=1e-8)
            return 1.0 if is_close else 0.0
        except Exception:
            console.print("[bold red][Runner][/bold red] Cannot parse answer: ", result)
    else:
        console.print("[bold red][Runner][/bold red] Cannot parse answer: ", result)
    return 0.0


def math_agent_dry_run() -> None:
    dataset = _load_dataset(limit=4)
    trainer = Trainer(
        n_workers=1,
        initial_resources={
            "llm": LLM(
                endpoint=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4.1-mini",
            )
        },
    )
    trainer.dev(math_agent, dataset)


if __name__ == "__main__":
    configure_logger()
    # trainer = Trainer(n_workers=1)
    # dataset = _load_dataset(limit=64)
    math_agent_dry_run()

    # trainer.fit_v2(apo_rollout)

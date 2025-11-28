# Copyright (c) Microsoft. All rights reserved.

"""This sample file contains the definition of a math agent operating on GSM-hard dataset.

To run it, first configure the environment variables:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url
```

Then, run the agent:

```bash
python math_agent.py
```
"""

import json
import os
import re
from typing import Any, Optional, TypedDict

import numpy as np
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from datasets import load_dataset  # type: ignore
from openai import AsyncOpenAI
from rich.console import Console
from trl import SFTConfig, SFTTrainer  # type: ignore

from agentlightning import Trainer, setup_logging
from agentlightning.litagent import rollout
from agentlightning.types import LLM, Dataset

console = Console()


class GsmProblem(TypedDict):
    """Type definition for a GSM-hard math problem.

    Reference link: https://huggingface.co/datasets/reasoning-machines/gsm-hard

    Attributes:
        input: The math problem question as a string.
        target: The expected numeric answer.
    """

    input: str
    target: float


def _download_dataset() -> None:  # pyright: ignore[reportUnusedFunction]
    """Download the GSM-hard dataset from Hugging Face.

    Downloads the first 64 samples from the dataset and saves them to data_gsmhard.jsonl.
    This function is provided as a utility to help set up the dataset for the first time.
    """
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")  # pyright: ignore[reportUnknownVariableType]
    df = ds.to_list()  # type: ignore
    with open("data_gsmhard.jsonl", "w") as f:
        for i, row in enumerate(df):  # type: ignore
            if i >= 64:
                break
            f.write(json.dumps(row) + "\n")
    console.print(f"Downloaded data to data_gsmhard.jsonl")


def load_math_dataset(limit: Optional[int] = None) -> Dataset[GsmProblem]:
    """Load the GSM-hard math dataset from the local JSONL file.

    Args:
        limit: Optional maximum number of problems to load. If None, loads all problems.

    Returns:
        A list of GsmProblem instances.
    """
    with open("data_gsmhard.jsonl", "r") as f:
        problems = [GsmProblem(**json.loads(line)) for line in f]
    if limit is not None:
        problems = problems[:limit]
    return problems


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
    """Compute the reward for a math agent's answer.

    The answer is expected to be formatted as: ### <answer> ###.
    The reward is 1.0 if the extracted answer is numerically close to the target, 0.0 otherwise.

    Args:
        result: The agent's output containing the answer.
        target: The expected correct answer.

    Returns:
        1.0 if the answer is correct (within numerical tolerance), 0.0 otherwise.
    """
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
    """Run a dry run of the math agent on a small dataset.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_math_dataset(limit=4)
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
    setup_logging()
    math_agent_dry_run()

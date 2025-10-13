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
from typing import Any, Optional, TypedDict, cast
import numpy as np

from openai import AsyncOpenAI
from rich.console import Console
from datasets import load_dataset  # type: ignore

from agentlightning import Trainer, configure_logger
from agentlightning.litagent import rollout
from agentlightning.types import LLM, Dataset

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio

from trl import SFTTrainer, SFTConfig  # type: ignore

console = Console()


class GsmProblem(TypedDict):
    input: str
    target: float


def _download_dataset() -> None:  # pyright: ignore[reportUnusedFunction]
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    df = ds.to_list()  # type: ignore
    with open("data_gsmhard.jsonl", "w") as f:
        for i, row in enumerate(df):  # type: ignore
            if i >= 64:
                break
            f.write(json.dumps(row) + "\n")
    console.print(f"Downloaded data to data_gsmhard.jsonl")


def load_math_dataset(limit: Optional[int] = None) -> Dataset[GsmProblem]:
    with open("data_gsmhard.jsonl", "r") as f:
        problems = [GsmProblem(**json.loads(line)) for line in f]
    if limit is not None:
        problems = problems[:limit]
    return cast(Dataset[GsmProblem], problems)


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
    configure_logger()
    math_agent_dry_run()

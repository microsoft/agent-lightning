# Copyright (c) Microsoft. All rights reserved.

"""An example of agent using Azure OpenAI with tool calls to look up capital cities.

Running this script directly will run a few sample tasks using the `capital_agent`,
which will test the healthiness of your Azure OpenAI setup.

Remember to have the following environment variables set:

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL.
"""

import asyncio
import json
import os
from typing import List, TypedDict, cast

import openai
import pandas as pd
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from rich.console import Console

from agentlightning import LLM, AgentOpsTracer, InMemoryLightningStore, LitAgentRunner, rollout

CAPITALS = {
    "japan": "Tokyo",
    "france": "Paris",
    "canada": "Ottawa",
    "australia": "Canberra",
    "brazil": "Brasília",
    "egypt": "Cairo",
    "kenya": "Nairobi",
    "spain": "Madrid",
    "italy": "Rome",
    "germany": "Berlin",
    "south korea": "Seoul",
    "india": "New Delhi",
}

console = Console()


def country_capital_lookup(country: str) -> str:
    return CAPITALS.get(country.strip().lower(), "Unknown")


class CapitalTask(TypedDict):
    input: str
    output: str


TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "country_capital_lookup",
            "description": "Get the capital city of a given country.",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}}, "required": ["country"]},
        },
    }
]

SYSTEM = (
    "You are a concise assistant. "
    "If the user asks for a country's capital, ALWAYS call the tool 'country_capital_lookup'. "
    "Otherwise, answer briefly."
)


@rollout
def capital_agent(task: CapitalTask, llm: LLM) -> float:
    """Run one evaluation task with capital agent.

    Returns 1.0 if output contains expected substring, else 0.0.
    """
    console.print("[bold blue]======== Runner Start ========[/bold blue]")
    console.print("[bold blue]Runner[/bold blue] [Step 1] Running task with input:", task)
    prompt = task["input"]
    expected = task["output"]

    openai_client = openai.OpenAI(base_url=llm.endpoint, api_key=os.getenv("AZURE_OPENAI_API_KEY", ""))

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    # --- Call #1 ---
    first = openai_client.chat.completions.create(
        model=llm.model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=1.0,
    )

    msg = first.choices[0].message
    console.print("[bold blue]Runner[/bold blue] [Step 2] First call response:", msg)

    if msg.tool_calls:
        assistant_tool_calls: List[ChatCompletionMessageFunctionToolCallParam] = []
        tool_results: List[ChatCompletionToolMessageParam] = []
        for tc in msg.tool_calls:
            if tc.type == "function" and tc.function.name == "country_capital_lookup":
                args = json.loads(tc.function.arguments or "{}")
                result = country_capital_lookup(args.get("country", ""))
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": assistant_tool_calls,
            }
        )
        messages.extend(tool_results)
        console.print("[bold blue]Runner[/bold blue] [Step 3] Messages after tool call:", messages)

        # --- Call #2 ---
        second = openai_client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=1.0,
        )
        final_text = second.choices[0].message.content or ""
        console.print("[bold blue]Runner[/bold blue] [Step 4] Second call response:", final_text)
    else:
        console.print("[bold blue]Runner[/bold blue] [Step 3] No tool calls made.")
        final_text = msg.content or ""

    final_text = final_text.strip()
    reward = 1.0 if expected.lower() in final_text.lower() else 0.0
    console.print(f"[bold blue]Runner[/bold blue] [Step Final] Final output: {final_text} | Reward: {reward}")
    return reward


async def main():
    # We don't put API key in LLM object for security reasons.
    llm = LLM(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        model="gpt-4.1-mini",
    )

    data = pd.read_csv("capital_samples.csv")  # type: ignore
    tracer = AgentOpsTracer()
    runner = LitAgentRunner[CapitalTask](tracer=tracer)
    store = InMemoryLightningStore()
    with runner.run_context(agent=capital_agent, store=store):
        for index in range(5):
            sample = cast(CapitalTask, data.iloc[index].to_dict())  # type: ignore
            await runner.step(sample, resources={"main_llm": llm})


if __name__ == "__main__":
    asyncio.run(main())

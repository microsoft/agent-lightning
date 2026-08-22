# Copyright (c) Microsoft. All rights reserved.

"""Minimal Calc-X agent entrypoint."""

# AutoGen is an optional runtime dependency for this example container.
# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import os
import re

import httpx
from eval_utils import scalar_are_results_same


class Agent:
    async def run(self) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.models import ModelFamily
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

        question = os.environ["QUESTION"]
        result = os.environ["RESULT"]
        agl_key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        openai_base_url = os.environ["AGL_OPENAI_BASE_URL"]

        calculator_mcp_server = StdioServerParams(
            command="python",
            args=["-m", "mcp_server_calculator"],
            read_timeout_seconds=120,
        )
        output_format = (
            "Output the answer when you are ready. The answer should be surrounded by "
            "three sharps (`###`), in the form of ### ANSWER: <answer> ###."
        )

        async with McpWorkbench(calculator_mcp_server) as workbench:
            model_client = OpenAIChatCompletionClient(
                model="auto",
                base_url=openai_base_url,
                api_key=agl_key,
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                    "family": ModelFamily.UNKNOWN,
                    "structured_output": False,
                },
                max_retries=6,
            )
            agent = AssistantAgent(
                name="calc",
                model_client=model_client,
                workbench=workbench,
                reflect_on_tool_use=True,
            )
            response = await asyncio.wait_for(agent.run(task=f"{question} {output_format}"), timeout=300.0)

        raw_answer = str(response.messages[-1].content)
        match = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", raw_answer)
        answer = match.group(1).strip() if match else raw_answer.strip()
        reward = 1.0 if scalar_are_results_same(answer, result, 1e-2) else 0.0

        httpx.post(
            event_url,
            json={
                "event_type": "reward",
                "data": {"value": reward},
            },
            headers={"Authorization": f"Bearer {agl_key}"},
            timeout=10.0,
        ).raise_for_status()


if __name__ == "__main__":
    asyncio.run(Agent().run())

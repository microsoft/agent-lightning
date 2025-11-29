# Copyright (c) Microsoft. All rights reserved.

"""This is a Calc-X agent training script implemented with the legacy Agent-lightning API (v0.1).

It requires a shell script to run in the background to start the training server:

```bash
bash legacy_train.sh
```
"""

import os
import re
from typing import Any, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from eval_utils import evaluate_v0_1

from agentlightning import LLM, LitAgent, NamedResources, Trainer, setup_logging

setup_logging()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


def autogen_assistant_agent(
    model: str, openai_base_url: str, temperature: float, workbench: McpWorkbench
) -> AssistantAgent:
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
    )

    calc_agent = AssistantAgent(
        name="calc",
        model_client=model_client,
        workbench=workbench,
        reflect_on_tool_use=True,
    )
    return calc_agent


class LegacyCalcAgent(LitAgent[Any]):
    """This is a Calc-X agent implemented with the legacy Agent-lightning API (v0.1)."""

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))
        async with McpWorkbench(calculator_mcp_server) as workbench:
            calc_agent = autogen_assistant_agent(
                llm.model,
                llm.endpoint,
                llm.sampling_parameters.get("temperature", 0.7),
                workbench,
            )
            try:
                output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: <answer> ###."
                prompt = task["question"] + " " + output_format
                result = await calc_agent.run(task=prompt)
                # evaluate
                answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", result.messages[-1].content)  # type: ignore
                if answer:
                    answer = answer.group(1)
                else:
                    answer = result.messages[-1].content  # type: ignore
            except Exception as e:
                print("Failure:", str(e))
                answer = "None"
            reward = await evaluate_v0_1(
                answer, str(task["result"])  # type: ignore
            )  # reward is tracked with the decorator
            print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))  # type: ignore

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources)


if __name__ == "__main__":
    Trainer(n_workers=10).fit_v0(LegacyCalcAgent(), "http://localhost:9999/")

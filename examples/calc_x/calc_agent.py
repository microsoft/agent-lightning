# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to define a Calc-X agent trainable with Agent-lightning
with latest Agent-lightning API (v0.2+)."""

import asyncio
import os
import re
from typing import TypedDict, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from eval_utils import evaluate

import agentlightning as agl


class MathProblem(TypedDict):
    """This TypedDict defines the structure of each training sample.

    Your task structure should contain all the information needed for:

    - The agent to process the task (e.g., 'question')
    - Evaluation (e.g., 'result' for ground truth)

    This type is optional. Not necessary to make the example work.
    """

    # The fields come from the dataset
    id: str
    question: str  # The math problem for the agent to solve
    chain: str  # Step-by-step solution (not used in training)
    result: str  # Ground truth answer for evaluation
    source: str


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


@agl.rollout
async def calc_agent(task: MathProblem, llm: agl.LLM) -> None:
    """Calc-X agent rollout function.

    It would accept a math problem and a LLM endpoint resource.
    It's expected to return None, and emit reward via `agl.emit_reward`.
    It can also return the reward directly without `agl.emit_reward`.
    You can choose either way, but not both.
    """

    calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])

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
            # Sometimes MCP tools can timeout. In that case, the whole agent will block.
            # We thus set a timeout of 5 minutes so that the agent will not block indefinitely.
            result = await asyncio.wait_for(calc_agent.run(task=prompt), timeout=300.0)
            # evaluate
            last_message = cast(str, result.messages[-1].content)  # type: ignore
            answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", last_message)
            if answer:
                answer = answer.group(1)
            else:
                answer = last_message
        except asyncio.TimeoutError as e:
            print("Timeout occurred. Error:", str(e))
            answer = "None"
        except Exception as e:
            print("Failure:", str(e))
            answer = "None"
        reward = await evaluate(answer, str(task["result"]))
        agl.emit_reward(reward)  # Emit reward for tracing
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))


async def debug():
    """Here we show a more manual way for debugging, without Trainer.

    We get the data samples on our own, and run the agent with LitAgentRunner.
    You will need an `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variable set
    to run this function.
    """
    # Manually create a tracer as Runner will need it.
    # Use a dummy OtelTracer if you don't need to trace anything other than reward.
    tracer = agl.OtelTracer()
    # The runner processes MathProblem, which matches the agent's task type.
    runner = agl.LitAgentRunner[MathProblem](tracer)

    # A store is required here to store the data collected.
    store = agl.InMemoryLightningStore()

    # This is what needs to be tuned (i.e., LLM)
    resource = agl.LLM(
        endpoint=os.environ["OPENAI_BASE_URL"], model="gpt-4.1-nano", sampling_parameters={"temperature": 1.0}
    )

    made_up_task: MathProblem = {
        "id": "debug-1",
        "question": "What is 12 multiplied by 15?",
        "chain": "",
        "result": "180",
        "source": "debug",
    }

    another_made_up_task: MathProblem = {
        "id": "debug-2",
        "question": "What is the square root of 256?",
        "chain": "",
        "result": "16",
        "source": "debug",
    }

    # The agent here must be the same agent that will be used in the real run.
    with runner.run_context(agent=calc_agent, store=store):
        await runner.step(
            made_up_task,
            resources={
                # The key "main_llm" here can be arbitrary
                "main_llm": resource
            },
        )

        # Run another task
        await runner.step(
            another_made_up_task,
            resources={"main_llm": resource},
        )


if __name__ == "__main__":
    asyncio.run(debug())

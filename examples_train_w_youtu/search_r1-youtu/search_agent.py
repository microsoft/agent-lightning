import asyncio

import art
from openai import OpenAI
from utu.agents import OrchestratorAgent, SimpleAgent, get_agent
from utu.utils import AgentsUtils, PrintUtils
from agentlightning.types import RolloutLegacy as RolloutType
import json
import math
import os
import re
import string
from typing import Any, cast
from typing import List
import sympy
from datetime import datetime
import agentlightning
from agentlightning import LLM, LitAgent, NamedResources, Trainer, configure_logger, reward
from openai import AsyncOpenAI
from agents.mcp import MCPServerStreamableHttp
from agents.models.chatcmpl_converter import Converter
from autogen_agentchat.agents import AssistantAgent
from typing import Any, Dict, List, Literal, Optional, cast
from utu.config import ConfigLoader
import asyncio
import os
import re
from typing import TypedDict, cast
from agents import Runner, Agent, OpenAIChatCompletionsModel, set_tracing_disabled, ModelSettings
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
import agentlightning as agl
from qa_em import compute_score_em, em_check
from llm_as_a_judge import compute_score
from agentlightning.emitter.utils import get_tracer
import agentlightning as agl
from agentlightning.types import SpanNames
from agentlightning.emitter.utils import get_tracer
from copy import deepcopy
from copy import deepcopy
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool




AGENT_RUN_TIMEOUT = 300
INSTRUCTION_ANSWER_WRAPPING = "You are a Web Information Seeking Master."


async def evaluation(query:str, prediction_raw: str, ground_truth: List[str]) -> float:
    # reward_score_acc -1 (do not get answer); 0 (wrong answer); 1 (correct answer)
    # exact-match
    # reward_score_acc_rule, reward_score_format = compute_score_em(prediction, ground_truth, format_score=0.1)
    prediction = deepcopy(prediction_raw)
    if not ("<answer>" in prediction and "</answer>" in prediction):
        reward_score_acc_rule = -1
        reward_score_format = 0
    else:
        prediction = prediction[prediction.rfind("<answer>")+len("<answer>"):]
        if "</answer>" in prediction:
            prediction = prediction[:prediction.find("</answer>")]
            reward_score_acc_rule = float(em_check(prediction, ground_truth))
            reward_score_format = 0.1
        else:
            reward_score_acc_rule = -1
            reward_score_format = 0

    acc = False
    if reward_score_acc_rule == -1:
        # 没有抽取到答案 直接返回即可
        reward_score = reward_score_acc_rule
        reward_score_acc_llm = reward_score_acc_rule
    else:
        # 取严格匹配和抽取答案两者的最高分
        # llm as a judge
        reward_score_acc_llm = float(compute_score(prediction, ground_truth, query)["score"])
        reward_score = max(reward_score_acc_rule, reward_score_acc_llm)
    
    print(f"question: {query} | pred: {prediction} | {type(ground_truth)} gold_answer: {ground_truth} | reward-acc-em: {reward_score_acc_rule} | reward-acc-llm: {reward_score_acc_llm} | reward-acc: {reward_score} | reward-format: {reward_score_format}")
    # 最终的奖励包括准确性奖励以及格式奖励二者
    if reward_score >= 1:
        acc = True
    reward_score += reward_score_format
    result = {
        "score": reward_score,
        "acc": acc
    }
    return result






def get_agent_autogen(model: str, openai_base_url: str, temperature: float, top_p: float, tools: list = [],\
    workbench: McpWorkbench = None, max_tokens: int = 14000, max_tool_iterations: int = 8) -> AssistantAgent:
    """Create an AssistantAgent with sandbox tool.
    
    Args:
        model: Model name
        openai_base_url: API base URL
        temperature: Sampling temperature
        workbench: MCP workbench for tools
        max_tokens: Maximum tokens to generate (default: 16384)
    """
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "xxx"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,  # Limit generation length
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=tools,
        system_message=INSTRUCTION_ANSWER_WRAPPING,
        # workbench=workbench,  # ← Sandbox tool automatically registered
        # reflect_on_tool_use=True,
        max_tool_iterations=max_tool_iterations,
    )
    return agent



async def search_wiki(query: Annotated[list[str], "Array of query strings. Include multiple complementary search queries in a single call."]) -> str:
    retrieval_service_url = "http://{YOUR_RETRIEVAL_SERVICE_URL}/retrieve"  # NOTE: you should change this to your own deployment
    result_text, metadata = perform_single_search_batch(
                retrieval_service_url=retrieval_service_url,
                query_list=query,
                topk=3,
                concurrent_semaphore=None,  # Ray handles concurrency control
                timeout=60,
            )
    result_text_str = json.loads(result_text)["result"]
    # print(f"💻[wikipedia] >>> {query} searching total returned response: \n\n{result_text_str}")
    return result_text_str



# Create a function tool.
search_wiki_tool = FunctionTool(search_wiki, description="Performs batched searches on wikipedia: supply an array 'query'; the tool retrieves the top 3 results for each query in one call.")



class SearchYoutuAgent(LitAgent):
    async def training_rollout_async(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> Any:  # type: ignore
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        rollout_id = rollout.rollout_id
        prompt = task["question"]
        source = task["data_source"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])

        try:
            config = ConfigLoader.load_agent_config("examples/rl_train/qa_wiki")
            config.model.model_provider.model = llm.model
            config.model.model_provider.base_url = llm.endpoint
            if llm.api_key:
                config.model.model_provider.api_key = llm.api_key
            else:
                config.model.model_provider.api_key = "xxx"
            config.max_turns = llm.sampling_parameters.get("max_turns", 5)
            config.model.model_settings.temperature = llm.sampling_parameters.get("temperature", 1.0)
            config.model.model_settings.top_p = llm.sampling_parameters.get("top_p", 1.0)
            agent = get_agent(config=config)
            task_recorder = await agent.run(prompt)
            final_output = task_recorder.final_output
            messages = task_recorder.to_input_list()

        except Exception as e:
            print("Failure:", str(e))
            final_output = "None"
            messages = []

        num_toolcalls = 0
        for message in messages:
            if (type(message) is dict) and ("type" in message) and (message["type"] == "function_call"):
                num_toolcalls += 1
            elif (type(message) is dict) and ("role" in message) and (message["role"] == "tool"):
                num_toolcalls += 1
        print(f">>> {num_toolcalls=}")
        result = await evaluation(query=prompt, prediction_raw=final_output, ground_truth=answer_list)
        reward_acc = result["score"]
        reward_toolcall = min(1.0, 0.1 * num_toolcalls)
        if reward_acc >= 1:
            reward_value = float(reward_acc) + float(reward_toolcall > 0)   # 此时只要调用工具就算成功1分
        else:
            reward_value = float(reward_acc) + float(reward_toolcall)    # 此时工具调用次数成正比
        
        acc_value = float(result["acc"])
        tracer = get_tracer()
        metrics = {"acc": acc_value}
        attributes = {"reward": reward_value}
        attributes.update(metrics)
        span = tracer.start_span(SpanNames.REWARD.value, attributes=attributes)
        with span:
            pass
        

    async def validation_rollout_async(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> Any:  # type: ignore
        return await self.training_rollout_async(task, resources, rollout)



class SearchAgent(LitAgent):
    async def training_rollout_async(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> Any:  # type: ignore
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        rollout_id = rollout.rollout_id
        prompt = task["question"]
        source = task["data_source"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])

        try:
            ## autogen-agent
            agent = get_agent_autogen(llm.model,
                llm.endpoint,
                temperature=llm.sampling_parameters.get("temperature", 1.0),
                top_p=llm.sampling_parameters.get("top_p", 1.0),
                tools=[search_wiki_tool],
                max_tool_iterations=llm.sampling_parameters.get("max_turns", 5),
                max_tokens=llm.sampling_parameters.get("max_tokens", 16384),
            )
            task_recorder = await agent.run(task=prompt)
            messages = task_recorder.messages
            final_output = messages[-1].content

        except Exception as e:
            print("Failure:", str(e))
            final_output = "None"
            messages = []

        num_toolcalls = 0
        for message in messages:
            if (type(message) is dict) and ("type" in message) and (message["type"] == "function_call"):
                num_toolcalls += 1
            elif (type(message) is dict) and ("role" in message) and (message["role"] == "tool"):
                num_toolcalls += 1
            elif "type" in message or hasattr(message, "type"):
                if message.type == "ToolCallRequestEvent":
                    num_toolcalls += 1

        # print(f">>> {messages=}")
        print(f">>> {num_toolcalls=}")
        result = await evaluation(query=prompt, prediction_raw=final_output, ground_truth=answer_list)
        reward_acc = result["score"]
        reward_toolcall = min(1.0, 0.1 * num_toolcalls)
        if reward_acc >= 1:
            reward_value = float(reward_acc) + float(reward_toolcall > 0)   # 此时只要调用工具就算成功1分
        else:
            reward_value = float(reward_acc) + float(reward_toolcall)    # 此时工具调用次数成正比
        
        acc_value = float(result["acc"])
        tracer = get_tracer()
        metrics = {"acc": acc_value}
        attributes = {"reward": reward_value}
        attributes.update(metrics)
        span = tracer.start_span(SpanNames.REWARD.value, attributes=attributes)
        with span:
            pass
        
    async def validation_rollout_async(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> Any:  # type: ignore
        return await self.training_rollout_async(task, resources, rollout)


if __name__ == "__main__":
    print()
    query = ["machael jackson"]
    retrieval_service_url = "http://{YOUR_RETRIEVAL_SERVICE_URL}/retrieve"  # NOTE: you should change this to your own deployment
    result_text, metadata = perform_single_search_batch(
                retrieval_service_url=retrieval_service_url,
                query_list=query,
                topk=3,
                concurrent_semaphore=None,  # Ray handles concurrency control
                timeout=30,
            )
    print(f">>> {result_text=}")

import math
import os
import string
import re
from typing import Any

import sympy
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader
from openai import OpenAI
import time
from qa_em import compute_score_em
import requests

configure_logger()

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


@reward
async def eval(prediction: str, ground_truth: list) -> float:
    reward = float(compute_score_em(prediction, ground_truth))
    print(f"pred: {prediction} | {type(ground_truth)} gold_answer: {ground_truth} | res: {reward}")
    return reward


def postprocess_response(response):
    """Process responses to stop at search operation or answer operation."""
    if "</search>" in response:
        response = response.split("</search>")[0] + "</search>"
    elif "</answer>" in response:
        response = response.split("</answer>")[0] + "</answer>"
    return response


def extract_action(response):
    """Process (text-based) predictions from llm into actions and validity flags."""
    if isinstance(response, str):  # for llm output
        pattern = r"<(search|answer)>(.*?)</\1>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(2).strip()  # Return only the content inside the tags
            action = match.group(1)
        else:
            content = ""
            action = None
    else:
        raise ValueError(f"Invalid prediction type: {type(response)}")
    return action, content


def execute_response(response, do_search=True):
    """
    Execute predictions across multiple environments.
    """
    action, content = extract_action(response)
    if action == "answer":
        return ""
    elif action == "search":
        search_result = retrieve_doc(content) if do_search else ""
        return f"\n\n<information>{search_result}</information>\n\n"
    else:
        return f"\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"


def retrieve_doc(query):
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    retrieval_result = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()["result"][0]
    retrieval_result_str = passages2string(retrieval_result)
    return retrieval_result_str


def passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


def call_llm(llm_client, model_name, content, temperature=1.0, max_tokens=500):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


class Searchr1Agent(LitAgent):
    async def training_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources, temperature=1.0
    ) -> Any:
        prompt = INSTRUCTION_FORMAT + task["question"]
        answer_list = task["golden_answers"]

        llm: LLM = resources.get("main_llm")
        client = OpenAI(
            base_url=llm.endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        turn_id = 0
        finished_flag = False
        rollout_content = ""

        while turn_id < 4 and not finished_flag:
            turn_id += 1
            turn_response = call_llm(
                client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
            )
            valid_turn_response = postprocess_response(turn_response)
            turn_env_feedback = execute_response(valid_turn_response)
            if len(turn_env_feedback) == 0:
                finished_flag = True
            print(f"TURN ID {turn_id} | RESP: {turn_response} | ENV FEEDBACK: {turn_env_feedback}")
            rollout_content += turn_response + turn_env_feedback

        if not finished_flag:
            turn_response = call_llm(
                client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
            )
            rollout_content += turn_response
            print(f"LAST TURN GENERATE | RESP: {turn_response}")

        reward = await eval(rollout_content, answer_list)  # reward is tracked with the decorator
        print(
            "question: {} answer: {} ground_truth: {} reward: {}".format(
                task["question"], rollout_content, answer_list, reward
            )
        )
        return reward

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        llm: LLM = resources.get("main_llm")
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources, temperature=0.0)


if __name__ == "__main__":
    Trainer(n_workers=128).fit(Searchr1Agent(), "http://localhost:9999/")

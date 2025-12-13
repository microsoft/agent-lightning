# Copyright (c) Microsoft. All rights reserved.

import os
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import requests
from openai import OpenAI
from qa_em import compute_score_em

from agentlightning import LLM, LitAgent, NamedResources, Trainer, reward, setup_logging

setup_logging()

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


class Document(TypedDict):
    contents: str


class RetrievalItem(TypedDict):
    document: Document


@reward
async def eval(prediction: str, ground_truth: List[str]) -> float:
    reward_score = float(compute_score_em(prediction, ground_truth))
    print(f"pred: {prediction} | {type(ground_truth)} gold_answer: {ground_truth} | res: {reward_score}")
    return reward_score


def postprocess_response(response: str) -> str:
    """Process responses to stop at search operation or answer operation."""
    if "</search>" in response:
        response = response.split("</search>")[0] + "</search>"
    elif "</answer>" in response:
        response = response.split("</answer>")[0] + "</answer>"
    return response


def extract_action(response: str) -> Tuple[Optional[str], str]:
    """Process (text-based) predictions from llm into actions and validity flags."""
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action: Optional[str] = match.group(1)
    else:
        content = ""
        action = None
    return action, content


def execute_response(response: str, do_search: bool = True) -> str:
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
        return (
            "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        )


def retrieve_doc(query: str) -> str:
    payload: Dict[str, Any] = {"queries": [query], "topk": 3, "return_scores": True}
    response = requests.post("http://127.0.0.1:8000/retrieve", json=payload)
    response.raise_for_status()
    json_resp: Dict[str, Any] = cast(Dict[str, Any], response.json())
    retrieval_result: List[RetrievalItem] = cast(List[RetrievalItem], json_resp["result"][0])
    retrieval_result_str = passages2string(retrieval_result)
    return retrieval_result_str


def passages2string(retrieval_result: List[RetrievalItem]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(list(retrieval_result)):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


def call_llm(
    llm_client: OpenAI,
    model_name: str,
    content: str,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> str:
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


class Searchr1Agent(LitAgent[Any]):
    async def training_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
        temperature: float = 1.0,
    ) -> Any:
        prompt = INSTRUCTION_FORMAT + task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        llm: LLM = cast(LLM, resources.get("main_llm"))
        client = OpenAI(
            base_url=llm.endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        turn_id = 0
        finished_flag = False
        rollout_content: str = ""

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

        reward_score = await eval(rollout_content, answer_list)  # reward is tracked with the decorator
        print(
            "question: {} answer: {} ground_truth: {} reward: {}".format(
                task["question"], rollout_content, answer_list, reward_score
            )
        )
        return reward_score

    async def validation_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
    ) -> Any:
        # Use the same resources; set temperature to 0.0 for deterministic validation.
        return await self.training_rollout_async(task, resources, rollout, temperature=0.0)


if __name__ == "__main__":
    Trainer(n_workers=128).fit(Searchr1Agent(), "http://localhost:9999/")

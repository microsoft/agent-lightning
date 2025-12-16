# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import pandas as pd
import requests
from openai import OpenAI
from qa_em import compute_score_em

from agentlightning import LLM, LitAgent, NamedResources, Rollout, Trainer, configure_logger, setup_logging

setup_logging()
logger = configure_logger(name=__name__)

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


class Document(TypedDict):
    contents: str


class RetrievalItem(TypedDict):
    document: Document


def eval(prediction: str, ground_truth: List[str]) -> float:
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


class SearchR1Agent(LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 4,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.data_dir = os.environ.get("VERL_SEARCHR1_DATA_DIR", "data")
        self.max_turns = max_turns

    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        prompt = INSTRUCTION_FORMAT + task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Question: {task['question']}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {answer_list}")

        start_time = time.time()
        llm: LLM = cast(LLM, resources["main_llm"])
        client = OpenAI(
            base_url=llm.get_base_url(rollout_id, rollout.attempt.attempt_id),  # type: ignore
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        if rollout.mode == "train":
            temperature = llm.sampling_parameters.get("temperature", 1.0)
        else:
            temperature = self.val_temperature if self.val_temperature is not None else 0.0

        turn_id = 0
        finished_flag = False
        rollout_content: str = ""

        try:
            while turn_id < self.max_turns and not finished_flag:
                turn_id += 1
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                valid_turn_response = postprocess_response(turn_response)
                rollout_content += valid_turn_response
                turn_env_feedback = execute_response(valid_turn_response)
                if len(turn_env_feedback) == 0:
                    finished_flag = True
                else:
                    rollout_content += turn_env_feedback
                logger.info(f"TURN ID {turn_id} | RESP: {turn_response} | ENV FEEDBACK: {turn_env_feedback}")

            if not finished_flag:
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                rollout_content += turn_response
                logger.info(f"LAST TURN GENERATE | RESP: {turn_response}")

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None

        end_time_rollout = time.time()
        reward_score = eval(rollout_content, answer_list)
        logger.info("[Rollout %s] Reward: %s", rollout_id, reward_score)
        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )
        logger.info(
            "question: {} answer: {} ground_truth: {} reward: {}".format(
                task["question"], rollout_content, answer_list, reward_score
            )
        )
        return reward_score


def debug_search_r1_agent():
    searchr1_dev_data_path = os.path.join(os.environ.get("VERL_SEARCHR1_DATA_DIR", "data"), "test.parquet")
    if not os.path.exists(searchr1_dev_data_path):
        raise FileNotFoundError(f"Search_R1 dev data file {searchr1_dev_data_path} does not exist.")
    df = pd.read_parquet(searchr1_dev_data_path).head(10)  # type: ignore
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    print("Debug data:", df)

    trainer = Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="gpt-4.1-nano",
                sampling_parameters={"temperature": 0.0},
            )
        },
    )
    trainer.dev(SearchR1Agent(), df)


if __name__ == "__main__":
    debug_search_r1_agent()

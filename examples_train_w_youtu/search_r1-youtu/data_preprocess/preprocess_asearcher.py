# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Asearcher dataset to parquet format
"""
import json
import argparse
import os
import re
import datasets
import csv


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


ASEARCHER_SYSTEM_PROMPT="""You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.
As you proceed, adhere to the following principles:
1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.
2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.
3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins."""

INSTRUCTION_FORMAT = """You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer wrapped inside <answer> {your final answer here} </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """

# INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl_path")
    parser.add_argument("--local_dir")
    args = parser.parse_args()
    train_jsonl_path = args.input_jsonl_path
    assert(os.path.exists(train_jsonl_path))
    train_jsonl_basename = os.path.basename(train_jsonl_path).split(".json")[0]
    dataset = datasets.load_dataset('json', data_files=train_jsonl_path)    
    train_dataset = dataset["train"]
    # ['question', 'answer', 'source', 'aug_answer', 'qid']
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = INSTRUCTION_FORMAT + example.pop("question")
            question_source = example.pop("source")
            if question_source is None:
                question_source = ""
            source = train_jsonl_basename + "_" + question_source
            answer_all = example.pop("answer")
            if not (type(answer_all) is list):
                answer_all = [answer_all]
            if "aug_answer" in example:
                answers_aug = example.pop("aug_answer")
            else:
                answers_aug = []
            answer_all += answers_aug
            answer_all = [str(item) for item in answer_all]
            answer_all = list(set(answer_all))
            if "qid" in example:
                question_id = example.pop("qid")
            else:
                question_id = example.pop("id")

            data = {
                "data_source": str(source),
                "prompt": [
                    {
                        "role": "system",
                        "content": ASEARCHER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "question": str(question),
                "golden_answers": answer_all,
                "ability": "web",
                "qid": str(question_id),
                "reward_model": {"style": "rule", "ground_truth": answer_all},
                "agent_name": "tool_agent",
            }
            return data

        return process_fn
    

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, f"{train_jsonl_basename}_train.parquet"))

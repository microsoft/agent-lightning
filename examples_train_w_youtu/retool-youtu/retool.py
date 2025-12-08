# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import re
from typing import Any

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


WRAPPER_CODE="""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
import numpy as np

"""



class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        self.code_pattern2 = re.compile(r"```py(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters.get("code", "")
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()
        matches2 = self.code_pattern2.findall(code)
        if matches2:
            code = matches2[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # lines = code.split("\n")
        # for i, line in reversed(list(enumerate(lines))):
        #     if line == "":
        #         continue
        #     if not lines[i].startswith("print"):
        #         lines[i] = f"print({line})"
        #     break
        # code = "\n".join(lines)
        code = WRAPPER_CODE + code
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names, load_from_cache_file=False
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        logger.info(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        print(prompt)
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "question": prompt,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["question"] = content + answer_format
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row

def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["pred"] is None:
        result["pred"] = ""
    
    return result

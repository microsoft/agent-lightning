# Copyright (c) Microsoft. All rights reserved.

"""This script is the debugging script for the legacy Calc-X agent (v0.1)."""

import os

from legacy_calc_agent import LegacyCalcAgent

from agentlightning import LLM, DevTaskLoader, Trainer


def dev_task_loader() -> DevTaskLoader:
    return DevTaskLoader(
        tasks=[
            {
                "question": "What is 2 + 2?",
                "result": "4",
            },
            {
                "question": "What is 3 * 5?",
                "result": "15",
            },
            {
                "question": "What is the square root of 16?",
                "result": "4",
            },
        ],
        resources={
            "main_llm": LLM(
                endpoint=os.environ["OPENAI_BASE_URL"], model="gpt-4.1-nano", sampling_parameters={"temperature": 0.7}
            ),
        },
    )


if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=2).fit_v0(
        LegacyCalcAgent(), "http://localhost:9999/", dev_data=dev_task_loader()
    )

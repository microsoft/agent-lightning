import os
from agentlightning import Trainer, DevTaskLoader, LLM
from calc_agent import CalcAgent


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
                endpoint=os.environ["OPENAI_API_BASE"], model="gpt-4o-mini", sampling_parameters={"temperature": 0.7}
            ),
        },
    )


if __name__ == "__main__":
    Trainer(n_workers=4, dev=True, max_tasks=5).fit(CalcAgent(), "http://localhost:9999/", dev_task_loader())

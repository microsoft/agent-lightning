# Copyright (c) Microsoft. All rights reserved.

"""HotPotQA agent with multi-backend LLM support.

Supports Azure OpenAI (Entra ID or API key) and plain OpenAI. The backend is
selected via the ``LLM_PROVIDER`` env var — see ``llm_backend.py`` for details.

Usage::

    # Azure Entra ID (default):
    az login
    python hotpotqa_agent.py

    # Azure API key:
    LLM_PROVIDER=azure_key python hotpotqa_agent.py

    # OpenAI:
    LLM_PROVIDER=openai python hotpotqa_agent.py
"""

from __future__ import annotations

import asyncio
import re
import string
from collections import Counter
from typing import Any, List, Optional, Tuple, TypedDict, cast

from dspy.datasets import HotPotQA  # type: ignore[import-untyped]
from llm_backend import LLMProvider, get_model_names, get_provider, make_client
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console

from agentlightning.adapter import TraceToMessages
from agentlightning.litagent import rollout
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Dataset, PromptTemplate

console = Console()


class HotPotQATask(TypedDict):
    id: str
    question: str
    answer: str


ZERO_METRIC_ANSWERS = {"yes", "no", "noanswer"}


def prompt_template_baseline() -> PromptTemplate:
    return PromptTemplate(
        template=(
            "Answer the following question with a short factoid answer. "
            "Return only the answer, with no explanation.\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        engine="f-string",
    )


def _normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def hotpot_exact_match(prediction: Optional[str], gold: str) -> float:
    if not prediction:
        return 0.0
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


def hotpot_f1(prediction: Optional[str], gold: str) -> float:
    if not prediction:
        return 0.0

    normalized_prediction = _normalize_answer(prediction)
    normalized_gold = _normalize_answer(gold)

    if normalized_prediction in ZERO_METRIC_ANSWERS or normalized_gold in ZERO_METRIC_ANSWERS:
        return float(normalized_prediction == normalized_gold)

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hotpotqa_grader(final_message: Optional[str], expected_answer: str) -> float:
    em = hotpot_exact_match(final_message, expected_answer)
    f1 = hotpot_f1(final_message, expected_answer)

    console.print("[bold yellow]=== Gold Answer ===[/bold yellow]")
    console.print(expected_answer)
    console.print("[bold yellow]=== Prediction ===[/bold yellow]")
    console.print(final_message)
    console.print("[bold yellow]=== HotPotQA EM / F1 ===[/bold yellow]")
    console.print({"exact_match": em, "f1": f1})
    return f1


def _resolve_runtime_backend() -> Tuple[LLMProvider, str]:
    """Resolve the provider and model at call time.

    This allows scripts such as ``hotpotqa_gepa.py`` and
    ``gepa_autoresearch.py`` to set ``LLM_PROVIDER`` after import time and still
    have the rollout use the intended backend.
    """

    provider = get_provider()
    deployment_name, _ = get_model_names(provider)
    return provider, deployment_name


@rollout
def hotpotqa_agent(task: HotPotQATask, prompt_template: PromptTemplate) -> float:
    """Answer a HotPotQA question.

    The prompt template is optimized by Agent-lightning's GEPA algorithm.
    """

    provider, model = _resolve_runtime_backend()
    client = make_client(provider)

    user_message = prompt_template.format(question=task["question"])

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You answer factoid questions. Give the shortest correct answer you can. "
                "Do not add explanations unless the prompt explicitly asks for them."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    console.print("[bold yellow]=== Question ===[/bold yellow]")
    console.print(task["question"])
    console.print("[bold yellow]=== User Message ===[/bold yellow]")
    console.print(user_message)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    final_message = resp.choices[0].message.content

    console.print("[bold yellow]=== Final Assistant Message ===[/bold yellow]")
    console.print(final_message)

    return hotpotqa_grader(final_message, task["answer"])


def _example_to_task(split_name: str, index: int, example: Any) -> HotPotQATask:
    return {
        "id": str(example.get("id", f"{split_name}-{index}")),
        "question": str(example["question"]),
        "answer": str(example["answer"]),
    }


def load_hotpotqa_splits(
    train_size: int = 32,
    dev_size: int = 32,
    test_size: int = 0,
    train_seed: int = 1,
    eval_seed: int = 2023,
) -> Tuple[Dataset[HotPotQATask], Dataset[HotPotQATask], Dataset[HotPotQATask]]:
    dataset: Any = HotPotQA(  # type: ignore[reportUnknownVariableType]
        train_seed=train_seed,
        train_size=train_size,
        eval_seed=eval_seed,
        dev_size=dev_size,
        test_size=test_size,
    )

    train_tasks: List[HotPotQATask] = [
        _example_to_task("train", idx, example) for idx, example in enumerate(dataset.train)  # type: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    ]
    dev_tasks: List[HotPotQATask] = [
        _example_to_task("dev", idx, example) for idx, example in enumerate(dataset.dev)  # type: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    ]
    test_tasks: List[HotPotQATask] = [
        _example_to_task("test", idx, example) for idx, example in enumerate(dataset.test)  # type: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    ]
    return (
        cast(Dataset[HotPotQATask], train_tasks),
        cast(Dataset[HotPotQATask], dev_tasks),
        cast(Dataset[HotPotQATask], test_tasks),
    )


def load_hotpotqa_tasks(
    train_size: int = 32,
    dev_size: int = 32,
    train_seed: int = 1,
    eval_seed: int = 2023,
) -> Tuple[Dataset[HotPotQATask], Dataset[HotPotQATask]]:
    train_tasks, dev_tasks, _ = load_hotpotqa_splits(
        train_size=train_size,
        dev_size=dev_size,
        test_size=0,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    return train_tasks, dev_tasks


def load_hotpotqa_holdout_tasks(
    test_size: int = 32,
    eval_seed: int = 2023,
    train_seed: int = 1,
) -> Dataset[HotPotQATask]:
    """Load a held-out slice from the official HotPotQA validation split."""

    _, _, test_tasks = load_hotpotqa_splits(
        train_size=1,
        dev_size=1,
        test_size=test_size,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    return test_tasks


async def debug_hotpotqa_agent(limit: int = 1) -> None:
    runner = LitAgentRunner[HotPotQATask](AgentOpsTracer())
    store = InMemoryLightningStore()
    prompt_template = prompt_template_baseline()
    dataset, _ = load_hotpotqa_tasks(train_size=limit, dev_size=1)
    tasks = cast(List[HotPotQATask], dataset)
    with runner.run_context(agent=hotpotqa_agent, store=store):
        for task in tasks[:limit]:
            console.print("[bold green]=== Task ===[/bold green]", task, sep="\n")
            rollout = await runner.step(task, resources={"prompt_template": prompt_template})
            spans = await store.query_spans(rollout.rollout_id)
            adapter = TraceToMessages()
            messages = adapter.adapt(spans)
            for message_idx, message in enumerate(messages):
                console.print(f"[bold purple]=== Postmortem Message #{message_idx} ===[/bold purple]")
                console.print(message)
            reward = find_final_reward(spans)
            console.print("[bold purple]=== Postmortem Reward ===[/bold purple]", reward, sep="\n")


if __name__ == "__main__":
    asyncio.run(debug_hotpotqa_agent())

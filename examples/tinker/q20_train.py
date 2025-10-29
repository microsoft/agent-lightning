# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import traceback
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from agl_tinker.env import AGLDatasetBuilder
from agl_tinker.llm import create_llm_proxy
from agl_tinker.train import Config
from agl_tinker.train import main as entrypoint
from crewai import LLM as CrewLLM
from q20_agent import AnswererResponse, SearchTool, TwentyQuestionsFlow
from rich.console import Console

import agentlightning as agl


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Q20Task(TypedDict):
    category: str
    answer: str
    search_enabled: bool


LLM_TIMEOUT = 120.0

console = Console()


@agl.rollout
async def q20_agent(task: Q20Task, llm: agl.LLM, rollout: agl.Rollout) -> None:
    answer_llm_setting = os.getenv("ANSWERER_LLM", "gpt-5-mini")
    search_llm_setting = os.getenv("SEARCH_LLM", "gpt-4.1")
    player_llm = CrewLLM(model="openai/" + llm.model, base_url=llm.endpoint, api_key="dummy", timeout=LLM_TIMEOUT)
    answer_llm = CrewLLM(
        model="openai/" + answer_llm_setting,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="low",
        response_format=AnswererResponse,
        timeout=LLM_TIMEOUT,
    )
    if task["search_enabled"]:
        search_tool = SearchTool(
            model=CrewLLM(
                model="openai/" + search_llm_setting,
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                reasoning_effort="none",
                timeout=LLM_TIMEOUT,
            )
        )
    else:
        search_tool = None

    flow = TwentyQuestionsFlow(player_llm=player_llm, answer_llm=answer_llm, search_tool=search_tool)
    try:
        await flow.kickoff_async(cast(Any, task))
        agl.emit_reward(1.0 if flow.state.correct else 0.0)
    except Exception:
        console.print(f"Error in q20_agent: {traceback.format_exc()}")
        raise
        # Above, the exception is re-raised, so the rollout will appear failed, but reward will be none.
        # The handling below is another approach that will make the rollout appear succeeded, but with 0 reward.
        # I think algorithm should handle the case instead.
        # agl.emit_exception(e)
        # agl.emit_reward(0.0)


def dry_run():
    store = agl.InMemoryLightningStore()
    llm_proxy = create_llm_proxy("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct", store=store)
    trainer = agl.Trainer(
        n_runners=2,
        initial_resources={"llm": llm_proxy.as_resource()},
        store=store,
    )
    try:
        llm_proxy.start()
        sampled_csv = pd.read_csv("q20_nouns.csv").sample(n=4, random_state=42)  # type: ignore
        sampled_csv["search_enabled"] = False
        dataset = sampled_csv.to_dict(orient="records")  # type: ignore
        trainer.dev(q20_agent, cast(agl.Dataset[Q20Task], dataset))
    finally:
        llm_proxy.stop()


async def algo(search: bool, model: Literal["qwen4b", "qwen30b"], port: int):
    raw_data = pd.read_csv("q20_nouns.csv")  # type: ignore
    raw_data["search_enabled"] = search
    train_data, test_data = raw_data[raw_data["split"] == "train"], raw_data[raw_data["split"] == "test"]  # type: ignore

    train_dataset = cast(agl.Dataset[Q20Task], train_data.to_dict(orient="records"))  # type: ignore
    test_dataset = cast(agl.Dataset[Q20Task], test_data.to_dict(orient="records"))  # type: ignore

    if model == "qwen4b":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
        renderer_name = "qwen3"
    elif model == "qwen30b":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        renderer_name = "qwen3"
    else:
        raise ValueError(f"Invalid model: {model}")

    experiment_name = f"q20_{'search' if search else 'no_search'}_{model}_lr1e-6"

    llm_proxy_port = _find_available_port()

    config = Config(
        learning_rate=1e-6,
        dataset_builder=AGLDatasetBuilder(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=8,
            shuffle=True,
            group_size=4,
            seed=42,
            n_epochs=10,
        ),
        renderer_name=renderer_name,
        model_name=model_name,
        log_path=f"logs/{experiment_name}",
        concurrency=16,
        eval_every=4,
        wandb_project="AgentLightningQ20",
        wandb_name=experiment_name,
        store_address=f"http://localhost:{port}",
        llm_proxy_port=llm_proxy_port,
        adapter_from_llm_proxy=False,
        llm_proxy_retry_attempts=5,
    )
    await entrypoint(config)


def runner(port: int = 4747, n_runners: int = 2):
    # Run only the runners without algorithm
    store = agl.LightningStoreClient(f"http://localhost:{port}")
    trainer = agl.Trainer(
        algorithm=None,
        store=store,
        strategy={"type": "cs", "managed_store": False, "n_runners": n_runners, "role": "runner"},
    )
    trainer.fit(q20_agent)


def _run_dryrun(_args: argparse.Namespace) -> None:
    dry_run()


def _run_algo(args: argparse.Namespace) -> None:
    asyncio.run(algo(search=args.search, model=args.model, port=args.port))


def _run_runner(args: argparse.Namespace) -> None:
    runner(port=args.port, n_runners=args.n_runners)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Q20 AgentLightning experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dryrun_parser = subparsers.add_parser("dryrun", help="Run the in-memory dry run.")
    dryrun_parser.set_defaults(func=_run_dryrun)

    algo_parser = subparsers.add_parser("algo", help="Launch the full training algorithm.")
    algo_parser.add_argument("--port", type=int, default=4747, help="Port for the AgentLightning store.")
    algo_parser.add_argument("--search", action="store_true", help="Enable search tool.")
    algo_parser.add_argument(
        "--model",
        choices=("qwen4b", "qwen30b"),
        default="qwen30b",
        help="Model variant to train.",
    )
    algo_parser.set_defaults(func=_run_algo)

    runner_parser = subparsers.add_parser("runner", help="Run only the rollout runners.")
    runner_parser.add_argument("--port", type=int, default=4747, help="Port for the AgentLightning store.")
    runner_parser.add_argument("--n-runners", type=int, default=2, help="Number of runners to use.")
    runner_parser.set_defaults(func=_run_runner)

    args = parser.parse_args()
    agl.configure_logger()
    args.func(args)


if __name__ == "__main__":
    main()

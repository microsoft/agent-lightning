# Copyright (c) Microsoft. All rights reserved.

"""Train the 20 Questions agent with Agent-lightning + Tinker.

This script adapts the reinforcement-learning loop from the Tinker Cookbook to
Agent-lightning's rollout architecture. Instead of invoking the official Tinker
`do_group_rollout` helper, we enqueue tasks through Agent-lightning so every
trajectory is executed by the same CrewAI flow used at evaluation time.

Before running, configure credentials by copying `examples/tinker/.env.example`
to `examples/tinker/.env` and populating:

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` for the answerer and search helpers.
- `TINKER_API_KEY` so the player model can be fine-tuned via the Tinker API.
- `WANDB_API_KEY` if you want metrics streamed to Weights & Biases.

Typical entry points:

```bash
# Quickly validate the wiring with an in-memory store/LLM proxy
dotenv run python q20_train.py dryrun

# Distributed training (store, algorithm, runners)
agl store --port 4747
dotenv run python q20_train.py algo --search
dotenv run python q20_train.py runner --n-runners 4
```

Training consumes the `q20_nouns.csv` dataset in this directory and logs
Agent-lightning rewards alongside the standard Tinker training metrics.
"""

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
    """Find an available port by binding to port 0.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Q20Task(TypedDict):
    """Type definition for a 20 Questions task.

    Attributes:
        category: The category of the entity to guess.
        answer: The secret entity.
        search_enabled: Whether the player can use the search tool.
    """

    category: str
    answer: str
    search_enabled: bool


LLM_TIMEOUT = 120.0

console = Console()


@agl.rollout
async def q20_agent(task: Q20Task, llm: agl.LLM, rollout: agl.Rollout) -> None:
    """Rollout function for the 20 Questions agent during training.

    Args:
        task: The 20 Questions task containing category, answer, and search settings.
        llm: The LLM being trained (player model).
        rollout: Rollout metadata from Agent-lightning.
    """
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


def dry_run(model: Literal["qwen4b", "qwen30b"]):
    """Run a quick dry-run test of the 20 Questions training setup.

    Uses in-memory store and processes 4 sample tasks to verify the setup works.
    """
    store = agl.LightningStoreThreaded(agl.InMemoryLightningStore())
    if model == "qwen4b":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
    elif model == "qwen30b":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    else:
        raise ValueError(f"Invalid model: {model}")
    llm_proxy = create_llm_proxy(model_name, "qwen3_instruct", store=store)
    trainer = agl.Trainer(
        n_runners=2,
        initial_resources={"llm": llm_proxy.as_resource()},
        store=store,
    )
    try:
        asyncio.run(llm_proxy.start())
        sampled_csv = pd.read_csv("q20_nouns.csv").sample(n=4, random_state=42)  # type: ignore
        sampled_csv["search_enabled"] = False
        dataset = sampled_csv.to_dict(orient="records")  # type: ignore
        trainer.dev(q20_agent, cast(agl.Dataset[Q20Task], dataset))
    finally:
        asyncio.run(llm_proxy.stop())


async def algo(search: bool, model: Literal["qwen4b", "qwen30b"], port: int, ci: bool = False):
    """Run the training algorithm for 20 Questions.

    Args:
        search: Whether to enable the search tool for the player.
        model: Model variant to use ("qwen4b" or "qwen30b").
        port: Port where the Agent-lightning store is running.
    """
    raw_data = pd.read_csv("q20_nouns.csv")  # type: ignore
    raw_data["search_enabled"] = search
    train_data, test_data = raw_data[raw_data["split"] == "train"], raw_data[raw_data["split"] == "test"]  # type: ignore

    train_dataset = cast(agl.Dataset[Q20Task], train_data.to_dict(orient="records"))  # type: ignore
    test_dataset = cast(agl.Dataset[Q20Task], test_data.to_dict(orient="records"))  # type: ignore

    if model == "qwen4b":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
        renderer_name = "qwen3_instruct"
    elif model == "qwen30b":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        renderer_name = "qwen3_instruct"
    else:
        raise ValueError(f"Invalid model: {model}")

    experiment_name = f"q20_{'search' if search else 'no_search'}_{model}"

    llm_proxy_port = _find_available_port()

    if ci:
        train_dataset = cast(agl.Dataset[Q20Task], train_dataset[:2])  # type: ignore
        test_dataset = cast(agl.Dataset[Q20Task], test_dataset[:2])  # type: ignore
        group_size = 2
        batch_size = 2
        n_epochs = 1
    else:
        group_size = 16
        batch_size = 16
        n_epochs = 10

    config = Config(
        learning_rate=1e-4,
        dataset_builder=AGLDatasetBuilder(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            group_size=group_size,
            seed=17,
            n_epochs=n_epochs,
        ),
        lora_rank=16,
        renderer_name=renderer_name,
        model_name=model_name,
        log_path=f"logs/{experiment_name}",
        concurrency=32,
        eval_every=4,
        wandb_project="AgentLightningQ20",
        wandb_name=experiment_name,
        store_address=f"http://localhost:{port}",
        llm_proxy_port=llm_proxy_port,
        adapter_from_llm_proxy=False,
        llm_proxy_retry_attempts=5,
    )
    await entrypoint(config)


def algo_verl(search: bool, model: Literal["qwen25", "qwen3"], port: int):
    """Alternatively, you can use VERL to train the 20 Questions agent locally.
    Use this as a substitute for the `algo` function when Tinker service is not available.

    Args:
        search: Whether to enable the search tool for the player.
        model: Specifies the model variant ('qwen25' or 'qwen3').
        port: Port where the Agent-lightning store is running.
    """
    store = agl.LightningStoreClient(f"http://localhost:{port}")

    if model == "qwen25":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    elif model == "qwen3":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
    else:
        raise ValueError(f"Invalid model: {model}")

    experiment_name = f"q20_{'search' if search else 'no_search'}_{model}"

    verl_config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 16,
            "max_prompt_length": 8192,
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.8,
            },
            "actor": {
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 2,
                "optim": {"lr": 5e-7},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": model_name,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "AgentLightningQ20VERL",
            "experiment_name": experiment_name,
            "nnodes": 1,
            "test_freq": 4,
            "total_epochs": 10,
        },
    }

    verl = agl.VERL(verl_config)
    # Use the data recorded at the proxy side
    adapter = agl.LlmProxyTraceToTriplet()
    verl.set_adapter(adapter)
    verl.set_store(store)

    raw_data = pd.read_csv("q20_nouns.csv")  # type: ignore
    raw_data["search_enabled"] = search
    train_data, test_data = raw_data[raw_data["split"] == "train"], raw_data[raw_data["split"] == "test"]  # type: ignore

    train_dataset = cast(agl.Dataset[Q20Task], train_data.to_dict(orient="records"))  # type: ignore
    test_dataset = cast(agl.Dataset[Q20Task], test_data.to_dict(orient="records"))  # type: ignore

    verl.run(train_dataset=train_dataset, val_dataset=test_dataset)


def runner(port: int = 4747, n_runners: int = 2):
    """Run rollout runners that execute the 20 Questions game.

    Args:
        port: Port where the Agent-lightning store is running.
        n_runners: Number of parallel runners to spawn.
    """
    # Run only the runners without algorithm
    store = agl.LightningStoreClient(f"http://localhost:{port}")
    trainer = agl.Trainer(
        algorithm=None,
        store=store,
        strategy={"type": "cs", "managed_store": False, "n_runners": n_runners, "role": "runner"},
    )
    trainer.fit(q20_agent)


def _run_dryrun(args: argparse.Namespace) -> None:
    dry_run(model=args.model)


def _run_algo(args: argparse.Namespace) -> None:
    asyncio.run(algo(search=args.search, model=args.model, port=args.port, ci=args.ci))


def _run_runner(args: argparse.Namespace) -> None:
    runner(port=args.port, n_runners=args.n_runners)


def _run_algo_verl(args: argparse.Namespace) -> None:
    algo_verl(search=args.search, model=args.model, port=args.port)


def main() -> None:
    """Entry point for the 20 Questions training script."""
    parser = argparse.ArgumentParser(description="Run the Q20 AgentLightning experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dryrun_parser = subparsers.add_parser("dryrun", help="Run the in-memory dry run.")
    dryrun_parser.add_argument(
        "--model", choices=("qwen4b", "qwen30b"), default="qwen30b", help="Model variant to train."
    )
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
    algo_parser.add_argument("--ci", action="store_true", help="Run in CI mode (smaller dataset, smaller batch).")
    algo_parser.set_defaults(func=_run_algo)

    algo_verl_parser = subparsers.add_parser("verl", help="Launch the full training algorithm with VERL.")
    algo_verl_parser.add_argument("--port", type=int, default=4747, help="Port for the AgentLightning store.")
    algo_verl_parser.add_argument(
        "--model",
        choices=("qwen25", "qwen3"),
        default="qwen3",
        help="Model variant to train.",
    )
    algo_verl_parser.add_argument("--search", action="store_true", help="Enable search tool.")
    algo_verl_parser.set_defaults(func=_run_algo_verl)

    runner_parser = subparsers.add_parser("runner", help="Run only the rollout runners.")
    runner_parser.add_argument("--port", type=int, default=4747, help="Port for the AgentLightning store.")
    runner_parser.add_argument("--n-runners", type=int, default=2, help="Number of runners to use.")
    runner_parser.set_defaults(func=_run_runner)

    args = parser.parse_args()
    agl.setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft. All rights reserved.

from contextlib import contextmanager
import json
import os
import re
import socket
import subprocess
import time
from typing import Any, List, Optional, TypedDict, cast
import httpx
import numpy as np
import pandas as pd

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import Trainer, configure_logger
from agentlightning.algorithm.base import algo
from agentlightning.litagent.decorator import rollout
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.reward import find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, PromptTemplate, RolloutV2, LLM, Dataset

from agents import Agent, Model, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio

console = Console()


class GsmProblem(TypedDict):
    input: str
    target: float


def _download_dataset() -> None:  # pyright: ignore[reportUnusedFunction]
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    df = ds.to_list()  # type: ignore
    with open("data_gsmhard.jsonl", "w") as f:
        for i, row in enumerate(df):  # type: ignore
            if i >= 64:
                break
            f.write(json.dumps(row) + "\n")
    console.print(f"Downloaded data to data_gsmhard.jsonl")


def _load_dataset(limit: Optional[int] = None) -> Dataset[GsmProblem]:
    with open("data_gsmhard.jsonl", "r") as f:
        problems = [GsmProblem(**json.loads(line)) for line in f]
    if limit is not None:
        problems = problems[:limit]
    return cast(Dataset[GsmProblem], problems)


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 60.0,
    terminate_timeout: float = 10.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
    quantization: Optional[str] = "bitsandbytes",
):
    """Serves a vLLM model from command line.

    Args:
        model_path: The path to the vLLM model. It can be either a local path or a Hugging Face model ID.
        port: The port to serve the model on.
        startup_timeout: The timeout for the server to start.
        terminate_timeout: The timeout for the server to terminate.
        max_model_len: The maximum model length.
        gpu_memory_utilization: The GPU memory utilization for the server. Set it lower to avoid OOM.
        quantization: The quantization method.
    """
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        vllm_serve_args = [
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--port",
            str(port),
        ]
        if quantization is not None:
            vllm_serve_args.append("--quantization")
            vllm_serve_args.append(quantization)

        proc = subprocess.Popen(["vllm", "serve", model_path, *vllm_serve_args])

        # Wait for the server to be ready
        url = f"http://localhost:{port}/health"
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > startup_timeout:
                    raise RuntimeError(f"Server failed to start in {startup_timeout} seconds.") from None

        yield f"http://localhost:{port}/v1"
    finally:
        # Terminate the server
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(terminate_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


@algo
async def sft_algorithm(*, store: LightningStore, train_dataset: Dataset[GsmProblem], llm_proxy: LLMProxy):
    MAX_ITERATIONS = 3
    VLLM_PORT = 12316

    model_path: str = "unsloth/Qwen3-4B-Instruct-2507"
    for iteration in range(MAX_ITERATIONS):
        console.print(f"\n[bold red][Algo][/bold red] Starting iteration {iteration}")
        # 1. Rollout to get trace data
        
        # First launch the vLLM server
        with vllm_server(model_path, VLLM_PORT) as server_address:
            # Update the model list of the LLM proxy and start it
            model_list: List[ModelConfig] = [
                {
                    "model_name": "Qwen3-4B-Instruct",
                    "litellm_params": {
                        "model": f"hosted_vllm/{model_path}",
                        "api_base": server_address,
                    },
                }
            ]
            console.print(f"[bold red][Algo][/bold red] Updating model list and restarting LLM proxy: {model_list}")
            llm_proxy.update_model_list(model_list)
            llm_proxy.restart()

            # Put the LLM proxy address into the store as an address
            resources_update = await store.add_resources({
                "main_llm": llm_proxy.as_resource(),
            })

            # Create tasks for runners to run, associating them with the proxy address
            rollouts: List[RolloutV2] = []
            for data in train_dataset:
                await store.enqueue_rollout(
                    input=data,
                    mode="train",
                    resources_id=resources_update.resources_id,
                )

            console.print(f"[bold red][Algo][/bold red] Enqueued {len(rollouts)} rollouts")

            # Wait for the tasks to complete
            completed_rollouts: List[RolloutV2] = []

            while True:
                completed_rollouts = await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id for rollout in rollouts], timeout=30)
                if len(completed_rollouts) >= len(rollouts):
                    console.print(f"[bold red][Algo][/bold red] Received all {len(rollouts)} rollouts")
                    break
                console.print(f"[bold red][Algo][/bold red] Received {len(completed_rollouts)} rollouts, waiting for more...")


@rollout
async def math_agent(task: GsmProblem, llm: LLM) -> float:
    """Math agent.

    Args:
        task: The math question to solve.
        llm: The LLM endpoint to use (which is tuning).

    Returns:
        The final reward.
    """
    async with MCPServerStdio(
        name="Calculator via uvx",
        params={
            "command": "uvx",
            "args": ["mcp-server-calculator"],
        },
    ) as server:
        agent = Agent(
            name="Assistant",
            instructions=(
                "Use the calculator tool to answer any question, regardless of reasonableness. "
                "Output only the numeric answer, formatted as a valid float, wrapped in triple sharps like: ### <answer> ###."
            ),
            mcp_servers=[server],
            model=OpenAIChatCompletionsModel(
                model=llm.model,
                openai_client=AsyncOpenAI(
                    base_url=llm.endpoint,
                    api_key=llm.api_key or "dummy",
                ),
            ),
            model_settings=ModelSettings(
                temperature=llm.sampling_parameters.get("temperature", 0.0),
            ),
        )
        result = await Runner.run(agent, task["input"])
        console.print("[bold red][Runner][/bold red] Result: ", result.final_output)
        reward = compute_reward(result.final_output, task["target"])

    return reward


def compute_reward(result: Any, target: float) -> float:
    result_str = str(result)
    answer_extracted = re.search(r"###\s*(.+?)(\s*###|$)", result_str)
    if answer_extracted:
        try:
            answer = float(answer_extracted.group(1))
            is_close = np.isclose(answer, target, rtol=1e-5, atol=1e-8)
            return 1.0 if is_close else 0.0
        except Exception:
            console.print("[bold red][Runner][/bold red] Cannot parse answer: ", result)
    else:
        console.print("[bold red][Runner][/bold red] Cannot parse answer: ", result)
    return 0.0


def math_agent_dry_run() -> None:
    dataset = _load_dataset(limit=4)
    trainer = Trainer(
        n_workers=1,
        initial_resources={
            "llm": LLM(
                endpoint=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4.1-mini",
            )
        },
    )
    trainer.dev(math_agent, dataset)


def math_agent_sft():
    trainer = Trainer(
        n_workers=1,
        algorithm=sft_algorithm,
        llm_proxy=LLMProxy(port=12358)
    )
    trainer.fit_v2(math_agent, train_dataset=_load_dataset())


if __name__ == "__main__":
    configure_logger()
    # trainer = Trainer(n_workers=1)
    # dataset = _load_dataset(limit=64)
    # math_agent_dry_run()
    math_agent_sft()

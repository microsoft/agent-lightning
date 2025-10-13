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
from datasets import Dataset as HuggingFaceDataset, load_dataset  # type: ignore

from agentlightning import Trainer, configure_logger
from agentlightning.adapter.triplet import LlmProxyTripletAdapter
from agentlightning.algorithm.base import algo
from agentlightning.litagent.decorator import rollout
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.reward import find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, PromptTemplate, RolloutV2, LLM, Dataset, Triplet

from agents import Agent, Model, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig  # type: ignore

console = Console()


class GsmProblem(TypedDict):
    input: str
    target: float


def _download_dataset() -> None:  # pyright: ignore[reportUnusedFunction]
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


class HuggingFaceDatasetRecord(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 60.0,
    terminate_timeout: float = 10.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
    quantization: Optional[str] = "bitsandbytes",
    auto_tool_choice: bool = True,
    tool_call_parser: Optional[str] = "hermes",
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
        auto_tool_choice: Whether to enable auto tool choice.
        tool_call_parser: The tool call parser to use.
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
        if auto_tool_choice:
            vllm_serve_args.append("--enable-auto-tool-choice")
        if tool_call_parser is not None:
            vllm_serve_args.append("--tool-call-parser")
            vllm_serve_args.append(tool_call_parser)

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
async def sft_algorithm(*, store: LightningStore, train_dataset: Dataset[GsmProblem]):
    MAX_ITERATIONS = 1
    VLLM_PORT = 12316

    # Download the model before starting the script:
    # hf download unsloth/Qwen3-4B-Instruct-2507 --local-dir models/version_0
    model_path = "models/version_0"

    # Create the LLM proxy for rollout worker access and trace data collection
    llm_proxy = LLMProxy(port=12358, store=store)

    # This data adapter util is used to convert the trace data recorded by LLM proxy
    # into a format suitable for SFT
    data_adapter = LlmProxyTripletAdapter()

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
            # Restart the LLM proxy after backend model list update
            # If LLM proxy has never been started, it will be started
            llm_proxy.restart()

            # Put the LLM proxy address into the store as an address
            resources_update = await store.add_resources(
                {
                    "main_llm": llm_proxy.as_resource(),
                }
            )

            # Create tasks for runners to run, associating them with the proxy address
            rollouts: List[RolloutV2] = []
            for data in train_dataset:
                rollouts.append(
                    await store.enqueue_rollout(
                        input=data,
                        mode="train",
                        resources_id=resources_update.resources_id,
                    )
                )

            console.print(f"[bold red][Algo][/bold red] Enqueued {len(rollouts)} rollouts")

            # Wait for the tasks to complete
            completed_rollouts: List[RolloutV2] = []

            while True:
                completed_rollouts = await store.wait_for_rollouts(
                    rollout_ids=[rollout.rollout_id for rollout in rollouts], timeout=30
                )
                if len(completed_rollouts) >= len(rollouts):
                    console.print(f"[bold red][Algo][/bold red] Received all {len(rollouts)} rollouts")
                    break
                console.print(
                    f"[bold red][Algo][/bold red] Received {len(completed_rollouts)} rollouts, waiting for more..."
                )

        # LLM server can be shutdown now as we perform the training

        # 2. Prepare the dataset for SFT
        all_triplets: List[HuggingFaceDatasetRecord] = []
        for rollout in completed_rollouts:
            spans = await store.query_spans(rollout.rollout_id, "latest")
            triplets = data_adapter.adapt(spans)

            # Logging the prompt and response lengths and rewards for debugging
            prompt_lengths = [len(t.prompt["token_ids"]) if t.prompt["token_ids"] else 0 for t in triplets]
            response_lengths = [len(t.response["token_ids"]) if t.response["token_ids"] else 0 for t in triplets]
            console.print(
                f"[bold red][Algo][/bold red] Rollout {rollout.rollout_id} has {len(triplets)} triplets. "
                f"Prompt lengths: {prompt_lengths}. Response lengths: {response_lengths}. "
                f"Rewards are: {[t.reward for t in triplets]}"
            )

            # Converts the triplets to a HuggingFace Dataset
            for triplet in triplets:
                # Ensure that prompt and response are all not empty
                if triplet.prompt.get("token_ids") and triplet.response.get("token_ids"):
                    # HuggingFace Dataset format looks like:
                    # {
                    #     "input_ids": [151644, 872, 198, 3838, 374, 279, 74024],
                    #     "attention_mask": [1, 1, 1, 1, 1, 1, 1],
                    #     "labels": [-100, -100, -100, 3838, 374, 279, 74024],
                    # }
                    input_ids = triplet.prompt["token_ids"] + triplet.response["token_ids"]
                    labels = [-100 for _ in range(len(triplet.prompt["token_ids"]))] + triplet.response["token_ids"]
                    all_triplets.append(
                        {
                            "input_ids": input_ids,
                            "attention_mask": [1] * len(input_ids),
                            "labels": labels,
                        }
                    )

        sft_dataset = HuggingFaceDataset.from_list(all_triplets)  # type: ignore

        console.print(f"[bold red][Algo][/bold red] SFT dataset has {len(sft_dataset)} samples")

        # 3. Start the SFT training
        model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore
            model_name=model_path,
            max_seq_length=4096,  # Choose any for long context!
            load_in_4bit=True,  # 4 bit quantization to reduce memory
        )

        # Config the model to use LoRA
        model = FastLanguageModel.get_peft_model(  # type: ignore
            model,  # type: ignore
            r=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        trainer = SFTTrainer(
            model=model,  # type: ignore
            tokenizer=tokenizer,
            train_dataset=sft_dataset,
            args=SFTConfig(
                # dataset_text_field = "text",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,  # Use GA to mimic batch size!
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="none",  # Use this for WandB etc
            ),
        )

        trainer_stats = trainer.train()  # type: ignore
        console.print(f"[bold red][Algo][/bold red] Trainer stats: {trainer_stats}")

        next_version = f"models/version_{iteration + 1}"
        # Save in 16-bit for vLLM inference later
        model.save_pretrained_merged(next_version, tokenizer, save_method="merged_16bit")  # type: ignore


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
    trainer = Trainer(n_workers=1, algorithm=sft_algorithm, llm_proxy=LLMProxy(port=12358))
    dataset = _load_dataset(limit=4)
    trainer.fit_v2(math_agent, train_dataset=dataset)


if __name__ == "__main__":
    configure_logger()
    # trainer = Trainer(n_workers=1)
    # dataset = _load_dataset(limit=64)
    # math_agent_dry_run()
    math_agent_sft()

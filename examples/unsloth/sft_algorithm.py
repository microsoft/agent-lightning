# Copyright (c) Microsoft. All rights reserved.

"""This sample shows the implementation of a basic SFT algorithm.

It requires a model to be downloaded and a store server before running.

First download the model:

```bash
hf download unsloth/Qwen3-4B-Instruct-2507 --local-dir models/version_0
```

Then run the store server:

```bash
agl store --port 4747
```
"""

import asyncio
import multiprocessing
import os
import random
import subprocess
import time
from contextlib import contextmanager
from typing import List, Optional, TypedDict

import httpx
from datasets import Dataset as HuggingFaceDataset  # type: ignore
from math_agent import GsmProblem, load_math_dataset
from rich.console import Console
from unsloth_helper import unsloth_training

from agentlightning import setup_logging
from agentlightning.adapter import LlmProxyTraceToTriplet, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.types import Dataset, Rollout

console = Console()


class HuggingFaceDatasetRecord(TypedDict):
    """Type definition for a HuggingFace dataset record used in SFT training.

    Attributes:
        input_ids: Token IDs for the entire input sequence (prompt + response).
        attention_mask: Attention mask (all 1s for this use case).
        labels: Token IDs for training labels (-100 for prompt tokens, actual token IDs for response).
        reward: The reward associated with this training sample.
    """

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    reward: float


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 300.0,
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


async def sft_one_iter(
    *,
    iteration: int,
    store: LightningStore,
    model_path: str,
    train_dataset: Dataset[GsmProblem],
    llm_proxy: LLMProxy,
    data_adapter: TraceToTripletBase,
    triplet_fraction: float,
    vllm_port: int,
) -> str:
    """One iteration of SFT.

    The idea is to get all trace data from the rollouts, and then use the reward to select the top triplets to train on.

    Performs (1) rollout - data collection, (2) data conversion, (3) SFT training, and (4) model saving.

    Args:
        iteration: The iteration number.
        store: The LightningStore instance.
        model_path: The path to the model to train. Must be a local path.
        train_dataset: The dataset to train on.
        llm_proxy: The LLM proxy instance. Used to shield between the inference endpoint and the rollout runners.
        data_adapter: The data adapter instance. This is used to convert the trace data recorded by LLM proxy.
        triplet_fraction: The fraction of triplets to use for SFT.
        vllm_port: The port to serve vLLM chat completion endpoint.

    Returns:
        The path to the saved model (next generation).
    """

    console.print(f"\n[bold red][Algo][/bold red] Starting iteration {iteration}")

    # 1. Rollout to get trace data
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")

    # First launch the vLLM server
    with vllm_server(model_path, vllm_port) as server_address:
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
        await llm_proxy.restart()

        # Put the LLM proxy address into the store as an address
        resources_update = await store.add_resources(
            {
                "main_llm": llm_proxy.as_resource(),
            }
        )

        # Create tasks for runners to run, associating them with the proxy address
        rollouts: List[Rollout] = []
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
        completed_rollouts: List[Rollout] = []

        while True:
            completed_rollouts = await store.wait_for_rollouts(
                rollout_ids=[rollout.rollout_id for rollout in rollouts],
                timeout=0.0,  # Timeout must be a very small value to avoid blocking the store server
            )
            if len(completed_rollouts) >= len(rollouts):
                console.print(f"[bold red][Algo][/bold red] Received all {len(rollouts)} rollouts")
                break
            console.print(
                f"[bold red][Algo][/bold red] Received {len(completed_rollouts)} rollouts, waiting for more..."
            )
            await asyncio.sleep(5.0)

    # LLM server can be shutdown now as we perform the training

    # 2. Prepare the dataset for SFT
    all_triplets: List[HuggingFaceDatasetRecord] = []
    for rollout in completed_rollouts:
        spans = await store.query_spans(rollout.rollout_id, "latest")
        # Use data_adapter to adapt the spans to triplets. Triplets are a list of Pydantic models:
        # Triplet(
        #     prompt={
        #         "token_ids": [1, 2, 3],
        #     },
        #     response={
        #         "token_ids": [4, 5, 6],
        #     },
        #     reward=0.5,
        # )
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
        # Reverse the triplets so that the later rewards can propagate to the earlier triplets
        recent_reward: Optional[float] = None
        for triplet in reversed(triplets):
            # Ensure that prompt and response are all not empty
            if triplet.prompt.get("token_ids") and triplet.response.get("token_ids"):
                if triplet.reward is not None:
                    recent_reward = triplet.reward

                if recent_reward is None:
                    console.print(
                        f"[bold red][Algo][/bold red] Recent reward is None for triplet {triplet}. "
                        "Skip adding to SFT training data."
                    )
                    continue

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
                        "reward": recent_reward,
                    }
                )
            else:
                console.print(
                    f"[bold red][Algo][/bold red] Skip triplet because it has no prompt or response: {triplet}"
                )

    # IMPORTANT: Shuffle the triplets and rank them by reward
    if len(all_triplets) == 0:
        raise ValueError("No triplets to train on.")
    random.shuffle(all_triplets)
    all_triplets.sort(key=lambda x: x["reward"], reverse=True)
    sliced_triplets = all_triplets[: max(1, int(len(all_triplets) * triplet_fraction))]
    console.print(
        f"[bold red][Algo][/bold red] Generated {len(all_triplets)} triplets for SFT training. "
        f"Keeping {len(sliced_triplets)} with top rewards."
    )
    # Shuffle the sliced triplets again
    random.shuffle(sliced_triplets)

    sft_dataset = HuggingFaceDataset.from_list(sliced_triplets)  # type: ignore

    console.print(f"[bold red][Algo][/bold red] SFT dataset has {len(sft_dataset)} samples")

    # 3. Start the SFT training and save the model
    next_model_path = f"models/version_{iteration + 1}"

    context = multiprocessing.get_context("spawn")  # This has to be spawn, otherwise torch.cuda won't be initialized
    unsloth_process = context.Process(
        target=unsloth_training, args=(model_path, sft_dataset, next_model_path), daemon=True
    )
    unsloth_process.start()
    unsloth_process.join(timeout=600.0)
    if unsloth_process.is_alive():
        console.print(f"[bold red][Algo][/bold red] Unsloth training process hung. Terminating...")
        unsloth_process.terminate()
        unsloth_process.join(timeout=10.0)
        if unsloth_process.is_alive():
            console.print(
                f"[bold red][Algo][/bold red] Unsloth training process still alive after termination. Killing..."
            )
            unsloth_process.kill()
        raise RuntimeError("Unsloth training process did not finish in 600 seconds.")

    console.print(f"[bold red][Algo][/bold red] Saved model to {next_model_path}")

    return next_model_path


async def sft_algorithm(*, store: LightningStore) -> None:
    """Run the complete SFT algorithm with multiple iterations.

    This is the main entry point for running the SFT training pipeline. It sets up
    the LLM proxy, data adapter, and runs multiple iterations of model training.

    The function performs these steps for each iteration:
    1. Serves the current model via vLLM
    2. Collects rollout data using the model
    3. Converts trace data to training triplets
    4. Trains the model on top-performing examples
    5. Saves the improved model for the next iteration

    Args:
        store: The LightningStore instance for managing rollouts and trace data.
    """
    train_dataset = load_math_dataset()

    # Constants for the SFT algorithm
    MAX_ITERATIONS = 2
    VLLM_PORT = 12316
    LLM_PROXY_PORT = 12358
    TRAIN_TRIPLET_FRACTION = 0.5

    # Download the model before starting the script:
    # hf download unsloth/Qwen3-4B-Instruct-2507 --local-dir models/version_0
    model_path = "models/version_0"

    # Create the LLM proxy for rollout worker access and trace data collection
    llm_proxy = LLMProxy(port=LLM_PROXY_PORT, store=store)

    # This data adapter util is used to convert the trace data recorded by LLM proxy
    # into a format suitable for SFT
    data_adapter = LlmProxyTraceToTriplet()

    for iteration in range(MAX_ITERATIONS):
        model_path = await sft_one_iter(
            iteration=iteration,
            store=store,
            model_path=model_path,
            train_dataset=train_dataset,
            llm_proxy=llm_proxy,
            data_adapter=data_adapter,
            triplet_fraction=TRAIN_TRIPLET_FRACTION,
            vllm_port=VLLM_PORT,
        )

    console.print(f"[bold red][Algo][/bold red] Final model path: {model_path}")


if __name__ == "__main__":
    setup_logging()

    store = LightningStoreClient("http://localhost:4747")

    asyncio.run(sft_algorithm(store=store))

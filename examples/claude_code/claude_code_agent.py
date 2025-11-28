# Copyright (c) Microsoft. All rights reserved.

"""Main module for the Claude Code Agent implementation.

This module provides the core functionality for running Claude Code agent experiments
on SWE-bench datasets. It includes the ClaudeCodeAgent class that implements the agent logic,
functions for loading datasets, and asynchronous execution functions for running experiments.

Key components:

- ClaudeCodeAgent: Main agent implementation that handles rollout logic
- Dataset loading utilities
- Asynchronous execution functions for dry runs and full datasets
"""

import asyncio
import json
import logging
import os
import resource
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional, cast

from claude_code_controller import ClaudeController
from datasets import Dataset
from extended_adapter import ExtendedLlmProxyTraceToTriplet
from swebench.harness.utils import load_swebench_dataset  # pyright: ignore[reportUnknownVariableType]
from swebench_utils.evaluation import evaluate
from swebench_utils.logging import log_for_evaluation
from transformers import AutoTokenizer

from agentlightning import (
    InMemoryLightningStore,
    LightningStoreServer,
    LitAgentRunner,
    OtelTracer,
    configure_logger,
    setup_logging,
)
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.types import AttemptedRollout, NamedResources, ProxyLLM, Rollout, RolloutRawResult, Span

logger = logging.getLogger("claude_code_agent")


def load_dataset(path: str = "swe_debug.jsonl", epoch: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            instance = json.loads(line)
            instance["epoch"] = epoch
            instances.append(instance)

    if limit is not None:
        instances = instances[:limit]
    return instances


class ClaudeCodeAgent(LitAgent[Dict[str, Any]]):
    """Claude Code Agent implementation.

    This agent is a wrapper of the Claude Code controller. It is used to run the Claude Code agent on SWE-bench datasets.
    """

    def __init__(
        self,
        namespace: Literal["swebench", "starryzhang"] = "swebench",
        full_set: Literal["princeton-nlp/SWE-bench", "SWE-bench-Live/SWE-bench-Live"] = "princeton-nlp/SWE-bench",
        split: str = "test",
        max_step: int = 5,
        run_method: Literal["python", "cli"] = "cli",
        open_file_limit: int = 4096,
        cache_level: str = "env",  # ["none", "base", "env", "instance"]
        clean: bool = False,
        force_rebuild: bool = False,
        timeout: int = 1_800,  # in sec
        instance_image_tag: str = "latest",
        rewrite_reports: bool = False,
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.full_set = full_set
        self.split = split
        self.max_step = max_step
        self.run_method = run_method

        self.cache_level = cache_level
        self.clean = clean
        self.force_rebuild = force_rebuild
        self.timeout = timeout
        self.instance_image_tag = instance_image_tag
        self.rewrite_reports = rewrite_reports

        full_dataset = load_swebench_dataset(full_set, split)
        self.dataset = {each["instance_id"]: each for each in full_dataset}

        # Set the maximum number of open files to the specified limit.
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    async def rollout_async(
        self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout
    ) -> RolloutRawResult:
        if not isinstance(rollout, AttemptedRollout):
            # Technically, rollout should be an AttemptedRollout here.
            # but the API is not stabilized yet.
            raise ValueError("Rollout is not an AttemptedRollout.")

        run_id = f"epoch_{task.get('epoch', 0)}"
        image = f"{self.namespace}/sweb.eval.x86_64.{task['instance_id'].lower()}".replace("__", "_1776_")

        llm = cast(ProxyLLM, resources["llm"])

        try:
            # 1. init container
            controller = ClaudeController(
                image,
                task,
                run_id,
                llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                llm.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
            )
            # 2. execute task
            prediction = controller.run_instance(
                task, max_step=self.max_step, run_method=cast(Literal["python", "cli"], self.run_method)
            )
            del controller
        except Exception as e:
            log_for_evaluation(run_id, task["instance_id"], f"Exception during rollout: {e}")
            return 0.0

        # 3. obtain rewards (evaluation result)
        reward = 0.0
        # empty patch
        if prediction["model_patch"] in ["", None]:
            return reward

        instance_id = prediction["instance_id"]

        result = evaluate(
            prediction,
            self.dataset[instance_id],
            self.cache_level,
            self.clean,
            self.force_rebuild,
            run_id,
            self.timeout,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            rewrite_reports=self.rewrite_reports,
        )

        # error patch
        if result is None:
            return reward

        report = result[1]
        # resolved/unresolved patch
        if report[instance_id]["resolved"]:
            reward = 1.0
        return reward


def flatten_messages(messages: List[Any]) -> List[Dict[str, str]]:
    flattened: List[Dict[str, str]] = []
    for msg in messages:
        if msg["role"] in ["system", "user"] and isinstance(msg["content"], list):
            msg_content: List[str] = []
            for content in msg["content"]:
                msg_content.append(content["text"])

            msg["content"] = "".join(msg_content)
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            # NOTE:
            # Tool calls are list of dict, though in most case only one tool call is made per call
            # We serialize it as json string here to avoid nested structure
            msg["tool_calls"] = json.dumps(msg["tool_calls"])

        for k in msg:
            assert isinstance(msg[k], str), f"\n>>> {msg}"
        flattened.append(msg)
    return flattened


async def cc_agent_dry_run_sample(
    model_path: str,
    server_address: str,
    dataset_path: str,
    sonnet_name: str,
    haiku_name: str,
    max_step: int,
    output_dir: Optional[str],
) -> None:
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_dataset(dataset_path, limit=4)
    # from_pretrained has partially unknown typing in some stubs; cast via Any to satisfy type-checkers.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore

    logging = configure_logger(name="Claude Code Agent")

    tracer = OtelTracer()
    runner = LitAgentRunner[Dict[str, Any]](tracer)
    adapter = ExtendedLlmProxyTraceToTriplet()
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=7654)
    llm_proxy = LLMProxy(port=12358, store=store, callbacks=["return_token_ids", "opentelemetry", "logprobs"])

    await store.start()

    llm_proxy.update_model_list(
        [
            ModelConfig(
                model_name=f"{sonnet_name}",
                litellm_params={
                    "model": f"hosted_vllm/{model_path}",
                    "api_base": server_address,
                },
            ),
            ModelConfig(
                model_name=f"{haiku_name}",
                litellm_params={
                    "model": f"hosted_vllm/{model_path}",
                    "api_base": server_address,
                },
            ),
        ]
    )
    await llm_proxy.restart()

    # Put the LLM proxy address into the store as an address
    await store.add_resources(
        {
            "llm": llm_proxy.as_resource(model="local"),
        }
    )

    with runner.run_context(agent=ClaudeCodeAgent(max_step=max_step), store=store):
        rollout = await runner.step(
            dataset[0],
        )

        spans = await store.query_spans(rollout.rollout_id)
        triplets = adapter.adapt(cast(List[Span], spans))
        logging.info(f"dump {len(spans)} spans, extract {len(triplets)} triplets")
        if output_dir is not None:
            with open(os.path.join(output_dir, f"stream_{dataset[0]['instance_id']}.json"), "w") as f:
                for span in spans:
                    f.write(json.dumps(span.model_dump()) + "\n")

            all_triplets: List[Dict[str, Any]] = []
            recent_reward: Optional[float] = None
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    recent_reward = triplet.reward

                prompt = tokenizer.decode(triplet.prompt["token_ids"])  # type: ignore
                all_triplets.append(
                    {
                        "repo": rollout.input["repo"],
                        "instance_id": rollout.input["instance_id"],
                        "turn": triplet.metadata["sequence_id"],
                        "prompt_ids": triplet.prompt["token_ids"],
                        "gold_completion_ids": triplet.response["token_ids"],
                        "logprobs": triplet.response["logprobs"],
                        "reward": recent_reward,
                        "prompt": prompt,
                        "messages": flatten_messages(triplet.metadata["messages"]),
                    }
                )

            ds = Dataset.from_list(all_triplets)  # type: ignore
            ds.save_to_disk(os.path.join(output_dir, f"dataset-{dataset[0]['instance_id']}"))  # type: ignore
            logging.info(f"Saved dataset with {len(ds)} samples to dataset-{dataset[0]['instance_id']}")


async def gold_cc_agent_run_dataset(
    sonnet_name: str,
    haiku_name: str,
    max_step: int,
    dataset_path: str,
    output_dir: Optional[str],
):
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_dataset(dataset_path)

    logging = configure_logger(name="Claude Code Agent")

    tracer = OtelTracer()
    runner = LitAgentRunner[Dict[str, Any]](tracer)
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=7654)
    llm_proxy = LLMProxy(port=12358, store=store)

    await store.start()

    llm_proxy.update_model_list(
        [
            ModelConfig(
                model_name=f"{sonnet_name}",
                litellm_params={"model": f"anthropic/{sonnet_name}", "api_key": "os.environ/ANTHROPIC_API_KEY"},
            ),
            ModelConfig(
                model_name=f"{haiku_name}",
                litellm_params={"model": f"anthropic/{haiku_name}", "api_key": "os.environ/ANTHROPIC_API_KEY"},
            ),
        ]
    )
    await llm_proxy.restart()

    # Put the LLM proxy address into the store as an address
    await store.add_resources(
        {
            "llm": llm_proxy.as_resource(model="local"),
        }
    )

    for each in dataset:
        with runner.run_context(agent=ClaudeCodeAgent(max_step=max_step), store=store):
            rollout = await runner.step(each)
            spans = await store.query_spans(rollout.rollout_id)

        if output_dir is None:
            logging.info(f"instance {each['instance_id']} generate {len(spans)} spans")
        else:
            logging.info(f"instance {each['instance_id']} dump {len(spans)} spans to {output_dir}")
            with open(os.path.join(output_dir, f"{each['instance_id']}.json"), "w") as f:
                for span in spans:
                    f.write(json.dumps(span.model_dump()) + "\n")

        await asyncio.sleep(2 * 60)  # sleep for 2 minutes


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # extract spans from official Claude Code
    parser.add_argument("--official", action="store_true", help="Whether to run official claude code.")

    # extract spans from hosted LLM server via litellm proxy
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Model name or path."
    )
    parser.add_argument("--server_address", type=str, default="http://localhost:8000/v1", help="LLM server address.")

    # common setup
    parser.add_argument(
        "--sonnet_name", type=str, default="claude-sonnet-4-5-20250929", help="Name of the sonnet model."
    )
    parser.add_argument("--haiku_name", type=str, default="claude-haiku-4-5-20251001", help="Name of the haiku model.")
    parser.add_argument("--dataset_path", type=str, default="swe_debug.jsonl", help="Path to the dataset.")
    parser.add_argument("--max_step", type=int, default=5, help="Maximum steps per instance.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output logs.")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(apply_to=[logger.name])

    if not args.official:
        asyncio.run(
            cc_agent_dry_run_sample(
                model_path=args.model_name_or_path,
                server_address=args.server_address,
                dataset_path=args.dataset_path,
                sonnet_name=args.sonnet_name,
                haiku_name=args.haiku_name,
                max_step=args.max_step,
                output_dir=args.output_dir,
            )
        )
    else:
        asyncio.run(
            gold_cc_agent_run_dataset(
                sonnet_name=args.sonnet_name,
                haiku_name=args.haiku_name,
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                max_step=args.max_step,
            )
        )

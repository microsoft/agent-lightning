# Copyright (c) Microsoft. All rights reserved.

"""Instrumented driver for running Claude Code on SWE-bench with Agent-lightning.

This script wires together the Lightning Store, LLM proxy, and Claude Code controller so
that every SWE-bench instance is executed inside the official Claude container while
capturing full Agent-lightning traces. It supports three backend modes:

- `vllm`: wrap an OpenAI-compatible endpoint (e.g., vLLM) for hosted OSS models while
  collecting prompt/response token ids and logprobs.
- `anthropic`: call the official Claude Code API via `ANTHROPIC_API_KEY` for prompt
  tuning. Backend model defaults to the provided frontend names.
- `openai`: route through any OpenAI-compatible provider using `OPENAI_API_KEY`.

Typical usage: hosted vLLM (requires model paths and --base-url)

```bash
# Run vLLM in background
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 45993 &

python claude_code_agent.py vllm \
  --backend-model-high Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --backend-model-low Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --base-url http://localhost:45993/v1 \
  --dataset-path swebench_samples.jsonl \
```

Official Claude Code via Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-...
python claude_code_agent.py anthropic \
  --dataset-path swebench_samples.jsonl \
  --output-dir data_anthropic
```

Any OpenAI-compatible backend:

```bash
export OPENAI_API_KEY=sk-...
python claude_code_agent.py openai \
  --backend-model-high gpt-5.1-codex-mini \
  --backend-model-low gpt-4.1-mini \
  --dataset-path swebench_samples.jsonl
```

Use `--debug` to enable debug loggings.
"""

import asyncio
import json
import logging
import os
import resource
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional, Sequence, cast

from claude_code_controller import ClaudeController
from datasets import Dataset
from extended_adapter import ExtendedLlmProxyTraceToTriplet
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.utils import load_swebench_dataset  # pyright: ignore[reportUnknownVariableType]
from swebench_utils.evaluation import evaluate
from swebench_utils.logging import log_for_evaluation
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agentlightning import (
    InMemoryLightningStore,
    LightningStoreServer,
    LitAgentRunner,
    OtelTracer,
    setup_logging,
    setup_module_logging,
)
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store import LightningStore
from agentlightning.types import AttemptedRollout, NamedResources, ProxyLLM, Rollout, RolloutRawResult, Span

logger = logging.getLogger("claude_code_agent")


def _load_dataset(path: str, epoch: int = 0, limit: Optional[int] = None) -> List[SWEbenchInstance]:
    instances: List[SWEbenchInstance] = []
    with open(path) as f:
        for line in f:
            instance = json.loads(line)
            instance["epoch"] = epoch
            instances.append(instance)

    if limit is not None:
        instances = instances[:limit]
    return instances


def _flatten_messages(messages: List[Any]) -> List[Dict[str, str]]:
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


class ClaudeCodeAgent(LitAgent[SWEbenchInstance]):
    """Claude Code Agent implementation.

    This agent is a wrapper of the Claude Code controller,
    and it should be used to run the Claude Code agent on SWE-bench datasets.
    """

    def __init__(
        self,
        namespace: Literal["swebench", "starryzhang"] = "swebench",
        max_turns: int = 5,
        run_method: Literal["python", "cli"] = "cli",
        open_file_limit: int = 4096,
        cache_level: str = "env",  # ["none", "base", "env", "instance"]
        clean: bool = False,
        force_rebuild: bool = False,
        timeout: int = 1_800,  # in sec
        instance_image_tag: str = "latest",
        rewrite_reports: bool = False,
        swebench_full_dataset: Optional[List[SWEbenchInstance]] = None,
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.max_turns = max_turns
        self.run_method = run_method

        self.cache_level = cache_level
        self.clean = clean
        self.force_rebuild = force_rebuild
        self.timeout = timeout
        self.instance_image_tag = instance_image_tag
        self.rewrite_reports = rewrite_reports

        self.swebench_full_dataset = (
            {each["instance_id"]: each for each in swebench_full_dataset} if swebench_full_dataset is not None else {}
        )

        # Set the maximum number of open files to the specified limit.
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    async def rollout_async(
        self, task: SWEbenchInstance, resources: NamedResources, rollout: Rollout
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
                task, max_turns=self.max_turns, run_method=cast(Literal["python", "cli"], self.run_method)
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
            cast(Any, prediction),
            self.swebench_full_dataset[instance_id],
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


def sanity_check_spans(spans: Sequence[Span]) -> None:
    assert len(spans) > 1, f"At least two spans are expected for a valid rollout. Found {len(spans)} spans."
    assert any(span.name == "raw_gen_ai_request" for span in spans), "raw_gen_ai_request span not found"
    assert any(span.name == "agentlightning.annotation" for span in spans), "agentlightning.annotation span not found"


async def run_instance_async(
    instance: SWEbenchInstance,
    agent: ClaudeCodeAgent,
    runner: LitAgentRunner[SWEbenchInstance],
    store: LightningStore,
    output_dir: Optional[str],
    adapter: Optional[ExtendedLlmProxyTraceToTriplet],
    tokenizer: Optional[PreTrainedTokenizerBase],
) -> None:
    """Runs the agent on a specific SWE-bench instance.

    Running on specific SWE-bench instance and queries the traced spans.
    It then extracts the triplets and saves the dataset.
    """

    instance_id = instance["instance_id"]
    logger.info(f"Starting to run instance: {instance_id}")

    # Run the agent and query the traced spans.
    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(instance)

    logger.info(f"Finished running instance: {instance_id}")

    spans = await store.query_spans(rollout.rollout_id)

    if output_dir is None:
        logger.info(f"Generated {len(spans)} spans for {instance_id}")
        return

    # 1. Dump raw spans (Common for both types)
    raw_path = os.path.join(output_dir, f"stream_{instance_id}.json")
    with open(raw_path, "w") as f:
        for span in spans:
            f.write(json.dumps(span.model_dump()) + "\n")
    logger.info(f"Dumped {len(spans)} spans to {raw_path}")

    # 2. Extract Triplets and Save Dataset (vLLM specific)
    if adapter is not None and tokenizer is not None:
        try:
            triplets = adapter.adapt(cast(List[Span], spans))
            logger.info(f"Extracted {len(triplets)} triplets for {instance_id}")

            all_triplets: List[Dict[str, Any]] = []
            recent_reward: Optional[float] = None

            # Process in reverse to propagate rewards if necessary/logic dictates
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    recent_reward = triplet.reward

                prompt_text = tokenizer.decode(triplet.prompt["token_ids"])  # type: ignore
                all_triplets.append(
                    {
                        "repo": instance.get("repo", ""),
                        "instance_id": instance_id,
                        "turn": triplet.metadata["sequence_id"],
                        "prompt_ids": triplet.prompt["token_ids"],
                        "gold_completion_ids": triplet.response["token_ids"],
                        "logprobs": triplet.response["logprobs"],
                        "reward": recent_reward,
                        "prompt": prompt_text,
                        "messages": _flatten_messages(triplet.metadata["messages"]),
                    }
                )

            if all_triplets:
                ds = Dataset.from_list(all_triplets)  # type: ignore
                save_path = os.path.join(output_dir, f"dataset-{instance_id}")
                ds.save_to_disk(save_path)  # type: ignore
                logger.info(f"Saved HuggingFace dataset to {save_path}")

        except Exception as e:
            logger.error(f"Failed to extract triplets for {instance_id}: {e}")

    logger.info(f"Finished extracting spans and traces for instance: {instance_id}")

    # Quickly sanity check the spans
    sanity_check_spans(spans)
    logger.info(f"Sanity check passed for instance: {instance_id}")


async def dry_run_claude_code(
    *,
    dataset_path: str,
    haiku_frontend_name: str,
    haiku_backend_name: str,
    sonnet_frontend_name: str,
    sonnet_backend_name: str,
    backend_type: Literal["vllm", "anthropic", "openai"],
    api_base_url: Optional[str],
    output_dir: Optional[str],
    max_turns: int,
    limit: Optional[int],
    cooldown_seconds: float,
) -> None:
    """Executes a dry run of the Claude Code agent on a dataset.

    This function handles both 'official' runs (interacting with Anthropic APIs)
    and 'hosted' runs (interacting with vLLM or compatible servers). It manages
    initialization of the Lightning Store, LLM Proxy, and the execution loop.

    If running in 'vllm' mode, it will also attempt to extract triplets using
    the provided backend name as the tokenizer path and save a HuggingFace Dataset.

    Args:
        dataset_path: Path to the JSONL dataset file.
        haiku_frontend_name: The model name used in the code to request the 'fast' model.
        haiku_backend_name: The actual model name/path on the backend.
        sonnet_frontend_name: The model name used in the code to request the 'strong' model.
        sonnet_backend_name: The actual model name/path on the backend.
        backend_type: The type of backend to configure ("vllm", "anthropic" or "openai").
        api_base_url: Base URL for the API. Required for "vllm" or "openai".
        output_dir: Directory to save logs, spans, and datasets.
        max_turns: Maximum number of steps the agent can take per instance.
        limit: Optional limit on the number of instances to process.
    """
    dataset = _load_dataset(dataset_path, limit=limit)

    # Initialize Infrastructure
    tracer = OtelTracer()
    runner = LitAgentRunner[SWEbenchInstance](tracer)
    store = LightningStoreServer(InMemoryLightningStore(), host="0.0.0.0", port=7654)
    await store.start()

    # Enable callbacks for training data extraction if using vLLM
    callbacks = ["return_token_ids", "opentelemetry", "logprobs"] if backend_type == "vllm" else ["opentelemetry"]
    llm_proxy = LLMProxy(port=12358, store=store, callbacks=callbacks)

    # Configure Models based on backend type
    model_configs: List[ModelConfig] = []
    model_params: Dict[str, Any] = {}

    if backend_type == "vllm":
        model_namespace = "hosted_vllm"
        if api_base_url:
            model_params["api_base"] = api_base_url
        else:
            raise ValueError("api_base_url is required for vllm backend")
    elif backend_type == "anthropic":
        model_namespace = "anthropic"
        model_params["api_key"] = "os.environ/ANTHROPIC_API_KEY"
        if api_base_url:
            model_params["api_base"] = api_base_url
    elif backend_type == "openai":
        model_namespace = "openai"
        model_params["api_key"] = "os.environ/OPENAI_API_KEY"
        if api_base_url:
            # Users can still override this via environment variables,
            # even if they don't pass it in as an argument.
            model_params["api_base"] = api_base_url

    model_configs.extend(
        [
            ModelConfig(
                model_name=sonnet_frontend_name,
                litellm_params={
                    "model": f"{model_namespace}/{sonnet_backend_name}",
                    **model_params,
                },
            ),
            ModelConfig(
                model_name=haiku_frontend_name,
                litellm_params={
                    "model": f"{model_namespace}/{haiku_backend_name}",
                    **model_params,
                },
            ),
        ]
    )

    logger.info(f"Updating model list: {model_configs}")

    llm_proxy.update_model_list(model_configs)
    await llm_proxy.start()

    try:
        # Add the LLM proxy as a resource to the store
        await store.add_resources({"llm": llm_proxy.as_resource(model="local")})

        # Prepare for triplet extraction if vllm
        adapter = ExtendedLlmProxyTraceToTriplet() if backend_type == "vllm" else None
        tokenizer = None
        if backend_type == "vllm":
            try:
                tokenizer = AutoTokenizer.from_pretrained(sonnet_backend_name)  # type: ignore
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {sonnet_backend_name}: {e}")

        # Load full swebench dataset. Mainly for evaluation purposes.
        swebench_full_dataset = load_swebench_dataset("princeton-nlp/SWE-bench", split="test")

        # Initialize Claude Code Agent
        claude_code_agent = ClaudeCodeAgent(swebench_full_dataset=swebench_full_dataset, max_turns=max_turns)

        # Execution Loop
        for instance in dataset:
            await run_instance_async(
                instance,
                claude_code_agent,
                runner,
                store,
                output_dir,
                adapter,
                cast(PreTrainedTokenizerBase, tokenizer),
            )
            # Basic sleep to allow resource cleanup or rate limit cooling
            await asyncio.sleep(cooldown_seconds)

    finally:
        await llm_proxy.stop()
        await store.stop()


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Claude Code Agent experiments.")

    # Backend Selection
    parser.add_argument(
        "backend_type",
        type=str,
        choices=["vllm", "anthropic", "openai"],
        help="Backend type: 'vllm' for hosted models, 'anthropic' for official API, 'openai' for OpenAI API.",
    )

    # Model Configuration
    parser.add_argument(
        "--backend-model-high",
        type=str,
        default=None,
        help="Backend model path/name for expensive model usages (used as vLLM model name / OpenAI model name).",
    )
    parser.add_argument(
        "--backend-model-low",
        type=str,
        default=None,
        help="Backend model path/name for low-price model usages (used as vLLM model name / OpenAI model name).",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000/v1", help="LLM server address (required for vllm)."
    )

    # Frontend/Agent Configuration
    parser.add_argument(
        "--frontend-model-high",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="The frontend high-price model name provided to Claude Code.",
    )
    parser.add_argument(
        "--frontend-model-low",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="The frontend low-price model name provided to Claude Code.",
    )

    # Execution Configuration
    parser.add_argument("--dataset-path", type=str, default="swebench_samples.jsonl", help="Path to the dataset.")
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum turns per instance.")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save output logs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of instances to run (for debugging).")
    parser.add_argument("--cooldown-seconds", type=float, default=2.0, help="Cooldown seconds between instances.")
    parser.add_argument("--debug", action="store_true", help="Enable debug loggings.")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.debug:
        setup_logging()
        setup_module_logging("DEBUG", name="claude_code_agent")
    else:
        setup_logging(apply_to=[logger.name])

    # Map backend_type to the appropriate args
    backend_mode = cast(Literal["vllm", "anthropic", "openai"], args.backend_type)

    # If using anthropic, the backend name usually matches the frontend or is specific API string.
    # Otherwise, the backend name is the model name/path (e.g., Qwen/...) and must be provided.
    if args.backend_model_high is None:
        if args.backend_type == "anthropic":
            backend_model_high = args.frontend_model_high
        else:
            raise ValueError("--backend-model-high is required for non-anthropic backends")
    else:
        backend_model_high = args.backend_model_high

    if args.backend_model_low is None:
        if args.backend_type == "anthropic":
            backend_model_low = args.frontend_model_low
        else:
            raise ValueError("--backend-model-low is required for non-anthropic backends")
    else:
        backend_model_low = args.backend_model_low

    asyncio.run(
        dry_run_claude_code(
            dataset_path=args.dataset_path,
            haiku_frontend_name=args.frontend_model_low,
            haiku_backend_name=backend_model_low,
            sonnet_frontend_name=args.frontend_model_high,
            sonnet_backend_name=backend_model_high,
            backend_type=backend_mode,
            api_base_url=args.base_url if backend_mode == "vllm" else None,
            output_dir=args.output_dir,
            max_turns=args.max_turns,
            limit=args.limit,
            cooldown_seconds=args.cooldown_seconds,
        )
    )

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import sys
from ast import literal_eval
from typing import Any, Dict, List, Sequence, Type, Union, cast

sys.path.append("examples/claude_code")

import anthropic
import openai
import pytest
from litellm.integrations.custom_logger import CustomLogger
from portpicker import pick_unused_port
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.utils import load_swebench_dataset  # pyright: ignore[reportUnknownVariableType]
from transformers import AutoTokenizer

from agentlightning import LitAgentRunner, OtelTracer
from agentlightning.llm_proxy import LLMProxy, _reset_litellm_logging_worker  # pyright: ignore[reportPrivateUsage]
from agentlightning.store import LightningStore, LightningStoreServer, LightningStoreThreaded
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import LLM, Span
from examples.claude_code.claude_code_agent import ClaudeCodeAgent, _load_dataset

from ..common.tracer import clear_tracer_provider

pytest.skip(reason="Debug only", allow_module_level=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "otlp_enabled",
    [
        True,
    ],
)
async def test_claude_code(otlp_enabled: bool):
    # For unknown reasons, I don't have local machine for debugging,
    # this model is deployed remotely, so as the whole unit test.
    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    endpoint = "http://localhost:8000/v1"
    max_turns = 5

    clear_tracer_provider()
    _reset_litellm_logging_worker()  # type: ignore

    # Prepare utilities for testing
    dataset_path = "examples/claude_code/swebench_samples.jsonl"
    instance = _load_dataset(dataset_path, limit=1)[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load full swebench dataset. Mainly for evaluation purposes.
    swebench_full_dataset = load_swebench_dataset("princeton-nlp/SWE-bench", split="test")
    # Initialize Claude Code Agent
    claude_code_agent = ClaudeCodeAgent(swebench_full_dataset=swebench_full_dataset, max_turns=max_turns)

    system_prompt_piece = "Please do not commit your edits. We will do it later."

    # Initialize agl Infrastructure
    inmemory_store = InMemoryLightningStore()
    if otlp_enabled:
        store = LightningStoreServer(store=inmemory_store, host="127.0.0.1", port=pick_unused_port())
        await store.start()
    else:
        store = LightningStoreThreaded(inmemory_store)

    proxy = LLMProxy(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "hosted_vllm/" + model_name,
                    "api_base": endpoint,
                },
            },
            {
                "model_name": "claude-haiku-4-5-20251001",
                "litellm_params": {
                    "model": "hosted_vllm/" + model_name,
                    "api_base": endpoint,
                },
            },
        ],
        launch_mode="thread" if not otlp_enabled else "mp",
        port=pick_unused_port(),
        store=store,
    )
    proxy.server_launcher._access_host = "localhost"
    await proxy.start()

    rollout = await store.start_rollout(None)

    resource = proxy.as_resource(rollout.rollout_id, rollout.attempt.attempt_id, model="local")
    print(f">>> DEBUG: {proxy.server_launcher.access_endpoint=}")
    print(f">>> DEBUG: {resource.endpoint=}, {resource.model=}")

    # Dry run to generate spans
    await claude_code_agent.rollout_async(
        task=instance,
        resources={"llm": resource},
        rollout=rollout,
    )

    spans = await store.query_spans(rollout.rollout_id)

    # Preprocess raw spans
    valid_spans = []
    for span in spans:
        if span.name != "raw_gen_ai_request":
            continue

        prompt_ids = span.attributes["llm.hosted_vllm.prompt_token_ids"]
        prompt_text = tokenizer.decode(literal_eval(prompt_ids))
        if system_prompt_piece not in prompt_text:
            continue

        choice = literal_eval(span.attributes["llm.hosted_vllm.choices"])[0]
        response_ids = choice["token_ids"]
        response_text = tokenizer.decode(response_ids)

        prompt_messages = literal_eval(span.attributes["llm.hosted_vllm.messages"])
        response_message = choice["message"]

        valid_spans.append(
            {
                "prompt_text": prompt_text,
                "response_text": response_text,
                "prompt_messages": prompt_messages,
                "response_message": response_message,
            }
        )

    with open("logs/test_spans.jsonl", "w") as f:
        for span in spans:
            f.write(json.dumps(span.model_dump(), indent=2) + "\n")

    with open("logs/test_valid_spans.jsonl", "w") as f:
        for span in valid_spans:
            f.write(json.dumps(span, indent=2) + "\n")

    # Test case 1: At least two valid spans
    assert len(valid_spans) > 1
    print(f"Generated {len(spans)} spans with {len(valid_spans)} LLM requests.")

    # Test case 2:
    for i in range(1, len(valid_spans)):
        prev = valid_spans[i - 1]
        curr = valid_spans[i]

        # The current prompt should contain the previous response
        assert prev["response_text"] in curr["prompt_text"]

    await proxy.stop()
    if isinstance(store, LightningStoreServer):
        await store.stop()

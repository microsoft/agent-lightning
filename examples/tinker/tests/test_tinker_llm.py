# Copyright (c) Microsoft. All rights reserved.

"""This file includes some basic tests for the integration of Tinker's sampling client
with LiteLLM and Agent-lightning.

It should be included in CI in future if we decided to maintain this example.
"""

import asyncio
from typing import cast

import openai
import tinker
from agl_tinker.llm import TinkerLLM
from agl_tinker.rollout import reconstruct_transitions
from rich.console import Console
from tinker_cookbook.renderers import Qwen3Renderer
from transformers import AutoTokenizer, PreTrainedTokenizer

from agentlightning import (
    AgentOpsTracer,
    InMemoryLightningStore,
    LLMProxy,
    LlmProxyTraceToTriplet,
    TracerTraceToTriplet,
    emit_reward,
    setup_logging,
)
from agentlightning.store import LightningStoreThreaded

setup_logging(apply_to=["agl_tinker"])


async def test_tracer():
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3Renderer(tokenizer)  # type: ignore
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tinker_llm = TinkerLLM(
        model_name=model_name, renderer=renderer, tokenizer=tokenizer, sampling_client=sampling_client, max_tokens=20
    )
    tinker_llm.rewrite_litellm_custom_providers()

    store = LightningStoreThreaded(InMemoryLightningStore())
    rollout = await store.start_rollout("dummy", "train")
    llm_proxy = LLMProxy(
        port=4000,
        store=store,
        model_list=tinker_llm.as_model_list(),
        num_retries=0,
        launch_mode="thread",
    )

    try:
        tracer = AgentOpsTracer()
        tracer.init()
        tracer.init_worker(worker_id=0, store=store)

        # init tracer before llm_proxy to avoid tracer provider being not active.
        console.print("Starting LLM proxy...")
        await llm_proxy.start()
        console.print("LLM proxy started")

        # client = openai.OpenAI(
        #     base_url=f"http://localhost:4000/rollout/{rollout.rollout_id}/attempt/{rollout.attempt.attempt_id}",
        #     api_key="dummy",
        # )
        client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")

        async with tracer.trace_context(
            name="test_llm", rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello world!"}],
                max_tokens=10,
                temperature=0.5,
                top_p=0.9,
                seed=43,
            )
            print(response)
            emit_reward(8.0)

        print(f"Found {len(tracer.get_last_trace())} spans in the tracer")

        tracer.teardown_worker(0)
        tracer.teardown()

        for store_span in await store.query_spans(rollout.rollout_id):
            print(store_span)

        spans = await store.query_spans(rollout.rollout_id)
        console.print(f"Found {len(spans)} spans")
        adapter = TracerTraceToTriplet()
        trajectory = reconstruct_transitions(spans, adapter, rollout.rollout_id)
        print(trajectory)
        assert len(trajectory.transitions) > 0
        assert len(trajectory.transitions[0].ac.tokens) > 0
    finally:
        console.print("Stopping LLM proxy...")
        await llm_proxy.stop()
        console.print("LLM proxy stopped")


async def test_llm_proxy():
    # FIXME: The llm proxy adapter needs some fixes to make this test work
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3Renderer(tokenizer)  # type: ignore
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tinker_llm = TinkerLLM(
        model_name=model_name, renderer=renderer, tokenizer=tokenizer, sampling_client=sampling_client, max_tokens=20
    )
    tinker_llm.rewrite_litellm_custom_providers()

    store = LightningStoreThreaded(InMemoryLightningStore())
    rollout = await store.start_rollout("dummy", "train")
    llm_proxy = LLMProxy(
        port=4000,
        store=store,
        model_list=tinker_llm.as_model_list(),
        num_retries=0,
        launch_mode="thread",
    )

    try:
        # init tracer before llm_proxy to avoid tracer provider being not active.
        console.print("Starting LLM proxy...")
        await llm_proxy.start()
        console.print("LLM proxy started")

        client = openai.OpenAI(
            base_url=f"http://localhost:4000/rollout/{rollout.rollout_id}/attempt/{rollout.attempt.attempt_id}",
            api_key="dummy",
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello world!"}],
            max_tokens=10,
            temperature=0.5,
            top_p=0.9,
            seed=43,
        )
        print(response)

        for store_span in await store.query_spans(rollout.rollout_id):
            print(store_span)

        spans = await store.query_spans(rollout.rollout_id)
        console.print(f"Found {len(spans)} spans")
        adapter = LlmProxyTraceToTriplet()
        trajectory = reconstruct_transitions(spans, adapter, rollout.rollout_id)
        print(trajectory)
    finally:
        console.print("Stopping LLM proxy...")
        await llm_proxy.stop()
        console.print("LLM proxy stopped")


if __name__ == "__main__":
    asyncio.run(test_tracer())

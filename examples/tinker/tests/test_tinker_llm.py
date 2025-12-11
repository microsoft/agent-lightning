# Copyright (c) Microsoft. All rights reserved.

"""This file includes some basic tests for the integration of Tinker's sampling client
with LiteLLM and Agent-lightning.

It should be included in CI in future if we decided to maintain this example.
"""

import argparse
import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, cast

import openai
import tinker
from agl_tinker.llm import TinkerLLM
from agl_tinker.rollout import reconstruct_transitions
from rich.console import Console
from tinker_cookbook.renderers import Qwen3InstructRenderer
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

_tool_call_system_prompt = """
You must call the provided tool once before responding to the user.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "echo_text", "description": "Echo back any provided text.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to repeat back."}}, "required": ["text"]}}
</tools>

For each function call, return a json object with function name and args within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "args": <args-json-object>}
</tool_call>
"""


def _run_tool_call_roundtrip(client: openai.OpenAI, *, model_name: str) -> None:
    """Force a tool call, parse the args, and feed back the tool result."""
    prompt_messages: list[Dict[str, str]] = [
        # FIXME: Currently the tool call definition needs to be hard-coded into the system prompt.
        {"role": "system", "content": _tool_call_system_prompt},
        {"role": "user", "content": "Use the tool to echo 'Agent Lightning loves tool calls'."},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=cast(Any, prompt_messages),
        max_tokens=256,
        temperature=0.0,
        # tools=cast(Any, tools),
        # tool_choice="auto",
    )
    print("First response:", response)
    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        raise AssertionError("Model did not emit a tool call when forced to do so.")
    tool_call = tool_calls[0]
    if tool_call.type != "function" or tool_call.function is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise AssertionError("Unexpected tool call payload from model.")
    arguments = tool_call.function.arguments or "{}"
    tool_args = json.loads(arguments)
    tool_result = tool_args.get("text", "")
    followup_messages: list[Dict[str, Any]] = [
        *prompt_messages,
        {
            "role": "assistant",
            "content": "",  # FIXME: Content must be here to make validation happy.
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": f"Echoed text: {tool_result}",
        },
    ]
    followup_response = client.chat.completions.create(
        model=model_name,
        messages=cast(Any, followup_messages),
        max_tokens=64,
        temperature=0.5,
    )
    print("Followup response:", followup_response)


def _run_text_completion(client: openai.OpenAI, *, model_name: str) -> None:
    """Simple text-only completion to contrast the tool-call scenario."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello world!"}],
        max_tokens=20,
        temperature=0.5,
        top_p=0.9,
        seed=11,
    )
    print(response)


async def _run_tracer_test(*, use_tool_call: bool) -> None:
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3InstructRenderer(tokenizer)  # type: ignore
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

    scenario = "tool-call" if use_tool_call else "text-only"
    console.print(f"Running tracer test scenario: {scenario}")

    try:
        tracer = AgentOpsTracer()
        tracer.init()
        tracer.init_worker(worker_id=0, store=store)

        # init tracer before llm_proxy to avoid tracer provider being not active.
        console.print("Starting LLM proxy...")
        await llm_proxy.start()
        console.print("LLM proxy started")

        client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")

        async with tracer.trace_context(
            name=f"test_llm_{scenario}", rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            if use_tool_call:
                _run_tool_call_roundtrip(client, model_name=model_name)
            else:
                _run_text_completion(client, model_name=model_name)
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


async def test_tracer_text_only():
    await _run_tracer_test(use_tool_call=False)


async def test_tracer_tool_call():
    await _run_tracer_test(use_tool_call=True)


async def test_tracer():
    await test_tracer_tool_call()


async def test_llm_proxy():
    # FIXME: The llm proxy adapter needs some fixes to make this test work
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3InstructRenderer(tokenizer)  # type: ignore
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


CLI_VARIANTS: Dict[str, Callable[[], Awaitable[None]]] = {
    "tracer-tool": test_tracer_tool_call,
    "tracer-text": test_tracer_text_only,
    "llm-proxy": test_llm_proxy,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually run the async Tinker LLM integration tests.")
    parser.add_argument(
        "variant",
        choices=sorted(CLI_VARIANTS.keys()),
        help="Which async test to run.",
    )
    args = parser.parse_args()
    asyncio.run(CLI_VARIANTS[args.variant]())  # type: ignore

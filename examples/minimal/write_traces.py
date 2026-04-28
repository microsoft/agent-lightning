# Copyright (c) Microsoft. All rights reserved.

"""Example to write traces to a LightningStore via raw OpenTelemetry or AgentOpsTracer.

The example can be run with or without using a Lightning Store server.
When running this server, the traces will be written to the server via OTLP endpoint.

Prior to running this example with `--use-client` flag, please start a LightningStore server with OTLP enabled first:

```bash
agl store --port 45993 --log-level DEBUG
```

The CLI also ships an `operation` mode showing how to record a synthetic operation span with
[`operation`][agentlightning.operation], build link attributes via
[`make_link_attributes`][agentlightning.utils.otel.make_link_attributes], tag the
follow-up reward with [`make_tag_attributes`][agentlightning.utils.otel.make_tag_attributes],
emit a reward span tied back to that operation, and then verify the recorded spans by
extracting rewards, tags, and links from the store using `agentlightning.utils.otel` helpers.
"""

import argparse
import asyncio
import random
import time
from typing import Any, Dict, List, Sequence
from uuid import uuid4

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import AgentOpsTracer, LightningStoreClient, OtelTracer, Span, emit_reward, operation, setup_logging
from agentlightning.semconv import AGL_OPERATION, LightningSpanAttributes
from agentlightning.store import InMemoryLightningStore
from agentlightning.utils.otel import (
    extract_links_from_attributes,
    extract_tags_from_attributes,
    filter_and_unflatten_attributes,
    get_tracer_provider,
    make_link_attributes,
    make_tag_attributes,
    query_linked_spans,
)

console = Console()


async def send_traces_via_otel(use_client: bool = False):
    tracer = OtelTracer()
    if not use_client:
        store = InMemoryLightningStore()
    else:
        store = LightningStoreClient("http://localhost:45993")
    rollout = await store.start_rollout(input={"origin": "write_traces_example"})

    with tracer.lifespan(store):
        # Initialize the capture of one single trace for one single rollout
        async with tracer.trace_context(
            "trace-manual", store=store, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ) as tracer:
            with tracer.start_as_current_span("grpc-span-1"):
                time.sleep(0.01)

                # Nested Span
                with tracer.start_as_current_span("grpc-span-2"):
                    time.sleep(0.01)

            with tracer.start_as_current_span("grpc-span-3"):
                time.sleep(0.01)

            # This creates a reward span
            emit_reward(1.0)

    traces = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)

    # Quickly validate the traces
    assert len(traces) == 4
    span_names = [span.name for span in traces]
    assert "grpc-span-1" in span_names
    assert "grpc-span-2" in span_names
    assert "grpc-span-3" in span_names
    assert "agentlightning.annotation" in span_names

    last_span = traces[-1]
    assert last_span.name == "agentlightning.annotation"
    # NOTE: Try not to rely on this attribute like this example do. It may change in the future.
    # Use utils from agentlightning.emitter to get the reward value.
    assert last_span.attributes["agentlightning.reward.0.value"] == 1.0

    if use_client:
        # When using client, the resource should have rollout_id and attempt_id set
        for span in traces:
            assert "agentlightning.rollout_id" in span.resource.attributes
            assert "agentlightning.attempt_id" in span.resource.attributes

    if isinstance(store, LightningStoreClient):
        await store.close()


async def send_traces_via_agentops(use_client: bool = False):
    tracer = AgentOpsTracer()
    if not use_client:
        store = InMemoryLightningStore()
    else:
        store = LightningStoreClient("http://localhost:45993")
    rollout = await store.start_rollout(input={"origin": "write_traces_example"})

    # Initialize the tracer lifespan
    # One lifespan can contain multiple traces
    with tracer.lifespan(store):
        # Inspect current tracer provider
        get_tracer_provider(inspect=True)

        # Initialize the capture of one single trace for one single rollout
        async with tracer.trace_context(
            "trace-1", rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            openai_client = AsyncOpenAI()
            response = await openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, what's your name?"},
                ],
            )
            console.print(response)
            assert response.choices[0].message.content is not None
            assert "chatgpt" in response.choices[0].message.content.lower()

    traces = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)
    await _verify_agentops_traces(traces, use_client=use_client)
    if isinstance(store, LightningStoreClient):
        await store.close()


async def _verify_agentops_traces(spans: Sequence[Span], use_client: bool = False):
    """Expected traces to something like:

    ```python
    Span(
        rollout_id='ro-ef9ff8a429d1',
        attempt_id='at-37cc5f24',
        sequence_id=1,
        trace_id='b3a16b603f7805934215d467e717c9e7',
        span_id='2782d5d750f49b2d',
        parent_id='2fb97c818363bce3',
        name='openai.chat.completion',
        status=TraceStatus(status_code='OK', description=None),
        attributes={
            'gen_ai.request.type': 'chat',
            'gen_ai.system': 'OpenAI',
            'gen_ai.request.model': 'gpt-4.1-mini',
            'gen_ai.request.streaming': False,
            'gen_ai.prompt.0.role': 'system',
            'gen_ai.prompt.0.content': 'You are a helpful assistant.',
            'gen_ai.prompt.1.role': 'user',
            'gen_ai.prompt.1.content': "Hello, what's your name?",
            'gen_ai.response.id': 'chatcmpl-Cc1osPWiArOwCS8nUkp0kZuZPkpY4',
            'gen_ai.response.model': 'gpt-4.1-mini-2025-04-14',
            'gen_ai.completion.0.role': 'assistant',
            'gen_ai.completion.0.content': "Hello! I'm ChatGPT, your AI assistant. How can I help you today?",
        },
        resource=OtelResource(
            attributes={
                'agentops.project.id': 'temporary',
                'agentlightning.rollout_id': 'ro-ef9ff8a429d1',
                'agentlightning.attempt_id': 'at-37cc5f24'
            },
            schema_url=''
        )
    )
    ```
    """
    assert len(spans) == 2
    for span in spans:
        if span.name == "openai.chat.completion":
            assert span.attributes["gen_ai.request.model"] == "gpt-4.1-mini"
            assert span.attributes["gen_ai.request.streaming"] == False
            assert span.attributes["gen_ai.prompt.0.role"] == "system"
            assert span.attributes["gen_ai.prompt.0.content"] == "You are a helpful assistant."
            assert span.attributes["gen_ai.prompt.1.role"] == "user"
            assert span.attributes["gen_ai.prompt.1.content"] == "Hello, what's your name?"
            assert "chatgpt" in span.attributes["gen_ai.completion.0.content"].lower()  # type: ignore
            if use_client:
                assert "agentlightning.rollout_id" in span.resource.attributes
                assert "agentlightning.attempt_id" in span.resource.attributes
        else:
            assert "trace-1" in span.name
            assert span.attributes["agentops.span.kind"] == "session"


async def send_operation_links(use_client: bool = False) -> None:
    """Demonstrate operation spans wired to reward annotations and verify the stored spans."""

    tracer = OtelTracer()
    if not use_client:
        store = InMemoryLightningStore()
    else:
        store = LightningStoreClient("http://localhost:45993")
    conversation_id = "chat-42"
    tags: Sequence[str] = ("demo.operation", "reward.positive")
    reward_value = 0.9
    operation_id = f"{conversation_id}-{uuid4().hex[:8]}"
    rollout = await store.start_rollout(input={"origin": "write_traces_operation"})

    with tracer.lifespan(store):
        async with tracer.trace_context(
            "operation-demo", store=store, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            console.print(f"[operation] recording span conversation={conversation_id} operation_id={operation_id}")
            with operation(conversation_id=conversation_id, operation_id=operation_id) as op_ctx:
                op_ctx.set_input(
                    task={"conversation_id": conversation_id},
                    metadata={"operation_id": operation_id},
                )
                synthetic_payload = {
                    "operation_id": operation_id,
                    "status": "ok",
                    "latency_seconds": round(random.uniform(0.05, 0.2), 3),
                }
                await asyncio.sleep(0.05)
                op_ctx.set_output(synthetic_payload)

            link_attrs = make_link_attributes({"conversation_id": conversation_id, "operation_id": operation_id})
            tag_attrs = make_tag_attributes(list(tags))
            emit_reward(
                reward_value,
                attributes={**link_attrs, **tag_attrs},
            )

    spans = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(spans)
    _verify_operation_spans(spans, conversation_id, operation_id, tags, reward_value)

    if isinstance(store, LightningStoreClient):
        await store.close()


def _verify_operation_spans(
    spans: Sequence[Span],
    conversation_id: str,
    operation_id: str,
    tags: Sequence[str],
    expected_reward: float,
) -> None:
    """Verify spans recorded by the operation demo using OTEL helpers."""

    operation_spans = [span for span in spans if span.name == AGL_OPERATION]
    if not operation_spans:
        raise RuntimeError("No operation spans recorded.")
    console.print(f"[verify] found {len(operation_spans)} operation spans")

    reward_span: Span | None = None
    reward_payload: List[Dict[str, Any]] = []
    for span in spans:
        flattened = dict(span.attributes or {})
        reward_section = filter_and_unflatten_attributes(flattened, LightningSpanAttributes.REWARD.value)
        if reward_section:
            reward_span = span
            if isinstance(reward_section, list):
                reward_payload = [dict(item) for item in reward_section]  # type: ignore[arg-type]
            else:
                reward_payload = [dict(reward_section)]  # type: ignore[arg-type]
            break

    if reward_span is None or not reward_payload:
        raise RuntimeError("No reward span recorded for operation demo.")

    primary_reward = reward_payload[0].get("value")
    console.print(f"[verify] reward dimensions: {reward_payload}")
    if primary_reward != expected_reward:
        raise AssertionError(f"Expected reward {expected_reward}, observed {primary_reward}")

    reward_attributes = dict(reward_span.attributes or {})
    extracted_tags = extract_tags_from_attributes(reward_attributes)
    console.print(f"[verify] reward tags: {extracted_tags}")
    for tag in tags:
        if tag not in extracted_tags:
            raise AssertionError(f"Missing tag '{tag}' on reward span")

    link_models = extract_links_from_attributes(reward_attributes)
    matches = query_linked_spans(operation_spans, link_models)
    if not matches:
        raise AssertionError("No operation span matched the reward links")
    console.print(f"[verify] reward links resolved spans: {[span.span_id for span in matches]}")

    linked_attrs = dict(matches[0].attributes or {})
    if linked_attrs.get("conversation_id") != conversation_id or linked_attrs.get("operation_id") != operation_id:
        raise AssertionError("Linked operation span attributes do not match expected identifiers")
    console.print("[verify] linked operation span attributes validated")


def main():
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["otel", "agentops", "operation"])
    parser.add_argument("--use-client", action="store_true")
    args = parser.parse_args()

    if args.mode == "otel":
        asyncio.run(send_traces_via_otel(use_client=args.use_client))
    elif args.mode == "agentops":
        asyncio.run(send_traces_via_agentops(use_client=args.use_client))
    elif args.mode == "operation":
        asyncio.run(send_operation_links(use_client=args.use_client))
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

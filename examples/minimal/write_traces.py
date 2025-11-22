# Copyright (c) Microsoft. All rights reserved.

"""Example to write traces to a LightningStore via raw OpenTelemetry or AgentOpsTracer.

The example can be run with or without using a Lightning Store server.
When running this server, the traces will be written to the server via OTLP endpoint.

Prior to running this example with `--use-client` flag, please start a LightningStore server with OTLP enabled first:

```bash
agl store --port 45993 --log-level DEBUG
```
"""

import argparse
import asyncio
import time
from typing import Sequence

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import AgentOpsTracer, LightningStoreClient, OtelTracer, Span, emit_reward, setup_logging
from agentlightning.store import InMemoryLightningStore

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
    assert "agentlightning.reward" in span_names

    last_span = traces[-1]
    assert last_span.name == "agentlightning.reward"
    # NOTE: Try not to rely on this attribute. It may change in the future.
    # Use utils from agentlightning.emitter to get the reward value.
    assert last_span.attributes["reward"] == 1.0

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


def main():
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["otel", "agentops"])
    parser.add_argument("--use-client", action="store_true")
    args = parser.parse_args()

    if args.mode == "otel":
        asyncio.run(send_traces_via_otel(use_client=args.use_client))
    elif args.mode == "agentops":
        asyncio.run(send_traces_via_agentops(use_client=args.use_client))
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

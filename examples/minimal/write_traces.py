# Copyright (c) Microsoft. All rights reserved.

"""Example to write traces to a LightningStore via raw OpenTelemetry or AgentOpsTracer.

Prior to running this example, please start a LightningStore server with OTLP enabled first:

```bash
agl store --port 45993 --log-level DEBUG
```
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import AgentOpsTracer, LightningStoreClient, OtelTracer, emit_reward, setup_logging

console = Console()


async def send_traces_via_otel():
    tracer = OtelTracer()
    client = LightningStoreClient("http://localhost:45993")
    rollout = await client.start_rollout(input={"origin": "write_traces_example"})

    with tracer.lifespan():
        # Initialize the capture of one single trace for one single rollout
        async with tracer.trace_context(
            "trace-manual", store=client, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ) as tracer:

            with tracer.start_as_current_span("grpc-span-1"):
                time.sleep(0.01)

                with tracer.start_as_current_span("grpc-span-2"):
                    time.sleep(0.01)

            with tracer.start_as_current_span("grpc-span-3"):
                time.sleep(0.01)

            emit_reward(1.0)

    traces = await client.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)
    assert len(traces) == 4
    await client.close()


async def send_traces_via_agentops():
    tracer = AgentOpsTracer()
    client = LightningStoreClient("http://localhost:45993")
    rollout = await client.start_rollout(input={"origin": "write_traces_example"})

    # Initialize the tracer lifespan
    # One lifespan can contain multiple traces
    with tracer.lifespan():
        # Initialize the capture of one single trace for one single rollout
        async with tracer.trace_context(
            "trace-1", store=client, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            openai_client = AsyncOpenAI()
            response = await openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, what's your name?"},
                ],
            )
            assert response.choices[0].message.content is not None
            assert "chatgpt" in response.choices[0].message.content.lower()

    traces = await client.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)
    await client.close()


def main():
    setup_logging("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["otel", "agentops"])
    args = parser.parse_args()

    if args.mode == "otel":
        asyncio.run(send_traces_via_otel())
    elif args.mode == "agentops":
        asyncio.run(send_traces_via_agentops())
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft. All rights reserved.

"""Example to write traces to a LightningStore via raw OpenTelemetry or AgentOpsTracer.

Prior to running this example, please start a LightningStore server with OTLP enabled first:

```bash
agl store --port 45993
```
"""

import argparse
import asyncio
import time
from typing import Literal

from openai import AsyncOpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentlightning import AgentOpsTracer, LightningStoreClient


def manually_send_traces():
    provider = TracerProvider(resource=Resource.create({"service.name": "otlp-demo-service"}))
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint="http://localhost:45993/v1/traces")

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("grpc-span-1"):
        time.sleep(0.01)

    with tracer.start_as_current_span("grpc-span-2"):
        time.sleep(0.01)

    processor.force_flush()


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
    print(traces)
    await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["manual", "agentops"])
    args = parser.parse_args()

    if args.mode == "manual":
        manually_send_traces()
    else:
        asyncio.run(send_traces_via_agentops())


if __name__ == "__main__":
    main()

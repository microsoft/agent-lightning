# Copyright (c) Microsoft. All rights reserved.

import argparse
import time
from typing import Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def make_tracer(endpoint: str, mode: Literal["grpc", "http"]):
    provider = TracerProvider(resource=Resource.create({"service.name": "otlp-demo-service"}))
    trace.set_tracer_provider(provider)

    if mode == "grpc":
        exporter = OTLPGrpcExporter(
            endpoint=endpoint,
            insecure=True,  # assumes HTTP / no TLS; use credentials for TLS setups
        )
    elif mode == "http":
        exporter = OTLPHttpExporter(
            endpoint=endpoint,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    return trace.get_tracer(__name__), processor


def send_via_grpc(endpoint: str):
    tracer, processor = make_tracer(endpoint, "grpc")

    with tracer.start_as_current_span("grpc-span-1"):
        time.sleep(0.01)

    with tracer.start_as_current_span("grpc-span-2"):
        time.sleep(0.01)

    processor.force_flush()
    print(f"Sent spans via OTLP/gRPC → {endpoint}")


def send_via_http_json(endpoint: str):
    tracer, processor = make_tracer(endpoint, "http")

    with tracer.start_as_current_span("json-span-1"):
        time.sleep(0.01)

    with tracer.start_as_current_span("json-span-2"):
        time.sleep(0.01)

    processor.force_flush()
    print(f"Sent spans via OTLP/HTTP+JSON → {endpoint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grpc", "json"])
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/traces")
    args = parser.parse_args()

    if args.mode == "grpc":
        send_via_grpc(args.endpoint)
    else:
        send_via_http_json(args.endpoint)


if __name__ == "__main__":
    main()

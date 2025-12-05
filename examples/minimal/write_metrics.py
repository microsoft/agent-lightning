# Copyright (c) Microsoft. All rights reserved.

"""Demonstrate `MultiMetricsBackend` by emitting metrics to both console and Prometheus.

Usage:
    python write_metrics.py --duration 10 --prom-port 8000

The script registers a counter and a histogram, pushes events through both the
`ConsoleMetricsBackend` (for immediate feedback) and the `PrometheusMetricsBackend`
for scraping via `/metrics`.

Run a Prometheus server (for example via `docker/compose.prometheus-memory-store.yml`)
and add the host running this script as a scrape target. By default the metrics
endpoint binds to `0.0.0.0:9105`.
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import time
from typing import Sequence

from prometheus_client import start_http_server

from agentlightning import setup_logging
from agentlightning.utils.metrics import (
    ConsoleMetricsBackend,
    MetricsBackend,
    MultiMetricsBackend,
    PrometheusMetricsBackend,
)


def _register_metrics(backend: MetricsBackend) -> None:
    backend.register_counter("minimal_requests_total", ["operation", "status"])
    backend.register_histogram(
        "minimal_latency_seconds",
        ["operation"],
        buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    )


def _emit_metrics(backend: MetricsBackend, duration: float, operations: Sequence[str]) -> None:
    statuses = ["200", "404", "500"]
    end_time = time.time() + duration
    random.seed(1337)
    while time.time() < end_time:
        operation = random.choice(operations)
        status = random.choices(statuses, weights=[0.9, 0.05, 0.05], k=1)[0]
        latency = random.lognormvariate(-4.0, 0.5)
        backend.inc_counter("minimal_requests_total", labels={"operation": operation, "status": status})
        backend.observe_histogram("minimal_latency_seconds", value=latency, labels={"operation": operation})
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to emit metrics before shutting down.")
    parser.add_argument("--prom-port", type=int, default=9105, help="Port for the /metrics endpoint.")
    parser.add_argument("--prom-host", default="0.0.0.0", help="Host/IP for the /metrics endpoint.")
    parser.add_argument("--group-level", type=int, default=2, help="ConsoleMetricsBackend label grouping depth.")
    args = parser.parse_args()

    setup_logging()

    console_backend = ConsoleMetricsBackend(window_seconds=15.0, log_interval_seconds=2.0, group_level=args.group_level)
    prom_backend = PrometheusMetricsBackend()
    backend = MultiMetricsBackend([console_backend, prom_backend])
    _register_metrics(backend)

    start_http_server(args.prom_port, addr=args.prom_host)
    print(f"Prometheus metrics exposed on http://{args.prom_host}:{args.prom_port}/metrics")
    print(f"Emitting demo metrics for {args.duration:.1f}s ...")

    # Handle CTRL+C gracefully
    interrupted = False

    def _handle_interrupt(signum: int, frame: object | None) -> None:  # pragma: no cover - signal handler
        nonlocal interrupted
        print(f"Received signal {signum}, stopping...")
        interrupted = True

    original_handler = signal.signal(signal.SIGINT, _handle_interrupt)
    try:
        _emit_metrics(backend, duration=args.duration, operations=["search", "summary", "answer"])
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if interrupted:
        sys.exit(1)


if __name__ == "__main__":
    main()

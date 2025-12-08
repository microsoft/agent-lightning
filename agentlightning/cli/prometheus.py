# Copyright (c) Microsoft. All rights reserved.

"""Serve Prometheus metrics from the Agent Lightning multiprocess registry."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from fastapi import FastAPI
from prometheus_client import make_asgi_app  # pyright: ignore[reportUnknownVariableType]

from agentlightning.logging import setup as setup_logging
from agentlightning.store.utils import LATENCY_BUCKETS
from agentlightning.utils.metrics import PrometheusMetricsBackend, get_prometheus_registry
from agentlightning.utils.server_launcher import PythonServerLauncher, PythonServerLauncherArgs

logger = logging.getLogger(__name__)


def ensure_prometheus_dir() -> str:
    """Ensure PROMETHEUS_MULTIPROC_DIR is set and the directory exists."""

    directory = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if directory is None:
        raise ValueError("PROMETHEUS_MULTIPROC_DIR is not set.")

    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Serving Prometheus multiprocess metrics from %s", directory)
    return directory


def create_prometheus_app(metrics_path: str = "/v1/prometheus") -> FastAPI:
    """Create a FastAPI app that exposes Prometheus metrics and a health endpoint.

    Args:
        metrics_path: URL path to expose the Prometheus metrics endpoint on.

    Returns:
        A FastAPI application ready to serve metrics.
    """

    if not metrics_path.startswith("/"):
        raise ValueError("metrics_path must start with '/'.")

    normalized_path = metrics_path.rstrip("/")
    if normalized_path in ("", "/"):
        raise ValueError("metrics_path must not be '/'. Choose a sub-path such as /v1/prometheus.")

    app = FastAPI(title="Agent Lightning Prometheus exporter", docs_url=None, redoc_url=None)
    metrics_app = make_asgi_app(registry=get_prometheus_registry())  # pyright: ignore[reportUnknownVariableType]
    app.mount(normalized_path, metrics_app)  # pyright: ignore[reportUnknownArgumentType]

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    return app


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Prometheus metrics outside the LightningStore server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the metrics server to.")
    parser.add_argument("--port", type=int, default=4748, help="Port to expose the Prometheus metrics on.")
    parser.add_argument(
        "--metrics-path",
        default="/v1/prometheus",
        help="HTTP path used to expose metrics. Must start with '/' and not be the root path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Configure the logging level for the metrics server.",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable uvicorn access logs. Disabled by default to reduce noise.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging(args.log_level)
    ensure_prometheus_dir()

    try:
        app = create_prometheus_app(args.metrics_path)
    except ValueError as exc:
        logger.error("Failed to configure prometheus app: %s", exc)
        return 1

    launcher_args = PythonServerLauncherArgs(
        host=args.host,
        port=args.port,
        log_level=getattr(logging, args.log_level),
        access_log=args.access_log,
        healthcheck_url="/health",
    )
    launcher = PythonServerLauncher(app, launcher_args)

    try:
        asyncio.run(_serve_with_warmup(launcher))
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping Prometheus server.")
    except RuntimeError as exc:
        logger.error("Prometheus server failed to start: %s", exc, exc_info=True)
        return 1
    return 0


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    kind: str
    label_names: Sequence[str]
    buckets: Sequence[float] | None = None

    @property
    def prometheus_name(self) -> str:
        return self.name.replace(".", "_")


_DEFAULT_LABEL_VALUE = ""
_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec("agl.http.total", "counter", ("path", "method", "status")),
    _MetricSpec("agl.http.latency", "histogram", ("path", "method", "status"), LATENCY_BUCKETS),
    _MetricSpec("agl.store.total", "counter", ("method", "status")),
    _MetricSpec("agl.store.latency", "histogram", ("method", "status"), LATENCY_BUCKETS),
    _MetricSpec("agl.rollouts.total", "counter", ("status", "mode")),
    _MetricSpec("agl.rollouts.duration", "histogram", ("status", "mode"), LATENCY_BUCKETS),
    _MetricSpec("agl.collections.total", "counter", ("store_method", "operation", "collection", "status")),
    _MetricSpec(
        "agl.collections.latency",
        "histogram",
        ("store_method", "operation", "collection", "status"),
        LATENCY_BUCKETS,
    ),
)


async def _serve_with_warmup(launcher: PythonServerLauncher) -> None:
    warm_task = asyncio.create_task(_warm_prometheus_metrics())
    try:
        await launcher.run_forever()
    finally:
        warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await warm_task


async def _warm_prometheus_metrics() -> None:
    """Emit a zero-value sample for each Agent Lightning metric."""
    await asyncio.sleep(0.1)
    backend = PrometheusMetricsBackend()
    placeholder_cache: dict[tuple[str, ...], dict[str, str]] = {}
    for spec in _METRIC_SPECS:
        labels = placeholder_cache.setdefault(
            tuple(spec.label_names),
            {name: _DEFAULT_LABEL_VALUE for name in spec.label_names},
        )
        if spec.kind == "counter":
            backend.register_counter(spec.name, spec.label_names)
            await backend.inc_counter(spec.name, amount=0, labels=labels)
        elif spec.kind == "histogram":
            backend.register_histogram(spec.name, spec.label_names, buckets=spec.buckets)
            await backend.observe_histogram(spec.name, value=0.0, labels=labels)


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) Microsoft. All rights reserved.

"""Serve Prometheus metrics from the Agent Lightning multiprocess registry."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI
from prometheus_client import make_asgi_app  # pyright: ignore[reportUnknownVariableType]

from agentlightning.logging import setup as setup_logging
from agentlightning.utils.metrics import get_prometheus_registry
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
        asyncio.run(launcher.run_forever())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping Prometheus server.")
    except RuntimeError as exc:
        logger.error("Prometheus server failed to start: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) Microsoft. All rights reserved.

"""Run a LightningStore server for persistent access from multiple processes."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Iterable

from agentlightning import setup_logging
from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore

logger = logging.getLogger(__name__)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LightningStore server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=4747, help="Port to run the server on")
    parser.add_argument(
        "--cors-origin",
        dest="cors_origins",
        action="append",
        help="Allowed CORS origin. Repeat for multiple origins. Use '*' to allow all origins.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Configure the logging level for the store.",
    )
    parser.add_argument(
        "--prometheus",
        action="store_true",
        help="Enable Prometheus metrics.",
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help=(
            "Number of workers to run in the server. When it's greater than 1, the server will be run using `mp` launch mode. "
            "Only applicable for zero-copy stores such as MongoDB backend."
        ),
    )

    parser.add_argument(
        "--backend",
        choices=["memory", "mongo"],
        default="memory",
        help="Backend to use for the store.",
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/?replicaSet=rs0",
        help="MongoDB URI to use for the store. Applicable only if --backend is 'mongo'.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging(args.log_level)

    if args.backend == "memory":
        store = InMemoryLightningStore(prometheus=args.prometheus)
    elif args.backend == "mongo":
        from agentlightning.store.mongo import MongoLightningStore

        store = MongoLightningStore(client=args.mongo_uri, prometheus=args.prometheus)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    if args.n_workers > 1:
        logger.info(f"Running the server using `mp` launch mode with {args.n_workers} workers.")
        launch_mode = "mp"
    else:
        logger.info("Running the server using `asyncio` launch mode.")
        launch_mode = "asyncio"
    server = LightningStoreServer(
        store,
        host=args.host,
        port=args.port,
        cors_allow_origins=args.cors_origins,
        launch_mode=launch_mode,
        prometheus=args.prometheus,
        n_workers=args.n_workers,
    )
    try:
        asyncio.run(server.run_forever())
    except RuntimeError as exc:
        logger.error("LightningStore server failed to start: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

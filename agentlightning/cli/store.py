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
    parser.add_argument("--port", type=int, default=4747, help="Port to run the server on")
    parser.add_argument(
        "--cors-origin",
        dest="cors_origins",
        action="append",
        help="Allowed CORS origin. Repeat for multiple origins. Use '*' to allow all origins.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging()

    store = InMemoryLightningStore()
    server = LightningStoreServer(
        store,
        host="0.0.0.0",
        port=args.port,
        cors_allow_origins=args.cors_origins,
        launch_mode="asyncio",
    )
    try:
        asyncio.run(server.run_forever())
    except RuntimeError as exc:
        logger.error("LightningStore server failed to start: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) Microsoft. All rights reserved.

"""Run a LightningStore server for persistent access from multiple processes."""

from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

from agentlightning.logging import configure_logger
from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LightningStore server")
    parser.add_argument("--port", type=int, default=4747, help="Port to run the server on")
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logger()

    store = InMemoryLightningStore()
    server = LightningStoreServer(store, host="0.0.0.0", port=args.port)
    asyncio.run(server.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

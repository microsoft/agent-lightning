# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import time
from typing import Iterable

from agentlightning.instrumentation.agentops import AgentOpsServerManager


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start AgentOps server")
    parser.add_argument("--daemon", action="store_true", help="Run server as a daemon")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manager = AgentOpsServerManager(daemon=args.daemon, port=args.port)
    try:
        manager.start()
        # Wait forever
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

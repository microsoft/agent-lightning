# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning command line interface entry point."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict, Iterable, Tuple

_SUBCOMMANDS: Dict[str, Tuple[str, str]] = {
    "vllm": ("agentlightning.cli.vllm", "Run the vLLM CLI with Agent Lightning instrumentation."),
    "store": ("agentlightning.cli.store", "Run a LightningStore server."),
    "agentops": ("agentlightning.cli.agentops_server", "Start the AgentOps server manager."),
}

_DESCRIPTION = "Agent Lightning CLI entry point.\n\nAvailable subcommands:\n" + "\n".join(
    f"  {name:<10}{desc}" for name, (_, desc) in _SUBCOMMANDS.items()
)


def main(argv: Iterable[str] | None = None) -> int:
    """Dispatch to the requested Agent Lightning subcommand."""
    parser = argparse.ArgumentParser(
        prog="agl",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("subcommand", choices=_SUBCOMMANDS.keys(), help="Subcommand to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parsed = parser.parse_args(list(argv) if argv is not None else None)
    module_name, _ = _SUBCOMMANDS[parsed.subcommand]
    module = importlib.import_module(module_name)

    entry_point = getattr(module, "main", None)
    if entry_point is None:
        parser.error(f"Subcommand '{parsed.subcommand}' does not define a callable 'main'")

    dispatch_args = parsed.args
    original_argv = sys.argv
    sys.argv = [f"{parser.prog} {parsed.subcommand}", *dispatch_args]
    try:
        result = entry_point(dispatch_args or None)
    finally:
        sys.argv = original_argv

    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

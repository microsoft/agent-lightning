# Copyright (c) Microsoft. All rights reserved.

"""Shared command helpers for benchmark-specific SHAPER entry points."""

from __future__ import annotations

import os
import sys
from typing import Sequence
from urllib.parse import urlsplit


def cli_arguments(argv: Sequence[str] | None) -> list[str]:
    """Return explicit arguments or the current process arguments."""

    return list(argv) if argv is not None else sys.argv[1:]


def requests_help(arguments: Sequence[str]) -> bool:
    """Return whether an entry point should bypass environment preflight."""

    return any(argument in {"-h", "--help"} for argument in arguments)


def endpoint_socket(endpoint: str) -> tuple[str | None, int | None]:
    """Return a best-effort host/port pair for a planner endpoint."""

    parsed = urlsplit(endpoint)
    if not parsed.hostname:
        return None, None
    if parsed.port is not None:
        return parsed.hostname, parsed.port
    if parsed.scheme == "https":
        return parsed.hostname, 443
    if parsed.scheme == "http":
        return parsed.hostname, 80
    return None, None


def print_preflight_errors(errors: list[str]) -> int:
    """Print actionable preflight failures using a stable CLI format."""

    for error in errors:
        print(f"[missing] {error}")
    return 2 if errors else 0


def required_environment(name: str) -> str:
    """Read one required environment variable for a direct benchmark CLI."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before running this command.")
    return value

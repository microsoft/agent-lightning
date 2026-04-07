"""Streaming response assemblers — one module per LLM API format.

Each assembler is a pure function: ``(list[dict]) -> dict`` that converts
a list of parsed SSE JSON chunks into the canonical non-streaming response
shape for that format.

The registry (``select_assembler``) maps URL path suffixes to the right
assembler.  Unknown paths get ``None`` — the caller falls back to storing
raw chunks.
"""

from __future__ import annotations

from typing import Any, Callable

from agl_lite.gateway.assemblers.chat_completion import assemble_chat_completion
from agl_lite.gateway.assemblers.completion import assemble_completion
from agl_lite.gateway.assemblers.anthropic import assemble_anthropic_message

Assembler = Callable[[list[dict[str, Any]]], dict[str, Any]]

# Evaluated in order — first suffix match wins.
# "chat/completions" MUST come before "completions" (substring).
_ASSEMBLERS: list[tuple[str, Assembler]] = [
    ("chat/completions", assemble_chat_completion),
    ("completions", assemble_completion),
    ("messages", assemble_anthropic_message),
]


def select_assembler(path: str) -> Assembler | None:
    """Return the assembler for the given path, or None for raw fallback."""
    normalized = path.rstrip("/")
    for suffix, assembler in _ASSEMBLERS:
        if normalized.endswith(suffix):
            return assembler
    return None

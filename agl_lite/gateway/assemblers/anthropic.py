"""Assembler for Anthropic ``/v1/messages`` streaming responses.

Anthropic SSE uses typed events (``message_start``, ``content_block_delta``,
``message_delta``, ``message_stop``).  The ``event:`` line is not captured by
``parse_sse_chunks`` — but the ``data:`` JSON payload contains a ``type``
field that identifies the event, so the assembler dispatches on
``chunk["type"]``.

Assembled shape mirrors the Anthropic non-streaming ``Message``::

    {
      "id": "msg_...",
      "type": "message",
      "role": "assistant",
      "model": "<model>",
      "content": [{"type": "text", "text": "<full text>"}],
      "stop_reason": "end_turn",
      "usage": {"input_tokens": ..., "output_tokens": ...}
    }
"""

from __future__ import annotations

from typing import Any


def assemble_anthropic_message(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble Anthropic messages SSE chunks into a Message-shaped dict."""
    if not chunks:
        return {}

    result: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
    }
    # Collect text per content block index.
    content_blocks: dict[int, list[str]] = {}
    usage: dict[str, int] = {}

    for chunk in chunks:
        event_type = chunk.get("type", "")

        if event_type == "message_start":
            # The message shell with id, model, role, and initial usage.
            message = chunk.get("message", {})
            result["id"] = message.get("id", "")
            result["model"] = message.get("model", "")
            result["role"] = message.get("role", "assistant")
            if message.get("usage"):
                usage.update(message["usage"])

        elif event_type == "content_block_start":
            idx = chunk.get("index", 0)
            content_blocks.setdefault(idx, [])

        elif event_type == "content_block_delta":
            idx = chunk.get("index", 0)
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                content_blocks.setdefault(idx, []).append(delta.get("text", ""))

        elif event_type == "message_delta":
            delta = chunk.get("delta", {})
            if "stop_reason" in delta:
                result["stop_reason"] = delta["stop_reason"]
            # message_delta may carry output_tokens usage update.
            if chunk.get("usage"):
                usage.update(chunk["usage"])

        # message_stop and other events are ignored.

    # Build content array from collected blocks.
    content = [
        {"type": "text", "text": "".join(parts)}
        for _, parts in sorted(content_blocks.items())
    ]
    result["content"] = content if content else [{"type": "text", "text": ""}]

    if usage:
        result["usage"] = usage

    return result

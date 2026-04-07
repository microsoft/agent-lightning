"""Assembler for OpenAI ``/v1/chat/completions`` streaming responses.

Converts ``choices[i].delta.content`` chunks into a single
``ChatCompletion``-shaped dict matching the non-streaming response format.
Also preserves vLLM-specific fields (``prompt_token_ids``, per-choice
``token_ids``).
"""

from __future__ import annotations

from typing import Any


def assemble_chat_completion(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble OpenAI chat-completion SSE chunks into a ChatCompletion-shaped dict.

    Only call this for ``/v1/chat/completions`` paths.  Other endpoints use
    different delta structures (e.g. ``choices[0].text`` for legacy completions)
    and must not go through this function.

    Assembled shape mirrors the non-streaming OpenAI response::

        {
          "id": "chatcmpl-...",
          "object": "chat.completion",
          "created": <int>,
          "model": "<model>",
          "choices": [{"index": 0,
                       "message": {"role": "assistant", "content": "<full text>"},
                       "finish_reason": "stop"}],
          "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
        }

    ``usage`` is ``None`` when not provided by the upstream (vLLM only sends it
    when ``stream_options.include_usage=True`` is set by the caller).

    **vLLM extensions** (preserved when present, ignored when absent):

    - ``prompt_token_ids`` — tokenized prompt, sent in the first chunk
    - ``choices[i].token_ids`` — per-chunk response token IDs, concatenated
      across all chunks into each choice

    These are non-standard fields added by vLLM for training pipelines
    (see https://github.com/vllm-project/vllm/pull/22587).  Standard
    OpenAI endpoints never send them — the assembler simply omits them.
    """
    if not chunks:
        return {}

    first = chunks[0]
    last = chunks[-1]

    contents: dict[int, list[str]] = {}
    token_ids: dict[int, list[int]] = {}
    finish_reasons: dict[int, str | None] = {}
    role: str = "assistant"
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            delta = choice.get("delta", {})
            if "role" in delta:
                role = delta["role"]
            contents.setdefault(idx, []).append(delta.get("content") or "")
            # vLLM includes per-chunk token_ids in each choice.
            tids = choice.get("token_ids")
            if tids:
                token_ids.setdefault(idx, []).extend(tids)
            if choice.get("finish_reason"):
                finish_reasons[idx] = choice["finish_reason"]

    choices = [
        {
            "index": idx,
            "message": {"role": role, "content": "".join(parts)},
            "finish_reason": finish_reasons.get(idx),
            # Concatenated response token IDs (all chunks for this choice index).
            **({"token_ids": token_ids[idx]} if idx in token_ids else {}),
        }
        for idx, parts in sorted(contents.items())
    ]
    if not choices:
        choices = [{"index": 0, "message": {"role": role, "content": ""}, "finish_reason": None}]

    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model", ""),
        "choices": choices,
        "usage": last.get("usage"),
    }
    # vLLM sends prompt_token_ids in the first chunk.
    if first.get("prompt_token_ids"):
        result["prompt_token_ids"] = first["prompt_token_ids"]
    return result

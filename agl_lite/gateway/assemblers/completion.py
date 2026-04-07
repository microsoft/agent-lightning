"""Assembler for OpenAI legacy ``/v1/completions`` streaming responses.

Converts ``choices[i].text`` chunks into a single ``Completion``-shaped dict
matching the non-streaming response format.
"""

from __future__ import annotations

from typing import Any


def assemble_completion(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble legacy completion SSE chunks into a Completion-shaped dict.

    Streaming chunks use ``choices[i].text`` (a plain string, no delta
    wrapper).

    Assembled shape mirrors the non-streaming OpenAI response::

        {
          "id": "cmpl-...",
          "object": "text_completion",
          "created": <int>,
          "model": "<model>",
          "choices": [{"index": 0, "text": "<full text>", "finish_reason": "stop"}],
          "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
        }

    **vLLM extensions** (preserved when present, ignored when absent):

    - ``prompt_token_ids`` — tokenized prompt, sent in the first chunk
    - ``choices[i].token_ids`` — per-chunk response token IDs, concatenated

    See :func:`~agl_lite.gateway.assemblers.chat_completion.assemble_chat_completion`
    for details on these non-standard fields.
    """
    if not chunks:
        return {}

    first = chunks[0]
    last = chunks[-1]

    texts: dict[int, list[str]] = {}
    token_ids: dict[int, list[int]] = {}
    finish_reasons: dict[int, str | None] = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            texts.setdefault(idx, []).append(choice.get("text") or "")
            tids = choice.get("token_ids")
            if tids:
                token_ids.setdefault(idx, []).extend(tids)
            if choice.get("finish_reason"):
                finish_reasons[idx] = choice["finish_reason"]

    choices = [
        {
            "index": idx,
            "text": "".join(parts),
            "finish_reason": finish_reasons.get(idx),
            **({"token_ids": token_ids[idx]} if idx in token_ids else {}),
        }
        for idx, parts in sorted(texts.items())
    ]
    if not choices:
        choices = [{"index": 0, "text": "", "finish_reason": None}]

    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "text_completion",
        "created": first.get("created"),
        "model": first.get("model", ""),
        "choices": choices,
        "usage": last.get("usage"),
    }
    # vLLM sends prompt_token_ids in the first chunk.
    if first.get("prompt_token_ids"):
        result["prompt_token_ids"] = first["prompt_token_ids"]
    return result

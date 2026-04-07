"""Gateway proxy — HTTP forwarding to model servers with event capture."""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from agl_lite.schemas.model_server import ModelServer
from agl_lite.store.memory import InMemoryStore

log = structlog.get_logger()


async def forward_request(
    *,
    client: httpx.AsyncClient,
    server: ModelServer,
    path: str,
    body: dict[str, Any],
    store: InMemoryStore,
    rollout_id: str,
    attempt_id: str,
    original_body: dict[str, Any],
) -> Response:
    """Forward a request to a model server, capture event, return response.

    Dispatches to streaming or non-streaming based on body["stream"].
    """
    is_stream = body.get("stream", False)
    url = f"{server.endpoint.rstrip('/')}/{path.lstrip('/')}"

    headers: dict[str, str] = {"content-type": "application/json"}
    if server.token:
        headers["authorization"] = f"Bearer {server.token}"

    server_meta = {"model": server.model, "endpoint": server.endpoint, "version": server.version}

    if is_stream:
        return await _forward_streaming(
            client=client,
            url=url,
            path=path,
            headers=headers,
            body=body,
            store=store,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            original_body=original_body,
            server_meta=server_meta,
        )
    else:
        return await _forward_non_streaming(
            client=client,
            url=url,
            headers=headers,
            body=body,
            store=store,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            original_body=original_body,
            server_meta=server_meta,
        )


async def _forward_non_streaming(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    store: InMemoryStore,
    rollout_id: str,
    attempt_id: str,
    original_body: dict[str, Any],
    server_meta: dict[str, Any],
) -> JSONResponse:
    """Forward non-streaming request. Capture full response as event."""
    resp = await client.post(url, json=body, headers=headers)
    response_body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

    _capture_event(
        store=store,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        original_body=original_body,
        response_body=response_body,
        server_meta=server_meta,
    )

    return JSONResponse(content=response_body, status_code=resp.status_code)


async def _forward_streaming(
    *,
    client: httpx.AsyncClient,
    url: str,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
    store: InMemoryStore,
    rollout_id: str,
    attempt_id: str,
    original_body: dict[str, Any],
    server_meta: dict[str, Any],
) -> StreamingResponse:
    """Forward streaming request. Tee chunks to client while buffering for event capture."""
    upstream = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )

    buffer: list[bytes] = []

    async def stream_and_capture():
        try:
            async for chunk in upstream.aiter_bytes():
                buffer.append(chunk)
                yield chunk
        finally:
            await upstream.aclose()

            # Parse SSE into raw chunks (format-agnostic), then attempt
            # chat-completion assembly only for the known path.
            raw = b"".join(buffer)
            chunks = _parse_sse_chunks(raw)
            is_chat = path.rstrip("/").endswith("chat/completions")
            response_body: dict[str, Any] = (
                _assemble_chat_completion(chunks) if is_chat else {"chunks": chunks}
            )

            _capture_event(
                store=store,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                original_body=original_body,
                response_body=response_body,
                server_meta=server_meta,
            )

    return StreamingResponse(
        stream_and_capture(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _capture_event(
    *,
    store: InMemoryStore,
    rollout_id: str,
    attempt_id: str,
    original_body: dict[str, Any],
    response_body: dict[str, Any],
    server_meta: dict[str, Any],
) -> None:
    """Write a model_request event to the store."""
    store.add_event(
        rollout_id,
        attempt_id,
        "model_request",
        {
            "request": original_body,
            "response": response_body,
            "server": server_meta,
        },
    )


def _parse_sse_chunks(raw: bytes) -> list[dict[str, Any]]:
    """Parse raw SSE bytes into a list of JSON data objects.

    Format-agnostic: extracts every ``data: <json>`` line, skips ``[DONE]``.
    Does not interpret the payload structure.
    """
    import contextlib
    import json

    chunks: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            with contextlib.suppress(json.JSONDecodeError):
                chunks.append(json.loads(line[6:]))
    return chunks


def _assemble_chat_completion(chunks: list[dict[str, Any]]) -> dict[str, Any]:
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
    """
    if not chunks:
        return {}

    first = chunks[0]
    last = chunks[-1]

    contents: dict[int, list[str]] = {}
    finish_reasons: dict[int, str | None] = {}
    role: str = "assistant"
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            delta = choice.get("delta", {})
            if "role" in delta:
                role = delta["role"]
            contents.setdefault(idx, []).append(delta.get("content") or "")
            if choice.get("finish_reason"):
                finish_reasons[idx] = choice["finish_reason"]

    choices = [
        {
            "index": idx,
            "message": {"role": role, "content": "".join(parts)},
            "finish_reason": finish_reasons.get(idx),
        }
        for idx, parts in sorted(contents.items())
    ]
    if not choices:
        choices = [{"index": 0, "message": {"role": role, "content": ""}, "finish_reason": None}]

    return {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model", ""),
        "choices": choices,
        "usage": last.get("usage"),
    }

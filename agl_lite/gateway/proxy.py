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

            # Assemble buffered chunks and capture event.
            raw = b"".join(buffer)
            response_body = _parse_sse_response(raw)

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
    response_body: dict[str, Any] | list[dict[str, Any]],
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


def _parse_sse_response(raw: bytes) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse SSE stream bytes into a list of data chunks.

    SSE format: lines starting with "data: " followed by JSON.
    The final line is "data: [DONE]".
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

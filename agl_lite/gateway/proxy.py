"""Gateway proxy — HTTP forwarding to model servers with event capture."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agl_lite.gateway.assemblers import select_assembler
from agl_lite.schemas.model_server import ModelServer
from agl_lite.store.memory import InMemoryStore

log = structlog.get_logger()

_UPSTREAM_MAX_ATTEMPTS = 6
_RETRY_STATUS_CODES = {408, 409, 429}
_RETRY_BACKOFF_BASE_SECONDS = 0.5
_RETRY_BACKOFF_CAP_SECONDS = 8.0


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRY_STATUS_CODES or status_code >= 500


def _retry_delay_seconds(attempt_index: int) -> float:
    delay = min(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index), _RETRY_BACKOFF_CAP_SECONDS)
    return delay * random.uniform(0.75, 1.25)


async def _send_upstream_with_retries(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    stream: bool,
    url: str,
) -> httpx.Response:
    """Send an upstream request with OpenAI-like transient-failure retry."""
    for attempt_index in range(_UPSTREAM_MAX_ATTEMPTS):
        try:
            response = await send()
        except httpx.TimeoutException as exc:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(status_code=504, detail="Upstream model server timed out") from exc
            delay = _retry_delay_seconds(attempt_index)
            log.warning(
                "Retrying upstream request after timeout",
                url=url,
                stream=stream,
                attempt=attempt_index + 1,
                max_attempts=_UPSTREAM_MAX_ATTEMPTS,
                delay_seconds=round(delay, 3),
            )
            await asyncio.sleep(delay)
            continue
        except httpx.TransportError as exc:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(status_code=502, detail="Upstream model server request failed") from exc
            delay = _retry_delay_seconds(attempt_index)
            log.warning(
                "Retrying upstream request after transport error",
                url=url,
                stream=stream,
                attempt=attempt_index + 1,
                max_attempts=_UPSTREAM_MAX_ATTEMPTS,
                delay_seconds=round(delay, 3),
                error=str(exc),
            )
            await asyncio.sleep(delay)
            continue

        if not _is_retryable_status(response.status_code) or attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
            return response

        delay = _retry_delay_seconds(attempt_index)
        log.warning(
            "Retrying upstream request after retryable status",
            url=url,
            stream=stream,
            attempt=attempt_index + 1,
            max_attempts=_UPSTREAM_MAX_ATTEMPTS,
            status_code=response.status_code,
            delay_seconds=round(delay, 3),
        )
        await response.aclose()
        await asyncio.sleep(delay)

    raise HTTPException(status_code=502, detail="Upstream model server request failed")


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

    log.debug(
        "Proxying request",
        rollout_id=rollout_id,
        model=server.model,
        stream=is_stream,
        path=path,
    )

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
    resp = await _send_upstream_with_retries(
        lambda: client.post(url, json=body, headers=headers),
        stream=False,
        url=url,
    )
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
    upstream = await _send_upstream_with_retries(
        lambda: client.send(client.build_request("POST", url, json=body, headers=headers), stream=True),
        stream=True,
        url=url,
    )

    buffer: list[bytes] = []

    async def stream_and_capture():
        try:
            async for chunk in upstream.aiter_bytes():
                buffer.append(chunk)
                yield chunk
        finally:
            await upstream.aclose()

            # Parse SSE into raw chunks (format-agnostic), then assemble
            # using the format-specific assembler for this path.
            raw = b"".join(buffer)
            chunks = _parse_sse_chunks(raw)
            assembler = select_assembler(path)
            response_body: dict[str, Any] = (
                assembler(chunks) if assembler else {"chunks": chunks}
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




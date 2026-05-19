"""Gateway proxy — HTTP forwarding to model servers with event capture."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agl_lite.gateway.assemblers import select_assembler
from agl_lite.schemas.model_server import ModelServer
from agl_lite.store.memory import InMemoryStore

log = structlog.get_logger()

_UPSTREAM_MAX_ATTEMPTS = 6
_RETRY_STATUS_CODES = {408, 409, 429}
_RETRY_BACKOFF_BASE_SECONDS = 0.5
_RETRY_BACKOFF_CAP_SECONDS = 8.0


@dataclass
class GatewayPauseState:
    """Process-level switch — when paused, llm_proxy returns 429.

    Owned by ``app.state.gateway_pause_state``. Mutated only by the admin
    routes and by the proxy forward path (inflight inc/dec). The state itself
    carries no business logic — it just holds a bool, a Retry-After hint, an
    in-flight counter, and an asyncio.Lock that protects all mutations.

    Used by the async-rollout feature: bridge flips ``paused=True`` after
    enough groups finish, then waits for ``inflight`` to drain before
    calling ``sleep_replicas()`` on vLLM.
    """

    paused: bool = False
    retry_after_seconds: int = 5
    reason: str | None = None
    inflight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.drained.set()


def _get_pause_state(request: Request | None) -> GatewayPauseState | None:
    """Return the app's pause state, or None when running without an app
    (e.g. unit-tested ``forward_request`` calls)."""
    if request is None:
        return None
    return getattr(request.app.state, "gateway_pause_state", None)


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
    pause_state: GatewayPauseState | None = None,
    request: Request | None = None,
) -> Response:
    """Forward a request to a model server, capture event, return response.

    Dispatches to streaming or non-streaming based on body["stream"].

    If ``pause_state.paused`` is True at entry, returns 429 with a
    ``Retry-After`` header instead of forwarding. Otherwise the request is
    registered into ``pause_state.inflight`` for the lifetime of the upstream
    call, so a concurrent ``pause_gateway()`` followed by drain polling sees
    a stable count of in-flight upstream requests.
    """
    if pause_state is not None:
        async with pause_state.lock:
            if pause_state.paused:
                retry_after = pause_state.retry_after_seconds
                reason = pause_state.reason
                return Response(
                    status_code=429,
                    headers={
                        "Retry-After": str(retry_after),
                        "X-Agl-Paused": "true",
                    },
                    content=json.dumps({"error": "gateway paused", "reason": reason}),
                    media_type="application/json",
                )
            # Not paused — register this request as in-flight under the same
            # lock, so a subsequent pause_gateway() either sees this request
            # already counted, or runs strictly before this branch.
            pause_state.inflight += 1
            pause_state.drained.clear()

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
        # Streaming path manages its own inflight lifecycle: decrement happens
        # either inside _forward_streaming (if upstream construction fails)
        # or inside the StreamingResponse generator's finally block (after
        # the upstream stream is fully consumed).
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
            pause_state=pause_state,
            request=request,
        )
    try:
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
    finally:
        # Non-streaming: upstream call is fully drained when
        # _forward_non_streaming returns (success or exception).
        if pause_state is not None:
            await _dec_inflight(pause_state)


async def _dec_inflight(pause_state: GatewayPauseState) -> None:
    async with pause_state.lock:
        pause_state.inflight = max(0, pause_state.inflight - 1)
        if pause_state.inflight == 0:
            pause_state.drained.set()


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
    pause_state: GatewayPauseState | None = None,
    request: Request | None = None,
) -> StreamingResponse:
    """Forward streaming request. Tee chunks to client while buffering for event capture."""
    try:
        upstream = await _send_upstream_with_retries(
            lambda: client.send(client.build_request("POST", url, json=body, headers=headers), stream=True),
            stream=True,
            url=url,
        )
    except BaseException:
        # Failure before stream construction — caller's finally won't run, so
        # release inflight slot here.
        if pause_state is not None:
            await _dec_inflight(pause_state)
        raise

    buffer: list[bytes] = []

    async def stream_and_capture():
        try:
            async for chunk in upstream.aiter_bytes():
                # Stop streaming if the client has disconnected, so the
                # generator's finally runs promptly and inflight is released.
                if request is not None:
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        pass
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

            # Release the pause-drain inflight slot only after the full
            # upstream stream has been consumed and closed. Releasing earlier
            # (e.g. when StreamingResponse returns) lets bridge see drained=0
            # while vLLM is still generating.
            if pause_state is not None:
                await _dec_inflight(pause_state)

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



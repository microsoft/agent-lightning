# Copyright (c) Microsoft. All rights reserved.

"""Server-side OpenAI chat-completions proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from agentlightning.schemas import Model
from agentlightning.server.routes.events import record_event
from agentlightning.server.store import _models

log = structlog.get_logger()

_UPSTREAM_MAX_ATTEMPTS = 6
_RETRY_STATUS_CODES = {408, 409, 429}
_RETRY_BACKOFF_BASE_SECONDS = 0.5
_RETRY_BACKOFF_CAP_SECONDS = 8.0


class NoServersError(Exception):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"No servers available for model '{model}'")


class ProxyRouter:
    """Selects the configured default model server and rewrites request params."""

    def __init__(self, default_proxy: Mapping[str, Any]) -> None:
        self._model_name = str(default_proxy["model_name"])
        self._train_temperature = float(default_proxy["train"]["temperature"])
        self._val_temperature = float(default_proxy["val"]["temperature"])
        self._include_log_probs = bool(default_proxy.get("include_log_probs", True))

    @property
    def model_name(self) -> str:
        return self._model_name

    def select_server(self, model: str, rollout_id: str) -> Model:
        servers = _models.get(model, {})
        if not servers:
            raise NoServersError(model)
        # Stable ordering pins each rollout to one endpoint for prefix-cache reuse.
        pool = [servers[endpoint] for endpoint in sorted(servers)]
        digest = hashlib.sha256(rollout_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(pool)
        return pool[index]

    def prepare_body(self, body: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "train":
            prepared = {
                **body,
                "model": self._model_name,
                "temperature": self._train_temperature,
                "return_token_ids": True,
            }
            if self._include_log_probs:
                prepared["logprobs"] = True
            return prepared
        if mode == "val":
            prepared = {
                **body,
                "model": self._model_name,
                "temperature": self._val_temperature,
                "return_token_ids": True,
            }
            return prepared
        raise ValueError(f"Unsupported proxy mode: {mode}")


@dataclass
class ProxyPauseState:
    paused: bool = False
    retry_after_seconds: int = 5
    reason: str | None = None
    inflight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def forward_request(
    *,
    client: httpx.AsyncClient,
    server: Model,
    body: dict[str, Any],
    upstream_path: str = "chat/completions",
    rollout_id: str,
    attempt_id: str,
    pause_state: ProxyPauseState | None = None,
) -> Response:
    if pause_state is not None:
        async with pause_state.lock:
            if pause_state.paused:
                retry_after = pause_state.retry_after_seconds
                reason = pause_state.reason
                return Response(
                    status_code=429,
                    headers={"Retry-After": str(retry_after), "X-Agl-Paused": "true"},
                    content=json.dumps({"error": "gateway paused", "reason": reason}),
                    media_type="application/json",
                )
            pause_state.inflight += 1

    try:
        if body.get("stream", False):
            raise HTTPException(status_code=400, detail="Streaming responses are not supported")

        url = f"{server.endpoint.rstrip('/')}/{upstream_path}"
        log.debug("Proxying request", rollout_id=rollout_id, model=server.model, path=upstream_path)

        started_at = time.perf_counter()
        response = await _send_upstream_with_retries(client=client, url=url, body=body)
        latency_ms = (time.perf_counter() - started_at) * 1000
        response_body = (
            response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        )

        _capture_event(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            request_body=body,
            response_body=response_body,
            server=server,
            latency_ms=latency_ms,
            http_status=response.status_code,
            status=_status_from_http_status(response.status_code),
            retry_count=int(response.extensions.get("agl_retry_count", 0)),
        )
        return JSONResponse(content=response_body, status_code=response.status_code)
    finally:
        if pause_state is not None:
            await _dec_inflight(pause_state)


async def _send_upstream_with_retries(
    *,
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
) -> httpx.Response:
    for attempt_index in range(_UPSTREAM_MAX_ATTEMPTS):
        try:
            response = await client.post(url, json=body, headers={"content-type": "application/json"})
        except httpx.TimeoutException as exc:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(status_code=504, detail="Upstream model server timed out") from exc
            await _sleep_before_retry(url=url, attempt_index=attempt_index, reason="timeout")
            continue
        except httpx.TransportError as exc:
            if attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
                raise HTTPException(status_code=502, detail="Upstream model server request failed") from exc
            await _sleep_before_retry(url=url, attempt_index=attempt_index, reason="transport error")
            continue

        if not _is_retryable_status(response.status_code) or attempt_index == _UPSTREAM_MAX_ATTEMPTS - 1:
            response.extensions["agl_retry_count"] = attempt_index
            return response

        await response.aclose()
        await _sleep_before_retry(
            url=url,
            attempt_index=attempt_index,
            reason=f"status {response.status_code}",
        )

    raise HTTPException(status_code=502, detail="Upstream model server request failed")


async def _sleep_before_retry(*, url: str, attempt_index: int, reason: str) -> None:
    delay = _retry_delay_seconds(attempt_index)
    log.warning(
        "Retrying upstream request",
        url=url,
        attempt=attempt_index + 1,
        max_attempts=_UPSTREAM_MAX_ATTEMPTS,
        delay_seconds=round(delay, 3),
        reason=reason,
    )
    await asyncio.sleep(delay)


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRY_STATUS_CODES or status_code >= 500


def _retry_delay_seconds(attempt_index: int) -> float:
    delay = min(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index), _RETRY_BACKOFF_CAP_SECONDS)
    return delay * random.uniform(0.75, 1.25)


async def _dec_inflight(pause_state: ProxyPauseState) -> None:
    async with pause_state.lock:
        pause_state.inflight = max(0, pause_state.inflight - 1)


def _capture_event(
    *,
    rollout_id: str,
    attempt_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    server: Model,
    latency_ms: float,
    http_status: int,
    status: str,
    retry_count: int,
) -> None:
    record_event(
        rollout_id,
        attempt_id,
        "model_request",
        {
            "model": server.model,
            "model_version": server.version,
            "request": request_body,
            "response": response_body,
            "server": {"model": server.model, "endpoint": server.endpoint, "version": server.version},
            "latency_ms": latency_ms,
            "http_status": http_status,
            "status": status,
            "retry_count": retry_count,
            "usage": _extract_usage(response_body),
            "finish_reason": _extract_finish_reason(response_body),
        },
    )


def _status_from_http_status(http_status: int) -> str:
    return "ok" if http_status < 400 else "error"


def _extract_usage(response_body: dict[str, Any]) -> dict[str, Any] | None:
    usage = response_body.get("usage")
    return usage if isinstance(usage, dict) else None


def _extract_finish_reason(response_body: dict[str, Any]) -> str | None:
    choices = response_body.get("choices")
    if isinstance(choices, list) and choices:
        reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        if isinstance(reason, str) and reason:
            return reason
    return None

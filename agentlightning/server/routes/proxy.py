# Copyright (c) Microsoft. All rights reserved.

"""Proxy forwarding and pause/drain management routes."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agentlightning.credit.state import AttemptCreditState
from agentlightning.credit.types import (
    CreditProtocolError,
    canonical_json,
    semantic_digest,
    validate_sampling_parameters,
)
from agentlightning.schemas import Model
from agentlightning.server.proxy import (
    NoServersError,
    ProxyPauseState,
    ProxyRouter,
    forward_process_request,
    forward_request,
)
from agentlightning.server.routes.credit import credit_http_error, get_credit_state
from agentlightning.server.store import _rollouts

log = structlog.get_logger()

router = APIRouter(tags=["gateway"])
management_router = APIRouter(tags=["gateway-management"], prefix="/proxy")


def _get_pause_state(request: Request) -> ProxyPauseState:
    state: ProxyPauseState | None = getattr(request.app.state, "proxy_pause_state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Gateway pause state not configured")
    return state


async def _wait_for_disconnect(request: Request) -> None:
    """Wait for the client to disconnect after the request body has been read."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


@router.post(
    "/proxy/rollout/{rollout_id}/attempt/{attempt_id}/mode/{mode}/openai/v1/{upstream_path:path}",
)
async def llm_proxy(rollout_id: str, attempt_id: str, mode: str, upstream_path: str, request: Request) -> Response:
    """LLM reverse proxy — forwards to model server, captures events."""
    if mode not in {"train", "val"}:
        raise HTTPException(status_code=404, detail=f"Unsupported proxy mode: {mode}")
    if upstream_path not in {"chat/completions", "completions"}:
        raise HTTPException(status_code=404, detail=f"Unsupported upstream path: {upstream_path}")

    # Validate rollout exists.
    if rollout_id not in _rollouts:
        raise HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")

    # Get gateway router and httpx client from app state.
    proxy_router: ProxyRouter | None = getattr(request.app.state, "proxy_router", None)
    http_client = getattr(request.app.state, "http_client", None)

    if proxy_router is None or http_client is None:
        raise HTTPException(status_code=503, detail="Proxy not configured")

    pause_state: ProxyPauseState | None = getattr(request.app.state, "proxy_pause_state", None)

    # Read and parse request body.
    raw_body = await request.body()
    try:
        body = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Select server.
    model_name = proxy_router.model_name
    try:
        server = proxy_router.select_server(model_name, rollout_id)
    except NoServersError:
        raise HTTPException(status_code=503, detail=f"No servers available for model '{model_name}'") from None

    prepared_body = proxy_router.prepare_body(body, mode)
    credit_state = get_credit_state(rollout_id, attempt_id)
    if credit_state is not None:
        return await _process_proxy(
            request, credit_state, server, body, prepared_body, upstream_path, http_client, pause_state
        )

    # Server endpoint includes the OpenAI base path (e.g., "http://vllm:8000/v1").
    forwarding = asyncio.create_task(
        forward_request(
            client=http_client,
            server=server,
            body=prepared_body,
            upstream_path=upstream_path,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            pause_state=pause_state,
        )
    )
    disconnected = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait((forwarding, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if forwarding in done:
            return forwarding.result()
        disconnected.result()
        log.info("Agent disconnected; cancelling model request", rollout_id=rollout_id, attempt_id=attempt_id)
        return Response(status_code=499)
    finally:
        for task in (forwarding, disconnected):
            if not task.done():
                task.cancel()
        await asyncio.gather(forwarding, disconnected, return_exceptions=True)


async def _process_proxy(
    request: Request,
    state: AttemptCreditState,
    server: Model,
    original_body: dict,
    prepared_body: dict,
    upstream_path: str,
    http_client,
    pause_state: ProxyPauseState | None,
) -> Response:
    try:
        if server.version != state.policy_version:
            raise CreditProtocolError("serving policy version differs from collection step")
        if prepared_body.get("stream"):
            raise CreditProtocolError("process credit requires complete non-streamed responses")
        validate_sampling_parameters(prepared_body)
        request_id = request.headers.get("X-AGL-Request-ID", "")
        action_index = int(request.headers.get("X-AGL-Action-Index", ""))
        payload = {"path": upstream_path, "body": original_body, "prepared": prepared_body, "index": action_index}
        if len(canonical_json(payload)) > state.config.max_artifact_bytes:
            raise CreditProtocolError("process request exceeds evidence byte limit")
        lease = state.begin_request(request_id, action_index, semantic_digest(payload))
    except ValueError as exc:
        raise credit_http_error(exc) from exc
    if lease.owner:
        state.retain_operation(
            asyncio.create_task(
                forward_process_request(
                    client=http_client,
                    server=server,
                    body=prepared_body,
                    upstream_path=upstream_path,
                    state=state,
                    lease=lease,
                    pause_state=pause_state,
                )
            )
        )
    # Cancel only this shielded waiter on disconnect. Other waiters and the
    # shared operation remain alive, including capture after all waiters leave.
    waiting = asyncio.ensure_future(asyncio.shield(lease.future))
    disconnected = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait((waiting, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if waiting in done:
            try:
                completion = waiting.result()
            except ValueError as exc:
                raise credit_http_error(exc) from exc
            return JSONResponse(
                content=completion.body,
                status_code=completion.status_code,
                headers={
                    "X-AGL-Event-ID": completion.event_id,
                    "X-AGL-Event-Sequence": str(completion.sequence),
                },
            )
        disconnected.result()
        return Response(status_code=499)
    finally:
        for task in (waiting, disconnected):
            if not task.done():
                task.cancel()
        await asyncio.gather(waiting, disconnected, return_exceptions=True)


# --- Management routes ------------------------------------------------------


class PauseRequest(BaseModel):
    retry_after_seconds: int = 5
    reason: str | None = None


class PauseStateResponse(BaseModel):
    paused: bool
    retry_after_seconds: int
    reason: str | None
    inflight: int


@management_router.post("/pause", response_model=PauseStateResponse)
async def pause_proxy(body: PauseRequest, request: Request) -> PauseStateResponse:
    """Pause new proxy forwarding requests while existing in-flight requests drain."""
    state = _get_pause_state(request)
    async with state.lock:
        state.paused = True
        state.retry_after_seconds = body.retry_after_seconds
        state.reason = body.reason
        return PauseStateResponse(
            paused=state.paused,
            retry_after_seconds=state.retry_after_seconds,
            reason=state.reason,
            inflight=state.inflight,
        )


@management_router.post("/resume", response_model=PauseStateResponse)
async def resume_proxy(request: Request) -> PauseStateResponse:
    """Resume proxy forwarding after a pause."""
    state = _get_pause_state(request)
    async with state.lock:
        state.paused = False
        state.reason = None
        return PauseStateResponse(
            paused=state.paused,
            retry_after_seconds=state.retry_after_seconds,
            reason=state.reason,
            inflight=state.inflight,
        )


@management_router.get("/state", response_model=PauseStateResponse)
async def proxy_state(request: Request) -> PauseStateResponse:
    """Return the proxy pause state and in-flight request count."""
    state = _get_pause_state(request)
    async with state.lock:
        return PauseStateResponse(
            paused=state.paused,
            retry_after_seconds=state.retry_after_seconds,
            reason=state.reason,
            inflight=state.inflight,
        )

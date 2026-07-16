"""Proxy forwarding and pause/drain management routes."""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from agl_lite.server.proxy import NoServersError, ProxyPauseState, ProxyRouter, forward_request
from agl_lite.server.store import _rollouts

log = structlog.get_logger()

router = APIRouter(tags=["gateway"])
management_router = APIRouter(tags=["gateway-management"], prefix="/proxy")


def _get_pause_state(request: Request) -> ProxyPauseState:
    state: ProxyPauseState | None = getattr(request.app.state, "proxy_pause_state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Gateway pause state not configured")
    return state


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

    # Select server.
    model_name = proxy_router.model_name
    try:
        server = proxy_router.select_server(model_name, rollout_id)
    except NoServersError:
        raise HTTPException(status_code=503, detail=f"No servers available for model '{model_name}'") from None

    prepared_body = proxy_router.prepare_body(body, mode)

    # Server endpoint includes the OpenAI base path (e.g., "http://vllm:8000/v1").
    return await forward_request(
        client=http_client,
        server=server,
        body=prepared_body,
        upstream_path=upstream_path,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        pause_state=pause_state,
    )


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

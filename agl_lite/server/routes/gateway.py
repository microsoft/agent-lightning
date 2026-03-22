"""Gateway routes — LLM reverse proxy + agent event ingestion.

Handles paths under /rollout/{rid}/attempt/{aid}/...
- /v1/... → LLM proxy (forwarded to model server)
- /events → agent event ingestion (rewards, custom events)
"""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.exceptions import HTTPException

from agl_lite.schemas.api import PostEventRequest
from agl_lite.schemas.event import Event
from agl_lite.store.memory import InMemoryStore

log = structlog.get_logger()

router = APIRouter(tags=["gateway"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.post("/rollout/{rollout_id}/attempt/{attempt_id}/events", response_model=Event)
async def post_event(rollout_id: str, attempt_id: str, body: PostEventRequest, request: Request) -> Event:
    """Agent posts an event (reward, custom type)."""
    store = _get_store(request)
    if not store.rollout_exists(rollout_id):
        raise HTTPException(status_code=404, detail=f"Rollout '{rollout_id}' not found")
    return store.add_event(rollout_id, attempt_id, body.event_type, body.data)


@router.api_route(
    "/rollout/{rollout_id}/attempt/{attempt_id}/v1/{path:path}",
    methods=["POST", "GET"],
)
async def llm_proxy(rollout_id: str, attempt_id: str, path: str, request: Request) -> Response:
    """LLM reverse proxy — forwards to model server, captures events."""
    from agl_lite.gateway.proxy import forward_request
    from agl_lite.gateway.router import GatewayRouter, NoServersError

    store = _get_store(request)

    # Validate rollout exists.
    if not store.rollout_exists(rollout_id):
        raise HTTPException(status_code=404, detail=f"Rollout '{rollout_id}' not found")

    # Get gateway router and httpx client from app state.
    gateway_router: GatewayRouter | None = getattr(request.app.state, "gateway_router", None)
    http_client = getattr(request.app.state, "http_client", None)

    if gateway_router is None or http_client is None:
        raise HTTPException(status_code=503, detail="Gateway not configured")

    # Read and parse request body.
    raw_body = await request.body()
    try:
        body = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body") from None

    # Extract model from body.
    model_in = body.get("model", "")
    if not model_in:
        raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

    # Route: model_in → model_out + param adjustments.
    model_out, route = gateway_router.resolve(model_in)

    # Select server.
    try:
        server = gateway_router.select_server(model_out)
    except NoServersError:
        raise HTTPException(status_code=503, detail=f"No servers available for model '{model_out}'") from None

    # Prepare body (rewrite model, apply params).
    prepared_body = gateway_router.prepare_body(body, model_out, route)

    # Forward request. Path is already relative (e.g., "chat/completions").
    # Server endpoint includes the base path (e.g., "http://vllm:8000/v1").
    return await forward_request(
        client=http_client,
        server=server,
        path=path,
        body=prepared_body,
        store=store,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        original_body=prepared_body,  # capture what was actually sent to model server
    )

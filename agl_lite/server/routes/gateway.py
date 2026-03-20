"""Gateway routes — LLM reverse proxy + agent event ingestion.

Handles paths under /rollout/{rid}/attempt/{aid}/...
- /v1/... → LLM proxy (forwarded to model server)
- /events → agent event ingestion (rewards, custom events)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.api import PostEventRequest
from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.event import Event
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["gateway"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.post("/rollout/{rollout_id}/attempt/{attempt_id}/events", response_model=Event)
async def post_event(rollout_id: str, attempt_id: str, body: PostEventRequest, request: Request) -> Event:
    """Agent posts an event (reward, custom type)."""
    store = _get_store(request)
    try:
        return store.add_event(rollout_id, attempt_id, body.event_type, body.data)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.api_route(
    "/rollout/{rollout_id}/attempt/{attempt_id}/v1/{path:path}",
    methods=["POST", "GET"],
)
async def llm_proxy(rollout_id: str, attempt_id: str, path: str, request: Request) -> Any:
    """LLM reverse proxy — forwards to model server, captures events.

    TODO: Implement in Phase 2.4/2.5 (gateway module).
    """
    store = _get_store(request)

    # Validate rollout exists.
    if not store.rollout_exists(rollout_id):
        raise HTTPException(status_code=404, detail=f"Rollout '{rollout_id}' not found")

    # Placeholder — will be replaced by gateway.proxy logic.
    raise HTTPException(status_code=501, detail="LLM proxy not yet implemented")

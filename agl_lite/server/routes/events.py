"""Event API routes (read-only).

Events are written through two paths, neither on /api:
  1. Agents post rewards/custom events via POST /rollout/{rid}/attempt/{aid}/events (gateway router)
  2. Gateway auto-captures model_request events internally during LLM proxying

This router provides read access for the algorithm to query collected events.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.event import Event
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["events"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.get("/events", response_model=list[Event])
async def query_events(
    request: Request,
    rollout_id: str,
    attempt_id: str | None = None,
    event_type: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[Event]:
    """Query events for a rollout. Smart attempt_id resolution if not specified."""
    store = _get_store(request)
    try:
        return store.query_events(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

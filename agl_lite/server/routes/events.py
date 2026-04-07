"""Event API routes (read-only).

Events are written through two paths, neither on /api:
  1. Agents post rewards/custom events via POST /rollout/{rid}/attempt/{aid}/events (gateway router)
  2. Gateway auto-captures model_request events internally during LLM proxying

This router provides read access for the algorithm to query collected events.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.event import Event
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["events"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


def _trim_model_request(data: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt_token_ids and response_token_ids from a model_request event.

    After gateway assembly, both streaming and non-streaming responses share the
    same dict shape with prompt_token_ids at top level and token_ids per choice.
    Legacy raw-chunk format (list) is also supported for backward compatibility.
    """
    resp = data.get("response")
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []

    if isinstance(resp, dict):
        # Assembled dict (streaming or non-streaming) — unified shape.
        prompt_token_ids = resp.get("prompt_token_ids", [])
        choices = resp.get("choices", [])
        if choices:
            response_token_ids = choices[0].get("token_ids", [])
    elif isinstance(resp, list):
        # Legacy: raw SSE chunks (pre-assembly format, backward compat).
        for chunk in resp:
            if not prompt_token_ids and chunk.get("prompt_token_ids"):
                prompt_token_ids = chunk["prompt_token_ids"]
            choices = chunk.get("choices", [])
            if choices:
                tids = choices[0].get("token_ids")
                if tids:
                    response_token_ids.extend(tids)

    srv = data.get("server", {})
    return {
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "server": {"model": srv.get("model"), "version": srv.get("version")},
    }


def _trim_reward(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only the scalar value from a reward event."""
    return {"value": data.get("value")}


def _to_triplet_format(event: Event) -> Event:
    """Trim event data for triplet consumption.

    - model_request: extract prompt_token_ids + response_token_ids only
    - reward: keep only the scalar value
    - other event types: pass through unchanged
    """
    if event.event_type == "model_request":
        trimmed = _trim_model_request(event.data)
        return event.model_copy(update={"data": trimmed})
    elif event.event_type == "reward":
        trimmed = _trim_reward(event.data)
        return event.model_copy(update={"data": trimmed})
    return event


@router.get("/events", response_model=list[Event])
async def query_events(
    request: Request,
    rollout_id: str,
    attempt_id: str | None = None,
    event_type: str | None = None,
    limit: int = 1000,
    offset: int = 0,
    format: str | None = Query(None, description="Set to 'triplet' to trim events for RL training"),
) -> list[Event]:
    """Query events for a rollout. Smart attempt_id resolution if not specified."""
    store = _get_store(request)
    try:
        events = store.query_events(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    if format == "triplet":
        events = [_to_triplet_format(e) for e in events]
    return events

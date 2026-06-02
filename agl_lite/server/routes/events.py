"""Event API routes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException

from agl_lite.schemas import DEFAULT_ATTEMPT_ID, Event, EventCreate
from agl_lite.server.store import _events, _rollouts

router = APIRouter(tags=["events"])


def _not_found(rollout_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")


def record_event(rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> Event:
    """Append a single event for an existing rollout."""
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    event = Event(
        event_type=event_type,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        timestamp=time.time(),
        data=data,
    )

    rid_events = _events[rollout_id]
    if attempt_id not in rid_events:
        rid_events[attempt_id] = []
    rid_events[attempt_id].append(event)
    return event


def _query_events(
    rollout_id: str,
    *,
    event_type: str | None = None,
) -> list[Event]:
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    rollout = _rollouts[rollout_id]
    attempt_id = rollout.status.last_attempt_id or DEFAULT_ATTEMPT_ID
    rid_events = _events.get(rollout_id, {})
    events = rid_events.get(attempt_id, [])
    if event_type is not None:
        events = [event for event in events if event.event_type == event_type]

    return events


def _trim_model_request(data: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt_token_ids and response_token_ids from a model_request event.

    Non-streaming gateway responses use a dict shape with prompt_token_ids at
    top level and token_ids per choice. Legacy raw-chunk format (list) is also
    supported for backward compatibility.
    """
    resp = data.get("response")
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []

    if isinstance(resp, dict):
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
    trimmed = {"value": data.get("value")}
    for key in ("source", "reason"):
        if key in data:
            trimmed[key] = data[key]
    return trimmed


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


@router.post("/rollouts/{rollout_id}/attempt/{attempt_id}/events", response_model=Event)
async def post_event(rollout_id: str, body: EventCreate, attempt_id: str) -> Event:
    """Post an event for one rollout attempt."""
    return record_event(rollout_id, attempt_id, body.event_type, body.data)


@router.get("/rollouts/{rollout_id}/events", response_model=list[Event])
async def query_events(
    rollout_id: str,
    event_type: str | None = None,
    format: str | None = Query(None, description="Set to 'triplet' to trim events for RL training"),
) -> list[Event]:
    """Query events for the default rollout attempt."""
    events = _query_events(
        rollout_id=rollout_id,
        event_type=event_type,
    )
    if format == "triplet":
        events = [_to_triplet_format(e) for e in events]
    return events

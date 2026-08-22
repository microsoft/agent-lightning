# Copyright (c) Microsoft. All rights reserved.

"""Event API routes."""

from __future__ import annotations

import math
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException

from agentlightning.schemas import DEFAULT_ATTEMPT_ID, Event, EventCreate
from agentlightning.server.store import _events, _rollouts

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


def _extract_choice_log_probs(choice: dict[str, Any]) -> list[float] | None:
    """Extract chosen-token logprobs from a single choice.

    Returns the per-token logprobs, or None when they are missing or unusable
    (no logprobs field, unrecognized schema, or any non-finite/non-float value).
    Never raises: a malformed response yields None so the triplet query stays a
    successful HTTP response and the training bridge drops the sample.
    """
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None

    raw: list[Any]
    if isinstance(lp.get("content"), list):
        # OpenAI chat schema: logprobs.content -> [{"logprob": float, ...}, ...]
        raw = []
        for item in lp["content"]:
            if not isinstance(item, dict) or "logprob" not in item:
                return None
            raw.append(item["logprob"])
    elif isinstance(lp.get("token_logprobs"), list):
        # Completions schema: logprobs.token_logprobs -> [float, ...]
        raw = list(lp["token_logprobs"])
    else:
        return None

    out: list[float] = []
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _trim_model_request(data: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt_token_ids and response_token_ids from a model_request event.

    Non-streaming gateway responses use a dict shape with prompt_token_ids at
    top level for chat completions or per choice for completions, and token_ids
    per choice. Legacy raw-chunk format (list) is also supported for backward
    compatibility.
    """
    resp = data.get("response")
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []
    response_log_probs: list[float] | None = None

    if isinstance(resp, dict):
        prompt_token_ids = resp.get("prompt_token_ids", [])
        choices = resp.get("choices", [])
        if choices:
            if not prompt_token_ids:
                prompt_token_ids = choices[0].get("prompt_token_ids", [])
            response_token_ids = choices[0].get("token_ids", [])
            response_log_probs = _extract_choice_log_probs(choices[0])
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
    trimmed = {
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "response_log_probs": response_log_probs,
        "server": {"model": srv.get("model"), "version": srv.get("version")},
    }
    for key in ("http_status", "status"):
        if key in data:
            trimmed[key] = data[key]
    if isinstance(resp, dict) and "error" in resp:
        trimmed["error"] = resp["error"]
    return trimmed


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


def _dedupe_model_requests_by_prompt_token_ids(events: list[Event]) -> list[Event]:
    """Keep only the last model_request event for each prompt_token_ids key."""
    last_index_by_prompt: dict[tuple[Any, ...], int] = {}
    for index, event in enumerate(events):
        if event.event_type != "model_request":
            continue
        prompt_token_ids = event.data.get("prompt_token_ids", [])
        prompt_key = tuple(prompt_token_ids) if isinstance(prompt_token_ids, list) else ()
        last_index_by_prompt[prompt_key] = index

    last_indexes = set(last_index_by_prompt.values())
    return [event for index, event in enumerate(events) if event.event_type != "model_request" or index in last_indexes]


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
        events = _dedupe_model_requests_by_prompt_token_ids(events)
    return events

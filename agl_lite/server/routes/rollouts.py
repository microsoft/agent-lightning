# Copyright (c) Microsoft. All rights reserved.

"""Rollout API routes."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from agl_lite.schemas import (
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    Rollout,
    RolloutConfig,
    RolloutCreate,
    RolloutLifecycleStatus,
    RolloutMetadata,
    RolloutPatch,
    RolloutState,
)
from agl_lite.server.store import _events, _rollouts, _terminal_order

router = APIRouter(tags=["rollouts"])


class RolloutDetail(BaseModel):
    """Rollout with attempt list."""

    rollout: Rollout
    attempts: list[str]


class TerminalRolloutItem(BaseModel):
    """Lightweight projection of a terminal rollout (no input/config payload)."""

    rollout_id: str
    state: RolloutState
    data_id: str
    is_train: bool


class TerminalRolloutsPage(BaseModel):
    """A page of terminal rollouts plus the cursor to fetch the next page."""

    items: list[TerminalRolloutItem]
    next_after: int
    total_terminal: int


def _not_found(rollout_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")


def _invalid_transition(rollout_id: str, from_status: str, to_status: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"Rollout {rollout_id}: cannot transition {from_status} -> {to_status}",
    )


def _get_rollout(rollout_id: str) -> Rollout:
    try:
        return _rollouts[rollout_id]
    except KeyError:
        raise _not_found(rollout_id) from None


def _metadata_from_request(req: RolloutCreate) -> RolloutMetadata:
    if isinstance(req.metadata, dict):
        return RolloutMetadata(**req.metadata)
    if req.metadata is not None:
        return req.metadata
    return RolloutMetadata()


def _list_attempts(rollout_id: str) -> list[str]:
    if rollout_id not in _rollouts:
        raise _not_found(rollout_id)

    rid_events = _events.get(rollout_id, {})
    if not rid_events:
        return []

    return sorted(
        rid_events.keys(),
        key=lambda attempt_id: rid_events[attempt_id][0].timestamp if rid_events[attempt_id] else float("inf"),
    )


@router.post("/rollouts", status_code=201, response_model=list[Rollout])
async def enqueue_rollouts(body: list[RolloutCreate]) -> list[Rollout]:
    """Enqueue rollouts. Each item in the list is self-contained.

    If a request carries a `rollout_id` that already exists, the existing
    rollout is returned unchanged (its events are left intact), making creation
    idempotent so callers can pre-assign ids and retry safely.
    """
    results: list[Rollout] = []
    for req in body:
        if req.rollout_id is not None and req.rollout_id in _rollouts:
            results.append(_rollouts[req.rollout_id])
            continue
        now = time.time()
        rollout_id = req.rollout_id or uuid.uuid4().hex
        metadata = _metadata_from_request(req)
        rollout = Rollout(
            rollout_id=rollout_id,
            input=req.input,
            is_train=req.is_train,
            config=req.config or RolloutConfig(),
            metadata=metadata,
            status=RolloutLifecycleStatus(created_at=now, updated_at=now),
        )
        _rollouts[rollout_id] = rollout
        _events[rollout_id] = {}
        results.append(rollout)
    return results


@router.get("/rollouts", response_model=list[Rollout])
async def list_rollouts(
    state_in: Annotated[list[RolloutState], Query()],
    limit: int = 500,
) -> list[Rollout]:
    """List rollouts by lifecycle states."""
    states = set(state_in)
    matches = [rollout for rollout in _rollouts.values() if rollout.status.state in states]
    return matches[:limit]


def _data_id_of(rollout: Rollout) -> str:
    inp = rollout.input
    if isinstance(inp, dict):
        return str(inp.get("data_id") or inp.get("instance_id") or "")
    return ""


@router.get("/rollouts/terminal", response_model=TerminalRolloutsPage)
async def list_terminal_rollouts(after: int = 0, limit: int = 1000) -> TerminalRolloutsPage:
    """Cursor-paginate terminal rollouts in completion order (lightweight projection).

    `after` is an index into the append-only completion log; pass back `next_after`
    to fetch only rollouts that completed since the last call. Out-of-order
    completions are never missed because the log is append-on-terminal-transition.
    Returns only id/state/data_id/is_train — fetch events per rollout for details.
    """
    if after < 0:
        after = 0
    if limit < 1:
        limit = 1
    total = len(_terminal_order)
    slice_ids = _terminal_order[after : after + limit]
    items: list[TerminalRolloutItem] = []
    for rid in slice_ids:
        rollout = _rollouts.get(rid)
        if rollout is None:
            continue
        items.append(
            TerminalRolloutItem(
                rollout_id=rid,
                state=rollout.status.state,
                data_id=_data_id_of(rollout),
                is_train=rollout.is_train,
            )
        )
    return TerminalRolloutsPage(items=items, next_after=after + len(slice_ids), total_terminal=total)


@router.get("/rollouts/{rollout_id}", response_model=RolloutDetail)
async def get_rollout(rollout_id: str) -> RolloutDetail:
    """Get a single rollout with its attempt list."""
    rollout = _get_rollout(rollout_id)
    attempts = _list_attempts(rollout_id)
    return RolloutDetail(rollout=rollout, attempts=attempts)


@router.patch("/rollouts/{rollout_id}", response_model=Rollout)
async def patch_rollout(rollout_id: str, body: RolloutPatch) -> Rollout:
    """Patch the lifecycle status of a rollout."""
    rollout = _get_rollout(rollout_id)
    updates = body.status.model_dump(exclude_unset=True) if body.status is not None else {}

    if not updates:
        return rollout

    if "state" in updates:
        new_state = updates["state"]
        if new_state not in VALID_TRANSITIONS[rollout.status.state]:
            raise _invalid_transition(rollout_id, rollout.status.state, str(new_state))

    updated_status = rollout.status.model_copy(
        update={
            **updates,
            "version": rollout.status.version + 1,
            "updated_at": time.time(),
        }
    )

    updated = rollout.model_copy(
        update={
            "status": updated_status,
        }
    )
    _rollouts[rollout_id] = updated
    if "state" in updates and updated_status.state in TERMINAL_STATES:
        # One-way terminal transition (guarded above) => append exactly once.
        _terminal_order.append(rollout_id)
    return updated


@router.delete("/rollouts/{rollout_id}", status_code=204)
async def delete_rollout(rollout_id: str) -> None:
    """Delete a rollout and its events. Idempotent: missing id is a no-op."""
    _rollouts.pop(rollout_id, None)
    _events.pop(rollout_id, None)



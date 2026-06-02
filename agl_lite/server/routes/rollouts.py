"""Rollout API routes."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from agl_lite.schemas import (
    VALID_TRANSITIONS,
    Rollout,
    RolloutConfig,
    RolloutCreate,
    RolloutLifecycleStatus,
    RolloutMetadata,
    RolloutPatch,
    RolloutState,
)
from agl_lite.server.store import _events, _rollouts

router = APIRouter(tags=["rollouts"])


class RolloutDetail(BaseModel):
    """Rollout with attempt list."""

    rollout: Rollout
    attempts: list[str]


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
    """Enqueue rollouts. Each item in the list is self-contained."""
    results: list[Rollout] = []
    for req in body:
        now = time.time()
        rollout_id = uuid.uuid4().hex
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
    return updated



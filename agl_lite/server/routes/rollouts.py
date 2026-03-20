"""Rollout API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from agl_lite.schemas.api import EnqueueBatchRequest, PatchRolloutRequest
from agl_lite.schemas.errors import InvalidTransitionError, NotFoundError
from agl_lite.schemas.rollout import Rollout, RolloutStatus
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["rollouts"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


class RolloutDetail(BaseModel):
    """Rollout with attempt list."""

    rollout: Rollout
    attempts: list[str]


@router.post("/rollouts", status_code=201, response_model=list[Rollout])
async def enqueue_rollouts(body: EnqueueBatchRequest, request: Request) -> list[Rollout]:
    """Enqueue rollouts. Each item in the list is self-contained."""
    store = _get_store(request)
    return store.enqueue_rollouts(body.rollouts)


@router.get("/rollouts", response_model=list[Rollout])
async def query_rollouts(
    request: Request,
    ids: str | None = None,
    status: str | None = None,
    cancel_requested: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Rollout]:
    """Query rollouts with optional filters."""
    store = _get_store(request)
    id_list = ids.split(",") if ids else None
    status_list = [RolloutStatus(s.strip()) for s in status.split(",")] if status else None
    return store.query_rollouts(
        ids=id_list,
        status_in=status_list,
        cancel_requested=cancel_requested,
        limit=limit,
        offset=offset,
    )


@router.get("/rollouts/{rollout_id}", response_model=RolloutDetail)
async def get_rollout(rollout_id: str, request: Request) -> RolloutDetail:
    """Get a single rollout with its attempt list."""
    store = _get_store(request)
    try:
        rollout = store.get_rollout(rollout_id)
        attempts = store.list_attempts(rollout_id)
        return RolloutDetail(rollout=rollout, attempts=attempts)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.patch("/rollouts/{rollout_id}", response_model=Rollout)
async def patch_rollout(rollout_id: str, body: PatchRolloutRequest, request: Request) -> Rollout:
    """Partial update a rollout."""
    store = _get_store(request)
    try:
        return store.update_rollout(rollout_id, body)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None


@router.post("/rollouts/{rollout_id}/cancel", response_model=Rollout)
async def cancel_rollout(rollout_id: str, request: Request) -> Rollout:
    """Set cancel_requested flag on a rollout."""
    store = _get_store(request)
    try:
        return store.cancel_rollout(rollout_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None

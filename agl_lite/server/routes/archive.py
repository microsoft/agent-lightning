"""Archive API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.api import ArchiveRequest, ArchiveResult
from agl_lite.schemas.errors import NotFoundError
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["archive"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.post("/rollouts/archive", response_model=ArchiveResult)
async def archive_rollouts(body: ArchiveRequest, request: Request) -> ArchiveResult:
    """Archive and purge terminal rollouts."""
    store = _get_store(request)
    try:
        return store.archive_rollouts(body.rollout_ids, backend=body.backend)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None

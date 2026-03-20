"""Resource API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.resources import ResourcesUpdate
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["resources"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.post("/resources", status_code=201, response_model=ResourcesUpdate)
async def add_resources(body: dict, request: Request) -> ResourcesUpdate:
    """Add a new resource snapshot. Returns the snapshot with generated ID."""
    store = _get_store(request)
    return store.add_resources(body)


@router.get("/resources/latest", response_model=ResourcesUpdate | None)
async def get_latest_resources(request: Request) -> ResourcesUpdate | None:
    """Get the most recently added resource snapshot."""
    store = _get_store(request)
    return store.get_latest_resources()


@router.get("/resources/{resources_id}", response_model=ResourcesUpdate)
async def get_resources(resources_id: str, request: Request) -> ResourcesUpdate:
    """Get a resource snapshot by ID."""
    store = _get_store(request)
    try:
        return store.get_resources(resources_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

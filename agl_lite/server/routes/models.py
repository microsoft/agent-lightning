"""Model server API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from agl_lite.schemas.api import DeleteModelServersRequest, RegisterModelRequest
from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.model_server import ModelServer
from agl_lite.store.memory import InMemoryStore

router = APIRouter(tags=["models"])


def _get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


@router.post("/models", status_code=201, response_model=list[ModelServer])
async def register_models(body: list[RegisterModelRequest], request: Request) -> list[ModelServer]:
    """Register model server(s). Upsert by (model, endpoint)."""
    store = _get_store(request)
    results: list[ModelServer] = []
    for item in body:
        m = store.register_model(
            model=item.model,
            endpoint=item.endpoint,
            version=item.version,
            token=item.token,
        )
        results.append(m)
    return results


@router.get("/models", response_model=list[ModelServer])
async def list_models(request: Request) -> list[ModelServer]:
    """List all registered model servers (flat list)."""
    store = _get_store(request)
    return store.list_models()


@router.delete("/models/{model}")
async def delete_model_servers(
    model: str, request: Request, body: DeleteModelServersRequest | None = None
) -> dict[str, str]:
    """Remove servers for a model. Optional body with specific endpoints."""
    store = _get_store(request)
    try:
        endpoints = body.endpoints if body and body.endpoints else None
        store.remove_model_servers(model, endpoints=endpoints)
        return {"status": "ok"}
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.delete("/models")
async def delete_all_models(request: Request) -> dict[str, str]:
    """Remove all model servers."""
    store = _get_store(request)
    store.remove_all_models()
    return {"status": "ok"}

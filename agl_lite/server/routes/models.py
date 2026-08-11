# Copyright (c) Microsoft. All rights reserved.

"""Model server API routes."""

from __future__ import annotations

from fastapi import APIRouter

from agl_lite.schemas import Model
from agl_lite.server.store import _models

router = APIRouter(tags=["models"])


@router.post("/models", status_code=201, response_model=list[Model])
async def register_models(body: list[Model]) -> list[Model]:
    """Register model server(s). Upsert by (model, endpoint)."""
    results: list[Model] = []
    for req in body:
        if req.model not in _models:
            _models[req.model] = {}
        _models[req.model][req.endpoint] = req
        results.append(req)
    return results


@router.delete("/models")
async def delete_all_models() -> dict[str, str]:
    """Remove all model servers."""
    _models.clear()
    return {"status": "ok"}

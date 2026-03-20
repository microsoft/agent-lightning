"""FastAPI application — lifespan, mount routes, wire store + gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from agl_lite.server.auth import build_auth_dependency
from agl_lite.server.config import ServerSettings
from agl_lite.server.routes import archive, events, gateway, models, resources, rollouts
from agl_lite.store.memory import InMemoryStore


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = ServerSettings()

    store = InMemoryStore()
    verify_key = build_auth_dependency(settings.agl_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Store is already created — attach to app.state for route access.
        app.state.store = store
        app.state.settings = settings
        yield

    app = FastAPI(title="agl-lite", version="0.1.0", lifespan=lifespan)

    # Health check — no auth.
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # Store API routes — all require auth.
    app.include_router(rollouts.router, prefix="/api", dependencies=[Depends(verify_key)])
    app.include_router(events.router, prefix="/api", dependencies=[Depends(verify_key)])
    app.include_router(models.router, prefix="/api", dependencies=[Depends(verify_key)])
    app.include_router(resources.router, prefix="/api", dependencies=[Depends(verify_key)])
    app.include_router(archive.router, prefix="/api", dependencies=[Depends(verify_key)])

    # Gateway routes (LLM proxy + event ingestion) — require auth.
    app.include_router(gateway.router, dependencies=[Depends(verify_key)])

    return app

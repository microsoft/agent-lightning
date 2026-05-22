"""FastAPI application — lifespan, mount routes, wire store + gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import Depends, FastAPI

from agl_lite.gateway.config import GatewayConfig, load_config
from agl_lite.gateway.proxy import GatewayPauseState
from agl_lite.gateway.router import GatewayRouter
from agl_lite.hooks import RolloutHooks, load_hooks
from agl_lite.server.auth import (
    build_admin_auth_dependency,
    build_auth_dependency,
    validate_admin_key_combo,
)
from agl_lite.server.config import ServerSettings
from agl_lite.server.routes import archive, events, gateway, models, resources, rollouts
from agl_lite.store.memory import InMemoryStore

log = structlog.get_logger()


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = ServerSettings()

    if not settings.key:
        log.warning("AGL_KEY not set — authentication disabled. Do not use in production.")

    # Fail loudly on inconsistent (key, admin_key) combinations. See
    # agl_lite/server/auth.py:validate_admin_key_combo for the matrix.
    validate_admin_key_combo(settings.key, settings.admin_key)

    # Load rollout lifecycle hooks (optional).
    hooks: RolloutHooks | None = None
    if settings.hooks:
        hooks = load_hooks(settings.hooks)
        log.info("Rollout hooks loaded", hooks_class=type(hooks).__name__, path=settings.hooks)

    store = InMemoryStore(hooks=hooks, log_dir=settings.log_dir)

    # Call on_startup after store is ready — hook may need store reference.
    if hooks:
        hooks.on_startup(store)
        log.info("Hook on_startup complete", hooks_class=type(hooks).__name__)
    verify_key = build_auth_dependency(settings.key)
    verify_admin_key = build_admin_auth_dependency(settings.admin_key, settings.key)

    # Load gateway config.
    if settings.gateway_config:
        gateway_config = load_config(settings.gateway_config)
        log.info("Gateway config loaded", num_routes=len(gateway_config.routes))
    else:
        gateway_config = GatewayConfig()
        log.warning("No gateway config — all model names pass through without routing or parameter adjustment.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store
        app.state.settings = settings

        # Gateway pause/drain state — initialized lazily inside the event
        # loop because GatewayPauseState owns asyncio primitives.
        app.state.gateway_pause_state = GatewayPauseState()

        # Gateway router + shared httpx client (connection pooling).
        app.state.gateway_router = GatewayRouter(gateway_config, store)
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=300.0)) as client:
            app.state.http_client = client
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

    # Gateway routes (LLM proxy + event ingestion) — require agent-facing auth.
    app.include_router(gateway.router, dependencies=[Depends(verify_key)])

    # Admin gateway routes (pause/resume/state) — require admin-only auth so
    # agent pods carrying the agent-facing key cannot flip pause flags.
    app.include_router(gateway.admin_router, dependencies=[Depends(verify_admin_key)])

    return app

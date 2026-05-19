"""API key authentication dependency."""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import HTTPException


def build_auth_dependency(agl_key: str):
    """Return a dependency that validates the API key.

    If agl_key is empty, auth is disabled (all requests pass).
    """

    async def verify_key(request: Request) -> None:
        if not agl_key:
            return  # auth disabled

        # Check Authorization: Bearer <key>
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == agl_key:
                return

        # Check x-api-key: <key> (Anthropic SDK)
        x_api_key = request.headers.get("x-api-key", "")
        if x_api_key == agl_key:
            return

        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_key


def build_admin_auth_dependency(admin_key: str | None, agl_key: str):
    """Return a dependency that validates the admin (trainer-only) API key.

    Auth matrix (per docs §3.2):

      agl_key  | admin_key | behavior
      -------- | --------- | ------------------------------------------------
      empty    | empty     | auth disabled (local dev); admin open as well
      not None | None      | rejected at startup (see validate_admin_key_combo)
      not None | not None  | admin endpoints require admin_key (separate from agl_key)

    The dependency itself only enforces "admin_key must match when set". The
    "fail loudly if admin_key is missing while agl_key is configured" check
    is a startup-time validation; see ``validate_admin_key_combo``.
    """

    async def verify_admin_key(request: Request) -> None:
        # Auth fully disabled: agl_key is empty AND admin_key is empty.
        if not agl_key and not admin_key:
            return

        if not admin_key:
            # Startup validation should have prevented this combination, but
            # defensively refuse all admin traffic if it ever reaches here.
            raise HTTPException(
                status_code=503,
                detail="Admin endpoint disabled: AGL_ADMIN_KEY not configured",
            )

        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == admin_key:
                return

        x_api_key = request.headers.get("x-admin-key", "")
        if x_api_key == admin_key:
            return

        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")

    return verify_admin_key


def validate_admin_key_combo(agl_key: str, admin_key: str | None) -> None:
    """Enforce the (agl_key, admin_key) combinations allowed at startup.

    Raises ``ValueError`` for forbidden combinations so the server fails to
    start instead of silently exposing admin endpoints under a shared key.
    """
    if agl_key and not admin_key:
        raise ValueError(
            "AGL_ADMIN_KEY is required when AGL_KEY is set. Admin endpoints "
            "(e.g. /admin/gateway/pause) must not be reachable with the same "
            "credential agent pods use to call the LLM proxy."
        )
    if (not agl_key) and admin_key:
        raise ValueError(
            "AGL_ADMIN_KEY is set but AGL_KEY is empty. This creates a confusing "
            "half-authed configuration (open agent API, locked admin API). "
            "Either set both keys or unset both."
        )

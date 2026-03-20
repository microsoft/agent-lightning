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

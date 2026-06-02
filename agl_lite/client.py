"""Thin httpx clients for agl-lite."""

from __future__ import annotations

from typing import Any

import httpx


def _headers_with_key(headers: httpx.Headers | dict[str, str] | None, key: str | None) -> dict[str, str]:
    merged = dict(headers or {})
    if key:
        merged["Authorization"] = f"Bearer {key}"
    return merged


class AglLiteAsyncClient(httpx.AsyncClient):
    """Async httpx client with optional bearer key."""

    def __init__(
        self,
        *,
        key: str | None = None,
        headers: httpx.Headers | dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            headers=_headers_with_key(headers, key),
            **kwargs,
        )


class AglLiteSyncClient(httpx.Client):
    """Sync httpx client with optional bearer key."""

    def __init__(
        self,
        *,
        key: str | None = None,
        headers: httpx.Headers | dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            headers=_headers_with_key(headers, key),
            **kwargs,
        )

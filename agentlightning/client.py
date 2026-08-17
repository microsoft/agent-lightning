# Copyright (c) Microsoft. All rights reserved.

"""Thin httpx clients for Agent Lightning."""

from __future__ import annotations

import time
from typing import Any

import httpx


def _headers_with_key(headers: httpx.Headers | dict[str, str] | None, key: str | None) -> dict[str, str]:
    merged = dict(headers or {})
    if key:
        merged["Authorization"] = f"Bearer {key}"
    return merged


class AgentLightningAsyncClient(httpx.AsyncClient):
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


class AgentLightningSyncClient(httpx.Client):
    """Sync httpx client with optional bearer key."""

    def __init__(
        self,
        *,
        key: str | None = None,
        headers: httpx.Headers | dict[str, str] | None = None,
        max_retries: int = 10,
        **kwargs: Any,
    ) -> None:
        self.max_retries = max_retries
        super().__init__(
            headers=_headers_with_key(headers, key),
            **kwargs,
        )

    def get(self, *args: Any, **kwargs: Any) -> httpx.Response:  # type: ignore[override]
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return super().get(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                print(f"GET failed (attempt {attempt + 1}/{self.max_retries + 1}): {exc}")
        assert last_exc is not None
        raise last_exc

    def post_with_retry(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """POST with retry + backoff, raising on non-2xx. Only for idempotent endpoints.

        Retries both transport errors and error status codes, so a transient 5xx
        is retried too. Callers get an already status-checked response back.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = super().post(*args, **kwargs)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                print(f"POST failed (attempt {attempt + 1}/{self.max_retries + 1}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt + 1), 30))
        assert last_exc is not None
        raise last_exc

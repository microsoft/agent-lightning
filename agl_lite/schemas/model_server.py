"""Model server schemas — inference server registry."""

from __future__ import annotations

from pydantic import BaseModel


class ModelServer(BaseModel):
    """A registered model inference server. Keyed by endpoint (natural key)."""

    endpoint: str  # e.g., "http://vllm-0:8000/v1" — the identity
    version: int  # training step (monotonically increasing)
    token: str | None = None  # optional auth token for gateway → model server
    created_at: float

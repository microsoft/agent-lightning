"""Model server schemas — inference server registry."""

from __future__ import annotations

from pydantic import BaseModel


class ModelServer(BaseModel):
    """A registered model inference server. Keyed by (model, endpoint)."""

    model: str  # grouping key for routing — e.g., "qwen-7b"
    endpoint: str  # e.g., "http://vllm-0:8000/v1"
    version: int  # training step — per server (supports online RL rolling updates)
    token: str | None = None  # optional auth token for gateway → model server
    created_at: float

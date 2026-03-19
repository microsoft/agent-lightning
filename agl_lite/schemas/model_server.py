"""Model server schemas — inference server registry."""

from __future__ import annotations

from pydantic import BaseModel


class ModelServer(BaseModel):
    """A registered model inference server."""

    model_id: str  # auto-generated UUID
    endpoint: str  # e.g., "http://vllm-0:8000/v1"
    version: int  # training step (monotonically increasing)
    created_at: float

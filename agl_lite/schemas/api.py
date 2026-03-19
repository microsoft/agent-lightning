"""API request/response body models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agl_lite.schemas.rollout import RolloutConfig, RolloutStatus

# --- Rollout API ---


class EnqueueRolloutRequest(BaseModel):
    """Single rollout enqueue."""

    input: dict[str, Any]
    config: RolloutConfig | None = None
    resources_id: str | None = None


class EnqueueBatchRequest(BaseModel):
    """Batch rollout enqueue. Batch-level config/resources_id apply to all; per-rollout fields override."""

    config: RolloutConfig
    resources_id: str | None = None
    rollouts: list[EnqueueRolloutRequest]


class PatchRolloutRequest(BaseModel):
    """Partial update for a rollout. Only fields present in the request body are applied.

    Use `model_dump(exclude_unset=True)` to get only the fields the caller explicitly set.
    """

    status: RolloutStatus | None = None
    job_name: str | None = None
    succeeded_attempt_id: str | None = None
    error_message: str | None = None


# --- Event API ---


class PostEventRequest(BaseModel):
    """Explicit event submission (reward, user-defined types)."""

    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)


# --- Model Server API ---


class RegisterModelRequest(BaseModel):
    """Register a model inference server."""

    endpoint: str
    version: int = 0


# --- Resource API ---


class AddResourcesRequest(BaseModel):
    """Add a new resource snapshot."""

    resources: dict[str, Any]  # {"job_defaults": {...}, "system_prompt": "...", ...}


# --- Archive API ---


class ArchiveRequest(BaseModel):
    """Archive and purge rollouts from hot store."""

    rollout_ids: list[str]
    backend: ArchiveBackend | None = None


class ArchiveBackend(BaseModel):
    """Archive backend configuration."""

    type: str = "jsonl"
    path: str  # file path, must end with .jsonl


class ArchiveResult(BaseModel):
    """Result of an archive operation."""

    archived: int
    purged: int
    path: str | None = None  # file path if backend was specified

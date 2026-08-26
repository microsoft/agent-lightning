# Copyright (c) Microsoft. All rights reserved.

"""Shared Pydantic schemas for Agent Lightning."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """Single event in a trajectory.

    Events are stored in insertion order per rollout. Position in the list
    is the identity — no separate event ID needed. Only two event types
    have well-known structure (model_request, reward). Everything else is
    opaque pass-through.
    """

    event_type: str  # "model_request", "reward", or any user-defined string
    rollout_id: str
    attempt_id: str
    timestamp: float  # assigned by store at write time
    data: dict[str, Any]  # event-type-specific payload


class EventCreate(BaseModel):
    """Input for appending a user-defined event."""

    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ModelRequestData(BaseModel):
    """Well-known structure for event_type='model_request'.

    Created automatically by the Gateway on every proxied LLM call.
    Not enforced by the Store — this is a documentation/validation helper.
    """

    model: str
    model_version: int | None = None  # training step of the serving model
    request: dict[str, Any]  # original request body (messages, temperature, etc.)
    adjusted_params: dict[str, Any] | None = None  # only if param adjustment changed anything
    response: dict[str, Any]  # full response body
    latency_ms: float | None = None
    http_status: int | None = None
    status: str = "ok"  # "ok" or "error"
    retry_count: int = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class RewardData(BaseModel):
    """Well-known structure for event_type='reward'.

    Reported by the environment, evaluator, or runner.
    Not enforced by the Store — this is a documentation/validation helper.
    """

    value: float  # scalar reward (required)
    message: str | None = None  # optional human-readable explanation
    source: str | None = None  # e.g. "agent" for explicit evaluator output, "fallback" for system fill-in
    reason: str | None = None  # optional machine-readable explanation


class Model(BaseModel):
    """A registered model inference endpoint. Keyed by (model, endpoint)."""

    model: str
    endpoint: str
    version: int = 0


class RolloutState(StrEnum):
    """Rollout lifecycle state values. Terminal states are final — no transitions out."""

    QUEUING = "queuing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Valid state transitions (Store-enforced).
VALID_TRANSITIONS: dict[RolloutState, set[RolloutState]] = {
    RolloutState.QUEUING: {RolloutState.RUNNING, RolloutState.FAILED},
    RolloutState.RUNNING: {RolloutState.SUCCEEDED, RolloutState.FAILED},
    # Terminal states — no transitions out.
    RolloutState.SUCCEEDED: set(),
    RolloutState.FAILED: set(),
}

TERMINAL_STATES: frozenset[RolloutState] = frozenset(
    {
        RolloutState.SUCCEEDED,
        RolloutState.FAILED,
    }
)

DEFAULT_ATTEMPT_ID = "0"


class K8sImageReadinessReport(BaseModel):
    """Controller report of images cached on every eligible K8s node."""

    images: list[str] = Field(default_factory=list)
    node_count: int = Field(ge=1)
    lease_seconds: float = Field(gt=0, le=300)


class K8sImageReadinessSnapshot(BaseModel):
    """Server-timestamped, leased K8s image inventory."""

    images: list[str] = Field(default_factory=list)
    node_count: int = Field(ge=1)
    observed_at: float
    expires_at: float


class RolloutLocalConfig(BaseModel):
    """Local runner config for a rollout."""

    agent_class: str | None = None
    env_map: dict[str, str] = Field(default_factory=dict)


class RolloutK8sConfig(BaseModel):
    """K8s runner config for a rollout."""

    job_template: str | None = None
    require_preloaded_images: bool = False


class RolloutConfig(BaseModel):
    """Controller-facing rollout config."""

    timeout_seconds: int = 3600
    local: RolloutLocalConfig | None = None
    k8s: RolloutK8sConfig | None = None


class RolloutMetadata(BaseModel):
    """Algorithm-facing batch context."""

    model_config = ConfigDict(extra="allow")

    batch_idx: int | None = None
    sample_idx_in_batch: int | None = None


class RolloutCreate(BaseModel):
    """Input for creating a rollout."""

    input: Any
    is_train: bool = True
    config: RolloutConfig | None = None
    metadata: RolloutMetadata | dict[str, Any] | None = None
    # A caller-supplied id makes rollout creation idempotent and safe to retry.
    rollout_id: str | None = None


class RolloutLifecycleStatus(BaseModel):
    """Controller-managed rollout lifecycle status."""

    state: RolloutState = RolloutState.QUEUING
    k8s_job_name: str | None = None
    last_attempt_id: str | None = None
    error_message: str | None = None
    version: int = 1
    created_at: float
    updated_at: float


class RolloutStatusPatch(BaseModel):
    """Partial update for the nested rollout status object."""

    model_config = ConfigDict(extra="forbid")

    state: RolloutState | None = None
    k8s_job_name: str | None = None
    last_attempt_id: str | None = None
    error_message: str | None = None


class RolloutPatch(BaseModel):
    """Partial rollout update. Only nested status may be patched."""

    status: RolloutStatusPatch | None = None


class Rollout(BaseModel):
    """Unit of work. Lifecycle managed by the K8s controller."""

    rollout_id: str
    input: Any
    is_train: bool = True
    config: RolloutConfig
    metadata: RolloutMetadata = Field(default_factory=RolloutMetadata)
    status: RolloutLifecycleStatus

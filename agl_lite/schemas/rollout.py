"""Rollout schemas — the fundamental unit of work in agl-lite."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RolloutStatus(StrEnum):
    """Rollout lifecycle states. Terminal states are final — no transitions out."""

    QUEUING = "queuing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    TERMINAL_FAILED = "terminal_failed"
    CANCELLED = "cancelled"


# Valid state transitions (Store-enforced).
VALID_TRANSITIONS: dict[RolloutStatus, set[RolloutStatus]] = {
    RolloutStatus.QUEUING: {RolloutStatus.RUNNING, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED},
    RolloutStatus.RUNNING: {RolloutStatus.SUCCEEDED, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED},
    # Terminal states — no transitions out.
    RolloutStatus.SUCCEEDED: set(),
    RolloutStatus.TERMINAL_FAILED: set(),
    RolloutStatus.CANCELLED: set(),
}

TERMINAL_STATUSES: frozenset[RolloutStatus] = frozenset(
    {
        RolloutStatus.SUCCEEDED,
        RolloutStatus.TERMINAL_FAILED,
        RolloutStatus.CANCELLED,
    }
)


class Mount(BaseModel):
    """Volume mount specification for agent containers."""

    name: str
    mount_path: str  # path inside container, e.g., "/data"
    source: str  # host path, PVC name, or ConfigMap name
    read_only: bool = True


class RolloutConfig(BaseModel):
    """Algorithm-facing config. Describes the containerized task.

    Named fields target the 'agent' container in the pod spec.
    The 'overrides' field provides per-rollout K8s overrides for the pod spec,
    including other containers via overrides.containers (name-matched merge).

    K8s pod-level infra details (nodeSelector, tolerations, etc.) come from
    the job_template in the resources snapshot — not here.
    """

    # Required — describe the agent container.
    image: str
    command: list[str] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    mount: list[Mount] = Field(default_factory=list)

    # Optional — execution policy.
    timeout: int | None = None  # seconds → K8s activeDeadlineSeconds
    max_retries: int | None = None  # retry count → K8s backoffLimit

    # Per-rollout K8s overrides (optional). Merged into the pod spec.
    # Use overrides.containers (list of {name: ..., ...}) for name-matched
    # merge into other containers (e.g., different scorer image per task).
    overrides: dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """Unit of work. Lifecycle managed by the K8s controller."""

    rollout_id: str
    status: RolloutStatus = RolloutStatus.QUEUING
    cancel_requested: bool = False

    input: Any  # task payload (delivered as AGL_TASK_INPUT env var, json-encoded)
    config: RolloutConfig
    resources_id: str | None = None  # links to immutable resource snapshot

    # Set by controller during lifecycle.
    job_name: str | None = None
    succeeded_attempt_id: str | None = None
    error_message: str | None = None

    # Concurrency control.
    version: int = 1  # optimistic locking — incremented on every update

    created_at: float
    updated_at: float

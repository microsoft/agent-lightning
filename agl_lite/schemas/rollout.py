"""Rollout schemas — the fundamental unit of work in agl-lite."""

from __future__ import annotations

from enum import StrEnum

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

    K8s-specific infra details (resources, nodeSelector, etc.) come from
    the controller's job defaults (from resources snapshot) — the algorithm
    never sees infra-level K8s details like nodeSelector or tolerations.
    """

    # Required — describe the container.
    image: str
    command: list[str] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    mount: list[Mount] = Field(default_factory=list)

    # Optional — execution policy (defaults from job_defaults in resources).
    timeout: int | None = None  # seconds → K8s activeDeadlineSeconds
    max_retries: int | None = None  # retry count → K8s backoffLimit


class Rollout(BaseModel):
    """Unit of work. Lifecycle managed by the K8s controller."""

    rollout_id: str
    status: RolloutStatus = RolloutStatus.QUEUING
    cancel_requested: bool = False

    input: dict  # task description (delivered as AGL_TASK_INPUT env var)
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

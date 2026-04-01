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


class RolloutConfig(BaseModel):
    """Algorithm-facing config. Describes the containerized task.

    pod_spec is the K8s pod spec fragment assembled by the on_enqueue hook —
    containers, volumes, nodeSelector, tolerations, etc. The hook deep-copies
    a per-dataset template and applies per-sample modifications before storing
    it here. The controller merges it with the manifest_template PodPatcher
    (which injects gateway env vars into every container) and wraps it in a
    K8s Job manifest.
    """

    pod_spec: dict[str, Any] | None = None  # full pod spec fragment, set by on_enqueue hook
    timeout: int | None = None             # seconds → K8s activeDeadlineSeconds
    max_retries: int | None = None         # retry count → K8s backoffLimit


class RolloutMetadata(BaseModel):
    """Algorithm + hook facing context. Not sent to container.

    Algorithm control fields help the daemon/trainer reconstruct batch structure.
    Extra fields are allowed (``extra="allow"``) for hooks to stash task-specific
    context (e.g., ground_truth for grading in ``on_succeeded``).
    """

    batch_idx: int | None = None
    sample_idx_in_batch: int | None = None
    trial_idx_in_group: int | None = None

    model_config = {"extra": "allow"}


class Rollout(BaseModel):
    """Unit of work. Lifecycle managed by the K8s controller."""

    rollout_id: str
    status: RolloutStatus = RolloutStatus.QUEUING
    cancel_requested: bool = False

    input: Any  # raw dataset content from algorithm (read by hooks, NOT sent to container)
    config: RolloutConfig
    metadata: RolloutMetadata = Field(default_factory=RolloutMetadata)
    resources_id: str | None = None  # links to immutable resource snapshot

    # Set by controller during lifecycle.
    job_name: str | None = None
    succeeded_attempt_id: str | None = None
    error_message: str | None = None

    # Concurrency control.
    version: int = 1  # optimistic locking — incremented on every update

    created_at: float
    updated_at: float

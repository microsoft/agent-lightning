"""Controller configuration — pure data carrier, no env reads."""

from __future__ import annotations

from pydantic import BaseModel


class ControllerSettings(BaseModel):
    """Settings for the agl-lite K8s controller. Constructed explicitly by cli.py."""

    # Connection to agl-lite server.
    base_url: str              # AGL_BASE_URL
    key: str = ""              # AGL_KEY — auth key; empty = auth disabled

    # K8s configuration.
    namespace: str             # AGL_NAMESPACE

    # Reconcile timing.
    poll_interval: int = 10    # AGL_POLL_INTERVAL — seconds between reconcile cycles
    max_queue_time: int = 3600 # AGL_MAX_QUEUE_TIME — max seconds a rollout stays in queuing

    # Job defaults.
    ttl_after_finished: int = 3600  # AGL_TTL_AFTER_FINISHED — ttlSecondsAfterFinished on Jobs

    # Job manifest template.
    job_manifest_template: str  # AGL_JOB_MANIFEST_TEMPLATE — always required

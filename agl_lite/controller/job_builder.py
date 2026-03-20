"""Job spec builder — pure function that converts rollout + resources into a K8s Job manifest.

No I/O, no K8s API calls. Easy to unit test.
"""

from __future__ import annotations

import json
from typing import Any

from agl_lite.controller.config import ControllerSettings
from agl_lite.schemas.resources import JobDefaults
from agl_lite.schemas.rollout import Rollout


def build_job_name(rollout_id: str) -> str:
    """Deterministic Job name from rollout ID."""
    return f"agl-rollout-{rollout_id}"


def build_job_spec(
    rollout: Rollout,
    job_defaults: JobDefaults | None,
    settings: ControllerSettings,
) -> dict[str, Any]:
    """Build a K8s Job manifest dict from rollout config + job defaults + controller settings.

    Merge order (last wins):
        1. job_defaults (infra-level defaults from resources snapshot)
        2. rollout.config (algorithm-specified overrides)
        3. controller settings (namespace, secret, ttl)
    """
    config = rollout.config
    defaults = job_defaults or JobDefaults()

    # --- Resolve execution policy ---
    timeout = config.timeout or defaults.timeout
    max_retries = config.max_retries if config.max_retries is not None else defaults.max_retries

    # --- Build container env vars ---
    gateway_base = f"{settings.lite_url}/rollout/{rollout.rollout_id}/attempt/$(AGL_POD_UID)"
    event_url = f"{gateway_base}/events"

    env: list[dict[str, Any]] = [
        # Pod UID via Downward API — used to construct attempt_id.
        {
            "name": "AGL_POD_UID",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
        },
        # Single AGL_KEY from Secret — used for all auth (gateway, event posts).
        # Also injected as OPENAI_API_KEY and ANTHROPIC_API_KEY so SDKs send it
        # as Authorization: Bearer / x-api-key headers automatically.
        {
            "name": "AGL_KEY",
            "valueFrom": {"secretKeyRef": {"name": settings.secret_name, "key": "AGL_KEY", "optional": True}},
        },
        {
            "name": "OPENAI_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": settings.secret_name, "key": "AGL_KEY", "optional": True}},
        },
        {
            "name": "ANTHROPIC_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": settings.secret_name, "key": "AGL_KEY", "optional": True}},
        },
        # SDK base URLs — point to agl-lite gateway.
        {"name": "OPENAI_BASE_URL", "value": f"{gateway_base}/v1"},
        {"name": "ANTHROPIC_BASE_URL", "value": f"{gateway_base}/v1"},
        # Task input and event URL.
        {"name": "AGL_TASK_INPUT", "value": json.dumps(rollout.input)},
        {"name": "AGL_EVENT_URL", "value": event_url},
    ]

    # User-specified env vars from rollout config (override defaults).
    for key, value in config.environment_variables.items():
        env.append({"name": key, "value": value})

    # --- Build container spec ---
    container: dict[str, Any] = {
        "name": "agent",
        "image": config.image,
        "env": env,
    }

    if config.command:
        container["command"] = config.command

    # Resource requests/limits from job_defaults.
    if defaults.resources:
        container["resources"] = {}
        if defaults.resources.requests:
            container["resources"]["requests"] = defaults.resources.requests
        if defaults.resources.limits:
            container["resources"]["limits"] = defaults.resources.limits

    # Volume mounts from rollout config.
    if config.mount:
        container["volumeMounts"] = [
            {"name": m.name, "mountPath": m.mount_path, "readOnly": m.read_only} for m in config.mount
        ]

    # --- Build pod spec ---
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
    }

    if defaults.service_account:
        pod_spec["serviceAccountName"] = defaults.service_account

    if defaults.node_selector:
        pod_spec["nodeSelector"] = defaults.node_selector

    if defaults.tolerations:
        pod_spec["tolerations"] = defaults.tolerations

    if defaults.image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": s} for s in defaults.image_pull_secrets]

    # Volumes from rollout config mounts.
    if config.mount:
        volumes: list[dict[str, Any]] = []
        for m in config.mount:
            vol: dict[str, Any] = {"name": m.name}
            # Heuristic: if source looks like a PVC name, use persistentVolumeClaim.
            # If it starts with /, use hostPath. Otherwise, use configMap.
            if m.source.startswith("/"):
                vol["hostPath"] = {"path": m.source}
            elif m.source.startswith("pvc:"):
                vol["persistentVolumeClaim"] = {"claimName": m.source[4:]}
            else:
                vol["configMap"] = {"name": m.source}
            volumes.append(vol)
        pod_spec["volumes"] = volumes

    # --- Build Job spec ---
    job_spec: dict[str, Any] = {
        "backoffLimit": max_retries if max_retries is not None else 0,
        "ttlSecondsAfterFinished": settings.ttl_after_finished,
        "template": {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/managed-by": "agl-lite",
                    "agl-lite/rollout-id": rollout.rollout_id,
                },
            },
            "spec": pod_spec,
        },
    }

    if timeout is not None:
        job_spec["activeDeadlineSeconds"] = timeout

    # --- Build full Job manifest ---
    job: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": build_job_name(rollout.rollout_id),
            "namespace": settings.namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "agl-lite",
                "agl-lite/rollout-id": rollout.rollout_id,
            },
        },
        "spec": job_spec,
    }

    # --- Apply overrides escape hatch ---
    if defaults.overrides:
        _deep_merge(job["spec"], defaults.overrides)

    return job


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place). Override wins on conflicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

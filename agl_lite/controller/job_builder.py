"""Job spec builder — pure function that converts rollout + job_template into a K8s Job manifest.

No I/O, no K8s API calls. Easy to unit test.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from agl_lite.controller.config import ControllerSettings
from agl_lite.schemas.rollout import Rollout


def build_job_name(rollout_id: str) -> str:
    """Deterministic Job name from rollout ID."""
    return f"agl-rollout-{rollout_id}"


def build_job_spec(
    rollout: Rollout,
    job_template: dict[str, Any] | None,
    settings: ControllerSettings,
) -> dict[str, Any]:
    """Build a K8s Job manifest from rollout config + job_template + controller settings.

    Merge order:
        1. job_template (raw pod spec from resources snapshot — infra/task environment)
        2. rollout.config.overrides (per-rollout K8s overrides, name-matched containers)
        3. rollout.config named fields (image, command, env vars → agent container)
        4. controller fields (namespace, labels, gateway env vars, secret refs)
    """
    config = rollout.config
    template = copy.deepcopy(job_template) if job_template else {}

    # --- Start from template as pod spec ---
    # Template can be:
    #   - Direct pod spec: {"containers": [...], "nodeSelector": {...}, ...}
    #   - Wrapped: {"spec": {"containers": [...], ...}}
    # Detect and extract.
    if "spec" in template and "containers" not in template:
        pod_spec = copy.deepcopy(template["spec"])
    else:
        pod_spec = copy.deepcopy(template)

    # Ensure restartPolicy is Never (controller requirement).
    pod_spec.setdefault("restartPolicy", "Never")

    # Ensure containers list exists.
    pod_spec.setdefault("containers", [])

    # --- Apply rollout overrides (name-matched container merge) ---
    if config.overrides:
        overrides = copy.deepcopy(config.overrides)
        # Handle containers specially: merge by name.
        override_containers = overrides.pop("containers", [])
        for oc in override_containers:
            oc_name = oc.get("name")
            if not oc_name:
                continue
            # Find matching container by name.
            matched = False
            for container in pod_spec["containers"]:
                if container.get("name") == oc_name:
                    _deep_merge(container, oc)
                    matched = True
                    break
            if not matched:
                # No matching container — skip (don't add unknown containers).
                pass

        # Merge remaining override fields into pod spec.
        _deep_merge(pod_spec, overrides)

    # --- Find or create the "agent" container ---
    agent_container = None
    for c in pod_spec["containers"]:
        if c.get("name") == "agent":
            agent_container = c
            break

    if agent_container is None:
        agent_container = {"name": "agent"}
        pod_spec["containers"].insert(0, agent_container)

    # --- Inject RolloutConfig named fields into agent container ---
    agent_container["image"] = config.image

    if config.command:
        agent_container["command"] = config.command

    # Volume mounts from rollout config.
    if config.mount:
        agent_container.setdefault("volumeMounts", [])
        for m in config.mount:
            agent_container["volumeMounts"].append(
                {"name": m.name, "mountPath": m.mount_path, "readOnly": m.read_only}
            )
        # Add volume definitions to pod spec.
        pod_spec.setdefault("volumes", [])
        for m in config.mount:
            vol: dict[str, Any] = {"name": m.name}
            if m.source.startswith("/"):
                vol["hostPath"] = {"path": m.source}
            elif m.source.startswith("pvc:"):
                vol["persistentVolumeClaim"] = {"claimName": m.source[4:]}
            else:
                vol["configMap"] = {"name": m.source}
            pod_spec["volumes"].append(vol)

    # --- Inject controller env vars into agent container ---
    gateway_base = f"{settings.lite_url}/rollout/{rollout.rollout_id}/attempt/$(AGL_POD_UID)"
    event_url = f"{gateway_base}/events"

    controller_env: list[dict[str, Any]] = [
        {
            "name": "AGL_POD_UID",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
        },
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
        {"name": "OPENAI_BASE_URL", "value": f"{gateway_base}/v1"},
        {"name": "ANTHROPIC_BASE_URL", "value": f"{gateway_base}/v1"},
        {"name": "AGL_TASK_INPUT", "value": json.dumps(rollout.input)},
        {"name": "AGL_EVENT_URL", "value": event_url},
    ]

    # User-specified env vars from rollout config.
    for key, value in config.environment_variables.items():
        controller_env.append({"name": key, "value": value})

    # Prepend controller env vars (so template env vars don't override them).
    existing_env = agent_container.get("env", [])
    agent_container["env"] = controller_env + existing_env

    # --- Build Job spec ---
    timeout = config.timeout
    max_retries = config.max_retries

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
    return {
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


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place). Override wins on conflicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

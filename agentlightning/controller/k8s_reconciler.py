# Copyright (c) Microsoft. All rights reserved.

"""K8s controller reconciler — manages rollout lifecycle via K8s Jobs.

Two concurrent tasks:
    1. periodic_reconcile() — poll queuing rollouts, create Jobs, expire stale
  2. watch_jobs() — react to Job completions/failures, update rollout status

Uses AgentLightningAsyncClient for store access and kr8s for K8s API.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx
import kr8s
import kr8s.asyncio
import structlog
from kr8s.asyncio import objects as k8s_objects
from omegaconf import DictConfig

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.k8s import extract_pod_images, normalize_image_reference, render_job_template
from agentlightning.schemas import (
    DEFAULT_ATTEMPT_ID,
    K8sImageReadinessReport,
    Rollout,
    RolloutPatch,
    RolloutState,
    RolloutStatusPatch,
)

log = structlog.get_logger()

MANAGED_BY_SELECTOR = "app.kubernetes.io/managed-by=agentlightning"
JOB_CREATION_WINDOW_SECONDS = 60


def build_job_name(rollout_id: str) -> str:
    """Deterministic Job name from rollout ID."""
    return f"agl-rollout-{rollout_id}"


def images_available_on_all_ready_nodes(
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[frozenset[str], int]:
    """Return the image-name intersection for Ready, schedulable nodes."""
    eligible: list[Mapping[str, Any]] = []
    for node in nodes:
        if node.get("spec", {}).get("unschedulable", False):
            continue
        conditions = node.get("status", {}).get("conditions", [])
        if not any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions):
            continue
        eligible.append(node)

    if not eligible:
        raise RuntimeError("no Ready schedulable Kubernetes nodes found")

    per_node: list[set[str]] = []
    for node in eligible:
        names = {
            normalize_image_reference(name)
            for image in node.get("status", {}).get("images", [])
            for name in image.get("names", [])
            if isinstance(name, str) and name.strip()
        }
        per_node.append(names)

    common = set(per_node[0])
    for names in per_node[1:]:
        common.intersection_update(names)
    return frozenset(common), len(eligible)


def build_job_spec(rollout: Rollout, controller_config: DictConfig) -> dict[str, Any]:
    """Build a K8s Job manifest from the rollout's complete Jinja2 Job template."""
    template = rollout.config.k8s.job_template if rollout.config.k8s else None
    if not template:
        raise ValueError("invalid rollout config: missing config.k8s.job_template")

    job = render_job_template(
        template,
        job_name=build_job_name(rollout.rollout_id),
        input_data=rollout.input,
    )

    metadata = job.setdefault("metadata", {})
    metadata["name"] = build_job_name(rollout.rollout_id)
    metadata["namespace"] = controller_config.k8s_runner.namespace
    labels = metadata.setdefault("labels", {})
    labels["app.kubernetes.io/managed-by"] = "agentlightning"
    labels["agentlightning/rollout-id"] = rollout.rollout_id
    labels["agentlightning/attempt-id"] = DEFAULT_ATTEMPT_ID

    spec = job.setdefault("spec", {})
    spec["backoffLimit"] = 0
    spec["ttlSecondsAfterFinished"] = controller_config.k8s_runner.ttl_after_finished
    if rollout.config.timeout_seconds:
        spec["activeDeadlineSeconds"] = rollout.config.timeout_seconds
    pod_spec = spec.setdefault("template", {}).setdefault("spec", {})
    pod_spec["restartPolicy"] = "Never"

    mode = "train" if rollout.is_train else "val"
    agent_base_url = str(
        controller_config.agl_server.get("agent_url", None) or controller_config.agl_server.url
    ).rstrip("/")
    agl_openai_base_url = (
        f"{agent_base_url}/proxy/rollout/{rollout.rollout_id}/attempt/{DEFAULT_ATTEMPT_ID}/mode/{mode}/openai/v1"
    )
    for container in pod_spec.get("containers", []):
        env = container.setdefault("env", [])
        for name, value in {
            "AGL_OPENAI_BASE_URL": agl_openai_base_url,
            "AGL_EVENT_URL": (
                f"{agent_base_url}/api/rollouts/{rollout.rollout_id}/attempt/{DEFAULT_ATTEMPT_ID}/events"
            ),
            "AGL_KEY": str(controller_config.agl_server.key or ""),
        }.items():
            existing = next((item for item in env if item.get("name") == name), None)
            if existing is None:
                env.append({"name": name, "value": value})
            else:
                existing.clear()
                existing.update({"name": name, "value": value})
    return job


class K8sReconciler:
    """Main controller loop. Reconciles rollouts into K8s Jobs.

    Args:
        api: AgentLightningAsyncClient for store access.
        config: Controller configuration.
    """

    def __init__(self, api: AgentLightningAsyncClient, config: DictConfig) -> None:
        self._api = api
        self._config = config
        self._runner_config = config.k8s_runner
        self._namespace = str(self._runner_config.namespace)
        self._k8s_api: Any | None = None
        self._stop = asyncio.Event()
        self._job_creation_timestamps: deque[float] = deque()
        self._ready_images: frozenset[str] | None = None
        self._ready_images_expires_at = 0.0

        readiness_config = self._runner_config.image_readiness
        if readiness_config.enabled:
            heartbeat = float(readiness_config.heartbeat_seconds)
            lease = float(readiness_config.lease_seconds)
            if not 0 < heartbeat < lease <= 300:
                raise ValueError("k8s_runner.image_readiness requires 0 < heartbeat_seconds < lease_seconds <= 300")

    async def _get_k8s_api(self) -> Any:
        if self._k8s_api is None:
            self._k8s_api = await kr8s.asyncio.api()
        return self._k8s_api

    async def run(self) -> None:
        """Start both reconcile loops. Blocks until stop() is called."""
        log.info(
            "Controller starting",
            namespace=self._namespace,
            poll_interval=self._runner_config.poll_interval,
        )
        try:
            tasks = []
            if self._runner_config.image_readiness.enabled:
                try:
                    await self._publish_image_readiness_once()
                except Exception:
                    log.exception("Initial K8s image readiness publish failed")
                tasks.append(self._image_readiness_loop())
            tasks.extend(
                [
                    self._periodic_reconcile_loop(),
                    self._watch_jobs_loop(),
                ]
            )
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("Controller stopped")

    def stop(self) -> None:
        """Signal the controller to stop."""
        self._stop.set()

    async def _scan_preloaded_images(self) -> tuple[frozenset[str], int]:
        api = await self._get_k8s_api()
        nodes = [cast(k8s_objects.Node, node).raw async for node in k8s_objects.Node.async_list(api=api)]
        return images_available_on_all_ready_nodes(nodes)

    async def _publish_image_readiness_once(self) -> None:
        images, node_count = await self._scan_preloaded_images()
        lease_seconds = float(self._runner_config.image_readiness.lease_seconds)
        scanned_at = time.monotonic()
        report = K8sImageReadinessReport(
            images=sorted(images),
            node_count=node_count,
            lease_seconds=lease_seconds,
        )
        response = await self._api.put(
            "/api/runner-readiness/k8s",
            json=report.model_dump(mode="json"),
        )
        response.raise_for_status()
        self._ready_images = images
        self._ready_images_expires_at = scanned_at + lease_seconds

    async def _image_readiness_loop(self) -> None:
        heartbeat = float(self._runner_config.image_readiness.heartbeat_seconds)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=heartbeat)
                return
            except TimeoutError:
                pass
            try:
                await self._publish_image_readiness_once()
            except Exception:
                log.exception("K8s image readiness publish failed")

    def _fresh_ready_images(self) -> frozenset[str] | None:
        if self._ready_images is None or time.monotonic() >= self._ready_images_expires_at:
            return None
        return self._ready_images

    # --- Periodic reconcile ---

    async def _periodic_reconcile_loop(self) -> None:
        """Poll queuing rollouts and reconcile."""
        while not self._stop.is_set():
            try:
                await self._reconcile_once()
            except Exception:
                log.exception("Periodic reconcile error")
            # Sleep with cancellation support.
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._runner_config.poll_interval)
                return  # stop was set
            except TimeoutError:
                pass

    async def _reconcile_once(self) -> None:
        """One reconcile cycle: align queuing/running rollouts with K8s Jobs."""
        rollouts = await self._query_rollouts(state_in=[RolloutState.QUEUING, RolloutState.RUNNING], limit=500)
        api = await self._get_k8s_api()
        jobs = [
            cast(k8s_objects.Job, job).raw
            async for job in k8s_objects.Job.async_list(
                namespace=self._namespace,
                label_selector=MANAGED_BY_SELECTOR,
                api=api,
            )
        ]
        jobs_by_name = {job.get("metadata", {}).get("name", ""): job for job in jobs}

        for rollout in rollouts:
            job_name = rollout.status.k8s_job_name or build_job_name(rollout.rollout_id)
            job = jobs_by_name.get(job_name)

            if job is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._create_job(rollout)
                    continue
                log.warning("Orphaned running rollout — Job gone", rollout_id=rollout.rollout_id, job_name=job_name)
                await self._patch_status(rollout.rollout_id, state=RolloutState.FAILED, error_message="Job disappeared")
                continue

            attempt_id = (
                job.get("metadata", {}).get("labels", {}).get("agentlightning/attempt-id") or DEFAULT_ATTEMPT_ID
            )

            job_status = job.get("status", {})
            state = None
            error_message = None
            for condition in job_status.get("conditions", []):
                if condition.get("status") != "True":
                    continue
                if condition.get("type") == "Complete":
                    state = RolloutState.SUCCEEDED
                    break
                if condition.get("type") == "Failed":
                    reason = condition.get("reason", "Unknown")
                    message = condition.get("message", "")
                    error_message = f"Job failed: {reason}"
                    if message:
                        error_message += f" — {message}"
                    state = RolloutState.FAILED
                    break

            if state is None and job_status.get("succeeded", 0) > 0:
                state = RolloutState.SUCCEEDED
            elif state is None and job_status.get("failed", 0) > 0:
                state = RolloutState.FAILED
                error_message = "Job failed"

            if state is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._patch_status(
                        rollout.rollout_id,
                        state=RolloutState.RUNNING,
                        k8s_job_name=job_name,
                        last_attempt_id=attempt_id,
                    )
                continue

            if rollout.status.state == RolloutState.QUEUING and state == RolloutState.SUCCEEDED:
                patched = await self._patch_status(
                    rollout.rollout_id,
                    state=RolloutState.RUNNING,
                    k8s_job_name=job_name,
                    last_attempt_id=attempt_id,
                )
                if not patched:
                    continue
            await self._patch_status(
                rollout.rollout_id,
                state=state,
                k8s_job_name=job_name,
                last_attempt_id=attempt_id,
                error_message=error_message,
            )

    async def _create_job(self, rollout: Rollout) -> None:
        """Create a K8s Job for a queuing rollout without changing rollout state."""
        job_name = build_job_name(rollout.rollout_id)
        try:
            manifest = build_job_spec(rollout, self._config)
            attempt_id = manifest["metadata"]["labels"]["agentlightning/attempt-id"]
            requires_preloaded = bool(rollout.config.k8s and rollout.config.k8s.require_preloaded_images)
            if requires_preloaded:
                ready_images = self._fresh_ready_images()
                if ready_images is None:
                    await self._patch_status(
                        rollout.rollout_id,
                        state=RolloutState.FAILED,
                        error_message="Fresh Kubernetes image readiness is unavailable",
                    )
                    return
                missing_images = sorted(extract_pod_images(manifest) - ready_images)
                if missing_images:
                    await self._patch_status(
                        rollout.rollout_id,
                        state=RolloutState.FAILED,
                        error_message=("Required Kubernetes image(s) are not preloaded: " + ", ".join(missing_images)),
                    )
                    return

            now = time.monotonic()
            window_start = now - JOB_CREATION_WINDOW_SECONDS
            while self._job_creation_timestamps and self._job_creation_timestamps[0] <= window_start:
                self._job_creation_timestamps.popleft()
            if len(self._job_creation_timestamps) >= self._runner_config.max_jobs_per_minute:
                log.info(
                    "Job creation rate limit reached — deferring queued rollouts",
                    rollout_id=rollout.rollout_id,
                    jobs_in_last_minute=len(self._job_creation_timestamps),
                    max_jobs_per_minute=self._runner_config.max_jobs_per_minute,
                )
                return

            api = await self._get_k8s_api()
            job = k8s_objects.Job(manifest, api=api)
            await job.async_create()
            self._job_creation_timestamps.append(time.monotonic())
            log.info("Job created", rollout_id=rollout.rollout_id, job_name=job_name, attempt_id=attempt_id)
        except ValueError as exc:
            error_str = str(exc)
            log.error("Invalid Job spec — marking failed", rollout_id=rollout.rollout_id, error=error_str)
            await self._patch_status(
                rollout.rollout_id,
                state=RolloutState.FAILED,
                error_message=f"Invalid Job spec: {error_str}",
            )
        except Exception as exc:
            error_str = str(exc)
            lower_error = error_str.lower()
            if "422" in lower_error or "unprocessable" in lower_error or "invalid" in lower_error:
                log.error("Invalid Job spec — marking failed", rollout_id=rollout.rollout_id, error=error_str)
                await self._patch_status(
                    rollout.rollout_id,
                    state=RolloutState.FAILED,
                    error_message=f"Invalid Job spec: {error_str}",
                )
            else:
                log.warning("Job creation failed — will retry", rollout_id=rollout.rollout_id, error=error_str)

    # --- Watch Jobs ---

    async def _watch_jobs_loop(self) -> None:
        """Watch K8s Job events and react to completions/failures."""
        while not self._stop.is_set():
            try:
                watcher = kr8s.asyncio.watch(
                    "jobs",
                    namespace=self._namespace,
                    label_selector=MANAGED_BY_SELECTOR,
                    api=await self._get_k8s_api(),
                )
                async for event_type, obj in watcher:
                    if self._stop.is_set():
                        return
                    if event_type in ("MODIFIED", "ADDED"):
                        await self._handle_job_event(obj.raw)
            except Exception:
                log.exception("Watch error — restarting watch")
                await asyncio.sleep(5)

    async def _handle_job_event(self, job: dict[str, Any]) -> None:
        """Process a Job event — check conditions, update rollout status."""
        labels = job.get("metadata", {}).get("labels", {})
        rollout_id = labels.get("agentlightning/rollout-id")
        if not rollout_id:
            return
        attempt_id = labels.get("agentlightning/attempt-id") or DEFAULT_ATTEMPT_ID

        conditions = job.get("status", {}).get("conditions", [])
        if not conditions:
            return

        for condition in conditions:
            cond_type = condition.get("type", "")
            cond_status = condition.get("status", "")
            if cond_status != "True":
                continue

            if cond_type == "Complete":
                log.info("Job completed", rollout_id=rollout_id, last_attempt_id=attempt_id)
                await self._patch_status(rollout_id, state=RolloutState.SUCCEEDED, last_attempt_id=attempt_id)
                return
            elif cond_type == "Failed":
                reason = condition.get("reason", "Unknown")
                message = condition.get("message", "")
                error_msg = f"Job failed: {reason}"
                if message:
                    error_msg += f" — {message}"
                log.info("Job failed", rollout_id=rollout_id, last_attempt_id=attempt_id, reason=reason)
                await self._patch_status(
                    rollout_id,
                    state=RolloutState.FAILED,
                    last_attempt_id=attempt_id,
                    error_message=error_msg,
                )
                return

    async def _query_rollouts(
        self,
        *,
        state_in: list[RolloutState],
        limit: int = 50,
    ) -> list[Rollout]:
        params = httpx.QueryParams()
        for state in state_in:
            params = params.add("state_in", state.value)
        params = params.add("limit", limit)
        response = await self._api.get("/api/rollouts", params=params)
        response.raise_for_status()
        return [Rollout.model_validate(item) for item in response.json()]

    async def _patch_status(self, rollout_id: str, **status: Any) -> bool:
        try:
            patch = RolloutPatch(status=RolloutStatusPatch.model_validate(status))
            response = await self._api.patch(
                f"/api/rollouts/{rollout_id}",
                json=patch.model_dump(mode="json", exclude_unset=True),
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            log.warning("Failed to patch rollout", rollout_id=rollout_id, error=str(exc))
            return False

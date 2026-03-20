"""K8s controller reconciler — manages rollout lifecycle via K8s Jobs.

Two concurrent tasks:
  1. periodic_reconcile() — poll queuing rollouts, create Jobs, handle cancels, expire stale
  2. watch_jobs() — react to Job completions/failures, update rollout status

Uses AglLiteClient for store access and kr8s for K8s API.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

import structlog

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.controller.config import ControllerSettings
from agl_lite.controller.job_builder import build_job_name, build_job_spec
from agl_lite.schemas.api import PatchRolloutRequest
from agl_lite.schemas.resources import JobDefaults, ResourcesUpdate
from agl_lite.schemas.rollout import Rollout, RolloutStatus

log = structlog.get_logger()


# --- K8s abstraction (for testability) ---


class K8sClient(Protocol):
    """Minimal K8s client interface — abstracts kr8s for testing."""

    async def create_job(self, manifest: dict[str, Any]) -> None: ...
    async def delete_job(self, name: str, namespace: str) -> None: ...
    async def get_job(self, name: str, namespace: str) -> dict[str, Any] | None: ...
    async def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]: ...
    async def list_pods(self, namespace: str, label_selector: str) -> list[dict[str, Any]]: ...
    async def watch_jobs(self, namespace: str, label_selector: str) -> AsyncJobWatcher: ...


class AsyncJobWatcher(Protocol):
    """Async iterator of (event_type, job_dict) tuples."""

    def __aiter__(self) -> AsyncJobWatcher: ...
    async def __anext__(self) -> tuple[str, dict[str, Any]]: ...


# --- Reconciler ---


class Reconciler:
    """Main controller loop. Reconciles rollouts into K8s Jobs.

    Args:
        api: AglLiteClient for store access.
        k8s: K8s client (kr8s wrapper or mock).
        settings: Controller configuration.
    """

    def __init__(self, api: AglLiteClient, k8s: K8sClient, settings: ControllerSettings) -> None:
        self._api = api
        self._k8s = k8s
        self._settings = settings
        self._resources_cache: dict[str, ResourcesUpdate] = {}
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Start both reconcile loops. Blocks until stop() is called."""
        log.info("Controller starting", namespace=self._settings.namespace, poll_interval=self._settings.poll_interval)
        try:
            await asyncio.gather(
                self._periodic_reconcile_loop(),
                self._watch_jobs_loop(),
            )
        except asyncio.CancelledError:
            log.info("Controller stopped")

    def stop(self) -> None:
        """Signal the controller to stop."""
        self._stop.set()

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
                await asyncio.wait_for(self._stop.wait(), timeout=self._settings.poll_interval)
                return  # stop was set
            except TimeoutError:
                pass

    async def _reconcile_once(self) -> None:
        """One reconcile cycle: create Jobs for queuing, handle cancels, expire stale."""
        # 1. Handle queuing rollouts → create Jobs.
        queuing = await self._api.query_rollouts(status_in=[RolloutStatus.QUEUING], limit=500)
        for rollout in queuing:
            if rollout.cancel_requested:
                await self._cancel_rollout(rollout)
            else:
                await self._maybe_create_job(rollout)

        # 2. Handle running rollouts with cancel_requested.
        running_cancelled = await self._api.query_rollouts(
            status_in=[RolloutStatus.RUNNING], cancel_requested=True, limit=500
        )
        for rollout in running_cancelled:
            await self._cancel_running_rollout(rollout)

        # 3. Crash recovery — check for orphaned running rollouts (Job gone).
        running = await self._api.query_rollouts(status_in=[RolloutStatus.RUNNING], limit=500)
        existing_jobs = await self._k8s.list_jobs(
            namespace=self._settings.namespace,
            label_selector="app.kubernetes.io/managed-by=agl-lite",
        )
        existing_job_names = {_job_name(j) for j in existing_jobs}
        for rollout in running:
            if rollout.job_name and rollout.job_name not in existing_job_names:
                log.warning(
                    "Orphaned running rollout — Job gone",
                    rollout_id=rollout.rollout_id,
                    job_name=rollout.job_name,
                )
                await self._patch_rollout(
                    rollout.rollout_id,
                    PatchRolloutRequest(
                        status=RolloutStatus.TERMINAL_FAILED,
                        error_message="Job disappeared — possible cluster issue or manual deletion",
                    ),
                )

    async def _maybe_create_job(self, rollout: Rollout) -> None:
        """Create a K8s Job for a queuing rollout."""
        # Check max queue time.
        if time.time() - rollout.created_at > self._settings.max_queue_time:
            log.warning("Rollout exceeded max queue time", rollout_id=rollout.rollout_id)
            await self._patch_rollout(
                rollout.rollout_id,
                PatchRolloutRequest(
                    status=RolloutStatus.TERMINAL_FAILED,
                    error_message=f"Exceeded max queue time ({self._settings.max_queue_time}s)",
                ),
            )
            return

        # Check if Job already exists (idempotency — crash recovery).
        job_name = build_job_name(rollout.rollout_id)
        existing = await self._k8s.get_job(job_name, self._settings.namespace)
        if existing is not None:
            # Job exists but rollout is still queuing — fix the status.
            log.info("Job already exists for queuing rollout — updating status", rollout_id=rollout.rollout_id)
            await self._patch_rollout(
                rollout.rollout_id,
                PatchRolloutRequest(status=RolloutStatus.RUNNING, job_name=job_name),
            )
            return

        # Fetch resources (with caching).
        job_defaults = await self._get_job_defaults(rollout.resources_id)

        # Build and create Job.
        manifest = build_job_spec(rollout, job_defaults, self._settings)
        try:
            await self._k8s.create_job(manifest)
            log.info("Job created", rollout_id=rollout.rollout_id, job_name=job_name)
            await self._patch_rollout(
                rollout.rollout_id,
                PatchRolloutRequest(status=RolloutStatus.RUNNING, job_name=job_name),
            )
        except Exception as e:
            # Job creation failed — stay in queuing, retry next cycle.
            log.warning("Job creation failed — will retry", rollout_id=rollout.rollout_id, error=str(e))

    async def _cancel_rollout(self, rollout: Rollout) -> None:
        """Cancel a queuing rollout (no Job exists)."""
        log.info("Cancelling queuing rollout", rollout_id=rollout.rollout_id)
        await self._patch_rollout(
            rollout.rollout_id,
            PatchRolloutRequest(status=RolloutStatus.CANCELLED),
        )

    async def _cancel_running_rollout(self, rollout: Rollout) -> None:
        """Cancel a running rollout — delete its Job."""
        log.info("Cancelling running rollout", rollout_id=rollout.rollout_id, job_name=rollout.job_name)
        if rollout.job_name:
            try:
                await self._k8s.delete_job(rollout.job_name, self._settings.namespace)
            except Exception:
                log.warning(
                    "Failed to delete Job during cancel",
                    rollout_id=rollout.rollout_id,
                    job_name=rollout.job_name,
                )
        await self._patch_rollout(
            rollout.rollout_id,
            PatchRolloutRequest(status=RolloutStatus.CANCELLED),
        )

    # --- Watch Jobs ---

    async def _watch_jobs_loop(self) -> None:
        """Watch K8s Job events and react to completions/failures."""
        while not self._stop.is_set():
            try:
                watcher = await self._k8s.watch_jobs(
                    namespace=self._settings.namespace,
                    label_selector="app.kubernetes.io/managed-by=agl-lite",
                )
                async for event_type, job in watcher:
                    if self._stop.is_set():
                        return
                    if event_type in ("MODIFIED", "ADDED"):
                        await self._handle_job_event(job)
            except Exception:
                log.exception("Watch error — restarting watch")
                await asyncio.sleep(5)

    async def _handle_job_event(self, job: dict[str, Any]) -> None:
        """Process a Job event — check conditions, update rollout status."""
        rollout_id = _rollout_id_from_job(job)
        if not rollout_id:
            return

        conditions = _job_conditions(job)
        if not conditions:
            return

        for condition in conditions:
            cond_type = condition.get("type", "")
            cond_status = condition.get("status", "")
            if cond_status != "True":
                continue

            if cond_type == "Complete":
                await self._handle_job_complete(rollout_id, job)
                return
            elif cond_type == "Failed":
                reason = condition.get("reason", "Unknown")
                message = condition.get("message", "")
                await self._handle_job_failed(rollout_id, reason, message)
                return

    async def _handle_job_complete(self, rollout_id: str, job: dict[str, Any]) -> None:
        """Job completed successfully — find succeeded pod UID, update rollout."""
        succeeded_uid = await self._find_succeeded_pod_uid(job)
        if not succeeded_uid:
            log.warning("Job complete but no succeeded pod found", rollout_id=rollout_id)

        log.info("Job completed", rollout_id=rollout_id, succeeded_attempt_id=succeeded_uid)
        await self._patch_rollout(
            rollout_id,
            PatchRolloutRequest(status=RolloutStatus.SUCCEEDED, succeeded_attempt_id=succeeded_uid),
        )

    async def _handle_job_failed(self, rollout_id: str, reason: str, message: str) -> None:
        """Job failed — update rollout with error details."""
        error_msg = f"Job failed: {reason}"
        if message:
            error_msg += f" — {message}"
        log.info("Job failed", rollout_id=rollout_id, reason=reason)
        await self._patch_rollout(
            rollout_id,
            PatchRolloutRequest(status=RolloutStatus.TERMINAL_FAILED, error_message=error_msg),
        )

    async def _find_succeeded_pod_uid(self, job: dict[str, Any]) -> str | None:
        """Find the UID of the succeeded pod for a completed Job."""
        job_name = _job_name(job)
        pods = await self._k8s.list_pods(
            namespace=self._settings.namespace,
            label_selector=f"job-name={job_name}",
        )
        for pod in pods:
            phase = _nested_get(pod, "status", "phase")
            if phase == "Succeeded":
                return _nested_get(pod, "metadata", "uid")
        return None

    # --- Helpers ---

    async def _get_job_defaults(self, resources_id: str | None) -> JobDefaults | None:
        """Fetch job_defaults from resources, with caching."""
        if not resources_id:
            return None

        # Check cache.
        if resources_id in self._resources_cache:
            res = self._resources_cache[resources_id]
        else:
            try:
                res = await self._api.get_resources(resources_id)
                self._resources_cache[resources_id] = res
            except AglLiteError:
                log.warning("Failed to fetch resources", resources_id=resources_id)
                return None

        raw = res.resources.get("job_defaults")
        if raw is None:
            return None
        return JobDefaults.model_validate(raw)

    async def _patch_rollout(self, rollout_id: str, patch: PatchRolloutRequest) -> None:
        """Patch a rollout, handling errors gracefully."""
        try:
            await self._api.patch_rollout(rollout_id, patch)
        except AglLiteError as e:
            log.warning("Failed to patch rollout", rollout_id=rollout_id, error=str(e))


# --- Job dict helpers ---


def _job_name(job: dict[str, Any]) -> str:
    return job.get("metadata", {}).get("name", "")


def _rollout_id_from_job(job: dict[str, Any]) -> str | None:
    return job.get("metadata", {}).get("labels", {}).get("agl-lite/rollout-id")


def _job_conditions(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("status", {}).get("conditions", [])


def _nested_get(d: dict, *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)  # type: ignore[assignment]
    return d

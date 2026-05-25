"""Local-process-pool reconciler — runs each rollout as a short-lived python subprocess.

A counterpart to ``Reconciler`` (K8s-backed). Maintains a single in-memory list of
in-flight subprocesses; each tick of ``_reconcile_loop`` runs three steps in order:
  1. reap exited processes → patch terminal status → drop from list.
  2. enforce timeout / cancel on still-running processes → SIGKILL the process group.
  3. admit new QUEUING rollouts → spawn subprocess → patch RUNNING.

Runner selection (``runner_type``) is a startup decision made in ``cli.py``;
this class is only constructed when ``settings.runner_type == LOCAL``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass

import structlog

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.controller.config import ControllerSettings, RunnerType
from agl_lite.schemas.api import PatchRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutStatus

log = structlog.get_logger()

_SHUTDOWN_WAIT_TIMEOUT = 5.0
_LOCAL_JOB_NAME_PREFIX = "local-"


def _local_job_name(rollout_id: str) -> str:
    return f"{_LOCAL_JOB_NAME_PREFIX}{rollout_id}"


@dataclass
class RunningProc:
    """One in-flight rollout subprocess."""

    rollout_id: str
    attempt_id: str
    proc: asyncio.subprocess.Process
    # monotonic timestamp; only used for timeout judgement. Using monotonic
    # rather than time.time() avoids NTP jumps firing or skipping the timeout.
    spawned_at: float
    # True once we've sent SIGKILL — avoids double-kill / duplicate log lines.
    killed: bool = False


class LocalReconciler:
    """Local-mode reconciler.

    Internal state is a ``list[RunningProc]``. The single ``_reconcile_loop``
    runs reap → enforce → admit on every tick.
    """

    def __init__(
        self,
        api: AglLiteClient,
        settings: ControllerSettings,
        base_env: dict[str, str],
    ) -> None:
        assert settings.runner_type == RunnerType.LOCAL
        assert settings.local_pool_size is not None
        assert settings.local_agent_class is not None
        self._api = api
        self._settings = settings
        # Narrow the optional fields once so the rest of the class can rely on
        # `int` / `str` types instead of repeating `assert is not None` everywhere.
        self._pool_size: int = settings.local_pool_size
        self._agent_class: str = settings.local_agent_class
        self._tick_interval: float = settings.local_tick_interval
        self._base_env = base_env
        self._running: list[RunningProc] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        log.info(
            "LocalReconciler starting",
            pool_size=self._pool_size,
            tick=self._tick_interval,
        )
        try:
            await self._startup_cleanup()
            await self._reconcile_loop()
        finally:
            await self._shutdown()

    def stop(self) -> None:
        self._stop.set()

    # -------- main loop --------

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._reconcile_once()
            except Exception:
                log.exception("Local reconcile error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._tick_interval)
                break  # stop signalled — fall through to run()'s finally for shutdown
            except TimeoutError:
                pass

    async def _reconcile_once(self) -> None:
        await self._reap_exited()  # (1)
        await self._enforce_termination()  # (2)
        await self._admit_queuing()  # (3)

    # -------- (1) reap exited subprocesses --------

    async def _reap_exited(self) -> None:
        """Drain finished subprocesses, patch terminal status, remove from list.

        Reap must run before admit in the same tick: if a process exited this
        tick, we need to free its slot now so the slot can be reused this tick.
        """
        still_running: list[RunningProc] = []
        for item in self._running:
            rc = item.proc.returncode
            if rc is None:
                still_running.append(item)
                continue
            await self._patch_terminal(item, rc)
        self._running = still_running

    async def _patch_terminal(self, item: RunningProc, returncode: int) -> None:
        try:
            current = await self._api.get_rollout(item.rollout_id)
        except AglLiteError:
            log.warning(
                "Cannot read rollout to patch terminal",
                rollout_id=item.rollout_id,
            )
            return

        # Allow falling from RUNNING (normal) or QUEUING (the spawn→RUNNING
        # patch failed and the subprocess exited before we retried — rare but
        # possible). Anything else means another path already terminated this
        # rollout; skip.
        if current.status not in (RolloutStatus.QUEUING, RolloutStatus.RUNNING):
            return

        job_name = _local_job_name(item.rollout_id)
        if returncode == 0:
            # success wins: even if cancel_requested=True we mark SUCCEEDED
            # because the trajectory is complete and usable for training.
            await self._patch(
                item.rollout_id,
                PatchRolloutRequest(
                    status=RolloutStatus.SUCCEEDED,
                    succeeded_attempt_id=item.attempt_id,
                    job_name=job_name,
                ),
            )
        elif current.cancel_requested:
            await self._patch(
                item.rollout_id,
                PatchRolloutRequest(
                    status=RolloutStatus.CANCELLED,
                    job_name=job_name,
                ),
            )
        else:
            await self._patch(
                item.rollout_id,
                PatchRolloutRequest(
                    status=RolloutStatus.TERMINAL_FAILED,
                    error_message=f"subprocess exited with code {returncode}",
                    job_name=job_name,
                ),
            )

    # -------- (2) timeout / cancel enforcement --------

    async def _enforce_termination(self) -> None:
        """SIGKILL the process group for any live proc that's cancelled or timed out."""
        now = time.monotonic()
        for item in self._running:
            if item.killed or item.proc.returncode is not None:
                continue

            try:
                current = await self._api.get_rollout(item.rollout_id)
            except AglLiteError:
                current = None

            should_kill = False
            if current is not None and current.cancel_requested:
                should_kill = True
            effective_timeout = self._effective_timeout(current)
            if effective_timeout is not None and (now - item.spawned_at) > effective_timeout:
                should_kill = True

            if should_kill:
                self._kill_process_group(item)

    def _effective_timeout(self, rollout: Rollout | None) -> float | None:
        """Per-rollout timeout in seconds, shared with K8s activeDeadlineSeconds.

        ``None`` means "no controller-side timeout for this rollout", matching
        the K8s ``activeDeadlineSeconds`` default semantics.
        """
        if rollout is None or rollout.config is None:
            return None
        return float(rollout.config.timeout) if rollout.config.timeout else None

    def _kill_process_group(self, item: RunningProc) -> None:
        """SIGKILL the worker's process group.

        Workers are spawned with ``start_new_session=True`` so each is its own
        process-group leader; killing the group also cleans up any grandchildren
        the agent spawned (MCP server, tool subprocesses, etc.).
        """
        if item.proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(item.proc.pid, signal.SIGKILL)
        item.killed = True
        log.info(
            "SIGKILL sent to subprocess group",
            rollout_id=item.rollout_id,
            pid=item.proc.pid,
        )

    # -------- (3) admit new rollouts --------

    async def _admit_queuing(self) -> None:
        if self._stop.is_set():
            # Shutting down — don't spawn any more subprocesses this tick.
            return
        capacity_left = self._pool_size - len(self._running)
        if capacity_left <= 0:
            return

        queuing = await self._api.query_rollouts(status_in=[RolloutStatus.QUEUING], limit=500)
        for r in queuing:
            if self._stop.is_set():
                break
            if r.cancel_requested:
                await self._patch(
                    r.rollout_id,
                    PatchRolloutRequest(
                        status=RolloutStatus.CANCELLED,
                        job_name=_local_job_name(r.rollout_id),
                    ),
                )
                continue
            if time.time() - r.created_at > self._settings.max_queue_time:
                await self._patch(
                    r.rollout_id,
                    PatchRolloutRequest(
                        status=RolloutStatus.TERMINAL_FAILED,
                        error_message=(f"Exceeded max queue time ({self._settings.max_queue_time}s)"),
                        job_name=_local_job_name(r.rollout_id),
                    ),
                )
                continue
            if capacity_left <= 0:
                break
            if await self._spawn_for(r):
                capacity_left -= 1

    async def _spawn_for(self, rollout: Rollout) -> bool:
        """Spawn one subprocess for the given rollout.

        Returns True on successful spawn (added to list + patched RUNNING),
        False on spawn failure (patched TERMINAL_FAILED, not added to list).
        """
        attempt_id = self._attempt_id_for(rollout.rollout_id)
        env = self._build_worker_env(rollout, attempt_id)
        job_name = _local_job_name(rollout.rollout_id)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "agl_lite.controller.local_worker",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=None,  # inherit controller stdout for local debugging
                stderr=None,
                env=env,
                start_new_session=True,
            )
        except Exception as e:
            log.exception("Spawn failed", rollout_id=rollout.rollout_id)
            await self._patch(
                rollout.rollout_id,
                PatchRolloutRequest(
                    status=RolloutStatus.TERMINAL_FAILED,
                    error_message=f"local subprocess spawn failed: {e}",
                    job_name=job_name,
                ),
            )
            return False

        self._running.append(
            RunningProc(
                rollout_id=rollout.rollout_id,
                attempt_id=attempt_id,
                proc=proc,
                spawned_at=time.monotonic(),
            )
        )
        await self._patch(
            rollout.rollout_id,
            PatchRolloutRequest(
                status=RolloutStatus.RUNNING,
                job_name=job_name,
            ),
        )
        log.info(
            "Spawned rollout subprocess",
            rollout_id=rollout.rollout_id,
            attempt_id=attempt_id,
            pid=proc.pid,
        )
        return True

    def _build_worker_env(self, rollout: Rollout, attempt_id: str) -> dict[str, str]:
        """Per-rollout env injected into the subprocess.

        Mirrors the K8s PodPatcher (deploy/controller/job-template.yaml.j2) so
        agent code reads the same variable names in both modes. The two extras
        (AGL_TASK_INPUT, AGL_LOCAL_AGENT_CLASS) replace what an on_enqueue hook
        would otherwise have written into the container's command line.
        """
        task_input_json = json.dumps(rollout.input)
        rollout_id = rollout.rollout_id
        base = self._settings.base_url
        key = self._settings.key or ""
        log_dir = f"/tmp/agl-lite/logs/{attempt_id}"
        is_train = bool(getattr(rollout.metadata, "is_train", True))
        return {
            **self._base_env,
            # ---- matches K8s PodPatcher one-to-one ----
            "AGL_ROLLOUT_ID": rollout_id,
            "AGL_ATTEMPT_ID": attempt_id,  # K8s: $(AGL_POD_UID)
            "AGL_KEY": key,
            "OPENAI_API_KEY": key,
            "ANTHROPIC_API_KEY": key,
            "OPENAI_BASE_URL": f"{base}/rollout/{rollout_id}/attempt/{attempt_id}/v1",
            "ANTHROPIC_BASE_URL": f"{base}/rollout/{rollout_id}/attempt/{attempt_id}",
            "AGL_EVENT_URL": f"{base}/rollout/{rollout_id}/attempt/{attempt_id}/events",
            "AGL_LOG_DIR": log_dir,
            # ---- local-mode extras (K8s mode injects these via on_enqueue hook) ----
            "AGL_TASK_INPUT": task_input_json,
            "AGL_LOCAL_AGENT_CLASS": self._agent_class,
            # AGL_IS_TRAIN is *local-only* (no K8s equivalent yet). See process_pool.md §7.
            "AGL_IS_TRAIN": "1" if is_train else "0",
        }

    def _attempt_id_for(self, rollout_id: str) -> str:
        """Single source of truth for the attempt_id derivation rule."""
        return _local_job_name(rollout_id)

    # -------- startup / shutdown --------

    async def _startup_cleanup(self) -> None:
        """Fail local RUNNING rollouts left over from a previous controller process.

        A restarted controller has no handle on yesterday's subprocesses, so
        any RUNNING rollout tagged with our local- prefix must be terminated to
        unblock the training side.
        """
        running = await self._api.query_rollouts(status_in=[RolloutStatus.RUNNING], limit=500)
        for r in running:
            if r.job_name and r.job_name.startswith(_LOCAL_JOB_NAME_PREFIX):
                status = RolloutStatus.CANCELLED if r.cancel_requested else RolloutStatus.TERMINAL_FAILED
                await self._patch(
                    r.rollout_id,
                    PatchRolloutRequest(
                        status=status,
                        error_message=("local controller restarted; previous subprocess was lost"),
                        job_name=_local_job_name(r.rollout_id),
                    ),
                )

    async def _shutdown(self) -> None:
        """Kill all live subprocesses and mark their rollouts CANCELLED.

        Called from ``run()``'s finally branch — the reconcile loop has exited,
        so nobody else mutates ``self._running``. We first reap procs that have
        already exited so they land on their natural terminal state (SUCCEEDED
        for rc=0 — success-wins — or TERMINAL_FAILED for rc!=0); only the procs
        we still have to kill end up CANCELLED with the shutdown error_message.
        SIGKILL is uninterceptable; ``proc.wait()`` just collects exit status
        and normally returns within milliseconds. The 5s cap is a safety net
        against a stuck wait pipe.
        """
        # 1) Reap procs that finished before/during shutdown. _reap_exited
        #    removes them from self._running and patches the natural terminal
        #    state (SUCCEEDED / TERMINAL_FAILED / CANCELLED if cancel_requested).
        try:
            await self._reap_exited()
        except Exception:
            log.exception("Final reap during shutdown failed")

        # 2) The remainder are still in flight — SIGKILL the process group.
        for item in self._running:
            if item.proc.returncode is None:
                self._kill_process_group(item)
        for item in self._running:
            try:
                await asyncio.wait_for(item.proc.wait(), timeout=_SHUTDOWN_WAIT_TIMEOUT)
            except TimeoutError:
                log.warning(
                    "Subprocess did not exit after SIGKILL within 5s",
                    rollout_id=item.rollout_id,
                    pid=item.proc.pid,
                )
        # 3) Mark the procs we just killed as CANCELLED with the shutdown
        #    error_message. These really were in-flight when shutdown started.
        for item in self._running:
            try:
                await self._patch(
                    item.rollout_id,
                    PatchRolloutRequest(
                        status=RolloutStatus.CANCELLED,
                        error_message="local controller shutdown",
                        job_name=_local_job_name(item.rollout_id),
                    ),
                )
            except Exception:
                log.exception(
                    "Failed to patch CANCELLED on shutdown",
                    rollout_id=item.rollout_id,
                )
        self._running.clear()

    async def _patch(self, rollout_id: str, patch: PatchRolloutRequest) -> None:
        try:
            await self._api.patch_rollout(rollout_id, patch)
        except AglLiteError as e:
            log.warning(
                "Failed to patch rollout",
                rollout_id=rollout_id,
                error=str(e),
            )

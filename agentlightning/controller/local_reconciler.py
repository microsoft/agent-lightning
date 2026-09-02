# Copyright (c) Microsoft. All rights reserved.

"""Local reconciler that runs rollouts as short-lived Python subprocesses."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass

import httpx
import structlog
from omegaconf import DictConfig

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.schemas import DEFAULT_ATTEMPT_ID, Rollout, RolloutPatch, RolloutState, RolloutStatusPatch

log = structlog.get_logger()

_SHUTDOWN_WAIT_TIMEOUT = 5.0


def _run_local_reconciler_worker(agent_class_path: str) -> int:
    try:
        if ":" in agent_class_path:
            module_name, class_name = agent_class_path.split(":", 1)
        else:
            module_name, class_name = agent_class_path.rsplit(".", 1)
        loaded = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(loaded, type):
            raise TypeError(f"{agent_class_path} is not a class")
        result = loaded().run()
        if inspect.isawaitable(result):
            asyncio.run(result)  # type: ignore[arg-type]
        return 0
    except Exception:
        traceback.print_exc()
        return 1


@dataclass
class Proc:
    """In-flight local subprocess."""

    attempt_id: str
    proc: asyncio.subprocess.Process
    spawned_at: float
    killed: bool = False


def _build_env_from_map(task_input: object, env_map: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name, path in env_map.items():
        value = _resolve_input_path(task_input, path)
        if isinstance(value, str):
            env[name] = value
            continue
        try:
            env[name] = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"local.env_map.{name} value is not JSON serializable") from exc
    return env


def _resolve_input_path(task_input: object, path: str) -> object:
    if path == "input":
        return task_input
    if not path.startswith("input."):
        return path

    value = task_input
    for part in path.split(".")[1:]:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError(f"local.env_map path not found: {path}")
    return value


class LocalReconciler:
    """Local-mode rollout reconciler."""

    def __init__(
        self,
        api: AgentLightningAsyncClient,
        config: DictConfig,
    ) -> None:
        assert config.runner_type == "local"
        self._api = api
        self._config = config
        self._runner_config = config.local_runner
        self._pool_size = int(self._runner_config.maximum_size)
        self._tick_interval = float(self._runner_config.poll_interval)
        self._rid_to_proc: dict[str, Proc] = {}
        self._stop = asyncio.Event()

    async def run(self) -> None:
        log.info("LocalReconciler starting", pool_size=self._pool_size, tick=self._tick_interval)
        try:
            await self._reconcile_loop()
        finally:
            await self._shutdown()

    def stop(self) -> None:
        self._stop.set()

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._reconcile_once()
            except Exception:
                log.exception("Local reconcile error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._tick_interval)
                break
            except TimeoutError:
                pass

    async def _reconcile_once(self) -> None:
        params = httpx.QueryParams()
        params = params.add("state_in", RolloutState.QUEUING.value)
        params = params.add("state_in", RolloutState.RUNNING.value)
        params = params.add("limit", 50)
        response = await self._api.get("/api/rollouts", params=params)
        response.raise_for_status()
        rollouts = [Rollout.model_validate(item) for item in response.json()]
        rollouts_by_id = {rollout.rollout_id: rollout for rollout in rollouts}
        live_count = sum(1 for item in self._rid_to_proc.values() if item.proc.returncode is None)

        for rollout in rollouts:
            item = self._rid_to_proc.get(rollout.rollout_id)

            if item is None:
                if rollout.status.state == RolloutState.QUEUING and live_count < self._pool_size:
                    if await self._spawn_for(rollout):
                        live_count += 1
                elif rollout.status.state == RolloutState.RUNNING:
                    await self._patch(rollout.rollout_id, RolloutState.FAILED, "local subprocess is not running")
                continue

            if item.proc.returncode is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._patch(rollout.rollout_id, RolloutState.RUNNING, last_attempt_id=item.attempt_id)
                continue

            await self._finish_proc(rollout, item)

        now = time.monotonic()
        for rollout_id, item in list(self._rid_to_proc.items()):
            if item.proc.returncode is not None:
                continue
            rollout = rollouts_by_id.get(rollout_id)
            timeout = float(rollout.config.timeout_seconds) if rollout and rollout.config.timeout_seconds else None
            if (
                timeout is not None
                and (now - item.spawned_at) > timeout
                and await self._kill_process_group(rollout_id, item)
            ):
                await self._patch(rollout_id, RolloutState.FAILED, "local subprocess timed out")

    async def _finish_proc(self, rollout: Rollout, item: Proc) -> bool:
        if rollout.status.state == RolloutState.QUEUING:
            patched = await self._patch(rollout.rollout_id, RolloutState.RUNNING, last_attempt_id=item.attempt_id)
            if not patched:
                return False
        if item.proc.returncode == 0:
            return await self._patch(rollout.rollout_id, RolloutState.SUCCEEDED, last_attempt_id=item.attempt_id)
        return await self._patch(
            rollout.rollout_id,
            RolloutState.FAILED,
            f"subprocess exited with code {item.proc.returncode}",
        )

    async def _kill_process_group(self, rollout_id: str, item: Proc) -> bool:
        """SIGKILL the worker process group and wait for exit."""
        if item.proc.returncode is not None:
            return True
        if not item.killed:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(item.proc.pid, signal.SIGKILL)
            item.killed = True
            log.info("SIGKILL sent to subprocess group", rollout_id=rollout_id, pid=item.proc.pid)
        try:
            await asyncio.wait_for(item.proc.wait(), timeout=_SHUTDOWN_WAIT_TIMEOUT)
            return True
        except TimeoutError:
            log.warning("Subprocess did not exit after SIGKILL within 5s", rollout_id=rollout_id, pid=item.proc.pid)
            return False

    async def _spawn_for(self, rollout: Rollout) -> bool:
        """Spawn one local subprocess for a rollout."""
        try:
            attempt_id = DEFAULT_ATTEMPT_ID
            if rollout.config.local is None or not rollout.config.local.agent_class:
                raise ValueError("invalid rollout config: missing config.local.agent_class")
            agent_class = rollout.config.local.agent_class
            mode = "train" if rollout.is_train else "val"
            agent_url = self._config.agl_server.get("agent_url", None)
            agent_base_url = str(agent_url or self._config.agl_server.url).rstrip("/")
            env = {
                **os.environ,
                "AGL_KEY": str(self._config.agl_server.key or ""),
                "AGL_OPENAI_BASE_URL": (
                    f"{agent_base_url}/proxy/rollout/{rollout.rollout_id}/attempt/{attempt_id}/mode/{mode}/openai/v1"
                ),
                "AGL_EVENT_URL": (f"{agent_base_url}/api/rollouts/{rollout.rollout_id}/attempt/{attempt_id}/events"),
            }
            env.update(_build_env_from_map(rollout.input, rollout.config.local.env_map))
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from agentlightning.controller.local_reconciler import _run_local_reconciler_worker; "
                    "sys.exit(_run_local_reconciler_worker(sys.argv[1]))"
                ),
                agent_class,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                env=env,
                start_new_session=True,
            )
        except Exception as e:
            log.exception("Spawn failed", rollout_id=rollout.rollout_id)
            await self._patch(rollout.rollout_id, RolloutState.FAILED, f"local subprocess spawn failed: {e}")
            return False

        self._rid_to_proc[rollout.rollout_id] = Proc(
            attempt_id=attempt_id,
            proc=proc,
            spawned_at=time.monotonic(),
        )
        await self._patch(rollout.rollout_id, RolloutState.RUNNING, last_attempt_id=attempt_id)
        log.info("Spawned rollout subprocess", rollout_id=rollout.rollout_id, attempt_id=attempt_id, pid=proc.pid)
        return True

    async def _shutdown(self) -> None:
        """Kill live subprocesses and mark them failed."""
        try:
            await self._reconcile_once()
        except Exception:
            log.exception("Final reconcile during shutdown failed")

        for rollout_id, item in list(self._rid_to_proc.items()):
            if item.proc.returncode is None and await self._kill_process_group(rollout_id, item):
                await self._patch(rollout_id, RolloutState.FAILED, "local controller shutdown")

    async def _patch(
        self,
        rollout_id: str,
        state: RolloutState,
        error_message: str | None = None,
        *,
        last_attempt_id: str | None = None,
    ) -> bool:
        status = RolloutStatusPatch(state=state)
        if error_message is not None:
            status.error_message = error_message
        if last_attempt_id is not None:
            status.last_attempt_id = last_attempt_id
        patch = RolloutPatch(status=status)
        try:
            response = await self._api.patch(
                f"/api/rollouts/{rollout_id}",
                json=patch.model_dump(mode="json", exclude_unset=True),
            )
            response.raise_for_status()
            return True
        except Exception as e:
            log.warning("Failed to patch rollout", rollout_id=rollout_id, error=str(e))
            return False

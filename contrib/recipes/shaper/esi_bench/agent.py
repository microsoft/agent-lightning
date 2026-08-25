# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning wrapper around a fresh-process official ESI-Bench run."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from agentlightning.litagent import LitAgent
from agentlightning.types import NamedResources, Rollout
from contrib.agentlightning.contrib.shaper import (
    EpisodeMetadata,
    RoundRecord,
    emit_episode_metadata,
    emit_round_record,
)

from ..common import require_llm, require_prompt
from ..harness_bridge import HarnessBridgeServer
from .contracts import make_harness_validator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ESIBenchRuntimeConfig:
    """Runtime configuration shared by one ESI-Bench runner process."""

    esi_bench_root: Path
    behavior_root: Path
    questions_jsonl: Path
    output_root: Path
    omnigibson_data_root: Path | None = None
    worker_python: Path = Path(sys.executable)
    planner_resource_name: str = "planner_llm"
    skill_resource_name: str = "skill"
    harness_resource_name: str = "harness"
    max_steps: int = 30
    min_steps: int = 3
    confidence_threshold: float = 0.85
    max_new_tokens: int = 32_768
    temperature: float = 1.0
    top_p: float = 0.95
    robot: str = "R1"
    episode_timeout_seconds: float = 1800.0
    environment_retries: int = 1
    harness_timeout_seconds: float = 3.0
    harness_memory_limit_mb: int = 768
    harness_max_output_chars: int = 24_000_000

    def __post_init__(self) -> None:
        for name in ("max_steps", "min_steps", "max_new_tokens"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps must not exceed max_steps.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one.")
        if self.episode_timeout_seconds <= 0:
            raise ValueError("episode_timeout_seconds must be positive.")
        if self.environment_retries < 0:
            raise ValueError("environment_retries must be non-negative.")


def _run_token(rollout: Rollout, task_id: str) -> str:
    value = f"{rollout.rollout_id}:{task_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def _worker_request(
    config: ESIBenchRuntimeConfig,
    task: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    api_key: str | None,
    sampling_parameters: Mapping[str, Any],
    skill: str,
    harness_socket: Path,
    harness_token: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Build the private worker request; this object is sent over stdin only."""

    return {
        "esi_bench_root": str(config.esi_bench_root),
        "behavior_root": str(config.behavior_root),
        "questions_jsonl": str(config.questions_jsonl),
        "run_dir": str(run_dir),
        "task": dict(task),
        "planner": {
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key,
            "sampling_parameters": dict(sampling_parameters),
        },
        "skill": skill,
        "harness_bridge": {
            "socket_path": str(harness_socket),
            "token": harness_token,
            "timeout_seconds": config.harness_timeout_seconds + 5.0,
            "max_response_bytes": config.harness_max_output_chars + 2_000_000,
        },
        "runtime": {
            "max_steps": int(task.get("max_steps", config.max_steps)),
            "min_steps": config.min_steps,
            "confidence_threshold": config.confidence_threshold,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "robot": config.robot,
            "harness_timeout_seconds": config.harness_timeout_seconds,
            "harness_memory_limit_mb": config.harness_memory_limit_mb,
            "harness_max_output_chars": config.harness_max_output_chars,
        },
    }


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the simulator and any child processes after a hard timeout."""

    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            return
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _worker_command(config: ESIBenchRuntimeConfig, response_path: Path) -> list[str]:
    """Build the command for the isolated simulator interpreter."""

    return [
        str(config.worker_python),
        "-m",
        "contrib.recipes.shaper.esi_bench.worker",
        "--response-path",
        str(response_path),
    ]


def _worker_environment(config: ESIBenchRuntimeConfig) -> dict[str, str]:
    """Build the isolated worker environment from validated runtime paths."""

    environment = os.environ.copy()
    recipe_checkout = Path(__file__).resolve().parents[4]
    omnigibson_checkout = config.behavior_root / "OmniGibson"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    worker_paths = [str(recipe_checkout), str(omnigibson_checkout)]
    if existing_pythonpath:
        worker_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(worker_paths)
    if config.omnigibson_data_root is not None:
        data_root = str(config.omnigibson_data_root.expanduser().resolve())
        environment["ESI_OMNIGIBSON_DATA_ROOT"] = data_root
        environment["OMNIGIBSON_DATA_PATH"] = data_root
    return environment


class ESIBenchAgent(LitAgent[dict[str, Any]]):
    """Execute SHAPER artifacts through ESI-Bench's official ``run_one``."""

    def __init__(self, config: ESIBenchRuntimeConfig) -> None:
        super().__init__()
        self.config = config

    def _run_worker(
        self,
        task: Mapping[str, Any],
        *,
        endpoint: str,
        model: str,
        api_key: str | None,
        sampling_parameters: Mapping[str, Any],
        skill: str,
        harness: str,
        attempt_dir: Path,
    ) -> tuple[dict[str, Any], int | None, bool, Path]:
        """Run one isolated simulator attempt and return its private response."""

        attempt_dir.mkdir(parents=True, exist_ok=True)
        response_path = attempt_dir / "worker_response.json"
        log_path = attempt_dir / "simulator.log"
        validator = make_harness_validator(
            timeout_seconds=self.config.harness_timeout_seconds,
            memory_limit_mb=self.config.harness_memory_limit_mb,
            max_output_chars=self.config.harness_max_output_chars,
        )
        runtime = validator.runtime(harness)
        timed_out = False
        with HarnessBridgeServer(
            runtime,
            max_response_bytes=self.config.harness_max_output_chars + 2_000_000,
        ) as bridge:
            assert bridge.socket_path is not None
            request = _worker_request(
                self.config,
                task,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                sampling_parameters=sampling_parameters,
                skill=skill,
                harness_socket=bridge.socket_path,
                harness_token=bridge.token,
                run_dir=attempt_dir,
            )
            command = _worker_command(self.config, response_path)
            worker_environment = _worker_environment(self.config)
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=os.name == "posix",
                    cwd=self.config.esi_bench_root,
                    env=worker_environment,
                )
                try:
                    process.communicate(
                        json.dumps(request, ensure_ascii=True),
                        timeout=self.config.episode_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)

        payload: dict[str, Any] = {}
        if response_path.is_file():
            try:
                value: object = json.loads(response_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    payload = cast(dict[str, Any], value)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Invalid ESI worker response in %s: %s", attempt_dir, exc)
        return payload, process.returncode, timed_out, log_path

    def rollout(self, task: dict[str, Any], resources: NamedResources, rollout: Rollout) -> float:
        planner = require_llm(resources, self.config.planner_resource_name, rollout)
        skill = require_prompt(resources, self.config.skill_resource_name)
        harness = require_prompt(resources, self.config.harness_resource_name)
        task_id = str(task.get("task_id", "esi/unknown"))
        run_dir = self.config.output_root / _run_token(rollout, task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        attempted_logs: list[str] = []
        reason = "worker_failure"
        message = "ESI-Bench worker did not produce a result."
        environment_invalid = False

        for attempt in range(self.config.environment_retries + 1):
            payload, return_code, timed_out, log_path = self._run_worker(
                task,
                endpoint=planner.get_base_url(),
                model=planner.model,
                api_key=planner.api_key,
                sampling_parameters=planner.sampling_parameters,
                skill=skill,
                harness=harness,
                attempt_dir=run_dir / f"attempt_{attempt + 1}",
            )
            attempted_logs.append(str(log_path))

            if payload.get("ok") is True:
                rounds = payload.get("rounds", [])
                if isinstance(rounds, list):
                    for value in cast(list[Any], rounds):
                        emit_round_record(RoundRecord.model_validate(value))
                metadata = EpisodeMetadata.model_validate(payload.get("metadata", {}))
                metadata.extra["simulator_attempts"] = attempt + 1
                metadata.extra["worker_logs"] = attempted_logs
                emit_episode_metadata(metadata)
                return float(payload.get("reward", 0.0))

            if timed_out:
                reason = "worker_hard_timeout"
                message = f"ESI-Bench worker exceeded {self.config.episode_timeout_seconds:.1f}s."
                environment_invalid = False
            elif return_code is not None and (return_code < 0 or return_code in {134, 139}):
                reason = "simulator_process_crash"
                message = f"ESI-Bench worker exited with code {return_code}."
                environment_invalid = True
            elif not payload:
                raise RuntimeError(
                    f"ESI-Bench worker produced no valid response for {task_id}; "
                    f"exit_code={return_code}, see {log_path}."
                )
            else:
                reason = str(payload.get("termination_reason", "worker_failure"))
                message = str(payload.get("error", f"ESI-Bench worker exited with code {return_code}."))
                environment_invalid = bool(payload.get("environment_invalid", False))
                if payload.get("failure_kind") == "infrastructure":
                    raise RuntimeError(f"ESI-Bench adapter/upstream failure for {task_id}; see {log_path}: {message}")
                if (
                    return_code not in {None, 0}
                    and payload.get("failure_kind")
                    not in {
                        "planner",
                        "artifact",
                    }
                    and not environment_invalid
                ):
                    raise RuntimeError(
                        f"ESI-Bench worker exited unexpectedly for {task_id}; "
                        f"exit_code={return_code}, see {log_path}: {message}"
                    )
            if environment_invalid and attempt < self.config.environment_retries:
                logger.warning(
                    "Retrying ESI-Bench rollout %s in a fresh process after environment failure: %s",
                    task_id,
                    message,
                )
                continue
            break

        emit_episode_metadata(
            EpisodeMetadata(
                environment_invalid=environment_invalid,
                termination_reason=reason,
                runtime_errors=[message],
                extra={
                    "task_id": task_id,
                    "simulator_attempts": len(attempted_logs),
                    "worker_logs": attempted_logs,
                },
            )
        )
        logger.warning("ESI-Bench rollout %s failed: %s", task_id, message)
        return 0.0

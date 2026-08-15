# Copyright (c) Microsoft. All rights reserved.

"""Hierarchical VLM-VLA Agent Lightning rollout for VLABench.

The official VLABench environment and reward remain authoritative. SHAPER's
planner issues one natural-language command per round; a frozen OpenPI policy
executes that command through the benchmark's websocket protocol.
"""

from __future__ import annotations

import collections
import importlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, cast

from agentlightning.contrib.shaper import (
    EpisodeMetadata,
    RoundRecord,
    emit_episode_metadata,
    emit_round_record,
)
from agentlightning.litagent import LitAgent
from agentlightning.types import NamedResources, Rollout

from ..common import (
    completion_text,
    ensure_jsonable,
    image_data_url,
    image_part,
    normalize_content,
    openai_client,
    require_llm,
    require_prompt,
    strip_thinking,
    text_part,
)
from .contracts import make_harness_validator
from .openpi_identity import (
    REPORTED_THREE_CAMERA,
    SUPPORTED_OBSERVATION_SCHEMAS,
    validate_server_metadata,
)

logger = logging.getLogger(__name__)
_ENVIRONMENT_CREATE_LOCK = threading.Lock()


class ContextBuilder(Protocol):
    """Restricted context-harness call interface."""

    def __call__(self, history: list[dict[str, Any]]) -> Any: ...


class ActorInfrastructureError(RuntimeError):
    """The frozen OpenPI actor service failed independently of a candidate."""


@dataclass(frozen=True)
class VLABenchRuntimeConfig:
    """Runtime configuration for one VLABench runner process."""

    vlabench_root: Path
    planner_resource_name: str = "planner_llm"
    skill_resource_name: str = "skill"
    harness_resource_name: str = "harness"
    vla_host: str = "127.0.0.1"
    vla_port: int = 8000
    vla_replan_steps: int = 5
    vla_inference_timeout_seconds: float = 300.0
    max_vlm_rounds: int = 10
    default_round_steps: int = 200
    min_round_steps: int = 1
    planner_max_completion_tokens: int = 32_768
    max_substeps: int = 1
    joint_tolerance: float = 0.01
    reset_wait_steps: int = 10
    harness_timeout_seconds: float = 3.0
    harness_memory_limit_mb: int = 768
    harness_max_output_chars: int = 32_000_000
    observation_schema: str = REPORTED_THREE_CAMERA
    expected_actor_id: str = ""
    expected_policy_config: str = ""

    def __post_init__(self) -> None:
        for name in (
            "vla_port",
            "vla_replan_steps",
            "max_vlm_rounds",
            "default_round_steps",
            "min_round_steps",
            "planner_max_completion_tokens",
            "max_substeps",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.joint_tolerance <= 0:
            raise ValueError("joint_tolerance must be positive.")
        if self.vla_inference_timeout_seconds <= 0:
            raise ValueError("vla_inference_timeout_seconds must be positive.")
        if self.observation_schema not in SUPPORTED_OBSERVATION_SCHEMAS:
            raise ValueError(
                "observation_schema must be one of "
                + ", ".join(sorted(SUPPORTED_OBSERVATION_SCHEMAS))
                + f"; got {self.observation_schema!r}."
            )
        identity_fields = (self.expected_actor_id, self.expected_policy_config)
        if any(identity_fields) and not all(identity_fields):
            raise ValueError("expected_actor_id and expected_policy_config must be configured together.")


class _TimedOpenPIClient:
    """Pinned OpenPI websocket protocol with finite connect and receive waits."""

    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        websocket_client = cast(Any, importlib.import_module("websockets.sync.client"))
        msgpack_numpy = cast(Any, importlib.import_module("openpi_client.msgpack_numpy"))
        self._timeout_seconds = timeout_seconds
        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        self._connection = websocket_client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout_seconds,
            close_timeout=min(timeout_seconds, 10.0),
        )
        metadata = self._connection.recv(timeout=timeout_seconds)
        self.server_metadata = self._unpack(metadata)

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._connection.send(self._packer.pack(dict(observation)))
        response = self._connection.recv(timeout=self._timeout_seconds)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server returned an error: {response}")
        value: object = self._unpack(response)
        if not isinstance(value, Mapping):
            raise TypeError("OpenPI response must be a mapping.")
        return cast(Mapping[str, Any], value)

    def close(self) -> None:
        self._connection.close()


class _OpenPIVLAPolicy:
    """Frozen OpenPI websocket actor using VLABench's official observation map."""

    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int,
        observation_schema: str,
        inference_timeout_seconds: float,
        expected_actor_id: str = "",
        expected_policy_config: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._client: Any = None
        self._replan_steps = replan_steps
        if observation_schema not in SUPPORTED_OBSERVATION_SCHEMAS:
            raise ValueError(f"Unsupported VLABench observation schema: {observation_schema!r}.")
        self._observation_schema = observation_schema
        self._inference_timeout_seconds = inference_timeout_seconds
        self._expected_actor_id = expected_actor_id
        self._expected_policy_config = expected_policy_config
        self._actions: collections.deque[Any] = collections.deque()
        self._connect_with_retry()

    def _connect(self) -> None:
        self._discard_client()
        client = _TimedOpenPIClient(
            self._host,
            self._port,
            self._inference_timeout_seconds,
        )
        if self._expected_actor_id:
            errors = validate_server_metadata(
                client.server_metadata,
                expected_actor_id=self._expected_actor_id,
                expected_policy_config=self._expected_policy_config,
                expected_observation_schema=self._observation_schema,
            )
            if errors:
                client.close()
                raise RuntimeError("OpenPI actor identity mismatch: " + "; ".join(errors))
        self._client = client

    def _discard_client(self) -> None:
        client = self._client
        self._client = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Ignoring an error while closing a failed OpenPI connection.", exc_info=True)

    def _connect_with_retry(self, attempts: int = 3) -> None:
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                self._connect()
                return
            except Exception as exc:
                last_error = exc
                self._discard_client()
                if attempt + 1 < attempts:
                    time.sleep(float(2**attempt))
        raise ActorInfrastructureError(
            f"Could not connect to OpenPI at {self._host}:{self._port} after {attempts} attempts."
        ) from last_error

    def reset(self) -> None:
        self._actions.clear()

    def predict(self, observation: Mapping[str, Any], instruction: str) -> tuple[Any, Any, Any]:
        import numpy as np

        vlabench_utils = cast(Any, importlib.import_module("VLABench.utils.utils"))

        if not self._actions:
            rgb = observation["rgb"]
            ee_state = observation["ee_state"]
            position = ee_state[:3] - np.array([0.0, -0.4, 0.78])
            state = np.concatenate(
                [
                    position,
                    vlabench_utils.quaternion_to_euler(ee_state[3:7]),
                    np.asarray(ee_state[-1]).reshape(-1),
                ]
            )
            if len(rgb) < 4:
                raise ValueError(f"VLABench actor requires four RGB views, received {len(rgb)}.")
            payload = {
                "observation/image": rgb[2],
                "observation/second_image": rgb[0],
                "observation/wrist_image": rgb[3],
                "observation/state": state,
                "prompt": instruction,
            }
            last_error: BaseException | None = None
            response: Mapping[str, Any] | None = None
            for attempt in range(3):
                try:
                    response = self._client.infer(payload)
                    break
                except Exception as exc:
                    last_error = exc
                    self._discard_client()
                    if attempt < 2:
                        time.sleep(float(2**attempt))
                        self._connect_with_retry()
            if response is None:
                raise RuntimeError("OpenPI inference failed after three connection attempts.") from last_error
            action_chunk = response.get("actions")
            if action_chunk is None or len(action_chunk) < self._replan_steps:
                raise ActorInfrastructureError(
                    f"OpenPI returned {0 if action_chunk is None else len(action_chunk)} actions; "
                    f"at least {self._replan_steps} are required."
                )
            self._actions.extend(action_chunk[: self._replan_steps])

        action = np.asarray(self._actions.popleft())
        target_position = action[:3].copy() + np.array([0.0, -0.4, 0.78])
        target_euler = action[3:6]
        gripper = np.ones(2) * 0.04 if float(action[-1]) >= 0.1 else np.zeros(2)
        return target_position, target_euler, gripper

    def close(self) -> None:
        self._discard_client()


def _load_environment(config: VLABenchRuntimeConfig, task: Mapping[str, Any]) -> Any:
    os.environ["VLABENCH_ROOT"] = str(config.vlabench_root)
    os.environ.setdefault("MUJOCO_GL", "egl")

    importlib.import_module("VLABench.robots")
    importlib.import_module("VLABench.tasks")
    environments = cast(Any, importlib.import_module("VLABench.envs"))

    # VLABench mutates shared task configuration while constructing an
    # environment, and concurrent EGL initialization is not thread-safe.
    with _ENVIRONMENT_CREATE_LOCK:
        return environments.load_env(
            str(task["task_name"]),
            episode_config=task["episode_config"],
            random_init=False,
            reset_wait_step=config.reset_wait_steps,
            run_mode="eval",
        )


def _observable_images(observation: Mapping[str, Any], observation_schema: str) -> tuple[Any, Any]:
    if observation_schema != REPORTED_THREE_CAMERA:
        raise ValueError(f"Unsupported VLABench observation schema: {observation_schema!r}.")
    rgb = observation["rgb"]
    return rgb[2], rgb[3]


def _observation_parts(
    observation: Mapping[str, Any],
    observation_schema: str,
) -> list[dict[str, Any]]:
    main, wrist = _observable_images(observation, observation_schema)
    return [
        text_part("Main camera (third-person)"),
        image_part(image_data_url(main)),
        text_part("Wrist camera (gripper)"),
        image_part(image_data_url(wrist)),
    ]


def _parse_plan(raw_text: str, fallback: str, default_steps: int) -> tuple[str, str, int]:
    visible = strip_thinking(raw_text)
    answer = re.search(r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?=\n\s*Steps\s*:|\Z)", visible, re.I | re.S)
    steps = re.search(r"(?:^|\n)\s*Steps\s*:\s*(\d+)", visible, re.I)
    command = answer.group(1).strip() if answer else fallback.strip()
    command = command.splitlines()[0].strip() or fallback.strip()
    estimated_steps = int(steps.group(1)) if steps else default_steps
    reasoning = visible[: answer.start()].strip() if answer else visible.strip()
    return reasoning, command, estimated_steps


def _planner_content(
    *,
    instruction: str,
    current_step: int,
    round_index: int,
    context: Any,
    observation: Mapping[str, Any],
    observation_schema: str,
) -> list[dict[str, Any]]:
    content = [
        text_part(
            "VLABench task instruction:\n"
            + instruction
            + f"\n\nEnvironment step: {current_step}\nPlanner round: {round_index + 1}"
        ),
        text_part("Observable execution context built by the harness:"),
        *normalize_content(context),
        text_part("Current observation:"),
        *_observation_parts(observation, observation_schema),
    ]
    return content


def _official_progress(environment: Any) -> tuple[bool, float]:
    """Read the benchmark reward only for scoring and termination."""

    try:
        progress = float(environment.get_task_progress())
        return progress >= 1.0, min(1.0, max(0.0, progress))
    except Exception:
        pass

    task = getattr(environment, "task", None)
    physics = getattr(environment, "physics", None)
    conditions = getattr(task, "conditions", None)
    if conditions is None:
        return False, 0.0

    try:
        completed = bool(conditions.is_met(physics))
    except Exception:
        completed = False
    if completed:
        return True, 1.0

    def normalized_progress(value: Any) -> float | None:
        if isinstance(value, tuple) and value:
            value = cast(tuple[Any, ...], value)[0]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if not isinstance(value, (int, float)):
            return None
        return min(1.0, max(0.0, float(value)))

    # VLABench's composite tasks expose one of three official condition
    # layouts. Reading them here preserves partial reward when the benchmark's
    # convenience wrapper fails; none of these values enter planner context.
    condition_history = getattr(conditions, "condition_has_been_met", None)
    if condition_history is not None:
        history = list(condition_history)
        if history:
            return False, sum(bool(item) for item in history) / len(history)
        return False, 0.0

    condition_sets = getattr(conditions, "condition_sets", None)
    if condition_sets is not None:
        progresses: list[float] = []
        for condition_set in condition_sets:
            try:
                progress = normalized_progress(condition_set.met_progress(physics))
            except Exception:
                try:
                    progress = normalized_progress(condition_set.is_met(physics))
                except Exception:
                    progress = None
            if progress is not None:
                progresses.append(progress)
        return False, max(progresses, default=0.0)

    met_progress = getattr(conditions, "met_progress", None)
    if callable(met_progress):
        try:
            progress = normalized_progress(met_progress(physics))
        except Exception:
            progress = None
        if progress is not None:
            return False, progress
    return False, 0.0


def _is_simulator_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "physics state is invalid",
        "badqacc",
        "mujoco fatal",
        "egl",
        "framebuffer",
        "render context",
        "failed to initialize",
    )
    return any(marker in text for marker in markers)


def _visible_instruction(task: Mapping[str, Any], environment: Any) -> str:
    """Return only the benchmark-visible natural-language instruction."""

    override = task.get("instruction")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return str(environment.task.get_instruction()).strip()


class VLABenchAgent(LitAgent[dict[str, Any]]):
    """Run SHAPER artifacts around a frozen VLABench/OpenPI stack."""

    def __init__(
        self,
        config: VLABenchRuntimeConfig,
        *,
        environment_loader: Callable[[VLABenchRuntimeConfig, Mapping[str, Any]], Any] = _load_environment,
        policy_factory: Callable[[str, int, int, str, float, str, str], Any] = _OpenPIVLAPolicy,
    ) -> None:
        super().__init__()
        self.config = config
        self._environment_loader = environment_loader
        self._policy_factory = policy_factory

    def _harness_validator(self) -> Any:
        return make_harness_validator(
            timeout_seconds=self.config.harness_timeout_seconds,
            memory_limit_mb=self.config.harness_memory_limit_mb,
            max_output_chars=self.config.harness_max_output_chars,
        )

    def rollout(self, task: dict[str, Any], resources: NamedResources, rollout: Rollout) -> float:
        import numpy as np

        planner_resource = require_llm(resources, self.config.planner_resource_name, rollout)
        skill = require_prompt(resources, self.config.skill_resource_name)
        harness_source = require_prompt(resources, self.config.harness_resource_name)
        context_builder: ContextBuilder = self._harness_validator().runtime(harness_source)
        planner_client = openai_client(planner_resource)

        environment: Any = None
        policy: Any = None
        history: list[dict[str, Any]] = []
        runtime_errors: list[str] = []
        total_steps = 0
        completed = False
        reward = 0.0
        termination_reason = "step_budget"
        environment_invalid = False
        failure_stage = "environment_startup"

        try:
            environment = self._environment_loader(self.config, task)
            # Match VLABench's official evaluator, which calls reset after
            # load_env returns the constructed environment.
            environment.reset()
            observation = environment.get_observation(require_pcd=False)
            instruction = _visible_instruction(task, environment)
            failure_stage = "actor_startup"
            policy = self._policy_factory(
                self.config.vla_host,
                self.config.vla_port,
                self.config.vla_replan_steps,
                self.config.observation_schema,
                self.config.vla_inference_timeout_seconds,
                self.config.expected_actor_id,
                self.config.expected_policy_config,
            )

            for round_index in range(self.config.max_vlm_rounds):
                if total_steps >= int(task.get("max_steps", 400)):
                    break
                before = _observation_parts(observation, self.config.observation_schema)
                failure_stage = "harness"
                context = context_builder(history)
                content = _planner_content(
                    instruction=instruction,
                    current_step=total_steps,
                    round_index=round_index,
                    context=context,
                    observation=observation,
                    observation_schema=self.config.observation_schema,
                )
                sampling = planner_resource.sampling_parameters
                request: dict[str, Any] = {
                    "model": planner_resource.model,
                    "messages": [
                        {"role": "system", "content": skill},
                        {"role": "user", "content": content},
                    ],
                    "max_completion_tokens": int(
                        sampling.get("max_completion_tokens", self.config.planner_max_completion_tokens)
                    ),
                }
                if "temperature" in sampling:
                    request["temperature"] = sampling["temperature"]
                if "top_p" in sampling:
                    request["top_p"] = sampling["top_p"]
                if "presence_penalty" in sampling:
                    request["presence_penalty"] = sampling["presence_penalty"]
                if isinstance(sampling.get("extra_body"), dict):
                    request["extra_body"] = sampling["extra_body"]
                failure_stage = "planner"
                response = cast(Any, planner_client.chat.completions.create(**request))
                raw_text, _ = completion_text(response)
                reasoning, command, requested_steps = _parse_plan(
                    raw_text,
                    instruction,
                    self.config.default_round_steps,
                )
                remaining = int(task.get("max_steps", 400)) - total_steps
                if round_index + 1 == self.config.max_vlm_rounds:
                    # The reported evaluator stops asking the planner after its
                    # final round but keeps executing that subgoal until the
                    # episode-level step budget is exhausted.
                    round_budget = remaining
                else:
                    round_budget = min(remaining, max(self.config.min_round_steps, requested_steps))
                policy.reset()
                round_errors: list[str] = []
                executed = 0

                for _ in range(round_budget):
                    try:
                        failure_stage = "actor"
                        target_position, target_euler, gripper = policy.predict(
                            observation,
                            command,
                        )
                        vlabench_utils = cast(Any, importlib.import_module("VLABench.utils.utils"))
                        quaternion = vlabench_utils.euler_to_quaternion(*target_euler)
                        ik_success, joints = environment.robot.get_qpos_from_ee_pos(
                            physics=environment.physics,
                            pos=target_position,
                            quat=quaternion,
                        )
                        if not ik_success:
                            round_errors.append("IK solver failed for the frozen actor action.")
                            termination_reason = "ik_failure"
                            break
                        full_action = np.concatenate([joints, gripper])
                        done = False
                        failure_stage = "simulator"
                        for _ in range(self.config.max_substeps):
                            timestep = environment.step(full_action)
                            if timestep.last():
                                done = True
                                break
                            current = np.asarray(environment.task.robot.get_qpos(environment.physics)).reshape(-1)
                            if float(np.max(np.abs(current - full_action[:7]))) < self.config.joint_tolerance:
                                break
                        observation = environment.get_observation(require_pcd=False)
                        total_steps += 1
                        executed += 1
                        completed, reward = _official_progress(environment)
                        if done:
                            # The pinned VLABench environment has an infinite
                            # time limit and its task termination hooks return
                            # true only for successful task conditions. Mirror
                            # the official evaluator, which treats
                            # timestep.last() as the authoritative success
                            # signal even if a composite task's convenience
                            # progress helper is stale or incomplete.
                            completed = True
                            reward = 1.0
                        if completed:
                            termination_reason = "completed"
                            break
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        round_errors.append(message)
                        if _is_simulator_failure(exc):
                            environment_invalid = True
                            termination_reason = "simulator_failure"
                        elif failure_stage == "actor":
                            raise ActorInfrastructureError(
                                "The frozen OpenPI actor failed while producing an action."
                            ) from exc
                        else:
                            raise
                        break

                after = _observation_parts(observation, self.config.observation_schema)
                emit_round_record(
                    RoundRecord(
                        round_index=round_index,
                        task_instruction=instruction,
                        planner_response=reasoning,
                        command=command,
                        observation_before=before,
                        observation_after=after,
                        context_payload=context,
                        execution_steps=executed,
                        action_result={"ik_success": not any("IK solver" in item for item in round_errors)},
                        runtime_errors=round_errors,
                    )
                )
                history.append(
                    {
                        "round_index": round_index,
                        "task_instruction": instruction,
                        "planner_response": reasoning,
                        "command": command,
                        "execution_steps": executed,
                        "observation_before": ensure_jsonable(before),
                        "observation_after": ensure_jsonable(after),
                        "action_result": {"ik_success": not any("IK solver" in item for item in round_errors)},
                        "runtime_errors": list(round_errors),
                    }
                )
                runtime_errors.extend(round_errors)
                if completed or round_errors:
                    break

            if not completed and termination_reason == "step_budget" and total_steps < int(task.get("max_steps", 400)):
                termination_reason = "round_budget"
        except Exception as exc:
            runtime_errors.append(f"{type(exc).__name__}: {exc}")
            simulator_failure = _is_simulator_failure(exc)
            candidate_failure = failure_stage in {"harness", "planner"}
            environment_invalid = failure_stage == "environment_startup" or simulator_failure
            if simulator_failure:
                termination_reason = "simulator_failure"
            elif candidate_failure:
                termination_reason = {
                    "harness": "harness_failure",
                    "planner": "planner_failure",
                }[failure_stage]
            elif failure_stage == "environment_startup":
                termination_reason = "environment_startup_failure"
            else:
                logger.exception(
                    "VLABench infrastructure failure for %s",
                    task.get("task_id", "unknown"),
                )
                raise
            reward = 0.0
            logger.exception("VLABench rollout %s failed", task.get("task_id", "unknown"))
        finally:
            if policy is not None:
                try:
                    policy.close()
                except Exception as exc:
                    runtime_errors.append(f"actor cleanup: {type(exc).__name__}: {exc}")
            if environment is not None:
                try:
                    environment.close()
                except Exception as exc:
                    runtime_errors.append(f"cleanup: {type(exc).__name__}: {exc}")

        emit_episode_metadata(
            EpisodeMetadata(
                environment_invalid=environment_invalid,
                termination_reason=termination_reason,
                runtime_errors=runtime_errors,
                extra={
                    "task_id": str(task.get("task_id", "")),
                    "environment_steps": total_steps,
                    "openpi_observation_schema": self.config.observation_schema,
                    "openpi_actor_id": self.config.expected_actor_id,
                    "openpi_policy_config": self.config.expected_policy_config,
                },
            )
        )
        return float(reward)

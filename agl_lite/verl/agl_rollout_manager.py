"""Rollout managers for agl-lite VERL training."""

from __future__ import annotations

import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from httpx_retries import Retry, RetryTransport
from pydantic import BaseModel, Field

from agl_lite.client import AglLiteSyncClient
from agl_lite.schemas import (
    TERMINAL_STATES,
    Event,
    EventCreate,
    Model,
    Rollout,
    RolloutCreate,
    RolloutState,
)

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional outside VERL installs.
    torch = None

if TYPE_CHECKING:
    from agl_lite.hooks import RolloutHooks


class Triplet(BaseModel):
    """Single prompt-response-reward turn."""

    prompt: Any
    response: Any
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnqueuedRollout(BaseModel):
    """Enqueued rollout request metadata."""

    data_id: str
    rollout_id: str
    step: int
    sample_idx_in_step: int
    enqueue_time: float
    input: Any = None
    # Server timestamps expose pod queue time and completion time.
    running_at: float | None = None
    finished_at: float | None = None


class CompletedRollout(BaseModel):
    """Completed rollout result."""

    rollout_id: str
    data_id: str
    step: int
    sample_idx_in_step: int
    enqueue_time: float
    input: Any = None
    running_at: float | None = None
    finished_at: float | None = None
    final_reward: float | None = None
    triplets: list[Triplet] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    triplet_events: list[dict[str, Any]] = Field(default_factory=list)
    rollout_state: RolloutState | None = None
    error_message: str | None = None


@dataclass
class _TraceEvent:
    rollout_id: str
    attempt_id: str
    event_type: str
    data: dict[str, Any]


class _TraceEventHelper:
    """Queues hook events before HTTP flush."""

    def __init__(self) -> None:
        self._queued: list[_TraceEvent] = []

    def add_event(self, rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> None:
        self._queued.append(_TraceEvent(rollout_id=rollout_id, attempt_id=attempt_id, event_type=event_type, data=data))

    def flush(self, manager: AglRolloutManagerBase) -> None:
        for event in self._queued:
            manager._post_event(
                event.rollout_id,
                event.attempt_id,
                EventCreate(event_type=event.event_type, data=event.data),
            )


def _as_reward_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.number):
        return float(value)
    return None


def _to_native(obj: Any) -> Any:
    """Convert numpy/torch values for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if isinstance(obj, Mapping):
        return {_to_native(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(item) for item in obj]
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    return obj


class AglRolloutManagerBase:
    """Base manager for agl-lite rollout HTTP operations."""

    def __init__(
        self,
        *,
        agl_base_url: str,
        agl_key: str,
        model: str,
        step: int,
        train_rollout_n: int = 1,
        rollout_timeout_seconds: float = 1200.0,
        poll_interval_seconds: float = 1.0,
        hooks: RolloutHooks | None = None,
        local_agent_class: str | None = None,
        local_env_map: dict[str, str] | None = None,
        k8s_job_template_path: str | None = None,
    ) -> None:
        self._model = model
        self._step = step
        self._train_rollout_n = train_rollout_n
        self._poll_interval_seconds = poll_interval_seconds
        self._hooks = hooks
        self._rollout_config: dict[str, Any] = {"timeout_seconds": int(rollout_timeout_seconds)}
        if local_agent_class:
            self._rollout_config["local"] = {
                "agent_class": local_agent_class,
                "env_map": local_env_map or {},
            }
        if k8s_job_template_path:
            self._rollout_config["k8s"] = {"job_template": Path(k8s_job_template_path).read_text()}

        self.client = AglLiteSyncClient(
            base_url=agl_base_url,
            key=agl_key,
            timeout=120.0,
            transport=RetryTransport(retry=Retry(total=10, allowed_methods=["GET"])),
        )

    def register_model(self, server_addresses: list[str]) -> list[Model]:
        """Register model server endpoints."""
        models: list[Model] = []
        for address in server_addresses:
            endpoint = address if address.startswith("http") else f"http://{address}/v1"
            models.append(Model(model=self._model, endpoint=endpoint))

        # Model registration is idempotent, so transient failures are safe to retry.
        payload = [model.model_dump(mode="json") for model in models]
        response = self.client.post_with_retry("/api/models", json=payload)
        return [Model.model_validate(item) for item in response.json()]

    def delete_model(self) -> dict[str, Any]:
        """Delete registered model endpoints. Best-effort: ignore errors."""
        try:
            response = self.client.delete("/api/models")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"RolloutManager: failed to delete models: {exc}")
            return {}

    def _get_rollout(self, rollout_id: str) -> Rollout:
        response = self.client.get(f"/api/rollouts/{rollout_id}")
        response.raise_for_status()
        payload = response.json()
        item = payload["rollout"] if isinstance(payload, dict) and "rollout" in payload else payload
        return Rollout.model_validate(item)

    def _delete_rollout(self, rollout_id: str) -> None:
        try:
            self.client.delete(f"/api/rollouts/{rollout_id}")
        except Exception as exc:
            print(f"RolloutManager: failed to delete rollout {rollout_id}: {exc}")

    @staticmethod
    def _record_lifecycle_timestamps(enqueued_rollout: EnqueuedRollout, rollout: Rollout) -> None:
        """Capture server-authoritative running/finished timestamps in place.

        Pods are launched in CPU-limited batches, so a rollout can sit QUEUING
        well after it was submitted; status.updated_at at the queuing->running
        flip is the moment its pod actually started. We record it the first time
        we observe RUNNING (or, if we polled too slowly and skipped straight to a
        terminal state, the terminal updated_at) so running_at - enqueue_time
        reflects the real queue/startup wait.
        """
        state = rollout.status.state
        updated_at = rollout.status.updated_at
        if enqueued_rollout.running_at is None and state in (
            RolloutState.RUNNING,
            RolloutState.SUCCEEDED,
            RolloutState.FAILED,
        ):
            enqueued_rollout.running_at = updated_at
        if state in TERMINAL_STATES:
            enqueued_rollout.finished_at = updated_at

    def _get_events(self, rollout_id: str, *, event_type: str | None = None, format: str | None = None) -> list[Event]:
        params = {
            key: value for key, value in {"event_type": event_type, "format": format}.items() if value is not None
        }
        response = self.client.get(f"/api/rollouts/{rollout_id}/events", params=params)
        response.raise_for_status()
        return [Event.model_validate(item) for item in response.json()]

    def _post_event(self, rollout_id: str, attempt_id: str, event: EventCreate) -> Event:
        response = self.client.post(
            f"/api/rollouts/{rollout_id}/attempt/{attempt_id}/events",
            json=event.model_dump(mode="json"),
        )
        response.raise_for_status()
        return Event.model_validate(response.json())

    def _create_rollouts(self, data: dict[str, Any], *, is_train: bool) -> list[EnqueuedRollout]:
        keys = list(data.keys())
        if not keys:
            return []

        num_samples = len(data[keys[0]])
        rollouts_per_sample = self._train_rollout_n if is_train else 1
        rollout_requests: list[RolloutCreate] = []
        enqueued_rollouts: list[EnqueuedRollout] = []
        for sample_idx in range(num_samples):
            original = {key: _to_native(data[key][sample_idx]) for key in keys}
            data_id = str(uuid.uuid4())
            for _ in range(rollouts_per_sample):
                request = RolloutCreate(
                    input=_to_native(original),
                    is_train=is_train,
                    config=self._rollout_config,
                    metadata={},
                )
                if self._hooks is not None:
                    request = self._hooks.on_enqueue(request)
                # Assign the id after hooks so creation remains idempotent.
                rollout_id = uuid.uuid4().hex
                request = request.model_copy(update={"rollout_id": rollout_id})
                enqueued_rollouts.append(
                    EnqueuedRollout(
                        data_id=data_id,
                        input=request.input,
                        rollout_id=rollout_id,
                        step=self._step,
                        sample_idx_in_step=sample_idx,
                        enqueue_time=time.time(),
                    )
                )
                rollout_requests.append(request)

        if not rollout_requests:
            return []

        # Preassigned ids make batch creation safe to retry without duplicates.
        payload = [request.model_dump(mode="json", exclude_none=True) for request in rollout_requests]
        response = self.client.post_with_retry("/api/rollouts", json=payload)
        created = [Rollout.model_validate(item) for item in response.json()]
        assert len(created) == len(rollout_requests), (
            f"agl-lite returned {len(created)} rollouts, expected {len(rollout_requests)}"
        )
        return [
            enqueued_rollout.model_copy(update={"rollout_id": rollout.rollout_id})
            for enqueued_rollout, rollout in zip(enqueued_rollouts, created, strict=True)
        ]

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        raw_events = self._get_events(rollout_id)
        triplet_events = self._get_events(rollout_id, format="triplet")
        return raw_events, triplet_events

    @staticmethod
    def _events_by_attempt(raw_events: list[Event], fallback_attempt_id: str) -> dict[str, list[Event]]:
        grouped: dict[str, list[Event]] = defaultdict(list)
        for event in raw_events:
            grouped[event.attempt_id or fallback_attempt_id].append(event)
        if not grouped:
            grouped[fallback_attempt_id] = []
        return dict(grouped)

    def _run_succeeded_hook(self, rollout: Rollout) -> None:
        if self._hooks is None:
            return
        attempt_id = rollout.status.last_attempt_id or "unknown"
        trace_event_helper = _TraceEventHelper()
        raw_events = self._get_events(rollout.rollout_id)
        events_by_attempt = self._events_by_attempt(raw_events, attempt_id)
        try:
            self._hooks.on_succeeded(rollout, events_by_attempt, trace_event_helper)
            trace_event_helper.flush(self)
        except Exception:
            traceback.print_exc()
            print(f"RolloutManager: on_succeeded hook failed for rollout {rollout.rollout_id}")

    def _run_failed_hook(self, rollout: Rollout) -> None:
        if self._hooks is None:
            return
        trace_event_helper = _TraceEventHelper()
        try:
            self._hooks.on_failed(rollout, trace_event_helper)
            trace_event_helper.flush(self)
        except Exception:
            traceback.print_exc()
            print(f"RolloutManager: on_failed hook failed for rollout {rollout.rollout_id}")

    def _build_completed_rollout(self, enqueued_rollout: EnqueuedRollout, rollout: Rollout) -> CompletedRollout:
        """Fetch triplets and reward for a terminal rollout."""
        raw_events, triplet_events = self._fetch_rollout_events(enqueued_rollout.rollout_id)

        triplets: list[Triplet] = []
        for event in triplet_events:
            if event.event_type != "model_request":
                continue
            data = event.data
            http_status = data.get("http_status")
            response_token_ids = data.get("response_token_ids", [])
            if data.get("status") == "error" or (isinstance(http_status, int) and http_status >= 400):
                continue
            if not response_token_ids:
                continue
            triplets.append(
                Triplet(
                    prompt={"token_ids": data.get("prompt_token_ids", [])},
                    response={
                        "token_ids": response_token_ids,
                        "log_probs": data.get("response_log_probs"),
                    },
                    reward=None,
                    metadata={"server": data.get("server", {})},
                )
            )

        final_reward: float | None = None
        reward_events = [event for event in triplet_events if event.event_type == "reward"]
        if reward_events:
            reward_data = reward_events[-1].data
            final_reward = _as_reward_value(reward_data.get("value"))

        if triplets and final_reward is not None:
            triplets[-1] = triplets[-1].model_copy(update={"reward": final_reward})

        metadata = rollout.metadata.model_dump()
        finished_at = enqueued_rollout.finished_at
        if finished_at is None:
            finished_at = rollout.status.updated_at
        return CompletedRollout(
            rollout_id=enqueued_rollout.rollout_id,
            data_id=enqueued_rollout.data_id,
            step=enqueued_rollout.step,
            sample_idx_in_step=enqueued_rollout.sample_idx_in_step,
            input=enqueued_rollout.input,
            enqueue_time=enqueued_rollout.enqueue_time,
            running_at=enqueued_rollout.running_at,
            finished_at=finished_at,
            final_reward=final_reward,
            triplets=triplets,
            metadata=metadata,
            events=[event.model_dump() for event in raw_events],
            triplet_events=[event.model_dump() for event in triplet_events],
            rollout_state=rollout.status.state,
            error_message=rollout.status.error_message,
        )


class AglRolloutManager(AglRolloutManagerBase):
    def enqueue_and_wait_until_completed(
        self,
        data: dict[str, Any],
        *,
        is_train: bool,
    ) -> list[CompletedRollout]:
        """Create rollouts, wait for completion, and return results."""
        enqueued_rollouts = self._create_rollouts(data, is_train=is_train)
        pending_rollouts = list(enqueued_rollouts)
        completed_rollouts: list[CompletedRollout] = []
        num_deleted = 0
        num_succeeded = 0
        num_failed = 0

        while len(completed_rollouts) < len(enqueued_rollouts):
            # Delete prior completions before polling to bound server-side state.
            for completed_rollout in completed_rollouts[num_deleted:]:
                self._delete_rollout(completed_rollout.rollout_id)
            num_deleted = len(completed_rollouts)

            for enqueued_rollout in list(pending_rollouts):
                rollout_id = enqueued_rollout.rollout_id
                rollout = self._get_rollout(rollout_id)
                state = rollout.status.state
                self._record_lifecycle_timestamps(enqueued_rollout, rollout)
                if state not in TERMINAL_STATES:
                    continue

                pending_rollouts.remove(enqueued_rollout)
                if state == RolloutState.SUCCEEDED:
                    num_succeeded += 1
                    self._run_succeeded_hook(rollout)
                elif state == RolloutState.FAILED:
                    num_failed += 1
                    self._run_failed_hook(rollout)

                completed_rollouts.append(self._build_completed_rollout(enqueued_rollout, rollout))

            print(
                f"AglRolloutManager: completed={len(completed_rollouts)}/{len(enqueued_rollouts)} "
                f"succeeded={num_succeeded} failed={num_failed}"
            )

            if pending_rollouts:
                time.sleep(self._poll_interval_seconds)

        # Delete whatever completed in the final round.
        for completed_rollout in completed_rollouts[num_deleted:]:
            self._delete_rollout(completed_rollout.rollout_id)

        return completed_rollouts


class AglAsyncRolloutManager(AglRolloutManagerBase):
    """Async rollout manager."""

    def enqueue_and_wait_until_group_completed(
        self,
        data: dict[str, Any],
        carry_over_enqueued_rollouts: list[EnqueuedRollout],
        *,
        is_train: bool,
        target_finished_group_num: int,
    ) -> tuple[list[CompletedRollout], list[EnqueuedRollout]]:
        """Enqueue rollouts and wait for enough completed rollout groups."""
        assert is_train is True
        enqueued_rollouts = self._create_rollouts(data, is_train=True)
        active_rollouts = carry_over_enqueued_rollouts + enqueued_rollouts
        if not active_rollouts:
            return [], []

        grouped_rollouts: dict[str, list[EnqueuedRollout]] = defaultdict(list)
        for enqueued_rollout in active_rollouts:
            grouped_rollouts[enqueued_rollout.data_id].append(enqueued_rollout)

        for group in grouped_rollouts.values():
            assert len(group) == self._train_rollout_n

        finished_rollout_ids: set[str] = set()
        terminal_rollouts: dict[str, Rollout] = {}
        completed_group_keys: set[str] = set()
        completed_rollouts: list[CompletedRollout] = []
        num_succeeded = 0
        num_failed = 0

        while len(completed_group_keys) < target_finished_group_num:
            for data_id, group in grouped_rollouts.items():
                if data_id in completed_group_keys:
                    continue

                for enqueued_rollout in group:
                    if enqueued_rollout.rollout_id in finished_rollout_ids:
                        continue

                    rollout = self._get_rollout(enqueued_rollout.rollout_id)
                    state = rollout.status.state
                    self._record_lifecycle_timestamps(enqueued_rollout, rollout)
                    if state not in TERMINAL_STATES:
                        continue

                    finished_rollout_ids.add(enqueued_rollout.rollout_id)
                    terminal_rollouts[enqueued_rollout.rollout_id] = rollout
                    if state == RolloutState.SUCCEEDED:
                        num_succeeded += 1
                        self._run_succeeded_hook(rollout)
                    elif state == RolloutState.FAILED:
                        num_failed += 1
                        self._run_failed_hook(rollout)

                if all(enqueued_rollout.rollout_id in finished_rollout_ids for enqueued_rollout in group):
                    completed_group_keys.add(data_id)
                    completed_rollouts.extend(
                        self._build_completed_rollout(
                            enqueued_rollout,
                            terminal_rollouts[enqueued_rollout.rollout_id],
                        )
                        for enqueued_rollout in group
                    )
                    # Free completed group state after reading it.
                    for enqueued_rollout in group:
                        self._delete_rollout(enqueued_rollout.rollout_id)
                    if len(completed_group_keys) >= target_finished_group_num:
                        break

            print(
                f"AglAsyncRolloutManager: completed_groups={len(completed_group_keys)}/{target_finished_group_num} "
                f"finished_rollouts={len(finished_rollout_ids)}/{len(active_rollouts)} "
                f"succeeded={num_succeeded} failed={num_failed}"
            )

            if len(completed_group_keys) < target_finished_group_num:
                time.sleep(self._poll_interval_seconds)

        new_carry_over_rollouts = [
            enqueued_rollout
            for data_id, group in grouped_rollouts.items()
            if data_id not in completed_group_keys
            for enqueued_rollout in group
        ]
        return completed_rollouts, new_carry_over_rollouts


__all__ = [
    "AglAsyncRolloutManager",
    "AglRolloutManager",
    "AglRolloutManagerBase",
    "CompletedRollout",
    "EnqueuedRollout",
    "Triplet",
]

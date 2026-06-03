# Copyright (c) Microsoft. All rights reserved.
# Adapted from Agent Lightning's AgentModeDaemon path.
from __future__ import annotations

import asyncio
import importlib

# AglLiteRolloutBridge talks to agl-lite over HTTP instead of using LightningStore,
# LLMProxy, and Adapter directly.
import math
import os
import re
import time
import traceback
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

# --- Optional heavy imports (only needed when actually training) ---
import torch

# --- Local types (replace agentlightning.types) ---
from pydantic import BaseModel, Field
from tensordict import TensorDict
from verl import DataProto

from agl_lite.client import AglLiteSyncClient
from agl_lite.schemas import (
    TERMINAL_STATES,
    Event,
    EventCreate,
    Model,
    Rollout,
    RolloutConfig,
    RolloutCreate,
    RolloutK8sConfig,
    RolloutLocalConfig,
    RolloutState,
)

if TYPE_CHECKING:
    from agl_lite.hooks import RolloutHooks


class Triplet(BaseModel):
    """Single interaction turn (prompt + response + reward)."""

    prompt: Any  # {"token_ids": [...], "image_urls": [...]}
    response: Any  # {"token_ids": [...]}
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Task echoed back in CompletedRollout."""

    rollout_id: str
    input: Any = None
    mode: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletedRollout(BaseModel):
    """Completed rollout with triplets, used by get_train_data_batch()."""

    rollout_id: str
    task: Task | None = None
    final_reward: float | None = None
    reward_source: str | None = None
    reward_reason: str | None = None
    triplets: list[Triplet] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    triplet_events: list[dict[str, Any]] = Field(default_factory=list)


class AgentJobK8sClient(Protocol):
    """K8s operations needed to clean up tracked agent Jobs."""

    async def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]: ...

    async def delete_job(self, name: str, namespace: str) -> None: ...


__all__ = [
    "AglLiteRolloutBridge",
    "get_left_padded_ids_and_attention_mask",
    "get_right_padded_ids_and_attention_mask",
]


_FALLBACK_REWARD_REASONS = {"no_reward_posted_by_agent", "terminal_failed"}


@dataclass
class _TraceEvent:
    rollout_id: str
    attempt_id: str
    event_type: str
    data: dict[str, Any]


class _SyncTraceSink:
    """Minimal synchronous sink for hooks; bridge flushes queued events over HTTP."""

    def __init__(self) -> None:
        self._queued: list[_TraceEvent] = []

    def add_event(self, rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> None:
        self._queued.append(_TraceEvent(rollout_id=rollout_id, attempt_id=attempt_id, event_type=event_type, data=data))

    def flush(self, bridge: "AglLiteRolloutBridge") -> None:
        for event in self._queued:
            bridge._post_event(
                event.rollout_id,
                event.attempt_id,
                EventCreate(event_type=event.event_type, data=event.data),
            )


_AGL_LITE_MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
_AGL_LITE_MANAGED_BY_VALUE = "agl-lite"
_AGL_LITE_MANAGED_BY_SELECTOR = f"{_AGL_LITE_MANAGED_BY_LABEL}={_AGL_LITE_MANAGED_BY_VALUE}"
_AGL_LITE_ROLLOUT_ID_LABEL = "agl-lite/rollout-id"
_FALLBACK_REWARD_REASONS = {"no_reward_posted_by_agent", "terminal_failed"}


def _safe_metric_name(value: Any, *, default: str = "unknown", max_length: int = 80) -> str:
    text = str(value) if value is not None else default
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._-")
    if not text:
        text = default
    return text[:max_length]


def _bounded_counts(values: list[Any], max_keys: int = 20) -> dict[str, int]:
    counter = Counter(_safe_metric_name(value) for value in values)
    result = {key: count for key, count in counter.most_common(max_keys)}
    remainder = sum(counter.values()) - sum(result.values())
    if remainder:
        result["other"] = remainder
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.number):
        return float(value)
    return None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _finish_reason_from_response(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str) and reason:
                return reason
    stop_reason = response.get("stop_reason")
    return stop_reason if isinstance(stop_reason, str) and stop_reason else None


def _token_count_from_choices(response: dict[str, Any]) -> int:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return 0
    total = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list):
            total += len(token_ids)
    return total


def _usage_from_model_request(data: dict[str, Any]) -> dict[str, int]:
    response_raw = data.get("response")
    response: dict[str, Any] = response_raw if isinstance(response_raw, dict) else {}
    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else response.get("usage", {})
    usage = usage_raw if isinstance(usage_raw, dict) else {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            result[key] = int(value)

    if "prompt_tokens" not in result:
        prompt_token_ids = response.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list):
            result["prompt_tokens"] = len(prompt_token_ids)
    if "completion_tokens" not in result:
        response_tokens = _token_count_from_choices(response)
        if response_tokens:
            result["completion_tokens"] = response_tokens
    if "total_tokens" not in result:
        prompt_tokens = result.get("prompt_tokens", result.get("input_tokens", 0))
        completion_tokens = result.get("completion_tokens", result.get("output_tokens", 0))
        if prompt_tokens or completion_tokens:
            result["total_tokens"] = prompt_tokens + completion_tokens
    return result


def ids_startswith(
    full_ids: list[int], prefix_ids: list[int], tokenizer: Any, debug: bool = False
) -> tuple[bool, tuple[bool, bool, bool]]:
    is_prefix: bool
    template_mismatch, retoken_mismatch, others_mismatch = False, False, False
    if full_ids[: len(prefix_ids)] == prefix_ids:
        is_prefix = True
        return True, (template_mismatch, retoken_mismatch, others_mismatch)
    else:
        is_prefix = False

    if not debug:
        return is_prefix, (template_mismatch, retoken_mismatch, others_mismatch)

    def _special_token_sequence(ids: list[int]) -> list[int]:
        return [id for id in ids if id in tokenizer.all_special_ids]

    def _none_special_token_sequence(ids: list[int]) -> list[int]:
        return [id for id in ids if id not in tokenizer.all_special_ids]

    # First, handle special tokens
    full_special_ids = _special_token_sequence(full_ids)
    prefix_special_ids = _special_token_sequence(prefix_ids)
    if sum(1 for a, b in zip(full_special_ids, prefix_special_ids, strict=False) if a != b) > 0:
        template_mismatch = True

    # Next, handle string content
    full_content_ids = _none_special_token_sequence(full_ids)
    prefix_content_ids = _none_special_token_sequence(prefix_ids)
    full_string = tokenizer.decode(full_ids, skip_special_tokens=True)
    prefix_string = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    if full_content_ids[: len(prefix_content_ids)] != prefix_content_ids and full_string.startswith(prefix_string):
        retoken_mismatch = True
    elif full_content_ids[: len(prefix_content_ids)] != prefix_content_ids and not full_string.startswith(
        prefix_string
    ):
        others_mismatch = True
    return is_prefix, (template_mismatch, retoken_mismatch, others_mismatch)


def log_mismatch_detail(
    diagnostic: tuple[bool, bool, bool],
    full_ids: list[int],
    prefix_ids: list[int],
    global_steps: int,
    rollout_id: str,
    turn_id: int,
    log_dir: str | None = None,
):
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    template_mismatch, retoken_mismatch, others_mismatch = diagnostic
    if template_mismatch:
        with open(os.path.join(log_dir, "template_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)
    if retoken_mismatch:
        with open(os.path.join(log_dir, "retoken_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)
    if others_mismatch:
        with open(os.path.join(log_dir, "others_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)


def get_left_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    """
    Left-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:             the original list of token IDs.
        max_length:      desired total length after padding/truncation.
        pad_token_id:    ID to use for padding.

    Returns:
        padded_ids (any):      list of length == max_length.
        attention_mask (any):  list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate from the left, keep the last max_length tokens
        trimmed = ids[-max_length:]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the left
    pad_len = max_length - seq_len
    padded_ids = [pad_token_id] * pad_len + ids
    attention_mask = [0] * pad_len + [1] * seq_len
    return padded_ids, attention_mask


def get_right_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    """
    Right-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:            the original list of token IDs.
        max_length:     desired total length after padding/truncation.
        pad_token_id:   ID to use for padding.

    Returns:
        padded_ids (any):     list of length == max_length.
        attention_mask (any): list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate to the first max_length tokens
        trimmed = ids[:max_length]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the right
    pad_len = max_length - seq_len
    padded_ids = ids + [pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    return padded_ids, attention_mask


def _to_native(obj: Any) -> Any:
    """Convert numpy/torch types to native Python for JSON serialization."""
    # 1) Arrays -> list (then recurse)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())

    # 2) NumPy scalar types -> Python scalars
    if isinstance(obj, np.generic):
        return _to_native(obj.item())

    # 3) Dict-like -> dict
    if isinstance(obj, Mapping):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}  # type: ignore

    # 4) Lists/Tuples/Sets -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(x) for x in obj]  # type: ignore

    # 5) Torch tensors
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()

    # 6) Anything else: leave as-is
    return obj


class AglLiteRolloutBridge:
    """Bridge between agl-lite HTTP API and VERL trainer.

    The trainer calls these methods for each rollout batch:
      1. set_up_data_and_server()  — register model + enqueue rollouts
      2. run_until_all_finished()  — poll until all rollouts complete
      3. get_train_data_batch() / get_test_metrics()  — assemble trainer outputs
            4. clear_data_and_server()  — reset local state only

        The bridge uses explicit synchronous HTTP calls through AglLiteSyncClient.
    """

    def __init__(
        self,
        agl_base_url: str,
        agl_key: str,
        train_rollout_n: int,
        train_information: dict[str, Any],
        tokenizer: Any,
        mini_batch_size: int,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        timeout_seconds: float = 1200.0,
        processor: Any = None,
        image_base_dir: str | None = None,
        trace_aggregator: dict[str, Any] | None = None,
        hooks: RolloutHooks | None = None,
        local_agent_class: str | None = None,
        local_env_map: dict[str, str] | None = None,
        k8s_job_template_path: str | None = None,
        cleanup_agent_jobs: bool = False,
        cleanup_namespace: str | None = None,
        cleanup_k8s_client: AgentJobK8sClient | None = None,
    ):
        if cleanup_agent_jobs and not cleanup_namespace:
            raise ValueError("cleanup_namespace is required when cleanup_agent_jobs is enabled")

        # --- agl-lite connection (replaces store + proxy + adapter) ---
        self._agl_base_url = agl_base_url
        self._agl_key = agl_key
        self.client = AglLiteSyncClient(
            base_url=agl_base_url,
            key=agl_key,
            timeout=120.0,
        )
        self.timeout_seconds = timeout_seconds

        # --- Training config ---
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fillna_value = reward_fillna_value
        self.image_base_dir = image_base_dir
        self.trace_aggregator = trace_aggregator or {"level": "transition"}
        self._hooks = hooks

        self.local_agent_class = local_agent_class
        self.local_env_map = local_env_map or {}
        self._k8s_job_template = Path(k8s_job_template_path).read_text() if k8s_job_template_path else None

        # --- K8s Job cleanup (only used by remote runner; local runner sets False) ---
        self._cleanup_agent_jobs_enabled = cleanup_agent_jobs
        self._cleanup_namespace = cleanup_namespace
        self._cleanup_k8s_client = cleanup_k8s_client
        self._cleanup_missing_client_warned = False

        # --- Multimodal ---
        self._use_mrope = self._is_mrope_model()

        # --- Internal state ---
        self._total_tasks_queued = 0
        self._completed_rollouts: dict[str, CompletedRollout] = {}
        self._task_id_to_original_sample: dict[str, dict[str, Any]] = {}
        self._enqueue_order: list[str] = []
        self._rollout_status: dict[str, RolloutState] = {}
        self._rollout_error: dict[str, str] = {}
        self._rollout_start_time: dict[str, float] = {}
        self._rollout_end_time: dict[str, float] = {}
        self._raw_events_by_rollout: dict[str, list[dict[str, Any]]] = {}
        self._triplet_events_by_rollout: dict[str, list[dict[str, Any]]] = {}
        self._timeout_rids: set[str] = set()
        self._num_succeeded = 0
        self._num_failed = 0
        self._num_timeout = 0
        self.is_train = True

        # --- Async-rollout (carry-over) state ---
        # rid -> data_id (the group id this rollout belongs to)
        self._rid_to_data_id: dict[str, str] = {}
        # data_id -> set of rids enqueued for that group (across all steps)
        self._data_id_to_rids: dict[str, set[str]] = defaultdict(set)
        # data_id -> wall-clock time when the group first reached "all rids
        # in TERMINAL_STATES". Tie-breaker is enqueue order, then data_id.
        self._group_finish_time: dict[str, float] = {}
        # Rids enqueued this step (cleared at the start of each async step).
        self._step_new_rids: set[str] = set()
        # Rids whose data_id group was selected this step (consumed by
        # async_get_train_data_batch / async_cleanup_consumed).
        self._selected_rids: set[str] = set()
        # Rids that have been enqueued but whose group has not yet been
        # selected by any past step. Persists across steps until consumed.
        self._carry_over_rids: set[str] = set()
        # Step index at which a rid first entered _carry_over_rids — used by
        # async metrics (carry_over_age_max_steps) and the soft-limit warning.
        self._carry_over_birth_step: dict[str, int] = {}
        # Snapshot of "this step's rollouts to run" for `training/n_rollouts`
        # under the async path: captured at async_set_up_data_and_server entry
        # and read by async_get_train_data_batch. None outside async steps.
        self._async_n_rollouts_this_step: int | None = None

    def _register_models(self, models: list[Model]) -> list[Model]:
        response = self.client.post(
            "/api/models",
            json=[model.model_dump(mode="json") for model in models],
        )
        response.raise_for_status()
        return [Model.model_validate(item) for item in response.json()]

    def _get_rollout(self, rollout_id: str) -> Rollout:
        response = self.client.get(f"/api/rollouts/{rollout_id}")
        response.raise_for_status()
        payload = response.json()
        return Rollout.model_validate(payload["rollout"] if isinstance(payload, dict) and "rollout" in payload else payload)

    def _get_events(self, rollout_id: str, *, event_type: str | None = None, format: str | None = None) -> list[Event]:
        params = {key: value for key, value in {"event_type": event_type, "format": format}.items() if value is not None}
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

    def _pause_gateway(self, *, retry_after_seconds: int, reason: str | None) -> dict[str, Any]:
        response = self.client.post(
            "/proxy/pause",
            json={"retry_after_seconds": retry_after_seconds, "reason": reason},
        )
        response.raise_for_status()
        return response.json()

    def resume_gateway(self) -> dict[str, Any]:
        response = self.client.post("/proxy/resume")
        response.raise_for_status()
        return response.json()

    def _wait_until_inflight_drained(self, *, timeout: float) -> int:
        deadline = time.time() + timeout
        residual = 0
        while True:
            response = self.client.get("/proxy/state")
            response.raise_for_status()
            residual = int(response.json().get("inflight", 0))
            if residual <= 0 or time.time() >= deadline:
                return residual
            time.sleep(0.25)

    # Multimodal utilities for M-RoPE position embeddings

    def _is_mrope_model(self) -> bool:
        """Check if processor requires M-RoPE position embeddings."""
        if self.processor is None or not hasattr(self.processor, "image_processor"):
            return False
        name = self.processor.image_processor.__class__.__name__
        return "Qwen2VLImageProcessor" in name or "Qwen3VLImageProcessor" in name

    def _resolve_image_path(self, path: str) -> str:
        """Resolve relative image path with base directory."""
        import os

        if os.path.isabs(path):
            return path
        if self.image_base_dir is None:
            raise ValueError(f"Relative path '{path}' requires 'image_base_dir' to be set.")
        return os.path.join(self.image_base_dir, path)

    def _get_image_grid_thw(self, image_urls: list[str]) -> torch.Tensor | None:
        """Compute image_grid_thw from image URLs for M-RoPE computation.

        Args:
            image_urls: List of image URLs extracted from triplet prompt payload.
                URLs can be http(s):// URLs or file:// URIs, or data: URIs.
        """
        from PIL import Image
        from verl.utils.dataset.vision_utils import process_image  # pyright: ignore[reportUnknownVariableType]

        if self.processor is None or not image_urls:
            return None

        def to_image_uri(url: str) -> str:
            # Already a proper URI (http, https, file, data)
            if url.startswith(("http://", "https://", "file://", "data:")):
                return url
            # Treat as a file path that needs resolution
            resolved = self._resolve_image_path(url)
            return f"file://{resolved}"

        images: list[Image.Image] = [process_image({"image": to_image_uri(url)}) for url in image_urls]
        model_inputs = self.processor(text=["dummy"], images=images, return_tensors="pt")
        return model_inputs.get("image_grid_thw")

    def _compute_mrope_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 4D position_ids for M-RoPE models."""
        from collections.abc import Callable

        module_name = (
            "verl.models.transformers.qwen3_vl"
            if "Qwen3VL" in self.processor.__class__.__name__
            else "verl.models.transformers.qwen2_vl"
        )
        module = importlib.import_module(module_name)
        get_rope_index: Callable[..., torch.Tensor] = module.get_rope_index  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

        vision_pos = get_rope_index(
            self.processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

        valid_mask = attention_mask.bool()
        text_pos = torch.zeros((1, len(input_ids)), dtype=torch.long, device=input_ids.device)
        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)

        return torch.cat([text_pos, vision_pos], dim=0)

    def _run_agl_lite_rollout(self, data: dict[str, Any], server_addresses: list[str], is_train: bool = True) -> None:
        """Register model servers and enqueue agl-lite rollouts.
        """
        self.clear_data_and_server()
        self.is_train = is_train
        self._register_and_enqueue(data, server_addresses, is_train)

    def finish_sync_rollout_batch(self) -> None:
        """Clean up external rollout resources, then reset local sync state."""
        try:
            self.cleanup_agent_jobs()
        finally:
            self.clear_data_and_server()

    def _register_and_enqueue(self, data: dict[str, Any], server_addresses: list[str], is_train: bool) -> None:
        """Shared setup implementation."""
        self._register_model_servers(server_addresses)
        rollouts_per_sample = self.train_rollout_n if is_train else 1
        created_with_requests = self._enqueue_rollouts(
            data,
            rollouts_per_sample=rollouts_per_sample,
            is_train=is_train,
        )
        created = [rollout for rollout, _request in created_with_requests]

        now = time.time()
        for r in created:
            rid = r.rollout_id
            self._task_id_to_original_sample[rid] = r.input if isinstance(r.input, dict) else {}
            self._enqueue_order.append(rid)
            self._rollout_start_time[rid] = now
        self._total_tasks_queued += len(created)

        print(f"AglLiteRolloutBridge: enqueued {len(created)} rollouts.")

    def _register_model_servers(self, server_addresses: list[str]) -> None:
        """Register VERL's current vLLM endpoints with the agl-lite gateway."""
        model_name = self.train_information.get("model", "default-model")
        regs: list[Model] = []
        for addr in server_addresses:
            endpoint = addr if addr.startswith("http") else f"http://{addr}/v1"
            regs.append(Model(model=model_name, endpoint=endpoint))
        self._register_models(regs)

    def _rollout_config(self) -> RolloutConfig:
        config = RolloutConfig(timeout_seconds=int(self.timeout_seconds))
        has_local = bool(self.local_agent_class)
        has_k8s = bool(self._k8s_job_template)
        if not has_local and not has_k8s:
            return config
        update = {}
        if has_local:
            update["local"] = RolloutLocalConfig(
                agent_class=self.local_agent_class,
                env_map=self.local_env_map,
            )
        if has_k8s:
            update["k8s"] = RolloutK8sConfig(job_template=self._k8s_job_template)
        return config.model_copy(update=update)

    def _enqueue_rollouts(
        self,
        data: dict[str, Any],
        *,
        rollouts_per_sample: int,
        is_train: bool,
    ) -> list[tuple[Rollout, RolloutCreate]]:
        """Build rollout requests, POST them, and return created rollouts with their requests."""
        keys = list(data.keys())
        if not keys:
            return []
        num_samples = len(data[keys[0]])
        rollout_config = self._rollout_config()

        rollout_requests: list[RolloutCreate] = []
        for i in range(num_samples):
            original = {key: _to_native(data[key][i]) for key in keys}
            data_id = str(original.get("data_id") or original.get("uid") or uuid.uuid4())
            original["data_id"] = data_id
            original["_sample_idx"] = i
            for trial_idx in range(rollouts_per_sample):
                request = RolloutCreate(
                    input=_to_native(original),
                    is_train=is_train,
                    config=rollout_config,
                    metadata={"sample_idx_in_batch": i},
                )
                if self._hooks is not None:
                    request = self._hooks.on_enqueue(request)
                rollout_requests.append(request)

        if not rollout_requests:
            return []
        response = self.client.post(
            "/api/rollouts",
            json=[request.model_dump(mode="json", exclude_none=True) for request in rollout_requests],
        )
        response.raise_for_status()
        created = [Rollout.model_validate(item) for item in response.json()]
        assert len(created) == len(rollout_requests), (
            f"agl-lite returned {len(created)} rollouts, expected {len(rollout_requests)}"
        )
        return list(zip(created, rollout_requests, strict=True))

    def set_up_data_and_server(self, data: dict[str, Any], server_addresses: list[str], is_train: bool = True) -> None:
        """Sync wrapper — same signature as AgentModeDaemon."""
        self._run_agl_lite_rollout(data, server_addresses, is_train)

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        raw_events = self._get_events(rollout_id)
        triplet_events = self._get_events(rollout_id, format="triplet")
        self._raw_events_by_rollout[rollout_id] = [event.model_dump() for event in raw_events]
        self._triplet_events_by_rollout[rollout_id] = [event.model_dump() for event in triplet_events]
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
        sink = _SyncTraceSink()
        raw_events = self._get_events(rollout.rollout_id)
        events_by_attempt = self._events_by_attempt(raw_events, attempt_id)
        try:
            self._hooks.on_succeeded(rollout, events_by_attempt, sink)
            sink.flush(self)
        except Exception:
            traceback.print_exc()
            print(f"AglLiteRolloutBridge: on_succeeded hook failed for rollout {rollout.rollout_id}")

    def _run_failed_hook(self, rollout: Rollout) -> None:
        if self._hooks is None:
            return
        sink = _SyncTraceSink()
        try:
            self._hooks.on_failed(rollout, sink)
            sink.flush(self)
        except Exception:
            traceback.print_exc()
            print(f"AglLiteRolloutBridge: on_failed hook failed for rollout {rollout.rollout_id}")

    def _fetch_rollout_result(self, rollout_id: str) -> CompletedRollout:
        """Fetch triplets for a completed rollout via agl-lite HTTP.

        Replaces AgentModeDaemon._validate_data_v1() which calls:
          store.query_spans() → adapter.adapt(spans) → List[Triplet]

        In agl-lite, the server does event→triplet conversion via format=triplet.
        """
        raw_events, events = self._fetch_rollout_events(rollout_id)

        # Convert trimmed events to Triplet objects
        triplets: list[Triplet] = []
        for evt in events:
            if evt.event_type == "model_request":
                d = evt.data
                triplets.append(
                    Triplet(
                        prompt={"token_ids": d.get("prompt_token_ids", [])},
                        response={
                            "token_ids": d.get("response_token_ids", []),
                            "log_probs": d.get("response_log_probs", []),
                        },
                        reward=None,
                        metadata={"server": d.get("server", {})},
                    )
                )

        # Match rewards to triplets (last reward wins)
        final_reward: float | None = None
        reward_source: str | None = None
        reward_reason: str | None = None
        reward_events = [e for e in events if e.event_type == "reward"]
        if reward_events:
            reward_data = reward_events[-1].data
            final_reward = reward_data.get("value")
            reward_source = reward_data.get("source")
            reward_reason = reward_data.get("reason")

        # Assign reward to last triplet (same as Agent Lightning convention)
        if triplets and final_reward is not None:
            triplets[-1] = triplets[-1].model_copy(update={"reward": final_reward})

        original = self._task_id_to_original_sample.get(rollout_id, {})
        result = CompletedRollout(
            rollout_id=rollout_id,
            task=Task(
                rollout_id=rollout_id,
                input=original,
                metadata=original.get("metadata", {}),
            ),
            final_reward=final_reward,
            reward_source=reward_source,
            reward_reason=reward_reason,
            triplets=triplets,
            metadata=original.get("metadata", {}),
            events=[event.model_dump() for event in raw_events],
            triplet_events=[event.model_dump() for event in events],
        )

        if result.final_reward is None:
            print(
                f"Warning: Reward is None for rollout {result.rollout_id}, "
                f"will be auto-set to {self.reward_fillna_value}."
            )
        if result.triplets is None:
            print(f"Warning: Triplet is None for rollout {result.rollout_id}.")
        elif len(result.triplets) == 0:
            print(f"Warning: Length of triplets is 0 for rollout {result.rollout_id}.")
        elif any(not r.response.get("token_ids", []) for r in result.triplets):
            print(f"Warning: Rollout {result.rollout_id} contains empty response: {result.triplets}")
        elif any(not r.prompt.get("token_ids", []) for r in result.triplets):
            print(f"Warning: Rollout {result.rollout_id} contains empty prompt: {result.triplets}")
        return result

    def _run_until_finished(
        self,
        verbose: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Poll agl-lite until all rollouts reach a terminal state.

        Terminal states (succeeded / failed) are recorded
        in self._rollout_status. Non-succeeded rollouts are NOT added to
        self._completed_rollouts — get_train_data_batch / get_test_metrics
        emit EOS-placeholder rows for those rids by iterating self._enqueue_order.

        If timeout_seconds is explicitly provided, any rollout that has not
        reached a terminal state by the deadline is recorded in
        self._timeout_rids. The polling loop exits without raising — the
        trainer still gets a row-aligned DataProto with placeholders for the
        timed-out rollouts. The default timeout_seconds=None waits forever.
        """
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds
        POLL_INTERVAL = 5.0

        while True:
            unfinished = [
                rid for rid in self._enqueue_order if rid not in self._rollout_status and rid not in self._timeout_rids
            ]
            if not unfinished:
                break
            if deadline is not None and time.time() >= deadline:
                now = time.time()
                for rid in unfinished:
                    self._timeout_rids.add(rid)
                    self._rollout_error[rid] = "trainer-side wall-clock timeout"
                    self._rollout_end_time[rid] = now
                    self._num_timeout += 1
                print(f"AglLiteRolloutBridge: timeout — marked {len(unfinished)} rollouts as timeout")
                break

            for rid in unfinished:
                rollout = self._get_rollout(rid)
                status = rollout.status.state
                if status not in TERMINAL_STATES:
                    continue
                self._rollout_status[rid] = status
                self._rollout_end_time[rid] = time.time()
                if status == RolloutState.SUCCEEDED:
                    self._run_succeeded_hook(rollout)
                    legacy = self._fetch_rollout_result(rid)
                    self._completed_rollouts[rid] = legacy
                    self._num_succeeded += 1
                elif status == RolloutState.FAILED:
                    self._run_failed_hook(rollout)
                    self._fetch_rollout_events(rid)
                    self._rollout_error[rid] = rollout.status.error_message or "failed"
                    self._num_failed += 1
            if verbose:
                n_done = len(self._rollout_status) + len(self._timeout_rids)
                print(
                    f"Completed {self._num_succeeded}/{self._total_tasks_queued} "
                    f"(failed={self._num_failed}, timeout={self._num_timeout}, "
                    f"unfinished={self._total_tasks_queued - n_done})"
                )
            time.sleep(POLL_INTERVAL)

        print("All tasks reached terminal state.")

    def run_until_all_finished(
        self,
        verbose: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Sync wrapper — waits forever unless per-call wall-clock timeout is set."""
        if self._total_tasks_queued == 0:
            print("Warning: No tasks were queued.")
            return

        self._run_until_finished(verbose, timeout_seconds)

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run.

        Iterates self._enqueue_order so non-succeeded rollouts (failed /
        timeout) count as fillna-reward samples. Polling metrics
        (num_succeeded / num_failed / num_timeout / avg_rollout_latency /
        rollout_completion_rate) are merged into the returned dict.
        """
        assert not self.is_train, "This method should only be called during validation."

        sample_stat_list: list[dict[str, Any]] = []
        sample_stat_list_by_source: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )  # FIXME: Evaluate whether grouping stats by source is actually needed.

        for rollout_id in self._enqueue_order:
            rollout = self._completed_rollouts.get(rollout_id)
            if rollout is None:
                # failed / timeout — fillna-reward, no triplets.
                sample_stat_list.append(
                    {
                        "reward": self.reward_fillna_value,
                        "has_reward": False,
                        "has_any_reward": False,
                        "has_fallback_reward": False,
                    }
                )
                continue
            final_reward_raw: float | None = rollout.final_reward
            final_reward = self._fillna_reward(rollout)
            has_agent_reward = self._has_agent_reward(rollout)
            has_fallback_reward = self._has_fallback_reward(rollout)
            if not rollout.triplets:
                print(f"Warning: No triplets found for test rollout {rollout.rollout_id}.")
                sample_stat_list.append(
                    {
                        "reward": final_reward,
                        "has_reward": has_agent_reward,
                        "has_any_reward": final_reward_raw is not None,
                        "has_fallback_reward": has_fallback_reward,
                    }
                )
                continue
            response_length_list = [len(triplet.response.get("token_ids") or []) for triplet in rollout.triplets]

            if "data_source" in self._task_id_to_original_sample[rollout_id]:
                # When a test sample includes a 'data_source' field, record per-source statistics for test results.
                # TODO: This is a flawed design. We should have a better way to handle this.
                data_source = _safe_metric_name(self._task_id_to_original_sample[rollout_id]["data_source"])
                sample_stat_list_by_source[data_source].append(
                    {
                        "sum_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                        "reward": final_reward,
                        "has_reward": has_agent_reward,
                        "has_any_reward": final_reward_raw is not None,
                        "has_fallback_reward": has_fallback_reward,
                    }
                )
            sample_stat_list.append(
                {
                    "sum_response_length": np.sum(response_length_list),
                    "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                    "turn_count": len(rollout.triplets),
                    "reward": final_reward,
                    "has_reward": has_agent_reward,
                    "has_any_reward": final_reward_raw is not None,
                    "has_fallback_reward": has_fallback_reward,
                }
            )
        metric_dict: dict[str, Any] = {}

        stats_w_trace = [stat for stat in sample_stat_list if "sum_response_length" in stat]
        stats_w_trace_by_source = {
            data_source: [stat for stat in sample_stats if "sum_response_length" in stat]
            for data_source, sample_stats in sample_stat_list_by_source.items()
        }
        for data_source, sample_stats in list(sample_stat_list_by_source.items())[:20]:
            metric_dict.update(
                {
                    f"val/{data_source}/n_rollouts": len(sample_stats),
                    f"val/{data_source}/n_rollouts_w_trace": len(stats_w_trace_by_source[data_source]),
                    f"val/{data_source}/n_rollouts_w_reward": len(
                        [stat for stat in sample_stats if stat["has_reward"]]
                    ),
                    f"val/{data_source}/n_rollouts_w_any_reward": len(
                        [stat for stat in sample_stats if stat["has_any_reward"]]
                    ),
                    f"val/{data_source}/n_rollouts_w_fallback_reward": len(
                        [stat for stat in sample_stats if stat["has_fallback_reward"]]
                    ),
                    f"val/{data_source}/reward": np.mean(
                        [stat["reward"] for stat in sample_stats]
                    ),  # each rollout must have a reward (fillna if missing)
                    f"val/{data_source}/mean_response_length": np.mean(
                        [stat["mean_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/sum_response_length": np.mean(
                        [stat["sum_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/turn_count": np.mean(
                        [stat["turn_count"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                }
            )
        metric_dict.update(
            {
                "val/n_rollouts": len(sample_stat_list),
                "val/n_rollouts_w_trace": len(stats_w_trace),
                "val/n_rollouts_w_reward": len([stat for stat in sample_stat_list if stat["has_reward"]]),
                "val/n_rollouts_w_any_reward": len([stat for stat in sample_stat_list if stat["has_any_reward"]]),
                "val/n_rollouts_w_fallback_reward": len(
                    [stat for stat in sample_stat_list if stat["has_fallback_reward"]]
                ),
                "val/reward": np.mean(
                    [stat["reward"] for stat in sample_stat_list]
                ),  # each rollout must have a reward (fillna if missing)
                "val/mean_response_length": np.mean([stat["mean_response_length"] for stat in stats_w_trace])
                if stats_w_trace
                else 0.0,
                "val/sum_response_length": np.mean([stat["sum_response_length"] for stat in stats_w_trace])
                if stats_w_trace
                else 0.0,
                "val/turn_count": np.mean([stat["turn_count"] for stat in stats_w_trace]) if stats_w_trace else 0.0,
            }
        )
        metric_dict.update(self._polling_metrics("val"))
        metric_dict.update(self._event_metrics("val"))
        return metric_dict

    def get_train_data_batch(
        self, max_prompt_length: int, max_response_length: int, device: torch.device, global_steps: int
    ):
        """Build a VERL training batch from all completed rollout steps.

        This follows Agent Lightning's agent-mode data path: ``transition``
        level emits one training row per triplet, while ``trajectory`` level
        merges compatible multi-turn traces and marks inserted observation
        tokens with response_mask=0. Gateway/event capture is intentionally not
        involved here; all masking is computed in the rollout bridge/trainer.
        """
        assert self.is_train, "This method should only be called during training."

        level = self.trace_aggregator.get("level", "transition")
        if level not in {"transition", "trajectory"}:
            raise ValueError(f"Unknown trace_aggregator level: {level}")
        if level == "trajectory":
            assert not self._use_mrope, "M-RoPE is not supported in trajectory level yet."

        finished_id_to_sample_info: dict[str, dict[str, Any]] = {}
        finished_id_to_final_reward: dict[str, float] = {}
        sample_with_agent_reward_count = 0
        sample_with_any_reward_count = 0
        sample_with_fallback_reward_count = 0

        for rid in self._enqueue_order:
            rollout = self._completed_rollouts.get(rid)
            original_sample = self._task_id_to_original_sample.get(rid, {})
            if rollout is None:
                finished_id_to_final_reward[rid] = self.reward_fillna_value
                continue

            sample_with_any_reward_count += int(rollout.final_reward is not None)
            sample_with_agent_reward_count += int(self._has_agent_reward(rollout))
            sample_with_fallback_reward_count += int(self._has_fallback_reward(rollout))
            final_reward = self._fillna_reward(rollout)
            finished_id_to_final_reward[rid] = final_reward

            if not rollout.triplets:
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, using placeholder.")
                continue

            trace_list = [
                {
                    "prompt_ids": list(t.prompt.get("token_ids", [])),
                    "response_ids": list(t.response.get("token_ids", [])),
                    "response_log_probs": list(t.response.get("log_probs", [])),
                    "image_urls": list(t.prompt.get("image_urls", [])),
                }
                for t in rollout.triplets
            ]
            data_id = str(original_sample.get("data_id") or original_sample.get("uid") or rid)
            finished_id_to_sample_info[rid] = {
                "reward": final_reward,
                "trace_list": trace_list,
                "data_id": data_id,
            }

        input_ids_list: list[list[int]] = []
        input_attention_mask_list: list[list[int]] = []
        response_ids_list: list[list[int]] = []
        response_attention_mask_list: list[list[int]] = []
        response_mask_list: list[list[int]] = []
        reward_list: list[float] = []
        data_id_list: list[str] = []
        rollout_id_list: list[str] = []
        turn_index_list: list[int] = []
        is_drop_list: list[bool] = []
        image_grid_thw_list: list[torch.Tensor | None] = []
        response_log_probs_list: list[list[float]] = []
        row_has_real_lp_list: list[bool] = []
        n_trunc_sample_because_of_response = 0
        unmerged_count = 0
        template_mismatch_count = 0
        retoken_mismatch_count = 0
        others_mismatch_count = 0
        response_per_turn_list: list[int] = []

        eos_id = (self.tokenizer.eos_token_id if self.tokenizer is not None else None) or self.pad_token_id

        def validate_trace_logprobs(trace: dict[str, Any]) -> list[float] | None:
            """Return per-token logprobs for one trace, or None when unusable."""
            lps = trace.get("response_log_probs") or []
            rids = trace.get("response_ids") or []
            if not lps or len(lps) != len(rids):
                return None
            out: list[float] = []
            for v in lps:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(f):
                    return None
                out.append(f)
            return out

        def append_training_row(
            *,
            rid: str,
            data_id: str,
            turn_index: int,
            prompt_ids: list[int],
            response_ids: list[int],
            reward: float,
            response_mask: list[int] | None = None,
            image_urls: list[str] | None = None,
            response_log_probs: list[float] | None = None,
        ) -> None:
            nonlocal n_trunc_sample_because_of_response
            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[:max_prompt_length]
                is_drop_list.append(True)
            else:
                is_drop_list.append(False)

            if len(response_ids) > max_response_length:
                response_ids = response_ids[:max_response_length]
                if response_mask is not None:
                    response_mask = response_mask[:max_response_length]
                if response_log_probs is not None:
                    response_log_probs = response_log_probs[:max_response_length]
                n_trunc_sample_because_of_response += 1

            # Validate logprob alignment defensively: a row only carries real
            # rollout logprobs when every response token has a finite value.
            if response_log_probs is not None and len(response_log_probs) != len(response_ids):
                response_log_probs = None

            one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                prompt_ids, max_prompt_length, self.pad_token_id
            )
            one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                response_ids, max_response_length, self.pad_token_id
            )
            input_ids_list.append(one_input_ids)
            input_attention_mask_list.append(one_input_attention_mask)
            response_ids_list.append(one_response_ids)
            response_attention_mask_list.append(one_response_attention_mask)
            if response_mask is not None:
                one_response_mask, _ = get_right_padded_ids_and_attention_mask(response_mask, max_response_length, 0)
                response_mask_list.append(one_response_mask)
            # Always emit a finite logprob row (0.0 filler) so the assembled tensor
            # never contains NaN/inf; rows without real logprobs are dropped later.
            if response_log_probs is None:
                row_has_real_lp_list.append(False)
                response_log_probs_list.append([0.0] * max_response_length)
            else:
                row_has_real_lp_list.append(True)
                padded_lp = response_log_probs + [0.0] * (max_response_length - len(response_log_probs))
                response_log_probs_list.append(padded_lp[:max_response_length])
            reward_list.append(reward)
            data_id_list.append(data_id)
            rollout_id_list.append(rid)
            if level == "transition":
                turn_index_list.append(turn_index)
            if self._use_mrope:
                image_grid_thw_list.append(self._get_image_grid_thw(image_urls or []))

        def append_placeholder(rid: str) -> None:
            original_sample = self._task_id_to_original_sample.get(rid, {})
            data_id = str(original_sample.get("data_id") or original_sample.get("uid") or rid)
            # Placeholder rows have no real rollout logprobs: leave response_log_probs
            # as None so the row is dropped when bypass mode is active.
            append_training_row(
                rid=rid,
                data_id=data_id,
                turn_index=-1,
                prompt_ids=[],
                response_ids=[eos_id],
                reward=finished_id_to_final_reward.get(rid, self.reward_fillna_value),
                response_mask=[1] if level == "trajectory" else None,
                response_log_probs=None,
            )

        if level == "transition":
            for rid in self._enqueue_order:
                sample_info = finished_id_to_sample_info.get(rid)
                if sample_info is None:
                    append_placeholder(rid)
                    continue
                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    trace_lp = validate_trace_logprobs(trace)
                    append_training_row(
                        rid=rid,
                        data_id=sample_info["data_id"],
                        turn_index=turn_index,
                        prompt_ids=trace["prompt_ids"],
                        response_ids=trace["response_ids"],
                        reward=sample_info["reward"],
                        image_urls=trace.get("image_urls", []),
                        response_log_probs=trace_lp,
                    )
        else:
            for rid in self._enqueue_order:
                sample_info = finished_id_to_sample_info.get(rid)
                if sample_info is None:
                    append_placeholder(rid)
                    continue

                merged_trace_idx: list[list[int]] = []
                current_merged_trace_idx: list[int] = []
                current_context: list[int] = []

                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    response_per_turn_list.append(len(trace["response_ids"]))
                    is_prefix, diagnostic = ids_startswith(
                        trace["prompt_ids"] + trace["response_ids"],
                        current_context,
                        self.tokenizer,
                        self.trace_aggregator.get("debug", False),
                    )
                    if not is_prefix and self.trace_aggregator.get("debug", False):
                        template_mismatch_count += int(diagnostic[0])
                        retoken_mismatch_count += int(diagnostic[1])
                        others_mismatch_count += int(diagnostic[2])
                        log_mismatch_detail(
                            diagnostic,
                            trace["prompt_ids"] + trace["response_ids"],
                            current_context,
                            global_steps,
                            rid,
                            turn_index,
                            self.trace_aggregator.get("mismatch_log_dir", None),
                        )

                    if is_prefix:
                        current_context = trace["prompt_ids"] + trace["response_ids"]
                        current_merged_trace_idx.append(turn_index)
                    else:
                        if current_merged_trace_idx:
                            merged_trace_idx.append(current_merged_trace_idx)
                        current_merged_trace_idx = [turn_index]
                        current_context = trace["prompt_ids"] + trace["response_ids"]

                if current_merged_trace_idx:
                    merged_trace_idx.append(current_merged_trace_idx)

                if len(merged_trace_idx) > 1:
                    unmerged_count += 1

                for current_group in merged_trace_idx:
                    first_trace = sample_info["trace_list"][current_group[0]]
                    prompt_ids = list(first_trace["prompt_ids"])
                    response_ids: list[int]
                    response_mask: list[int]
                    # Build response_log_probs in lockstep with response_mask:
                    # 0.0 at masked (observation / prompt-overflow) positions, real
                    # logprobs at assistant tokens. A row only keeps its logprobs
                    # when every assistant segment in the group is valid.
                    group_lp_ok = True
                    response_log_probs: list[float]
                    if current_group[0] > 0 and len(prompt_ids) > max_prompt_length:
                        response_ids = prompt_ids[max_prompt_length:]
                        prompt_ids = prompt_ids[:max_prompt_length]
                        response_mask = [1] * len(response_ids)
                        # Prompt-overflow tokens are mask==1 but have no rollout
                        # logprobs, so the row cannot carry valid logprobs.
                        response_log_probs = [0.0] * len(response_ids)
                        group_lp_ok = False
                    else:
                        response_ids = []
                        response_mask = []
                        response_log_probs = []

                    prompt_length = len(prompt_ids)
                    first_response_ids = list(first_trace["response_ids"])
                    response_ids += first_response_ids
                    response_mask += [1] * len(first_response_ids)
                    first_lp = validate_trace_logprobs(first_trace)
                    if first_lp is None:
                        group_lp_ok = False
                        response_log_probs += [0.0] * len(first_response_ids)
                    else:
                        response_log_probs += first_lp

                    for turn_index in current_group[1:]:
                        trace = sample_info["trace_list"][turn_index]
                        new_prompt_length = len(trace["prompt_ids"]) - len(response_ids) - prompt_length
                        if new_prompt_length > 0:
                            observation_ids = trace["prompt_ids"][-new_prompt_length:]
                            response_ids += observation_ids
                            response_mask += [0] * len(observation_ids)
                            response_log_probs += [0.0] * len(observation_ids)
                        response_ids += trace["response_ids"]
                        response_mask += [1] * len(trace["response_ids"])
                        turn_lp = validate_trace_logprobs(trace)
                        if turn_lp is None:
                            group_lp_ok = False
                            response_log_probs += [0.0] * len(trace["response_ids"])
                        else:
                            response_log_probs += turn_lp

                    append_training_row(
                        rid=rid,
                        data_id=sample_info["data_id"],
                        turn_index=current_group[0],
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        reward=sample_info["reward"],
                        response_mask=response_mask,
                        response_log_probs=response_log_probs if group_lp_ok else None,
                    )

        n_transition = len(input_ids_list)
        if n_transition == 0:
            raise RuntimeError("get_train_data_batch emitted zero training rows.")

        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)
        batch_response_mask = torch.LongTensor(response_mask_list).to(device) if level == "trajectory" else None

        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        if self._use_mrope:
            position_ids_list: list[torch.Tensor] = []
            for i in range(n_transition):
                pos_ids = self._compute_mrope_position_ids(
                    input_ids=batch_seq[i],
                    attention_mask=attention_mask[i],
                    image_grid_thw=image_grid_thw_list[i] if image_grid_thw_list else None,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        # Rollout-logprob gating: only emit rollout_log_probs once at least one
        # row carries real logprobs (i.e. bypass-mode data is present). Rows
        # without valid logprobs are dropped and counted; their filler is finite
        # 0.0 so they never poison full-tensor diffs before being masked out.
        emit_rollout_log_probs = any(row_has_real_lp_list)
        n_dropped_no_log_probs = 0
        if emit_rollout_log_probs:
            for i, has_lp in enumerate(row_has_real_lp_list):
                if has_lp:
                    continue
                is_drop_list[i] = True
                n_dropped_no_log_probs += 1

        is_drop_mask = torch.BoolTensor(is_drop_list).to(device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)

        # Place reward at the last attended token (eos position).
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device).unsqueeze(0)
        eos_mask_idx = torch.argmax(token_positions * attention_mask, dim=-1)
        token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -max_response_length:]

        batch_dict = {
            "prompts": batch_input_ids,
            "responses": batch_response_ids,
            "input_ids": batch_seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "is_drop_mask": is_drop_mask,
            "token_level_scores": token_level_scores.contiguous(),
        }
        if level == "trajectory":
            assert batch_response_mask is not None
            batch_dict["response_mask"] = batch_response_mask
        if emit_rollout_log_probs:
            batch_dict["rollout_log_probs"] = torch.tensor(
                response_log_probs_list, dtype=torch.float32
            ).to(device)

        batch = TensorDict(batch_dict, batch_size=n_transition)  # type: ignore[arg-type]
        data_proto = DataProto(batch=batch)
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        if level == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)

        n_response_turns = len(response_per_turn_list)

        data_metrics = {
            "training/reward": float(np.mean(list(finished_id_to_final_reward.values())))
            if finished_id_to_final_reward
            else 0.0,
            "training/n_rollouts": self._total_tasks_queued,
            "training/n_rollouts_w_trace": len(finished_id_to_sample_info),
            "training/n_rollouts_w_reward": sample_with_agent_reward_count,
            "training/n_rollouts_w_any_reward": sample_with_any_reward_count,
            "training/n_rollouts_w_fallback_reward": sample_with_fallback_reward_count,
            "training/n_truncated_triplets": n_trunc_sample_because_of_response,
            "training/n_triplets": n_transition,
            "training/n_dropped_no_log_probs": n_dropped_no_log_probs,
            **(
                {
                    "training/n_unmerged_rollouts": unmerged_count,
                    "training/n_triplets_by_turn": n_response_turns,
                    "response_length/training/avg_by_turn": float(np.mean(response_per_turn_list))
                    if response_per_turn_list
                    else 0.0,
                    "response_length/training/max_by_turn": int(np.max(response_per_turn_list))
                    if response_per_turn_list
                    else 0,
                    "response_length/training/min_by_turn": int(np.min(response_per_turn_list))
                    if response_per_turn_list
                    else 0,
                }
                if level == "trajectory"
                else {}
            ),
            **(
                {
                    "training/template_mismatch_triplets": template_mismatch_count,
                    "training/retoken_mismatch_triplets": retoken_mismatch_count,
                    "training/others_mismatch_triplets": others_mismatch_count,
                    "training/template_mismatch_ratio": template_mismatch_count / n_response_turns
                    if n_response_turns
                    else 0.0,
                    "training/retoken_mismatch_ratio": retoken_mismatch_count / n_response_turns
                    if n_response_turns
                    else 0.0,
                    "training/others_mismatch_ratio": others_mismatch_count / n_response_turns
                    if n_response_turns
                    else 0.0,
                }
                if level == "trajectory" and self.trace_aggregator.get("debug", False)
                else {}
            ),
        }
        data_metrics.update(self._polling_metrics("training"))
        data_metrics.update(self._event_metrics("training"))

        return data_proto, data_metrics

    def _iter_event_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for rid in self._enqueue_order:
            rollout = self._completed_rollouts.get(rid)
            if rollout is not None and rollout.events:
                records.extend(rollout.events)
            else:
                records.extend(self._raw_events_by_rollout.get(rid, []))
        return records

    def _event_metrics(self, phase: str) -> dict[str, Any]:
        records = self._iter_event_records()
        model_requests = [record for record in records if record.get("event_type") == "model_request"]
        return self._gateway_metrics(phase, model_requests)

    def _gateway_metrics(self, phase: str, model_requests: list[dict[str, Any]]) -> dict[str, Any]:
        gateway_prefix = f"gateway/{phase}"
        llm_prefix = f"llm/{phase}"
        metrics: dict[str, Any] = {
            f"{gateway_prefix}/request_count": len(model_requests),
            f"{gateway_prefix}/success_count": 0,
            f"{gateway_prefix}/error_count": 0,
        }
        if not model_requests:
            return metrics

        latencies: list[float] = []
        retry_count = 0
        status_codes: list[Any] = []
        finish_reasons: list[Any] = []
        models: list[Any] = []
        token_totals = Counter(
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        token_request_count = 0
        for record in model_requests:
            data = record.get("data", {}) if isinstance(record.get("data"), dict) else {}
            http_status = data.get("http_status")
            status_codes.append(http_status or "unknown")
            status = data.get("status") or ("ok" if not isinstance(http_status, int) or http_status < 400 else "error")
            is_success = status == "ok" and (not isinstance(http_status, int) or http_status < 400)
            metrics[f"{gateway_prefix}/success_count"] += int(is_success)
            metrics[f"{gateway_prefix}/error_count"] += int(not is_success)
            latency = _as_float(data.get("latency_ms"))
            if latency is not None:
                latencies.append(latency)
            retry_value = _as_float(data.get("retry_count"))
            retry_count += int(retry_value or 0)
            finish_reason = data.get("finish_reason") or _finish_reason_from_response(data.get("response"))
            if finish_reason:
                finish_reasons.append(finish_reason)
            server = data.get("server", {}) if isinstance(data.get("server"), dict) else {}
            models.append(data.get("model") or server.get("model") or "unknown")
            usage = _usage_from_model_request(data)
            if usage:
                token_request_count += 1
                token_totals.update(usage)

        if latencies:
            metrics[f"{gateway_prefix}/latency_ms_mean"] = float(np.mean(latencies))
            metrics[f"{gateway_prefix}/latency_ms_p50"] = _percentile(latencies, 50)
            metrics[f"{gateway_prefix}/latency_ms_p95"] = _percentile(latencies, 95)
        metrics[f"{gateway_prefix}/retry_count"] = retry_count
        for status_code, count in _bounded_counts(status_codes, max_keys=16).items():
            metrics[f"{gateway_prefix}/http_status/{status_code}_count"] = count
        for reason, count in _bounded_counts(finish_reasons, max_keys=16).items():
            metrics[f"{gateway_prefix}/finish_reason/{reason}_count"] = count
        for model, count in _bounded_counts(models, max_keys=20).items():
            metrics[f"{gateway_prefix}/model/{model}/request_count"] = count
        for token_name, value in token_totals.items():
            metrics[f"{llm_prefix}/{token_name}"] = int(value)
        if token_request_count:
            metrics[f"{llm_prefix}/tokens_per_request_mean"] = float(token_totals["total_tokens"] / token_request_count)
        return metrics

    def _polling_metrics(self, phase: str) -> dict[str, Any]:
        """Aggregate per-rollout terminal-state + latency metrics."""
        gateway_prefix = f"gateway/{phase}"
        total = self._total_tasks_queued
        latencies = [
            self._rollout_end_time[r] - self._rollout_start_time[r]
            for r in self._rollout_end_time
            if r in self._rollout_start_time
        ]
        return {
            f"{gateway_prefix}/num_succeeded_rollouts": self._num_succeeded,
            f"{gateway_prefix}/num_failed_rollouts": self._num_failed,
            f"{gateway_prefix}/num_timeout_rollouts": self._num_timeout,
            f"{gateway_prefix}/avg_rollout_latency": float(np.mean(latencies)) if latencies else 0.0,
            f"{gateway_prefix}/rollout_completion_rate": (self._num_succeeded / total if total else 0.0),
        }

    def clear_data_and_server(self) -> None:
        """Reset local state for the next iteration."""
        self._total_tasks_queued = 0
        self._completed_rollouts.clear()
        self._task_id_to_original_sample.clear()
        self._enqueue_order.clear()
        self._rollout_status.clear()
        self._rollout_error.clear()
        self._rollout_start_time.clear()
        self._rollout_end_time.clear()
        self._raw_events_by_rollout.clear()
        self._triplet_events_by_rollout.clear()
        self._timeout_rids.clear()
        self._num_succeeded = 0
        self._num_failed = 0
        self._num_timeout = 0
        self.is_train = True
        self._async_n_rollouts_this_step = None

    def _fillna_reward(self, rollout: CompletedRollout) -> float:
        """Return final_reward or the fill value."""
        if rollout.final_reward is not None:
            return rollout.final_reward
        return self.reward_fillna_value

    @staticmethod
    def _has_fallback_reward(rollout: CompletedRollout) -> bool:
        if rollout.final_reward is None:
            return False
        if rollout.reward_source == "fallback":
            return True
        return rollout.reward_reason in _FALLBACK_REWARD_REASONS

    def _has_agent_reward(self, rollout: CompletedRollout) -> bool:
        return rollout.final_reward is not None and not self._has_fallback_reward(rollout)

    # =========================================================================
    # Async-rollout (carry-over) path — parallel methods, sync path untouched.
    # =========================================================================
    #
    # These methods implement the "overshoot sampling + group-finish early
    # stop + cross-step carry-over" feature described in
    # doc_auto/new_async.md. They live alongside the sync methods above and
    # share state fields only where it is safe to do so. The sync path never
    # calls anything below, and these methods never call the sync `clear_*` /
    # `cleanup_agent_jobs` / `get_train_data_batch`.

    def _active_rids(self) -> set[str]:
        """All rids currently in the active pool: this step's new rids plus
        carry-over rids retained from earlier steps."""
        return set(self._step_new_rids) | set(self._carry_over_rids)

    def _active_groups(self) -> dict[str, set[str]]:
        """Map data_id -> rids in the active pool."""
        groups: dict[str, set[str]] = defaultdict(set)
        for rid in self._active_rids():
            did = self._rid_to_data_id.get(rid)
            if did is not None:
                groups[did].add(rid)
        return groups

    def _finished_data_ids(self, active_groups: dict[str, set[str]]) -> set[str]:
        """Data groups whose active rids are all consumable.

        Store/controller terminal states live in ``_rollout_status``. Trainer-
        side wall-clock timeouts live in ``_timeout_rids`` and are consumable
        via the same placeholder path as failed rollouts.
        """
        return {
            did for did, rids in active_groups.items()
            if rids and all((r in self._rollout_status) or (r in self._timeout_rids) for r in rids)
        }

    def _mark_active_unfinished_as_timeout(self, active_rids: set[str]) -> None:
        """Mark active non-terminal rids as trainer-side timeout.

        This mirrors the sync ``run_until_all_finished`` timeout path: timeout
        rids do not enter ``_completed_rollouts`` and therefore become
        placeholder rows when selected for training.
        """
        now = time.time()
        for rid in active_rids:
            if rid in self._rollout_status or rid in self._timeout_rids:
                continue
            self._timeout_rids.add(rid)
            self._rollout_error[rid] = "trainer-side wall-clock timeout"
            self._rollout_end_time[rid] = now
            self._num_timeout += 1

    def _register_and_enqueue_diff(
        self,
        data: dict[str, Any],
        server_addresses: list[str],
        async_train_batch_size: int,
    ) -> int:
        """Enqueue only the rollouts needed to top up the active pool to
        ``async_train_batch_size`` distinct data_ids.

        Returns the number of *new* data_ids enqueued this step. Carry-over
        data_ids are not re-enqueued — their agent pods are still running
        (or in 429 backoff) on K8s.
        """
        # Always (re-)register the current vLLM endpoints so weight-swap
        # between steps lands on the correct addresses.
        self._register_model_servers(server_addresses)

        # Compute how many new data_ids this step needs.
        carry_over_dids = {self._rid_to_data_id[r] for r in self._carry_over_rids if r in self._rid_to_data_id}
        n_carry_over_dids = len(carry_over_dids)
        n_new_dids = async_train_batch_size - n_carry_over_dids
        if n_new_dids <= 0:
            raise RuntimeError(
                "async rollout carry-over saturated: "
                f"async_train_batch_size={async_train_batch_size}, "
                f"n_carry_over_data_ids={n_carry_over_dids}. "
                "Carry-over-only enqueue/poll steps are not supported."
            )

        keys = list(data.keys())
        if not keys:
            return 0
        num_samples = len(data[keys[0]])
        # The trainer is responsible for providing exactly n_new_dids samples
        # via the take(n) sample_iterator; enforce that here defensively.
        if num_samples != n_new_dids:
            raise RuntimeError(
                f"async_register_and_enqueue_diff: dataloader provided {num_samples} samples but "
                f"async_train_batch_size - n_carry_over_dids = {n_new_dids}"
            )

        created = self._enqueue_rollouts(
            data,
            rollouts_per_sample=self.train_rollout_n,
            is_train=True,
        )

        now = time.time()
        for r, req in created:
            rid = r.rollout_id
            original = r.input if isinstance(r.input, dict) else {}
            did = str(original.get("data_id") or original.get("uid") or rid)
            self._task_id_to_original_sample[rid] = r.input if isinstance(r.input, dict) else {}
            self._enqueue_order.append(rid)
            self._rollout_start_time[rid] = now
            self._rid_to_data_id[rid] = did
            self._data_id_to_rids[did].add(rid)
            self._step_new_rids.add(rid)
        self._total_tasks_queued += len(created)
        print(f"AglLiteRolloutBridge.async: enqueued {len(created)} new rollouts ({n_new_dids} new data_ids).")
        return n_new_dids

    def async_set_up_data_and_server(
        self,
        data: dict[str, Any],
        server_addresses: list[str],
        async_train_batch_size: int,
    ) -> int:
        """Sync wrapper around the async-path enqueue. Returns n_new_dids.

        Unlike ``set_up_data_and_server()``, this does NOT clear bridge state —
        carry-over rids (and their book-keeping in _rid_to_data_id /
        _data_id_to_rids / _enqueue_order / _rollout_status / etc.) are
        preserved across steps so the bridge can poll them again.
        """
        self.is_train = True
        self._step_new_rids.clear()
        self._selected_rids.clear()

        # Snapshot "this step's rollouts to run" BEFORE we enqueue new rids:
        # carry-over rids from the previous step that have not yet reached a
        # terminal state (succeeded / failed / timeout). These will
        # still occupy the active pool and consume scheduling capacity this
        # step. Already-terminal carry-overs are excluded — they sit in the
        # pool only as ready-to-select results, not as work to run.
        terminal_rids = set(self._rollout_status.keys()) | self._timeout_rids
        prev_carry_over_unfinished = self._carry_over_rids - terminal_rids

        n_new_dids = self._register_and_enqueue_diff(data, server_addresses, async_train_batch_size)
        # n_new_rids = n_new_dids * rollout_n (set in _register_and_enqueue_diff).
        self._async_n_rollouts_this_step = (
            n_new_dids * self.train_rollout_n + len(prev_carry_over_unfinished)
        )
        return n_new_dids

    def _poll_status_for(self, rids: set[str]) -> None:
        """Refresh status for the given rids, updating bridge bookkeeping.

        Terminal rollouts are recorded in ``self._rollout_status`` and their
        results (if SUCCEEDED) are fetched into ``self._completed_rollouts``.
        Already-terminal rids are skipped — status is sticky across steps.
        """
        for rid in rids:
            if rid in self._rollout_status or rid in self._timeout_rids:
                continue
            rollout = self._get_rollout(rid)
            status = rollout.status.state
            if status not in TERMINAL_STATES:
                continue
            self._rollout_status[rid] = status
            self._rollout_end_time[rid] = time.time()
            if status == RolloutState.SUCCEEDED:
                self._run_succeeded_hook(rollout)
                legacy = self._fetch_rollout_result(rid)
                self._completed_rollouts[rid] = legacy
                self._num_succeeded += 1
            elif status == RolloutState.FAILED:
                self._run_failed_hook(rollout)
                self._fetch_rollout_events(rid)
                self._rollout_error[rid] = rollout.status.error_message or "failed"
                self._num_failed += 1

    def _pick_first_n_by_finish_time(
        self,
        finished_dids: set[str],
        n: int,
    ) -> list[str]:
        """Stable selection: by ``_group_finish_time`` ascending, then by
        earliest rid index in ``_enqueue_order``, then by data_id string."""
        order_index: dict[str, int] = {rid: i for i, rid in enumerate(self._enqueue_order)}

        def earliest_enqueue_index(did: str) -> int:
            return min(order_index.get(r, 10**9) for r in self._data_id_to_rids.get(did, set()))

        sorted_dids = sorted(
            finished_dids,
            key=lambda did: (
                self._group_finish_time.get(did, float("inf")),
                earliest_enqueue_index(did),
                did,
            ),
        )
        return sorted_dids[:n]

    def _run_until_groups_finished(
        self,
        *,
        target_groups: int,
        rollout_n: int,
        drain_timeout: float = 30.0,
        timeout_seconds: float | None = None,
        retry_after_seconds: int = 5,
        step_label: str = "",
        verbose: bool = True,
    ) -> tuple[set[str], set[str], dict[str, Any]]:
        """Poll the active pool until ``target_groups`` complete groups exist,
        then pause+drain the gateway and return the selection.

        Returns ``(selected_rids, carry_over_rids, async_metrics)``.

        Notes:
          - "Consumable" here means ``rid in self._rollout_status`` (i.e.
            SUCCEEDED / FAILED) or ``rid in
            self._timeout_rids``. Timeout rids follow the old sync placeholder
            path when selected for training.
          - From the moment ``len(finished_groups) >= target_groups`` is
            observed to the computation of ``selected_dids``, no ``await``
            happens; the first await is ``pause_gateway()``. This freezes
            the snapshot per docs §8.1.
          - If timeout_seconds is explicitly set, wall-clock timeout is a
            system-fault fallback. When it triggers, all active unfinished rids
            are marked into ``_timeout_rids``; this makes their groups
            consumable via placeholder rows while keeping the selected group
            count at ``target_groups``. The default timeout_seconds=None waits
            forever.
        """
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds
        POLL_INTERVAL = 5.0
        groups_finished_reached = 1
        active_pool_size_at_start = len(self._active_rids())
        inflight_at_pause = 0

        while True:
            active_rids = self._active_rids()
            self._poll_status_for(active_rids)

            active_groups = self._active_groups()
            finished_dids = self._finished_data_ids(active_groups)
            for did in finished_dids:
                if did not in self._group_finish_time:
                    self._group_finish_time[did] = time.time()

            if verbose:
                print(
                    f"AglLiteRolloutBridge.async{step_label}: "
                    f"active_rids={len(active_rids)} active_groups={len(active_groups)} "
                    f"finished_groups={len(finished_dids)}/{target_groups}"
                )

            if len(finished_dids) >= target_groups:
                finished_snapshot = set(finished_dids)
                selected_dids = self._pick_first_n_by_finish_time(finished_snapshot, target_groups)
                selected_rids: set[str] = set()
                for did in selected_dids:
                    selected_rids.update(active_groups.get(did, set()))

                pause_state = self._pause_gateway(
                    retry_after_seconds=retry_after_seconds,
                    reason=f"async-rollout step {step_label or ''}".strip(),
                )
                inflight_at_pause = int(pause_state.get("inflight", 0))
                drain_start = time.time()
                residual = self._wait_until_inflight_drained(timeout=drain_timeout)
                drain_seconds = time.time() - drain_start
                drain_timed_out = 1 if residual > 0 else 0
                if residual > 0:
                    print(
                        f"AglLiteRolloutBridge.async{step_label}: drain timeout — "
                        f"{residual} in-flight upstream requests remaining."
                    )
                unselected = self._active_rids() - selected_rids
                metrics = {
                    "training/async/groups_finished_reached": groups_finished_reached,
                    "training/async/n_active_data_ids": len(active_groups),
                    "training/async/n_active_rollouts": active_pool_size_at_start,
                    "training/async/n_selected_groups": len(selected_dids),
                    "training/async/drain_wait_seconds": float(drain_seconds),
                    "training/async/drain_timeout": int(drain_timed_out),
                    "training/async/inflight_at_pause": inflight_at_pause,
                    "training/async/group_finish_skew_s": float(self._compute_group_finish_skew(selected_dids)),
                }
                return selected_rids, unselected, metrics

            if deadline is not None and time.time() >= deadline and len(finished_dids) < target_groups:
                groups_finished_reached = 0
                self._mark_active_unfinished_as_timeout(active_rids)
                active_groups = self._active_groups()
                finished_dids = self._finished_data_ids(active_groups)
                for did in finished_dids:
                    if did not in self._group_finish_time:
                        self._group_finish_time[did] = time.time()
                if len(finished_dids) < target_groups:
                    raise RuntimeError(
                        "async rollout timed out but active pool did not contain enough complete groups "
                        f"after timeout placeholder marking: {len(finished_dids)}/{target_groups}"
                    )
                finished_snapshot = set(finished_dids)
                selected_dids = self._pick_first_n_by_finish_time(finished_snapshot, target_groups)
                selected_rids = set()
                for did in selected_dids:
                    selected_rids.update(active_groups.get(did, set()))

                pause_state = self._pause_gateway(
                    retry_after_seconds=retry_after_seconds,
                    reason=f"async-rollout step {step_label or ''} (timeout-placeholder)".strip(),
                )
                inflight_at_pause = int(pause_state.get("inflight", 0))
                drain_start = time.time()
                residual = self._wait_until_inflight_drained(timeout=drain_timeout)
                drain_seconds = time.time() - drain_start
                drain_timed_out = 1 if residual > 0 else 0
                unselected = self._active_rids() - selected_rids
                metrics = {
                    "training/async/groups_finished_reached": groups_finished_reached,
                    "training/async/n_active_data_ids": len(active_groups),
                    "training/async/n_active_rollouts": active_pool_size_at_start,
                    "training/async/n_selected_groups": len(selected_dids),
                    "training/async/drain_wait_seconds": float(drain_seconds),
                    "training/async/drain_timeout": int(drain_timed_out),
                    "training/async/inflight_at_pause": inflight_at_pause,
                    "training/async/group_finish_skew_s": float(self._compute_group_finish_skew(selected_dids)),
                }
                return selected_rids, unselected, metrics

            time.sleep(POLL_INTERVAL)

    def _compute_group_finish_skew(self, selected_dids: list[str]) -> float:
        """Median of (max - min) end-time within each selected group.
        Used as the ``group_finish_skew_s`` metric to observe within-group
        bucket effects (long-tail rollouts inside a group)."""
        if not selected_dids:
            return 0.0
        per_group_spread: list[float] = []
        for did in selected_dids:
            rids = self._data_id_to_rids.get(did, set())
            ends = [self._rollout_end_time[r] for r in rids if r in self._rollout_end_time]
            if len(ends) >= 2:
                per_group_spread.append(max(ends) - min(ends))
        if not per_group_spread:
            return 0.0
        return float(np.median(per_group_spread))

    def run_until_groups_finished(
        self,
        *,
        target_groups: int,
        rollout_n: int,
        drain_timeout: float = 30.0,
        timeout_seconds: float | None = None,
        retry_after_seconds: int = 5,
        step_label: str = "",
        verbose: bool = True,
    ) -> tuple[set[str], set[str], dict[str, Any]]:
        """Sync wrapper for the async-path group-finish poll.

        Uses direct synchronous HTTP calls for polling and pause/drain.
        """
        return self._run_until_groups_finished(
            target_groups=target_groups,
            rollout_n=rollout_n,
            drain_timeout=drain_timeout,
            timeout_seconds=timeout_seconds,
            retry_after_seconds=retry_after_seconds,
            step_label=step_label,
            verbose=verbose,
        )

    def commit_async_step_selection(
        self,
        selected_rids: set[str],
        unselected_rids: set[str],
        current_step: int,
    ) -> dict[str, Any]:
        """Record the selection produced by ``run_until_groups_finished``.

        Updates ``_selected_rids`` (consumed by the next call to
        ``async_get_train_data_batch``) and rewrites ``_carry_over_rids``
        per the rule in docs §3.5(4):

            carry_over_in  = previous _carry_over_rids
            carry_over_out = (active_rids - selected_rids)

        Returns the carry-over-related metrics for this step.
        """
        prev_carry_over = set(self._carry_over_rids)
        n_carry_over_resumed = len(prev_carry_over & selected_rids)

        self._selected_rids = set(selected_rids)
        # carry_over_out = all active rids not selected this step
        new_carry_over = set(unselected_rids)

        # Birth-step bookkeeping for carry_over_age metric.
        for rid in new_carry_over - prev_carry_over:
            self._carry_over_birth_step.setdefault(rid, current_step)
        for rid in selected_rids:
            self._carry_over_birth_step.pop(rid, None)
        self._carry_over_rids = new_carry_over

        ages = [
            max(0, current_step - self._carry_over_birth_step[r])
            for r in self._carry_over_rids
            if r in self._carry_over_birth_step
        ]
        return {
            "training/async/n_carry_over_in": len(prev_carry_over),
            "training/async/n_carry_over_out": len(self._carry_over_rids),
            "training/async/n_carry_over_resumed": n_carry_over_resumed,
            "training/async/carry_over_age_max_steps": max(ages) if ages else 0,
        }

    def async_get_train_data_batch(
        self,
        max_prompt_length: int,
        max_response_length: int,
        device: torch.device,
        global_steps: int,
    ):
        """Assemble a training batch for the rids selected this step.

        Mirrors ``get_train_data_batch`` exactly (same trace aggregator,
        same placeholder semantics, same MRoPE handling) but iterates only
        over the rids in ``self._selected_rids`` and *not* the full
        ``_enqueue_order``. After the batch is built, the selected rids and
        their per-rid bookkeeping are cleared from the bridge state so the
        carry-over pool can keep growing past consumed rids.
        """
        assert self.is_train, "This method should only be called during training."
        if not self._selected_rids:
            raise RuntimeError("async_get_train_data_batch called with empty _selected_rids")

        # Iterate selected rids in their original enqueue order so the
        # batch layout is deterministic and (where possible) matches the
        # sync path's behavior.
        selected_order = [rid for rid in self._enqueue_order if rid in self._selected_rids]
        # Defensive: any selected rid missing from _enqueue_order is appended
        # in arbitrary (sorted) order at the end. This should not happen.
        missing = self._selected_rids - set(selected_order)
        selected_order.extend(sorted(missing))

        prev_enqueue_order = self._enqueue_order
        try:
            # get_train_data_batch iterates self._enqueue_order — temporarily
            # narrow it to the selected rids so we reuse the full assembly
            # logic without copy-paste.
            self._enqueue_order = selected_order
            data_proto, metrics = self.get_train_data_batch(
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                device=device,
                global_steps=global_steps,
            )
        finally:
            self._enqueue_order = prev_enqueue_order

        # Override `training/n_rollouts` with the async-path semantics: the
        # number of rollouts this step actually had to run, i.e. new rids
        # enqueued this step plus carry-over rids from the previous step that
        # were still unfinished at step entry. The base sync semantics
        # (_total_tasks_queued) is preserved only for the legacy `_rollout`
        # path; in async mode it is a cumulative counter periodically reset
        # by `_validate` -> `clear_data_and_server`, which is misleading here.
        if self._async_n_rollouts_this_step is not None:
            metrics["training/n_rollouts"] = self._async_n_rollouts_this_step
            self._async_n_rollouts_this_step = None

        # Selected rids are now consumed — drop their per-rid bookkeeping.
        # Carry-over rids (which are NOT in selected_order) keep their state.
        self._async_drop_consumed(set(selected_order))
        return data_proto, metrics

    def _async_drop_consumed(self, consumed_rids: set[str]) -> None:
        """Remove per-rid bookkeeping for rids fully consumed by this step.

        Mirrors the cleanup section of ``clear_data_and_server()`` but only
        for the given rid set — carry-over state is preserved.
        """
        consumed_dids: set[str] = set()
        for rid in consumed_rids:
            did = self._rid_to_data_id.pop(rid, None)
            if did is not None:
                consumed_dids.add(did)
                rids_in_group = self._data_id_to_rids.get(did)
                if rids_in_group is not None:
                    rids_in_group.discard(rid)
                    if not rids_in_group:
                        self._data_id_to_rids.pop(did, None)
            self._completed_rollouts.pop(rid, None)
            self._task_id_to_original_sample.pop(rid, None)
            self._rollout_status.pop(rid, None)
            self._rollout_error.pop(rid, None)
            self._rollout_start_time.pop(rid, None)
            self._rollout_end_time.pop(rid, None)
            self._timeout_rids.discard(rid)
            self._carry_over_birth_step.pop(rid, None)
            self._selected_rids.discard(rid)
            self._step_new_rids.discard(rid)
        # Compact _enqueue_order to drop consumed rids — cheap with set lookup.
        if consumed_rids:
            self._enqueue_order = [r for r in self._enqueue_order if r not in consumed_rids]
        # Drop group_finish_time only for fully-consumed groups (no remaining
        # active rids). A group is "fully consumed" if no rid for that did is
        # left in any tracked set.
        for did in consumed_dids:
            if did not in self._data_id_to_rids:
                self._group_finish_time.pop(did, None)

    def _get_cleanup_k8s_client(self) -> AgentJobK8sClient | None:
        """Return the K8s client used for optional agent Job cleanup.

        Cleanup is best-effort. If cleanup is enabled but no client is wired,
        skip cleanup work instead of failing rollout/training.
        """
        if self._cleanup_k8s_client is not None:
            return self._cleanup_k8s_client
        if self._cleanup_agent_jobs_enabled and not self._cleanup_missing_client_warned:
            print("AglLiteRolloutBridge: cleanup_agent_jobs enabled but cleanup_k8s_client is missing; skipping cleanup.")
            self._cleanup_missing_client_warned = True
        return None

    async def _async_cleanup_tracked_agent_jobs(self) -> None:
        """Delete only agl-lite Jobs for rollout IDs tracked by this bridge batch.

        Used by the sync path (`fit()` / validation) which clears the whole
        ``_enqueue_order`` between steps. The async path uses
        ``_async_cleanup_consumed_jobs`` instead so carry-over pods survive.
        """
        if not self._cleanup_agent_jobs_enabled:
            return
        if not self._enqueue_order:
            return
        if not self._cleanup_namespace:
            raise RuntimeError("cleanup_namespace is required when cleanup_agent_jobs is enabled")

        tracked_rollout_ids = set(self._enqueue_order)
        k8s = self._get_cleanup_k8s_client()
        if k8s is None:
            return
        jobs = await k8s.list_jobs(
            namespace=self._cleanup_namespace,
            label_selector=_AGL_LITE_MANAGED_BY_SELECTOR,
        )

        deleted_count = 0
        for job in jobs:
            metadata = job.get("metadata", {})
            labels = metadata.get("labels", {})
            if labels.get(_AGL_LITE_MANAGED_BY_LABEL) != _AGL_LITE_MANAGED_BY_VALUE:
                continue
            rollout_id = labels.get(_AGL_LITE_ROLLOUT_ID_LABEL)
            job_name = metadata.get("name")
            if rollout_id in tracked_rollout_ids and job_name:
                await k8s.delete_job(job_name, self._cleanup_namespace)
                deleted_count += 1

        if deleted_count:
            print(f"AglLiteRolloutBridge: deleted {deleted_count} tracked agent Jobs.")

    def cleanup_agent_jobs(self) -> None:
        """Synchronously clean up tracked agl-lite agent Jobs when enabled.

        Sync-path public API; the async path uses ``async_cleanup_consumed``.
        """
        if not self._cleanup_agent_jobs_enabled:
            return
        asyncio.run(self._async_cleanup_tracked_agent_jobs())

    async def _async_cleanup_consumed_jobs(self, consumed_rids: set[str]) -> None:
        """Delete K8s Jobs for the given rids only (preserve carry-over pods)."""
        if not self._cleanup_agent_jobs_enabled:
            return
        if not consumed_rids:
            return
        if not self._cleanup_namespace:
            raise RuntimeError("cleanup_namespace is required when cleanup_agent_jobs is enabled")

        k8s = self._get_cleanup_k8s_client()
        if k8s is None:
            return
        jobs = await k8s.list_jobs(
            namespace=self._cleanup_namespace,
            label_selector=_AGL_LITE_MANAGED_BY_SELECTOR,
        )
        deleted = 0
        for job in jobs:
            metadata = job.get("metadata", {})
            labels = metadata.get("labels", {})
            if labels.get(_AGL_LITE_MANAGED_BY_LABEL) != _AGL_LITE_MANAGED_BY_VALUE:
                continue
            rollout_id = labels.get(_AGL_LITE_ROLLOUT_ID_LABEL)
            job_name = metadata.get("name")
            if rollout_id in consumed_rids and job_name:
                await k8s.delete_job(job_name, self._cleanup_namespace)
                deleted += 1
        if deleted:
            print(f"AglLiteRolloutBridge.async: deleted {deleted} consumed agent Jobs.")

    def async_cleanup_consumed(self, consumed_rids: set[str]) -> None:
        """Sync wrapper. Carry-over rids' Jobs are NEVER touched here."""
        if not self._cleanup_agent_jobs_enabled:
            return
        if not consumed_rids:
            return
        asyncio.run(self._async_cleanup_consumed_jobs(set(consumed_rids)))

    def n_carry_over_data_ids(self) -> int:
        """Distinct data_ids currently represented in the carry-over pool."""
        return len({self._rid_to_data_id[r] for r in self._carry_over_rids if r in self._rid_to_data_id})

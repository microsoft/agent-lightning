"""AglLiteDaemon — bridge between agl-lite HTTP API and VERL trainer.

This module replaces Agent Lightning's AgentModeDaemon + LightningStore + LLMProxy
+ Adapter stack with a single class that talks to agl-lite over HTTP using
the AglLiteClient.

The trainer calls 4 methods:
  1. set_up_data_and_server()  — register model + enqueue rollouts
  2. run_until_all_finished()  — poll until all rollouts complete
  3. get_train_data_batch()    — fetch triplets, build padded tensors → DataProto
  4. clear_data_and_server()   — reset state

Compared to AgentModeDaemon (1154 lines):
  - Store interaction (209 lines): REPLACED with AglLiteClient calls
  - Proxy server (141 lines): DROPPED (agl-lite gateway handles this)
  - Tensor construction (328 lines): COPIED from agent-lightning (unchanged)
  - Multimodal/mrope (63 lines): COPIED from agent-lightning (unchanged)
  - Utilities (157 lines): COPIED from agent-lightning (unchanged)
  - Validation/metrics (106 lines): COPIED from agent-lightning (unchanged)

Origin: agentlightning/verl/daemon.py (commit to be pinned)
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agl_lite.client import AglLiteClient
from agl_lite.schemas.api import EnqueueRolloutRequest, RegisterModelRequest

# --- Optional heavy imports (only needed when actually training) ---
try:
    import torch
    from tensordict import TensorDict
    from verl import DataProto
except ImportError:
    torch = None  # type: ignore[assignment]
    TensorDict = None  # type: ignore[assignment,misc]
    DataProto = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Types used by the daemon.
#
# These mirror the Agent Lightning types that get_train_data_batch() expects.
# Defined here to avoid importing agentlightning at runtime.
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class Triplet(BaseModel):
    """Single interaction turn (prompt + response + reward).

    Compatible with agentlightning.types.Triplet.
    """
    prompt: Any  # {"token_ids": [...], "image_urls": [...]}
    response: Any  # {"token_ids": [...]}
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Task echoed back in RolloutLegacy."""
    rollout_id: str
    input: Any = None
    mode: Optional[str] = None
    resources_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RolloutLegacy(BaseModel):
    """Completed rollout with triplets, used by get_train_data_batch().

    Compatible with agentlightning.types.RolloutLegacy.
    """
    rollout_id: str
    task: Optional[Task] = None
    final_reward: Optional[float] = None
    triplets: Optional[List[Triplet]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Utilities — copied from agentlightning/verl/daemon.py
# ============================================================================

def ids_startswith(
    ids: List[int], context: List[int], tokenizer: Any = None, debug: bool = False
) -> Tuple[bool, Tuple[int, int, int]]:
    """Check if context is a prefix of ids, with tolerance for special-token differences."""

    def _special_token_sequence(ids: List[int]) -> List[int]:
        return [idx for idx, token_id in enumerate(ids) if token_id in (tokenizer.all_special_ids if tokenizer else [])]

    def _none_special_token_sequence(ids: List[int]) -> List[int]:
        return [idx for idx, token_id in enumerate(ids) if token_id not in (tokenizer.all_special_ids if tokenizer else [])]

    if not context:
        return True, (0, 0, 0)

    if len(ids) < len(context):
        return False, (0, 0, 0)

    # Check if context is a prefix of ids
    if ids[: len(context)] == context:
        return True, (0, 0, 0)

    # Retry ignoring special tokens
    context_non_special = _none_special_token_sequence(context)
    ids_non_special = _none_special_token_sequence(ids[: len(context)])

    if ids_non_special == context_non_special:
        template_mismatch = 1
        return False, (template_mismatch, 0, 0)

    # Try retokenization match
    if tokenizer is not None:
        context_text = tokenizer.decode(context, skip_special_tokens=True)
        ids_text = tokenizer.decode(ids[: len(context)], skip_special_tokens=True)
        if context_text == ids_text:
            return False, (0, 1, 0)

    return False, (0, 0, 1)


def log_mismatch_detail(
    diagnostic: Tuple[int, int, int],
    ids: List[int],
    context: List[int],
    global_steps: int,
    rollout_id: str,
    turn_id: int,
    mismatch_log_dir: Optional[str] = None,
) -> None:
    """Log details about token sequence mismatches for debugging."""
    template_mismatch, retoken_mismatch, others_mismatch = diagnostic
    msgs: List[str] = []
    if template_mismatch:
        msgs.append(
            "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
        )
        msgs.append(f"Template mismatch: ids[:len(context)]={ids[: len(context)]}, context={context}")
    if retoken_mismatch:
        msgs.append(
            "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
        )
        msgs.append(f"Retokenization mismatch: ids[:len(context)]={ids[: len(context)]}, context={context}")
    if others_mismatch:
        msgs.append(
            "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
        )
        msgs.append(f"Others mismatch: ids[:len(context)]={ids[: len(context)]}, context={context}")
    for msg in msgs:
        print(msg)


def get_left_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """Left-pad token ids to max_length and create attention mask.

    Args:
        ids:          Variable-length list of token IDs.
        max_length:   Target length after padding.
        pad_token_id: ID to use for padding.

    Returns:
        Tuple of (padded_ids, attention_mask).
    """
    if len(ids) > max_length:
        ids = ids[:max_length]

    pad_len = max_length - len(ids)
    attention_mask = [0] * pad_len + [1] * len(ids)
    padded_ids = [pad_token_id] * pad_len + ids
    return padded_ids, attention_mask


def get_right_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """Right-pad token ids to max_length and create attention mask.

    Args:
        ids:          Variable-length list of token IDs.
        max_length:   Target length after padding.
        pad_token_id: ID to use for padding.

    Returns:
        Tuple of (padded_ids, attention_mask).
    """
    if len(ids) > max_length:
        ids = ids[:max_length]

    pad_len = max_length - len(ids)
    attention_mask = [1] * len(ids) + [0] * pad_len
    padded_ids = ids + [pad_token_id] * pad_len
    return padded_ids, attention_mask


def _to_native(obj: Any) -> Any:
    """Convert numpy/torch types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    return obj


# ============================================================================
# AglLiteDaemon — the main class
# ============================================================================

class AglLiteDaemon:
    """Bridge between agl-lite HTTP API and VERL trainer.

    Drop-in replacement for AgentModeDaemon. The trainer calls the same 4 methods:
    set_up_data_and_server, run_until_all_finished, get_train_data_batch,
    clear_data_and_server.

    Compared to AgentModeDaemon:
    - No LightningStore — talks to agl-lite over HTTP
    - No LLMProxy — agl-lite gateway handles proxying
    - No Adapter — agl-lite's format=triplet does event→triplet conversion
    - No proxy server thread — not needed
    - Tensor construction (get_train_data_batch) is identical
    """

    def __init__(
        self,
        agl_lite_url: str,
        agl_key: str,
        train_rollout_n: int,
        train_information: Dict[str, Any],
        tokenizer: Any,
        mini_batch_size: int,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        timeout_seconds: float = 1200.0,
        processor: Any = None,
        image_base_dir: Optional[str] = None,
        trace_aggregator: Dict[str, Any] | None = None,
    ):
        # --- agl-lite connection (REPLACES store + proxy + adapter) ---
        self.client = AglLiteClient(base_url=agl_lite_url, agl_key=agl_key)

        # --- Training config (same as AgentModeDaemon) ---
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fillna_value = reward_fillna_value
        self.image_base_dir = image_base_dir
        self.trace_aggregator = trace_aggregator or {"level": "transition"}
        self.timeout_seconds = timeout_seconds

        # --- Multimodal (copied from AgentModeDaemon) ---
        self._use_mrope = self._is_mrope_model()

        # --- Internal state (same as AgentModeDaemon) ---
        self._total_tasks_queued = 0
        self._completed_rollouts_v0: Dict[str, RolloutLegacy] = {}
        self._task_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self.is_train = True

        # --- Async event loop for _async methods ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ==================================================================
    # NEW: agl-lite HTTP interaction (REPLACES store + proxy + adapter)
    # ==================================================================

    def start(self) -> None:
        """No-op. AgentModeDaemon starts proxy server here; agl-lite gateway handles that."""
        pass

    async def _async_set_up(
        self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True
    ) -> None:
        """Register model server and enqueue rollouts via AglLiteClient.

        Replaces AgentModeDaemon._async_set_up() which calls:
          store.add_resources()
          store.enqueue_many_rollouts()
        """
        self.clear_data_and_server()
        self.is_train = is_train

        # 1. Register model server with agl-lite
        model_name = self.train_information.get("model", "default-model")
        regs = []
        for addr in server_addresses:
            endpoint = f"http://{addr}/v1" if not addr.startswith("http") else addr
            regs.append(RegisterModelRequest(model=model_name, endpoint=endpoint))
        await self.client.register_models(regs)

        # 2. Enqueue rollouts
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        rollout_requests: List[EnqueueRolloutRequest] = []
        data_id_to_original: Dict[str, Dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original = {key: _to_native(data[key][i]) for key in keys}
            original["data_id"] = data_id
            data_id_to_original[data_id] = original

            for _ in range(rollouts_per_sample):
                rollout_requests.append(EnqueueRolloutRequest(
                    input=_to_native(original),
                    config={"timeout": int(self.timeout_seconds)},
                ))

        created = await self.client.enqueue_rollouts(rollout_requests)

        for r in created:
            rid = r.rollout_id
            # data_id is embedded in the input (original sample dict)
            data_id = r.input.get("data_id") if isinstance(r.input, dict) else None
            if data_id and data_id in data_id_to_original:
                self._task_id_to_original_sample[rid] = data_id_to_original[data_id]
        self._total_tasks_queued += len(created)

    def set_up_data_and_server(
        self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True
    ) -> None:
        """Sync wrapper — same signature as AgentModeDaemon."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._async_set_up(data, server_addresses, is_train), self._loop
        )
        future.result()

    async def _async_validate_data(self, rollout_id: str) -> RolloutLegacy:
        """Fetch triplets for a completed rollout via AglLiteClient.

        Replaces AgentModeDaemon._validate_data_v1() which calls:
          store.query_spans() → adapter.adapt(spans) → List[Triplet]

        In agl-lite, the server does event→triplet conversion via format=triplet.
        """
        events = await self.client.get_events(rollout_id, format="triplet")

        # Convert trimmed events to Triplet objects
        triplets: List[Triplet] = []
        for evt in events:
            if evt.event_type == "model_request":
                d = evt.data
                triplets.append(Triplet(
                    prompt={"token_ids": d.get("prompt_token_ids", [])},
                    response={"token_ids": d.get("response_token_ids", [])},
                    reward=None,
                    metadata={"server": d.get("server", {})},
                ))

        # Match rewards to triplets (sequential: last reward wins)
        final_reward: Optional[float] = None
        reward_events = [e for e in events if e.event_type == "reward"]
        if reward_events:
            final_reward = reward_events[-1].data.get("value")

        # Assign reward to last triplet (same as Agent Lightning convention)
        if triplets and final_reward is not None:
            triplets[-1] = triplets[-1].model_copy(update={"reward": final_reward})

        original = self._task_id_to_original_sample.get(rollout_id, {})
        return RolloutLegacy(
            rollout_id=rollout_id,
            task=Task(
                rollout_id=rollout_id,
                input=original,
                metadata=original.get("metadata", {}),
            ),
            final_reward=final_reward,
            triplets=triplets,
            metadata=original.get("metadata", {}),
        )

    async def _async_run_until_finished(self, verbose: bool = True) -> None:
        """Poll agl-lite until all rollouts complete.

        Replaces AgentModeDaemon._async_run_until_finished() which calls:
          store.wait_for_rollouts()
        """
        while len(self._completed_rollouts_v0) < self._total_tasks_queued:
            rollout_ids = list(self._task_id_to_original_sample.keys())
            for rid in rollout_ids:
                if rid in self._completed_rollouts_v0:
                    continue
                rollout = await self.client.get_rollout(rid)
                if rollout.status == "succeeded":
                    legacy = await self._async_validate_data(rid)
                    self._completed_rollouts_v0[rid] = legacy

            if verbose:
                print(
                    f"Completed {len(self._completed_rollouts_v0)}/{self._total_tasks_queued} tasks..."
                )
            await asyncio.sleep(5)

        print("All tasks finished.")

    def run_until_all_finished(self, verbose: bool = True) -> None:
        """Sync wrapper — same signature as AgentModeDaemon."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._async_run_until_finished(verbose), self._loop
        )
        future.result()

    def clear_data_and_server(self) -> None:
        """Reset internal state for next iteration.

        Same as AgentModeDaemon.clear_data_and_server().
        """
        self._total_tasks_queued = 0
        self._completed_rollouts_v0.clear()
        self._task_id_to_original_sample.clear()
        self.is_train = True

    # ==================================================================
    # Validation & metrics — copied from AgentModeDaemon
    # ==================================================================

    def _validate_data(self, rollout: RolloutLegacy) -> None:
        """Basic validation on a completed rollout."""
        if rollout.triplets is None or len(rollout.triplets) == 0:
            print(f"Warning: No triplets found for rollout {rollout.rollout_id}")
        if rollout.final_reward is None:
            print(f"Warning: No reward found for rollout {rollout.rollout_id}")

    def _fillna_reward(self, rollout: RolloutLegacy) -> float:
        """Return final_reward or the fill value."""
        if rollout.final_reward is not None:
            return rollout.final_reward
        return self.reward_fillna_value

    def get_test_metrics(self) -> Dict[str, Any]:
        """Compute test/validation metrics from completed rollouts.

        Copied from AgentModeDaemon.get_test_metrics().
        """
        assert not self.is_train
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        rewards: List[float] = []
        n_with_triplets = 0
        n_with_reward = 0

        for rollout in self._completed_rollouts_v0.values():
            reward = self._fillna_reward(rollout)
            rewards.append(reward)
            if rollout.triplets:
                n_with_triplets += 1
            if rollout.final_reward is not None:
                n_with_reward += 1

        metrics: Dict[str, Any] = {
            "val/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "val/reward_max": float(np.max(rewards)) if rewards else 0.0,
            "val/reward_min": float(np.min(rewards)) if rewards else 0.0,
            "val/n_rollouts": len(self._completed_rollouts_v0),
            "val/n_rollouts_w_triplets": n_with_triplets,
            "val/n_rollouts_w_reward": n_with_reward,
        }
        return metrics

    # ==================================================================
    # Multimodal support — copied from AgentModeDaemon
    # ==================================================================

    def _is_mrope_model(self) -> bool:
        """Check if the model uses multi-dimensional rotary position embeddings."""
        if self.processor is None:
            return False
        model_type = getattr(getattr(self.processor, "image_processor", None), "model_type", None)
        return model_type in ("qwen2_vl", "qwen3_vl")

    def _resolve_image_path(self, path: str) -> str:
        """Resolve relative image paths against image_base_dir."""
        import os
        if self.image_base_dir and not os.path.isabs(path) and not path.startswith(("http://", "https://")):
            return os.path.join(self.image_base_dir, path)
        return path

    def _get_image_grid_thw(self, image_urls: List[str]) -> Optional["torch.Tensor"]:
        """Compute image grid (T, H, W) for Qwen2-VL mrope position encoding."""
        if not image_urls or not self._use_mrope:
            return None
        try:
            from verl.utils.dataset.vision_utils import process_image  # pyright: ignore

            def to_image_uri(url: str) -> str:
                resolved = self._resolve_image_path(url)
                if resolved.startswith(("http://", "https://")):
                    return resolved
                return f"file://{resolved}"

            image_data = process_image(
                {"image": [to_image_uri(u) for u in image_urls]},
                self.processor,
            )
            return image_data.get("image_grid_thw")
        except Exception as e:
            print(f"Warning: Failed to process images for mrope: {e}")
            return None

    def _compute_mrope_position_ids(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        image_grid_thw: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute multi-dimensional rotary position IDs for Qwen2-VL."""
        model_type = getattr(getattr(self.processor, "image_processor", None), "model_type", None)
        if model_type == "qwen3_vl":
            from verl.models.transformers.qwen3_vl import get_rope_index  # pyright: ignore
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index  # pyright: ignore

        position_ids, _ = get_rope_index(
            self.processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
        valid_mask = attention_mask.bool()
        text_pos = torch.zeros((1, len(input_ids)), dtype=torch.long, device=input_ids.device)
        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)
        return position_ids.squeeze(0)

    # ==================================================================
    # Tensor construction — copied from AgentModeDaemon.get_train_data_batch()
    #
    # This is the core 328-line method that converts triplets into padded
    # tensors for PPO training. Identical to the original.
    # ==================================================================

    def get_train_data_batch(
        self, max_prompt_length: int, max_response_length: int, device: "torch.device", global_steps: int
    ) -> Tuple[Any, Dict[str, Any]]:
        """Processes completed rollouts to generate a training data batch.

        This function reconstructs the logic from the original AgentModeDaemon,
        using data retrieved from agl-lite. It handles padding, truncation, and
        tensor creation for the PPO training loop.
        """
        assert self.is_train, "This method should only be called during training."
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        # 1. Reconstruct the `finished_id_to_sample_info` structure from completed rollouts
        finished_id_to_sample_info: Dict[str, Dict[str, Any]] = {}
        finished_id_to_final_reward: Dict[str, float] = {}
        sample_with_reward_count = 0
        for rollout_id, rollout in self._completed_rollouts_v0.items():
            original_sample = self._task_id_to_original_sample[rollout_id]
            sample_with_reward_count += int(rollout.final_reward is not None)
            final_reward = self._fillna_reward(rollout)

            if not rollout.triplets:
                finished_id_to_final_reward[rollout_id] = final_reward
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue

            trace_list = [
                {
                    "prompt_ids": t.prompt.get("token_ids", []),
                    "response_ids": t.response.get("token_ids", []),
                    "image_urls": t.prompt.get("image_urls", []),
                }
                for t in rollout.triplets
            ]
            info = {
                "reward": final_reward,
                "trace_list": trace_list,
                "data_id": original_sample["data_id"],
            }
            finished_id_to_sample_info[rollout_id] = info
            finished_id_to_final_reward[rollout_id] = final_reward

        # 2. Build padded tensors from triplets
        input_ids_list: List[List[int]] = []
        input_attention_mask_list: List[List[int]] = []
        response_ids_list: List[List[int]] = []
        response_attention_mask_list: List[List[int]] = []
        reward_list: List[float] = []
        data_id_list: List[str] = []
        rollout_id_list: List[str] = []
        turn_index_list: List[int] = []
        is_drop_list: List[bool] = []
        image_grid_thw_list: List[Optional["torch.Tensor"]] = []
        n_trunc_sample_because_of_response = 0

        if self.trace_aggregator.get("level", "transition") == "transition":
            for rollout_id, sample_info in finished_id_to_sample_info.items():
                for turn_index, trace in enumerate(sample_info["trace_list"]):

                    reward_list.append(sample_info["reward"])
                    prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]

                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        n_trunc_sample_because_of_response += 1

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
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
                    turn_index_list.append(turn_index)

                    if self._use_mrope:
                        image_urls = trace.get("image_urls", [])
                        image_grid_thw_list.append(self._get_image_grid_thw(image_urls))

        elif self.trace_aggregator.get("level", "transition") == "trajectory":
            assert not self._use_mrope, "M-RoPE is not supported in trajectory level yet."

            response_mask_list: List[List[int]] = []
            unmerged_count: int = 0
            template_mismatch_count, retoken_mismatch_count, others_mismatch_count = 0, 0, 0
            response_per_turn_list: List[int] = []

            for rollout_id, sample_info in finished_id_to_sample_info.items():
                merged_trace_idx: List[List[int]] = []

                current_merged_trace_idx: List[int] = []
                current_context: List[int] = []
                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    response_per_turn_list.append(len(trace["response_ids"]))
                    is_prefix, diagnostic = ids_startswith(
                        trace["prompt_ids"] + trace["response_ids"],
                        current_context,
                        self.tokenizer,
                        self.trace_aggregator.get("debug", False),
                    )
                    if not is_prefix and self.trace_aggregator.get("debug", False) is True:
                        template_mismatch_count += diagnostic[0]
                        retoken_mismatch_count += diagnostic[1]
                        others_mismatch_count += diagnostic[2]
                        log_mismatch_detail(
                            diagnostic,
                            trace["prompt_ids"] + trace["response_ids"],
                            current_context,
                            global_steps,
                            rollout_id,
                            turn_index,
                            self.trace_aggregator.get("mismatch_log_dir", None),
                        )

                    if is_prefix:
                        current_context = trace["prompt_ids"] + trace["response_ids"]
                        current_merged_trace_idx.append(turn_index)
                    else:
                        merged_trace_idx.append(current_merged_trace_idx)
                        current_merged_trace_idx = [turn_index]
                        current_context = trace["prompt_ids"] + trace["response_ids"]

                if current_merged_trace_idx not in merged_trace_idx:
                    merged_trace_idx.append(current_merged_trace_idx)

                if len(merged_trace_idx) > 1:
                    unmerged_count += 1

                for current_merged_trace_idx in merged_trace_idx:
                    prompt_ids = sample_info["trace_list"][current_merged_trace_idx[0]]["prompt_ids"]

                    if current_merged_trace_idx[0] > 0 and len(prompt_ids) > max_prompt_length:
                        response_ids = prompt_ids[max_prompt_length:]
                        prompt_ids = prompt_ids[:max_prompt_length]
                        response_mask = [1] * len(response_ids)
                    else:
                        response_ids = []
                        response_mask = []

                    prompt_length = len(prompt_ids)
                    response_ids += sample_info["trace_list"][current_merged_trace_idx[0]]["response_ids"]
                    response_mask += [1] * len(response_ids)
                    for turn_index in current_merged_trace_idx[1:]:
                        trace = sample_info["trace_list"][turn_index]
                        new_prompt_length = len(trace["prompt_ids"]) - len(response_ids) - prompt_length
                        response_ids += trace["prompt_ids"][-new_prompt_length:]
                        response_ids += trace["response_ids"]
                        response_mask += [0] * new_prompt_length
                        response_mask += [1] * len(trace["response_ids"])

                    reward_list.append(sample_info["reward"])

                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        response_mask = response_mask[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                        prompt_ids, max_prompt_length, self.pad_token_id
                    )
                    one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                        response_ids, max_response_length, self.pad_token_id
                    )
                    one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                        response_mask, max_response_length, 0
                    )

                    input_ids_list.append(one_input_ids)
                    input_attention_mask_list.append(one_input_attention_mask)
                    response_ids_list.append(one_response_ids)
                    response_attention_mask_list.append(one_response_attention_mask)
                    response_mask_list.append(one_response_mask)
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
        else:
            raise ValueError(f"Unknown trace_aggregator level: {self.trace_aggregator.get('level')}")

        # 3. Convert to tensors
        n_transition = len(input_ids_list)
        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)
        response_mask = (
            torch.LongTensor(response_mask_list).to(device)
            if self.trace_aggregator.get("level", "transition") == "trajectory"
            else None
        )

        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)

        # Position IDs
        if self._use_mrope:
            position_ids_list_t: list[torch.Tensor] = []
            for i in range(n_transition):
                pos_ids = self._compute_mrope_position_ids(
                    input_ids=batch_seq[i],
                    attention_mask=attention_mask[i],
                    image_grid_thw=image_grid_thw_list[i] if image_grid_thw_list else None,
                )
                position_ids_list_t.append(pos_ids)
            position_ids = torch.stack(position_ids_list_t, dim=0)
        else:
            position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        is_drop_mask = torch.BoolTensor(is_drop_list).to(device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)

        # Token-level scores: place final reward at last token position
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        if self._use_mrope:
            text_position_ids = position_ids[:, 0, :]
            eos_mask_idx = torch.argmax(text_position_ids * attention_mask, dim=-1)
        else:
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -max_response_length:]

        # 4. Build DataProto
        batch = TensorDict(
            {
                "prompts": batch_input_ids,
                "responses": batch_response_ids,
                "input_ids": batch_seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "is_drop_mask": is_drop_mask,
                "token_level_scores": token_level_scores.contiguous(),
                **(
                    {"response_mask": response_mask}
                    if self.trace_aggregator.get("level", "transition") == "trajectory"
                    else {}
                ),
            },
            batch_size=n_transition,
        )
        data_proto = DataProto(batch=batch)

        # 5. Metrics
        data_metrics: Dict[str, Any] = {
            "training/reward": np.mean(list(finished_id_to_final_reward.values())),
            "training/n_rollouts": len(finished_id_to_final_reward),
            "training/n_rollouts_w_trace": len(finished_id_to_sample_info),
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_triplets": n_trunc_sample_because_of_response,
            "training/n_triplets": n_transition,
        }
        if self.trace_aggregator.get("level", "transition") == "trajectory":
            data_metrics.update({
                "training/n_unmerged_rollouts": unmerged_count,  # type: ignore[possibly-undefined]
                "training/n_triplets_by_turn": len(response_per_turn_list),  # type: ignore[possibly-undefined]
                "training/avg_response_length_by_turn": np.mean(response_per_turn_list),  # type: ignore[possibly-undefined]
                "training/max_response_length_by_turn": np.max(response_per_turn_list),  # type: ignore[possibly-undefined]
                "training/min_response_length_by_turn": np.min(response_per_turn_list),  # type: ignore[possibly-undefined]
            })
            if self.trace_aggregator.get("debug", False):
                data_metrics.update({
                    "training/template_mismatch_triplets": template_mismatch_count,  # type: ignore[possibly-undefined]
                    "training/retoken_mismatch_triplets": retoken_mismatch_count,  # type: ignore[possibly-undefined]
                    "training/others_mismatch_triplets": others_mismatch_count,  # type: ignore[possibly-undefined]
                    "training/template_mismatch_ratio": template_mismatch_count / len(response_per_turn_list),  # type: ignore[possibly-undefined]
                    "training/retoken_mismatch_ratio": retoken_mismatch_count / len(response_per_turn_list),  # type: ignore[possibly-undefined]
                    "training/others_mismatch_ratio": others_mismatch_count / len(response_per_turn_list),  # type: ignore[possibly-undefined]
                })

        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        if self.trace_aggregator.get("level", "transition") == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)

        return data_proto, data_metrics

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import socket
import threading
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from agentlightning import NamedResources
from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store.base import LightningStore
from agentlightning.types import EnqueueRolloutRequest, Rollout, RolloutConfig, Triplet

__all__ = [
    "AgentModeDaemon",
    "get_left_padded_ids_and_attention_mask",
    "get_right_padded_ids_and_attention_mask",
]


@dataclass
class _CompletedRollout:
    rollout_id: str
    final_reward: Optional[float]
    triplets: List[Triplet]
    metadata: Dict[str, Any]


def ids_startswith(
    full_ids: List[int], prefix_ids: List[int], tokenizer: Any, debug: bool = False
) -> Tuple[bool, Tuple[bool, bool, bool]]:
    is_prefix: bool
    template_mismatch, retoken_mismatch, others_mismatch = False, False, False
    if full_ids[: len(prefix_ids)] == prefix_ids:
        is_prefix = True
        return True, (template_mismatch, retoken_mismatch, others_mismatch)
    else:
        is_prefix = False

    if not debug:
        return is_prefix, (template_mismatch, retoken_mismatch, others_mismatch)

    def _special_token_sequence(ids: List[int]) -> List[int]:
        return [id for id in ids if id in tokenizer.all_special_ids]

    def _none_special_token_sequence(ids: List[int]) -> List[int]:
        return [id for id in ids if id not in tokenizer.all_special_ids]

    # First, handle special tokens
    full_special_ids = _special_token_sequence(full_ids)
    prefix_special_ids = _special_token_sequence(prefix_ids)
    if sum(1 for a, b in zip(full_special_ids, prefix_special_ids) if a != b) > 0:
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
    diagnostic: Tuple[bool, bool, bool],
    full_ids: List[int],
    prefix_ids: List[int],
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
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
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
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
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


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _to_native(obj: Any) -> Any:
    """Convert data retrieved from Parquet to data usable in AGL server."""
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

    # 5) Anything else: leave as-is
    return obj


class AgentModeDaemon:
    """
    AgentModeDaemon using the AgentLightningServer SDK.

    This class manages the server lifecycle, task queuing, and results
    retrieval, while also running a proxy server for LLM requests. It maintains
    the original interface for compatibility with the RayPPOTrainer.
    """

    def __init__(
        self,
        port: Optional[int],
        train_rollout_n: int,
        train_information: Dict[str, Any],
        tokenizer: Any,
        mini_batch_size: int,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        llm_timeout_seconds: float = 1200.0,
        llm_proxy: LLMProxy | None = None,
        store: LightningStore | None = None,
        adapter: TraceToTripletBase | None = None,
        processor: Any = None,
        image_base_dir: Optional[str] = None,
        trace_aggregator: Dict[str, Any] = {"level": "transition"},
    ):
        if store is None:
            raise ValueError("VERL v1 execution requires a store instance.")

        self.store = store
        self.llm_timeout_seconds = llm_timeout_seconds

        if llm_proxy is None:
            self.llm_proxy = LLMProxy(
                port=_find_available_port(),
                model_list=[],
                store=store,
            )
        else:
            # Reuse the existing LLM proxy (probably configured by user)
            self.llm_proxy = llm_proxy
        if adapter is None:
            self.adapter = TracerTraceToTriplet()
        else:
            # Reuse the one from trainer
            self.adapter = adapter
        self._internal_loop: Optional[asyncio.AbstractEventLoop] = None
        self._internal_loop_thread = threading.Thread(target=self._internal_loop_runner, daemon=True)
        self._internal_loop_thread.start()

        # Training and Data Configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fillna_value = reward_fillna_value
        self.image_base_dir = image_base_dir
        self.trace_aggregator = trace_aggregator

        # Check if model requires multimodal position_ids (e.g., Qwen2-VL)
        self._use_mrope = self._is_mrope_model()

        # Internal State
        self.backend_llm_server_addresses: List[str] = []
        self._total_tasks_queued = 0
        self._completed_rollouts: Dict[str, _CompletedRollout] = {}
        self._task_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self.is_train = True

    def _internal_loop_runner(self):
        """Run the internal loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._internal_loop = loop
        loop.run_forever()
        loop.close()

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

    def _get_image_grid_thw(self, image_urls: List[str]) -> Optional[torch.Tensor]:
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

        images: List[Image.Image] = [process_image({"image": to_image_uri(url)}) for url in image_urls]
        model_inputs = self.processor(text=["dummy"], images=images, return_tensors="pt")
        return model_inputs.get("image_grid_thw")

    def _compute_mrope_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 4D position_ids for M-RoPE models."""
        from typing import Callable

        get_rope_index: Callable[..., torch.Tensor]
        if "Qwen3VL" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index  # pyright: ignore[reportUnknownVariableType]
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index  # pyright: ignore[reportUnknownVariableType]

        vision_pos = get_rope_index(
            self.processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

        valid_mask = attention_mask.bool()
        text_pos = torch.zeros((1, len(input_ids)), dtype=torch.long, device=input_ids.device)
        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)

        return torch.cat([text_pos, vision_pos], dim=0)

    async def _update_proxy_server_v1(self):
        model_name = self.train_information.get("model")
        if not model_name:
            raise ValueError("Model name is not set.")
        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "hosted_vllm/" + model_name,
                            "api_base": f"http://{address}/v1/",
                        },
                    }
                )
                for address in self.backend_llm_server_addresses
            ],
        )

        await self.llm_proxy.restart()

    def start(self):
        """No-op: v1 execution uses an external async LLM proxy only."""
        return None

    async def _async_set_up(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()
        if server_addresses != self.backend_llm_server_addresses:
            self.backend_llm_server_addresses = server_addresses
            if not self.llm_proxy.is_running():
                await self._update_proxy_server_v1()
        self.is_train = is_train

        # 1. Update resources on the server for clients to use
        llm_resource = self.llm_proxy.as_resource(
            sampling_parameters={
                "temperature": self.train_information.get("temperature", 0.7 if is_train else 0.0)
            },
        )

        resources: NamedResources = {"main_llm": llm_resource}

        resources_update = await self.store.add_resources(resources)
        resources_id = resources_update.resources_id

        # 2. Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        enqueue_rollout_requests: List[EnqueueRolloutRequest] = []
        data_id_to_original_sample: Dict[str, Dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            data_id_to_original_sample[data_id] = original_sample

            # For training, each sample is rolled out multiple times
            # Data ID is different from Rollout ID, as one data can have multiple rollouts.
            for _ in range(rollouts_per_sample):
                task_metadata = {"data_id": data_id, "is_train": is_train}
                # Collect tasks to enqueue in batch and queue them later
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=_to_native(original_sample),
                        mode="train" if is_train else "val",
                        resources_id=resources_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                        metadata=task_metadata,
                    )
                )

        # Enqueue all the tasks in a single batch
        rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
        self._task_id_to_original_sample.update(
            {
                # Recover the original data and store it for later use.
                rollout.rollout_id: data_id_to_original_sample[rollout.metadata["data_id"]]  # type: ignore[index]
                for rollout in rollouts
            }
        )
        self._total_tasks_queued += len(rollouts)

    def set_up_data_and_server(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        """Synchronous wrapper for setting up data and server resources."""
        coro = self._async_set_up(data, server_addresses, is_train)

        if self._internal_loop is None:
            raise RuntimeError("Internal loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._internal_loop)
        try:
            future.result(timeout=300)  # Wait for completion with a timeout
        except Exception as e:
            print(f"Failed to set up data on server: {e}")
            raise

    def _validate_data(self, rollout: _CompletedRollout):
        if rollout.final_reward is None:
            print(
                f"Warning: Reward is None for rollout {rollout.rollout_id}, will be auto-set to {self.reward_fillna_value}."
            )
        if len(rollout.triplets) == 0:
            print(f"Warning: Length of triplets is 0 for rollout {rollout.rollout_id}.")
        elif any(not r.response.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty response: {rollout.triplets}")
        elif any(not r.prompt.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty prompt: {rollout.triplets}")

    async def _build_completed_rollout(self, rollout: Rollout) -> _CompletedRollout:
        """Build completion payload and validate rollout/spans from v1 store representation."""
        # Query spans for this rollout (latest attempt)
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")

        # Convert spans to triplets using the adapter
        if not spans:
            # No triplets found, will emit a warning later.
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)

        # Extract final reward from triplets
        final_reward: Optional[float] = None
        if triplets:
            # Search backwards through triplets for the first non-None reward
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        result_rollout = _CompletedRollout(
            rollout_id=rollout.rollout_id,
            final_reward=final_reward,
            triplets=triplets,
            metadata=rollout.metadata or {},
        )

        # Validate converted rollout before returning it to the training pipeline.
        self._validate_data(result_rollout)

        return result_rollout

    async def _async_run_until_finished(self, verbose: bool = True):
        """Async helper to wait for all tasks to complete."""
        while len(self._completed_rollouts) < self._total_tasks_queued:
            completed_batch = await self.store.wait_for_rollouts(
                rollout_ids=list(self._task_id_to_original_sample.keys()), timeout=0
            )
            for rollout in completed_batch:
                if rollout.rollout_id in self._completed_rollouts:
                    # Already processed, skip
                    continue
                rollout = await self._build_completed_rollout(rollout)
                if rollout.rollout_id not in self._task_id_to_original_sample:
                    print(f"Warning: Received unknown rollout ID {rollout.rollout_id}, skipping.")
                else:
                    self._completed_rollouts[rollout.rollout_id] = rollout
            if verbose:
                print(f"Completed {len(self._completed_rollouts)}/{self._total_tasks_queued} tasks...")
            await asyncio.sleep(5)

        print("All tasks finished.")

    def run_until_all_finished(self, verbose: bool = True):
        """Synchronously waits for all queued tasks to be completed and reported."""
        if self._total_tasks_queued == 0:
            print("Warning: No tasks were queued.")
            return

        loop = self._internal_loop
        assert loop is not None

        coro = self._async_run_until_finished(verbose)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            future.result()  # Wait indefinitely for all tasks to complete
        except Exception as e:
            print(f"Error while waiting for tasks to finish: {e}")
            raise

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        assert not self.is_train, "This method should only be called during validation."
        assert len(self._completed_rollouts) == self._total_tasks_queued

        sample_stat_list: List[Dict[str, Any]] = []
        sample_stat_list_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # FIXME: Evaluate whether grouping stats by source is actually needed.

        for rollout_id, rollout in self._completed_rollouts.items():
            final_reward_raw: Optional[float] = rollout.final_reward
            final_reward = self._fillna_reward(rollout)
            if not rollout.triplets:
                print(f"Warning: No triplets found for test rollout {rollout.rollout_id}.")
                sample_stat_list.append({"reward": final_reward, "has_reward": final_reward_raw is not None})
                continue
            response_length_list = [len(triplet.response.get("token_ids", [])) for triplet in rollout.triplets]

            if "data_source" in self._task_id_to_original_sample[rollout_id]:
                # When a test sample includes a 'data_source' field, record per-source statistics for test results.
                # TODO: This is a flawed design. We should have a better way to handle this.
                data_source = self._task_id_to_original_sample[rollout_id]["data_source"]
                sample_stat_list_by_source[data_source].append(
                    {
                        "sum_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                        "reward": final_reward,
                        "has_reward": final_reward_raw is not None,
                    }
                )
            sample_stat_list.append(
                {
                    "sum_response_length": np.sum(response_length_list),
                    "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                    "turn_count": len(rollout.triplets),
                    "reward": final_reward,
                    "has_reward": final_reward_raw is not None,
                }
            )
        metric_dict: Dict[str, Any] = {}

        stats_w_trace = [stat for stat in sample_stat_list if "sum_response_length" in stat]
        stats_w_trace_by_source = {
            data_source: [stat for stat in sample_stats if "sum_response_length" in stat]
            for data_source, sample_stats in sample_stat_list_by_source.items()
        }
        for data_source, sample_stats in sample_stat_list_by_source.items():
            metric_dict.update(
                {
                    f"val/{data_source}/n_rollouts": len(sample_stats),
                    f"val/{data_source}/n_rollouts_w_trace": len(stats_w_trace_by_source[data_source]),
                    f"val/{data_source}/n_rollouts_w_reward": len(
                        [stat for stat in sample_stats if stat["has_reward"]]
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
                "val/reward": np.mean(
                    [stat["reward"] for stat in sample_stat_list]
                ),  # each rollout must have a reward (fillna if missing)
                "val/mean_response_length": np.mean([stat["mean_response_length"] for stat in stats_w_trace]),
                "val/sum_response_length": np.mean([stat["sum_response_length"] for stat in stats_w_trace]),
                "val/turn_count": np.mean([stat["turn_count"] for stat in stats_w_trace]),
            }
        )
        return metric_dict

    def get_train_data_batch(
        self, max_prompt_length: int, max_response_length: int, device: torch.device, global_steps: int
    ):
        """
        Processes completed rollouts to generate a training data batch.

        This function reconstructs the logic from the original AgentModeDaemon,
        using data retrieved from the new server architecture. It handles padding,
        truncation, and tensor creation for the PPO training loop.
        """
        assert self.is_train, "This method should only be called during training."
        assert len(self._completed_rollouts) == self._total_tasks_queued

        # 1. Reconstruct the `finished_id_to_sample_info` structure from completed rollouts
        finished_id_to_sample_info: Dict[str, Dict[str, Any]] = {}
        finished_id_to_final_reward: Dict[str, float] = {}
        sample_with_reward_count = 0
        for rollout_id, rollout in self._completed_rollouts.items():
            original_sample = self._task_id_to_original_sample[rollout_id]
            sample_with_reward_count += int(rollout.final_reward is not None)
            final_reward = self._fillna_reward(rollout)

            if not rollout.triplets:
                finished_id_to_final_reward[rollout_id] = final_reward
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue

            # The client should report triplets that contain prompt_ids and response_ids.
            # Example triplet.prompt: {"token_ids": [...], "image_urls": [...]}
            # Example triplet.response: {"token_ids": [...]}
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
        #
        # --- Data processing and tensor creation logic ---
        # Get all the reported data.
        # prompt_ids are left-padded.
        # response_ids are right-padded.
        # They are concatenated in the middle.
        # Discard handling:
        #   - Those exceeding max_prompt_length will be marked for discard, but not
        #     discarded here. They are only truncated and marked, to be discarded later.
        #     This is for the correctness of the advantage calculation.
        #   - The discard for the PPO mini-batch should also be handled this way.
        input_ids_list: List[List[int]] = []
        input_attention_mask_list: List[List[int]] = []
        response_ids_list: List[List[int]] = []
        response_attention_mask_list: List[List[int]] = []
        reward_list: List[float] = []
        data_id_list: List[str] = []
        rollout_id_list: List[str] = []
        turn_index_list: List[int] = []
        is_drop_list: List[bool] = []
        image_grid_thw_list: List[Optional[torch.Tensor]] = []  # For Qwen2-VL mrope
        n_trunc_sample_because_of_response = 0

        if self.trace_aggregator.get("level", "transition") == "transition":
            for rollout_id, sample_info in finished_id_to_sample_info.items():
                for turn_index, trace in enumerate(sample_info["trace_list"]):

                    reward_list.append(sample_info["reward"])
                    prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]

                    # Mark samples with prompts exceeding max_prompt_length to be dropped later
                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    # Truncate responses that exceed max_response_length
                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # Pad prompts to the left and responses to the right
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

                    # Compute image_grid_thw for this triplet using image_urls from prompt
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

                # Identify which turns can be merged based on token ids prefix matching
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
                    if not is_prefix and self.trace_aggregator.get("debug", False) == True:
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

                # Merge all trace segments in merged_trace_idx into training samples
                for current_merged_trace_idx in merged_trace_idx:
                    prompt_ids = sample_info["trace_list"][current_merged_trace_idx[0]]["prompt_ids"]

                    # if the merged_trace_idx doesn't start with the beginning of the prompt_ids, we need to adjust it
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

                    # Mark samples with prompts exceeding max_prompt_length to be dropped later
                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    # Truncate responses that exceed max_response_length
                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        response_mask = response_mask[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # Pad prompts to the left and responses to the right
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
                    # turn_index_list.append(current_merged_trace_idx)
        else:
            raise ValueError(f"Unknown trace_aggregator level: {self.trace_aggregator.get('level')}")

        n_transition = len(input_ids_list)
        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)
        response_mask = (
            torch.LongTensor(response_mask_list).to(device) if self.trace_aggregator.get("level", "transition") == "trajectory" else None  # type: ignore
        )

        # Concatenate prompts and responses to form the full sequence
        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)

        # Compute position_ids - use mrope for Qwen2-VL, standard 2D otherwise
        if self._use_mrope:
            # For Qwen2-VL: compute 4D position_ids (batch_size, 4, seq_length)
            position_ids_list: list[torch.Tensor] = []
            for i in range(n_transition):
                pos_ids = self._compute_mrope_position_ids(
                    input_ids=batch_seq[i],
                    attention_mask=attention_mask[i],
                    image_grid_thw=image_grid_thw_list[i] if image_grid_thw_list else None,
                )  # (4, seq_length)
                position_ids_list.append(pos_ids)
            # Stack to (batch_size, 4, seq_length)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            # Standard 2D position_ids (batch_size, seq_length)
            position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        is_drop_mask = torch.BoolTensor(is_drop_list).to(device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)

        # Create token-level scores by placing the final reward at the last token position
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        # For mrope (3D position_ids), use the first dimension (text position_ids) for eos calculation
        if self._use_mrope:
            # position_ids is (batch_size, 4, seq_length), use first dim for text positions
            text_position_ids = position_ids[:, 0, :]  # (batch_size, seq_length)
            eos_mask_idx = torch.argmax(text_position_ids * attention_mask, dim=-1)  # (bsz,)
        else:
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        # At the eos_mask_idx position of each sample, fill in the corresponding scores.
        # torch.arange(n_transition) generates [0,1,2,...,bsz-1] as indices for the batch dimension.
        token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
        # Only take the last response_length part of the sequence to get the token-level scores for the model's response part.
        token_level_scores = token_level_scores[:, -max_response_length:]

        # Form the final batch using TensorDict
        batch = TensorDict(
            {
                "prompts": batch_input_ids,
                "responses": batch_response_ids,
                "input_ids": batch_seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "is_drop_mask": is_drop_mask,
                "token_level_scores": token_level_scores.contiguous(),
                **(
                    {"response_mask": response_mask}
                    if self.trace_aggregator.get("level", "transition") == "trajectory"
                    else {}
                ),
            },  # type: ignore
            batch_size=n_transition,
        )
        data_proto = DataProto(batch=batch)

        data_metrics = {
            "training/reward": np.mean(list(finished_id_to_final_reward.values())),
            "training/n_rollouts": len(finished_id_to_final_reward),
            "training/n_rollouts_w_trace": len(finished_id_to_sample_info),
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_triplets": n_trunc_sample_because_of_response,
            "training/n_triplets": n_transition,
            # log data, only for debug testing
            **(
                {
                    "training/n_unmerged_rollouts": unmerged_count,  # type: ignore
                    "training/n_triplets_by_turn": len(response_per_turn_list),  # type: ignore
                    "training/avg_response_length_by_turn": np.mean(response_per_turn_list),  # type: ignore
                    "training/max_response_length_by_turn": np.max(response_per_turn_list),  # type: ignore
                    "training/min_response_length_by_turn": np.min(response_per_turn_list),  # type: ignore
                }
                if self.trace_aggregator.get("level", "transition") == "trajectory"
                else {}
            ),
            **(
                {
                    "training/template_mismatch_triplets": template_mismatch_count,  # type: ignore
                    "training/retoken_mismatch_triplets": retoken_mismatch_count,  # type: ignore
                    "training/others_mismatch_triplets": others_mismatch_count,  # type: ignore
                    "training/template_mismatch_ratio": template_mismatch_count / len(response_per_turn_list),  # type: ignore
                    "training/retoken_mismatch_ratio": retoken_mismatch_count / len(response_per_turn_list),  # type: ignore
                    "training/others_mismatch_ratio": others_mismatch_count / len(response_per_turn_list),  # type: ignore
                }
                if self.trace_aggregator.get("level", "transition") == "trajectory"
                and self.trace_aggregator.get("debug", False)
                else {}
            ),
        }

        # Add non-tensor data for advantage calculation and logging
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)  # type: ignore
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)  # type: ignore
        if self.trace_aggregator.get("level", "transition") == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)  # type: ignore

        return data_proto, data_metrics

    def clear_data_and_server(self):
        """Resets the internal state of the daemon for the next run."""
        self.backend_llm_server_addresses = []
        self._completed_rollouts.clear()
        self._task_id_to_original_sample.clear()
        self._total_tasks_queued = 0
        # For a true reset, the server's internal queues would also need clearing.
        # This implementation assumes that `set_up_data_and_server` is called
        # for each new run, effectively starting a fresh batch.

    def _fillna_reward(self, rollout: _CompletedRollout):
        if rollout.final_reward is None:
            if self.reward_fillna_value is not None:  # type: ignore
                final_reward = self.reward_fillna_value
            else:
                raise ValueError(f"Reward is None for rollout {rollout.rollout_id}, please check the reward function.")
        else:
            final_reward = rollout.final_reward
        return final_reward

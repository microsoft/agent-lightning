# Copyright (c) Microsoft. All rights reserved.

"""Adapters from completed Agent Lightning rollouts to VERL training data."""

from __future__ import annotations

import base64  # [multimodal-patch]
import io
import json
import zipfile
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from agentlightning.verl.agl_rollout_manager import CompletedRollout

_TRACE_MERGE_MISMATCH_WANDB_LIMIT = 100
_TRACE_MERGE_MISMATCH_TEXT_LIMIT = 4000
_ROLLOUT_TRAJECTORY_WANDB_LIMIT = 24
_TRACE_MERGE_MISMATCH_COLUMNS = [
    "global_steps",
    "rollout_id",
    "data_id",
    "turn_index",
    "template_mismatch",
    "retoken_mismatch",
    "others_mismatch",
    "prompt_length",
    "response_length",
    "previous_trace_length",
    "current_trace_length",
    "previous_trace",
    "current_trace",
]
_ROLLOUT_TRAJECTORY_COLUMNS = [
    "global_steps",
    "trajectory_artifact",
    "trajectory_artifact_path",
    "row_count",
]


def ids_startswith(full_ids: list[int], prefix_ids: list[int]) -> bool:
    return full_ids[: len(prefix_ids)] == prefix_ids


def _decode_token_ids(tokenizer: Any | None, ids: list[int]) -> str:
    if tokenizer is not None:
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except TypeError:
            text = tokenizer.decode(ids)
        except Exception:
            text = " ".join(str(i) for i in ids)
    else:
        text = " ".join(str(i) for i in ids)
    return text


def _decode_trace_text(tokenizer: Any | None, ids: list[int]) -> str:
    text = _decode_token_ids(tokenizer, ids)
    if len(text) > _TRACE_MERGE_MISMATCH_TEXT_LIMIT:
        truncated = len(text) - _TRACE_MERGE_MISMATCH_TEXT_LIMIT
        return text[:_TRACE_MERGE_MISMATCH_TEXT_LIMIT] + f"\n...[truncated {truncated} chars]"
    return text


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) and isinstance(value.get("token_ids"), list):
        return value["token_ids"]
    return []


def _artifact_safe_name(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isascii() and (char.isalnum() or char in {"-", "_", "."}) else "_" for char in text)
    return safe or "unknown"


def _build_compact_rollout_trajectory_records(
    rollouts: list[CompletedRollout],
    *,
    tokenizer: Any | None,
    reward_fillna_value: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sorted_rollouts = sorted(rollouts, key=lambda rollout: (rollout.step, rollout.sample_idx_in_step))
    for rollout in sorted_rollouts:
        if limit is not None and len(records) >= limit:
            break
        if not rollout.triplets:
            continue
        last_triplet = rollout.triplets[-1]
        records.append(
            {
                "rollout_id": rollout.rollout_id,
                "reward": rollout.final_reward if rollout.final_reward is not None else reward_fillna_value,
                "prompt": _decode_token_ids(tokenizer, _token_ids(last_triplet.prompt)),
                "response": _decode_token_ids(tokenizer, _token_ids(last_triplet.response)),
            }
        )
    return records


def _build_zipped_jsonl(records: list[dict[str, Any]], jsonl_name: str) -> bytes:
    jsonl_text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(jsonl_name, jsonl_text.encode("utf-8"))
    return buffer.getvalue()


def _upload_trace_merge_mismatches_to_wandb(rows: list[dict[str, Any]], global_steps: int) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        table = wandb.Table(columns=cast(list[str | int], _TRACE_MERGE_MISMATCH_COLUMNS))
        for row in rows:
            table.add_data(*(row.get(column) for column in _TRACE_MERGE_MISMATCH_COLUMNS))
        wandb.log({"training/trace_merge_mismatches": table}, step=global_steps)
    except Exception as exc:
        print(f"Warning: failed to upload trace merge mismatches to wandb: {exc}")


def _upload_compact_rollout_trajectories_to_wandb(
    records: list[dict[str, Any]],
    global_steps: int,
    *,
    is_validation: bool = False,
) -> None:
    split = "validation" if is_validation else "train"
    try:
        import wandb

        if wandb.run is None:
            return
        run = wandb.run
        artifact_type = f"{split}_trajectories"
        artifact_path = f"step_{global_steps}/{artifact_type}.jsonl.zip"
        artifact_name = (
            f"{split}-trajectories-{_artifact_safe_name(getattr(run, 'id', None) or 'run')}-step-{global_steps}"
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata={"global_steps": global_steps, "row_count": len(records), "format": "jsonl.zip"},
        )
        with artifact.new_file(artifact_path, mode="wb") as trajectory_file:
            trajectory_file.write(_build_zipped_jsonl(records, f"{artifact_type}.jsonl"))
        run.log_artifact(artifact)

        table = wandb.Table(columns=cast(list[str | int], _ROLLOUT_TRAJECTORY_COLUMNS))
        table.add_data(global_steps, artifact_name, artifact_path, len(records))
        table_key = "val/rollout_trajectories" if is_validation else "training/rollout_trajectories"
        wandb.log({table_key: table}, step=global_steps)
    except Exception as exc:
        print(f"Warning: failed to upload {split} trajectories to wandb: {exc}")


def get_left_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[-max_length:], [1] * max_length

    pad_len = max_length - seq_len
    return [pad_token_id] * pad_len + ids, [0] * pad_len + [1] * seq_len


def get_right_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[:max_length], [1] * max_length

    pad_len = max_length - seq_len
    return ids + [pad_token_id] * pad_len, [1] * seq_len + [0] * pad_len


# ---------------------------------------------------------------------------
# [multimodal-patch] Multimodal (image) support for mrope VLM training.
# Mirrors verl 0.8.0's AgentLoopWorker._compute_multi_modal_inputs /
# _compute_position_ids (verl/experimental/agent_loop/agent_loop.py) so the
# FSDP engine receives per-row `multi_modal_inputs` (pixel_values +
# image_grid_thw) and (batch, 4, seq_len) mrope position ids for rows whose
# prompt contains images. Rows without images keep the plain behavior.
# ---------------------------------------------------------------------------

_MROPE_PROCESSOR_TAGS = ("Qwen2VL", "Qwen2_5_VL", "Qwen3VL", "Qwen3_5")


def _is_mrope_processor(processor: Any) -> bool:
    """Check whether the processor belongs to an mrope (Qwen-VL style) model."""
    if processor is None:
        return False
    # verl.utils.tokenizer.hf_processor binds the HF model class' get_rope_index.
    if getattr(processor, "get_rope_index", None) is not None:
        return True
    class_names = [processor.__class__.__name__]
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        class_names.append(image_processor.__class__.__name__)
    return any(tag in name for name in class_names for tag in _MROPE_PROCESSOR_TAGS)


def _load_pil_image(url: str) -> Any:
    """Decode one image URL (data: base64 / file:// / http(s)://) into a PIL image."""
    from PIL import Image

    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://") :]).convert("RGB")
    if url.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    raise ValueError(f"[multimodal-patch] unsupported image url scheme: {url[:64]}")


def _build_multi_modal_inputs(processor: Any, image_urls: list[str]) -> tuple[dict[str, Any], bool]:
    """Run the HF processor on the row's images.

    Returns (multi_modal_inputs, has_mm_token_type_ids): the dict holds the vision
    tensors for verl's extract_multi_modal_inputs (e.g. pixel_values, and
    image_grid_thw for Qwen-VL style models); text-side keys are dropped.
    image_grid_thw is optional so non-mrope VLMs whose processor only returns
    pixel_values still get their vision features attached (their position_ids
    stay on the plain 2D cumsum path). has_mm_token_type_ids flags
    transformers>=5.3 processors, where mm_token_type_ids is needed (rebuilt
    per row) for position ids.
    """
    images = [_load_pil_image(url) for url in image_urls]
    model_inputs = processor(text=["dummy"], images=images, return_tensors="pt")
    has_mm_token_type_ids = "mm_token_type_ids" in model_inputs
    multi_modal_inputs = {
        key: value
        for key, value in dict(model_inputs).items()
        if key not in ("input_ids", "attention_mask", "mm_token_type_ids")
    }
    if not multi_modal_inputs:
        raise RuntimeError(
            f"[multimodal-patch] processor returned no vision tensors for {len(images)} image(s), "
            f"got keys: {sorted(dict(model_inputs))}"
        )
    return multi_modal_inputs, has_mm_token_type_ids


def _compute_mrope_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Any | None,
    has_mm_token_type_ids: bool = False,
) -> torch.Tensor:
    """Compute (4, seq_len) mrope position ids for one padded row.

    Row layout follows verl 0.8.0's agent loop: one text row + three vision rows
    (get_rope_index output). image_grid_thw=None yields the pure-text variant.
    """
    bound_get_rope_index = getattr(processor, "get_rope_index", None)
    if bound_get_rope_index is not None:
        # HF model-class get_rope_index bound by hf_processor; takes batched input.
        rope_kwargs: dict[str, Any] = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": None,
        }
        if has_mm_token_type_ids:
            rope_kwargs["mm_token_type_ids"] = _build_mm_token_type_ids(processor, input_ids).unsqueeze(0)
        result = bound_get_rope_index(**rope_kwargs)
        vision_position_ids = result[0] if isinstance(result, tuple) else result  # (3, 1, seq_len)
        vision_position_ids = vision_position_ids[:, 0, :]
    else:
        # Fallback: verl's per-family helpers take a 1D single-example input.
        class_names = processor.__class__.__name__
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            class_names += image_processor.__class__.__name__
        if "Qwen3VL" in class_names or "Qwen3_5" in class_names:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index
        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )  # (3, seq_len)

    valid_mask = attention_mask.bool()
    text_position_ids = torch.ones((1, input_ids.shape[0]), dtype=torch.long, device=input_ids.device)
    text_position_ids[0, valid_mask] = torch.arange(int(valid_mask.sum().item()), device=input_ids.device)
    return torch.cat([text_position_ids, vision_position_ids.to(device=input_ids.device, dtype=torch.long)], dim=0)


def _text_only_mrope_position_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(4, seq_len) mrope position ids for a row treated as pure text.

    All four rows hold the plain cumsum positions (1 on padding, matching the
    text-row convention of _compute_mrope_position_ids). Last-resort fallback
    when even get_rope_index(image_grid_thw=None) fails for a row.
    """
    valid_mask = attention_mask.bool()
    positions = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    positions[valid_mask] = torch.arange(int(valid_mask.sum().item()), device=input_ids.device)
    return positions.unsqueeze(0).repeat(4, 1)


def _build_mm_token_type_ids(processor: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Build per-token modality ids (1=image, 2=video); only used for position ids."""
    from verl.utils.tokenizer import get_processor_token_id

    mm_token_type_ids = torch.zeros_like(input_ids)
    image_token_id = get_processor_token_id(processor, "image")
    video_token_id = get_processor_token_id(processor, "video")
    if image_token_id is not None:
        mm_token_type_ids[input_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[input_ids == video_token_id] = 2
    return mm_token_type_ids


class RolloutAdapter:
    """Convert completed rollout results into VERL data structures."""

    def __init__(
        self,
        *,
        max_prompt_length: int,
        max_response_length: int,
        device: torch.device,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        trace_aggregator_level: str = "transition",
        tokenizer: Any | None = None,
        processor: Any | None = None,  # [multimodal-patch] HF processor; None keeps text-only behavior
    ) -> None:
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.device = device
        self.pad_token_id = pad_token_id
        self.reward_fillna_value = reward_fillna_value
        self.trace_aggregator_level = trace_aggregator_level
        self.tokenizer = tokenizer
        self.processor = processor  # [multimodal-patch]

    def get_train_data_batch(
        self,
        completed_rollouts: list[CompletedRollout],
        *,
        global_steps: int = 0,
    ) -> tuple[DataProto, dict[str, Any]]:
        """Build a VERL training batch from completed rollouts."""
        level = self.trace_aggregator_level
        if level not in {"transition", "trajectory"}:
            raise ValueError(f"Unknown trace_aggregator_level: {level}")

        # Keep rollout randomness within each sample instead of ordering samples by completion time.
        sorted_rollouts = sorted(completed_rollouts, key=lambda rollout: (rollout.step, rollout.sample_idx_in_step))

        # [multimodal-patch] Image-to-row alignment is only implemented for the transition
        # level: trajectory-level aggregation merges multi-turn prompts into one row, which
        # breaks the correspondence between image placeholder tokens and image_urls, and
        # append_training_row is never given image_urls on that path. Fail loudly instead of
        # silently training without the vision signal.
        if level == "trajectory" and any(
            triplet.image_urls for rollout in sorted_rollouts for triplet in (rollout.triplets or [])
        ):
            raise ValueError(
                "[multimodal-patch] Rollout traces contain images (triplets with image_urls), but "
                "trace_aggregator level 'trajectory' merges multi-turn prompts and cannot keep the "
                "image-to-token alignment. Set agentlightning.trace_aggregator.level: transition "
                "in the config for multimodal training."
            )

        final_rewards: list[float] = []
        sample_with_reward_count = 0
        sample_with_trace_count = 0

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
        response_log_probs_list: list[list[float] | None] = []
        image_urls_list: list[list[str] | None] = []  # [multimodal-patch] per kept training row
        n_trunc_sample_because_of_response = 0
        n_skipped_empty_training_rows = 0
        unmerged_count = 0
        response_len_per_turn_list: list[int] = []
        merge_mismatch_rows: list[dict[str, Any]] = []

        def append_training_row(
            *,
            rollout_id: str,
            data_id: str,
            turn_index: int,
            prompt_ids: list[int],
            response_ids: list[int],
            reward: float,
            response_mask: list[int] | None = None,
            response_log_probs: list[float] | None = None,
            image_urls: list[str] | None = None,  # [multimodal-patch]
        ) -> None:
            nonlocal n_skipped_empty_training_rows, n_trunc_sample_because_of_response
            if len(prompt_ids) > self.max_prompt_length:
                prompt_ids = prompt_ids[: self.max_prompt_length]
                is_drop = True
            else:
                is_drop = False

            if len(response_ids) > self.max_response_length:
                response_ids = response_ids[: self.max_response_length]
                if response_mask is not None:
                    response_mask = response_mask[: self.max_response_length]
                if response_log_probs is not None:
                    response_log_probs = response_log_probs[: self.max_response_length]
                n_trunc_sample_because_of_response += 1

            if response_log_probs is not None and len(response_log_probs) != len(response_ids):
                response_log_probs = None

            train_token_count = sum(response_mask) if response_mask is not None else len(response_ids)
            if train_token_count == 0:
                n_skipped_empty_training_rows += 1
                return

            one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                prompt_ids, self.max_prompt_length, self.pad_token_id
            )
            one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                response_ids, self.max_response_length, self.pad_token_id
            )
            input_ids_list.append(one_input_ids)
            input_attention_mask_list.append(one_input_attention_mask)
            response_ids_list.append(one_response_ids)
            response_attention_mask_list.append(one_response_attention_mask)
            is_drop_list.append(is_drop)
            image_urls_list.append(image_urls)  # [multimodal-patch] stays aligned with kept rows
            if response_mask is not None:
                one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                    response_mask, self.max_response_length, 0
                )
                response_mask_list.append(one_response_mask)

            response_log_probs_list.append(response_log_probs)

            reward_list.append(reward)
            data_id_list.append(data_id)
            rollout_id_list.append(rollout_id)
            if level == "transition":
                turn_index_list.append(turn_index)

        for rollout in sorted_rollouts:
            final_reward = self._fillna_reward(rollout)
            final_rewards.append(final_reward)
            if rollout.final_reward is not None:
                sample_with_reward_count += 1

            if not rollout.triplets:
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue
            sample_with_trace_count += 1

            if level == "transition":
                for turn_index, triplet in enumerate(rollout.triplets):
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=turn_index,
                        prompt_ids=triplet.prompt["token_ids"],
                        response_ids=response_ids,
                        reward=final_reward,
                        response_log_probs=log_probs,
                        image_urls=triplet.image_urls,  # [multimodal-patch]
                    )
                continue
            else:
                first_triplet = rollout.triplets[0]
                group_start_turn_index = 0
                current_prompt_ids = list(first_triplet.prompt["token_ids"])
                current_response_ids = list(first_triplet.response["token_ids"])
                current_context = current_prompt_ids + current_response_ids
                current_response_mask = [1] * len(current_response_ids)
                current_response_log_probs: list[float] | None = first_triplet.response["log_probs"]
                response_len_per_turn_list.append(len(current_response_ids))
                merged_group_count = 0

                for turn_index, triplet in enumerate(rollout.triplets[1:], start=1):
                    prompt_ids = triplet.prompt["token_ids"]
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    next_context = prompt_ids + response_ids

                    if ids_startswith(prompt_ids, current_context):
                        if len(prompt_ids) > len(current_context):
                            observation_ids = prompt_ids[len(current_context) :]
                            current_response_ids += observation_ids
                            current_response_mask += [0] * len(observation_ids)
                            if current_response_log_probs is not None:
                                current_response_log_probs += [0.0] * len(observation_ids)
                        current_response_ids += response_ids
                        current_response_mask += [1] * len(response_ids)
                        if current_response_log_probs is not None:
                            if log_probs is None or len(log_probs) != len(response_ids):
                                current_response_log_probs = None
                            else:
                                current_response_log_probs += list(log_probs)
                        current_context = next_context
                        continue

                    if len(merge_mismatch_rows) < _TRACE_MERGE_MISMATCH_WANDB_LIMIT:
                        merge_mismatch_rows.append(
                            {
                                "global_steps": global_steps,
                                "rollout_id": rollout.rollout_id,
                                "data_id": rollout.data_id,
                                "turn_index": turn_index,
                                # Token-prefix failures are classified as other mismatches.
                                "template_mismatch": False,
                                "retoken_mismatch": False,
                                "others_mismatch": True,
                                "prompt_length": len(prompt_ids),
                                "response_length": len(response_ids),
                                "previous_trace_length": len(current_context),
                                "current_trace_length": len(next_context),
                                "previous_trace": _decode_trace_text(self.tokenizer, current_context),
                                "current_trace": _decode_trace_text(self.tokenizer, next_context),
                            }
                        )

                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=group_start_turn_index,
                        prompt_ids=current_prompt_ids,
                        response_ids=current_response_ids,
                        reward=final_reward,
                        response_mask=current_response_mask,
                        response_log_probs=current_response_log_probs,
                    )
                    merged_group_count += 1

                    group_start_turn_index = turn_index
                    current_context = next_context
                    current_prompt_ids = list(prompt_ids)
                    current_response_ids = list(response_ids)
                    current_response_mask = [1] * len(response_ids)
                    current_response_log_probs = log_probs

                append_training_row(
                    rollout_id=rollout.rollout_id,
                    data_id=rollout.data_id,
                    turn_index=group_start_turn_index,
                    prompt_ids=current_prompt_ids,
                    response_ids=current_response_ids,
                    reward=final_reward,
                    response_mask=current_response_mask,
                    response_log_probs=current_response_log_probs,
                )
                merged_group_count += 1

                if merged_group_count > 1:
                    unmerged_count += 1

        rollout_trajectory_records = _build_compact_rollout_trajectory_records(
            sorted_rollouts,
            tokenizer=self.tokenizer,
            reward_fillna_value=self.reward_fillna_value,
            limit=_ROLLOUT_TRAJECTORY_WANDB_LIMIT,
        )
        _upload_trace_merge_mismatches_to_wandb(merge_mismatch_rows, global_steps)
        _upload_compact_rollout_trajectories_to_wandb(rollout_trajectory_records, global_steps)

        n_sample = len(input_ids_list)
        if n_sample == 0:
            raise RuntimeError("get_train_data_batch emitted zero training rows.")

        batch_input_ids = torch.LongTensor(input_ids_list).to(self.device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(self.device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(self.device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(self.device)
        batch_response_mask = torch.LongTensor(response_mask_list).to(self.device) if level == "trajectory" else None

        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        # [multimodal-patch] Build per-row multi_modal_inputs and mrope position ids.
        has_image_rows = any(image_urls for image_urls in image_urls_list)
        multi_modal_inputs_list: list[dict[str, Any] | None] | None = None
        if has_image_rows and self.processor is None:
            print(
                "Warning: [multimodal-patch] rollout traces contain images but RolloutAdapter has no "
                "processor; training rows will NOT include pixel_values (vision signal is lost)."
            )
        elif has_image_rows:
            use_mrope = _is_mrope_processor(self.processor)
            if not use_mrope:
                print(
                    "Warning: [multimodal-patch] processor is not a recognized mrope (Qwen-VL) "
                    "processor; multi_modal_inputs will be attached but position_ids stay 2D."
                )
            multi_modal_inputs_list = []
            mrope_position_ids_list: list[torch.Tensor] = []
            for row_index in range(n_sample):
                image_urls = image_urls_list[row_index]
                row_multi_modal_inputs: dict[str, Any] | None = None
                row_image_grid_thw = None
                row_has_mm_token_type_ids = False
                if image_urls and is_drop_list[row_index]:
                    # The over-long prompt was truncated and may have cut through the image
                    # placeholder tokens, so the image grid metadata no longer matches the
                    # token sequence. Fall back to a text-only row; this is safe because
                    # is_drop rows are filtered out by is_drop_mask before the training
                    # forward (see trainer.py) and never contribute gradients.
                    print(
                        f"Warning: [multimodal-patch] row {row_index} (rollout "
                        f"{rollout_id_list[row_index]}) has a truncated (is_drop) prompt with "
                        "images; falling back to text-only for this row."
                    )
                    image_urls = None
                if image_urls:
                    try:
                        row_multi_modal_inputs, row_has_mm_token_type_ids = _build_multi_modal_inputs(
                            self.processor, image_urls
                        )
                        row_image_grid_thw = row_multi_modal_inputs.get("image_grid_thw")
                    except Exception as exc:
                        # A broken row must not abort the whole step; it trains as text-only.
                        print(
                            f"Warning: [multimodal-patch] failed to process images for row {row_index} "
                            f"(rollout {rollout_id_list[row_index]}): {exc}"
                        )
                        row_multi_modal_inputs = None
                multi_modal_inputs_list.append(row_multi_modal_inputs)
                if use_mrope:
                    if image_urls and row_image_grid_thw is None and row_multi_modal_inputs is not None:
                        # mrope processor but no image_grid_thw: vision position ids cannot be
                        # computed for this row, only the text variant is possible.
                        print(
                            f"Warning: [multimodal-patch] row {row_index} (rollout "
                            f"{rollout_id_list[row_index]}) has images but no image_grid_thw; "
                            "using text-only mrope position ids for this row."
                        )
                    try:
                        mrope_position_ids_list.append(
                            _compute_mrope_position_ids(
                                self.processor,
                                input_ids=batch_seq[row_index],
                                attention_mask=attention_mask[row_index],
                                image_grid_thw=row_image_grid_thw,
                                has_mm_token_type_ids=row_has_mm_token_type_ids,
                            )
                        )
                    except Exception as exc:
                        print(
                            f"Warning: [multimodal-patch] mrope position ids failed for row {row_index} "
                            f"(rollout {rollout_id_list[row_index]}), using text-only variant: {exc}"
                        )
                        try:
                            mrope_position_ids_list.append(
                                _compute_mrope_position_ids(
                                    self.processor,
                                    input_ids=batch_seq[row_index],
                                    attention_mask=attention_mask[row_index],
                                    image_grid_thw=None,
                                )
                            )
                        except Exception:
                            # Last resort (e.g. leftover image tokens in a truncated prompt make
                            # even the text variant of get_rope_index fail): plain cumsum positions
                            # on all four mrope rows. The row is almost always an is_drop row that
                            # is filtered before the training forward anyway.
                            print(
                                f"Warning: [multimodal-patch] text-only mrope fallback also failed for "
                                f"row {row_index} (rollout {rollout_id_list[row_index]}); "
                                "using plain cumsum position ids."
                            )
                            mrope_position_ids_list.append(
                                _text_only_mrope_position_ids(batch_seq[row_index], attention_mask[row_index])
                            )
            if use_mrope:
                # (n_sample, 4, seq_len): verl's engine detects mrope via position_ids.dim() == 3.
                position_ids = torch.stack(mrope_position_ids_list, dim=0)

        row_has_log_probs_list = [log_probs is not None for log_probs in response_log_probs_list]
        emit_rollout_log_probs = all(row_has_log_probs_list)
        if not emit_rollout_log_probs and any(row_has_log_probs_list):
            print("Warning: Mixed rollout log_probs availability, omitting rollout_log_probs from batch.")

        is_drop_mask = torch.BoolTensor(is_drop_list).to(self.device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(self.device)

        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device).unsqueeze(0)
        eos_mask_idx = torch.argmax(token_positions * attention_mask, dim=-1)
        token_level_scores[torch.arange(n_sample), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -self.max_response_length :]

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
            padded_log_probs_list = [
                log_probs + [0.0] * (self.max_response_length - len(log_probs))
                for log_probs in response_log_probs_list
                if log_probs is not None
            ]
            batch_dict["rollout_log_probs"] = torch.tensor(padded_log_probs_list, dtype=torch.float32).to(self.device)

        batch = TensorDict(batch_dict, batch_size=n_sample)  # type: ignore[arg-type]
        data_proto = DataProto(batch=batch)
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        if level == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)
        if multi_modal_inputs_list is not None:
            # [multimodal-patch] Per-row dict (or None for text rows), matching
            # verl 0.8.0 extract_multi_modal_inputs expectations.
            data_proto.non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        n_response_turns = len(response_len_per_turn_list)
        data_metrics = {
            "training/reward": float(np.mean(final_rewards)) if final_rewards else 0.0,
            "training/n_sample": n_sample,
            "training/n_rollouts": len(sorted_rollouts),
            "training/n_rollouts_w_trace": sample_with_trace_count,
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_sample": n_trunc_sample_because_of_response,
            "training/n_skipped_empty_rows": n_skipped_empty_training_rows,
            "training/n_turns": n_response_turns,
            "response_length/training/avg_by_turn": float(np.mean(response_len_per_turn_list)),
            "response_length/training/max_by_turn": int(np.max(response_len_per_turn_list)),
            "response_length/training/min_by_turn": int(np.min(response_len_per_turn_list)),
        }
        if level == "trajectory":
            data_metrics["training/n_unmerged_rollouts"] = unmerged_count
            data_metrics["training/n_trace_merge_mismatch_rows"] = len(merge_mismatch_rows)

        return data_proto, data_metrics

    def get_test_metrics(self, completed_rollouts: list[CompletedRollout], *, global_steps: int = 0) -> dict[str, Any]:
        """Build validation metrics from completed rollouts."""
        sample_stat_list: list[dict[str, Any]] = []

        for rollout in completed_rollouts:
            final_reward = self._fillna_reward(rollout)
            sample_stat: dict[str, Any] = {
                "reward": final_reward,
                "has_reward": rollout.final_reward is not None,
            }
            if rollout.triplets:
                response_length_list = [len(triplet.response.get("token_ids") or []) for triplet in rollout.triplets]
                sample_stat.update(
                    {
                        "total_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                    }
                )
            sample_stat_list.append(sample_stat)

        stats_w_trace = [stat for stat in sample_stat_list if "total_response_length" in stat]
        if not stats_w_trace:
            raise RuntimeError("get_test_metrics received zero completed rollouts with trace.")

        validation_trajectory_records = _build_compact_rollout_trajectory_records(
            completed_rollouts,
            tokenizer=self.tokenizer,
            reward_fillna_value=self.reward_fillna_value,
        )
        _upload_compact_rollout_trajectories_to_wandb(
            validation_trajectory_records,
            global_steps,
            is_validation=True,
        )

        return {
            "val/reward": float(np.mean([stat["reward"] for stat in sample_stat_list])),
            "val/n_rollouts": len(sample_stat_list),
            "val/n_rollouts_w_trace": len(stats_w_trace),
            "val/n_rollouts_w_reward": len([stat for stat in sample_stat_list if stat["has_reward"]]),
            "val/mean_response_length_per_turn": float(
                np.mean([stat["mean_response_length"] for stat in stats_w_trace])
            ),
            "val/mean_total_response_length_per_rollout": float(
                np.mean([stat["total_response_length"] for stat in stats_w_trace])
            ),
            "val/turn_count": float(np.mean([stat["turn_count"] for stat in stats_w_trace])),
        }

    def _fillna_reward(self, rollout: CompletedRollout) -> float:
        if rollout.final_reward is not None:
            return rollout.final_reward
        return self.reward_fillna_value


__all__ = ["RolloutAdapter"]

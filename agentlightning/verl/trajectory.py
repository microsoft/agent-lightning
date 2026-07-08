# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple


def _int_list_factory() -> List[int]:
    return []


@dataclass
class TrajectoryMergedTrace:
    """Merged trajectory segment ready for sequence construction."""

    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    turn_indices: List[int]


@dataclass
class TrajectoryAggregationStats:
    """Diagnostics emitted while aggregating trajectory-level traces."""

    unmerged_rollouts: int = 0
    response_lengths_by_turn: List[int] = field(default_factory=_int_list_factory)
    template_mismatch_triplets: int = 0
    retoken_mismatch_triplets: int = 0
    others_mismatch_triplets: int = 0


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
) -> None:
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


def aggregate_trajectory_traces(
    trace_list: List[Mapping[str, List[int]]],
    *,
    max_prompt_length: int,
    tokenizer: Any,
    debug: bool = False,
    global_steps: int = 0,
    rollout_id: str = "",
    mismatch_log_dir: str | None = None,
) -> Tuple[List[TrajectoryMergedTrace], TrajectoryAggregationStats]:
    """Merge turn-level traces into trajectory-level training samples."""

    stats = TrajectoryAggregationStats()
    merged_trace_idx: List[List[int]] = []
    current_merged_trace_idx: List[int] = []
    current_context: List[int] = []

    for turn_index, trace in enumerate(trace_list):
        prompt_ids = trace["prompt_ids"]
        response_ids = trace["response_ids"]
        stats.response_lengths_by_turn.append(len(response_ids))
        full_ids = prompt_ids + response_ids
        is_prefix, diagnostic = ids_startswith(full_ids, current_context, tokenizer, debug)
        if not is_prefix and debug:
            stats.template_mismatch_triplets += int(diagnostic[0])
            stats.retoken_mismatch_triplets += int(diagnostic[1])
            stats.others_mismatch_triplets += int(diagnostic[2])
            log_mismatch_detail(
                diagnostic,
                full_ids,
                current_context,
                global_steps,
                rollout_id,
                turn_index,
                mismatch_log_dir,
            )

        if is_prefix:
            current_context = full_ids
            current_merged_trace_idx.append(turn_index)
        else:
            if current_merged_trace_idx:
                merged_trace_idx.append(current_merged_trace_idx)
            current_merged_trace_idx = [turn_index]
            current_context = full_ids

    if current_merged_trace_idx:
        merged_trace_idx.append(current_merged_trace_idx)

    if len(merged_trace_idx) > 1:
        stats.unmerged_rollouts = 1

    merged_traces: List[TrajectoryMergedTrace] = []
    for turn_indices in merged_trace_idx:
        first_trace = trace_list[turn_indices[0]]
        prompt_ids = list(first_trace["prompt_ids"])

        # If a merged trajectory segment starts after the first turn and the
        # prompt overflows, the truncated tail becomes supervised response.
        if turn_indices[0] > 0 and len(prompt_ids) > max_prompt_length:
            response_ids = prompt_ids[max_prompt_length:]
            prompt_ids = prompt_ids[:max_prompt_length]
            response_mask = [1] * len(response_ids)
        else:
            response_ids = []
            response_mask = []

        prompt_length = len(prompt_ids)
        first_response_ids = first_trace["response_ids"]
        response_ids.extend(first_response_ids)
        response_mask.extend([1] * len(first_response_ids))

        for turn_index in turn_indices[1:]:
            trace = trace_list[turn_index]
            new_prompt_length = max(len(trace["prompt_ids"]) - len(response_ids) - prompt_length, 0)
            if new_prompt_length > 0:
                response_ids.extend(trace["prompt_ids"][-new_prompt_length:])
                response_mask.extend([0] * new_prompt_length)
            response_ids.extend(trace["response_ids"])
            response_mask.extend([1] * len(trace["response_ids"]))

        merged_traces.append(
            TrajectoryMergedTrace(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                turn_indices=turn_indices,
            )
        )

    return merged_traces, stats

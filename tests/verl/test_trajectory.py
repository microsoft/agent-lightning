# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import List

from agentlightning.verl.trajectory import aggregate_trajectory_traces, ids_startswith


class FakeTokenizer:
    all_special_ids: List[int] = [0]

    def decode(self, ids: List[int], *, skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [id for id in ids if id not in self.all_special_ids]
        return "|".join(str(id) for id in ids)


def test_aggregate_trajectory_traces_merges_prefix_turns() -> None:
    merged, stats = aggregate_trajectory_traces(
        [
            {"prompt_ids": [1, 2], "response_ids": [3]},
            {"prompt_ids": [1, 2, 3, 4], "response_ids": [5]},
        ],
        max_prompt_length=8,
        tokenizer=FakeTokenizer(),
    )

    assert len(merged) == 1
    assert merged[0].prompt_ids == [1, 2]
    assert merged[0].response_ids == [3, 4, 5]
    assert merged[0].response_mask == [1, 0, 1]
    assert merged[0].turn_indices == [0, 1]
    assert stats.unmerged_rollouts == 0
    assert stats.response_lengths_by_turn == [1, 1]


def test_aggregate_trajectory_traces_splits_non_prefix_segments_and_masks_overflow_tail() -> None:
    merged, stats = aggregate_trajectory_traces(
        [
            {"prompt_ids": [1], "response_ids": [2]},
            {"prompt_ids": [9, 9, 9, 9], "response_ids": [7, 8]},
        ],
        max_prompt_length=2,
        tokenizer=FakeTokenizer(),
    )

    assert len(merged) == 2
    assert merged[1].prompt_ids == [9, 9]
    assert merged[1].response_ids == [9, 9, 7, 8]
    assert merged[1].response_mask == [1, 1, 1, 1]
    assert len(merged[1].response_mask) == len(merged[1].response_ids)
    assert stats.unmerged_rollouts == 1


def test_ids_startswith_reports_other_mismatch_in_debug_mode() -> None:
    is_prefix, diagnostic = ids_startswith([1, 2], [1, 3], FakeTokenizer(), debug=True)

    assert not is_prefix
    assert diagnostic == (False, False, True)

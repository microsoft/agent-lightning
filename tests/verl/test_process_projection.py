# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agentlightning.credit.artifacts import make_credit_artifact
from agentlightning.credit.types import IncompletePolicyTraceError, ProcessCreditConfig
from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.process_batch import ProjectionLimits, project_process_rollout


def completed(snapshot):
    artifact = make_credit_artifact(snapshot, ProcessCreditConfig(enabled=True, gamma=0.5))
    triplets = []
    for index, record in enumerate(record for record in snapshot.events if record.event_type == "model_request"):
        choice = record.data["response"]["choices"][0]
        triplets.append(
            Triplet(
                prompt={"token_ids": record.data["response"]["prompt_token_ids"]},
                response={"token_ids": choice["token_ids"], "log_probs": choice["logprobs"]["token_logprobs"]},
                metadata={
                    "process_credit": {
                        "action_event_id": record.envelope.event_id,
                        "action_index": index,
                        "finish_reason": choice["finish_reason"],
                    }
                },
            )
        )
    return CompletedRollout(
        rollout_id=snapshot.rollout_id,
        data_id=snapshot.group_id,
        step=7,
        sample_idx_in_step=0,
        enqueue_time=0,
        final_reward=artifact.resolved.terminal_outcome,
        triplets=triplets,
        process_credit=artifact,
    )


@pytest.mark.asyncio
async def test_transition_and_trajectory_preserve_exact_original_origins(sealed_snapshot):
    rollout = completed(await sealed_snapshot())
    limits = ProjectionLimits(max_prompt_length=20, max_response_length=20, eos_token_id=2)
    transition = project_process_rollout(rollout, level="transition", limits=limits)
    trajectory = project_process_rollout(rollout, level="trajectory", limits=limits)
    assert len(transition) == 3
    assert len(trajectory) == 1
    assert trajectory[0].response_mask == (1, 1, 0, 1, 1, 0, 1, 1)
    assert [origin for row in transition for origin in row.origins if origin] == [
        origin for row in trajectory for origin in row.origins if origin
    ]
    assert all(row.response_ids[-1] == 2 for row in transition)
    assert trajectory[0].origins[2] is None
    assert "process_credit" not in rollout.model_dump()


@pytest.mark.asyncio
async def test_incomplete_action_geometry_or_overlong_rows_fail_not_truncate(sealed_snapshot):
    rollout = completed(await sealed_snapshot())
    with pytest.raises(IncompletePolicyTraceError, match="limit"):
        project_process_rollout(
            rollout,
            level="transition",
            limits=ProjectionLimits(max_prompt_length=1, max_response_length=1, eos_token_id=2),
        )
    rollout.triplets[0].response["log_probs"] = None
    with pytest.raises(IncompletePolicyTraceError, match="logprob"):
        project_process_rollout(
            rollout,
            level="transition",
            limits=ProjectionLimits(max_prompt_length=20, max_response_length=20, eos_token_id=2),
        )


@pytest.mark.asyncio
async def test_zero_action_has_no_fake_row(sealed_snapshot):
    rollout = completed(await sealed_snapshot(rewards=(0,), action_count=0))
    assert (
        project_process_rollout(
            rollout,
            level="transition",
            limits=ProjectionLimits(max_prompt_length=5, max_response_length=5, eos_token_id=2),
        )
        == []
    )

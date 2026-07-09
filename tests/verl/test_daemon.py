# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from agentlightning.types import Rollout, Span, Triplet

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("verl")

from agentlightning.verl.daemon import AgentModeDaemon


class _DelayedSpanStore:
    def __init__(self, span: Span):
        self.span = span
        self.query_count = 0

    async def query_spans(self, rollout_id: str, attempt_id: str = "latest") -> list[Span]:
        self.query_count += 1
        if self.query_count == 1:
            return []
        return [self.span]


class _TripletAdapter:
    def adapt(self, spans: Sequence[Span]) -> list[Triplet]:
        if not spans:
            return []
        return [Triplet(prompt={"token_ids": [1, 2]}, response={"token_ids": [3]}, reward=1.0)]


def _make_span() -> Span:
    return Span.from_attributes(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        sequence_id=1,
        trace_id="trace-1",
        span_id="span-1",
        parent_id=None,
        name="raw_gen_ai_request",
        attributes={},
        start_time=1,
        end_time=2,
    )


@pytest.mark.asyncio
async def test_build_completed_rollout_waits_for_delayed_triplets() -> None:
    daemon = AgentModeDaemon.__new__(AgentModeDaemon)
    daemon.store = _DelayedSpanStore(_make_span())
    daemon.adapter = _TripletAdapter()
    daemon.reward_fillna_value = 0.0

    rollout = Rollout(rollout_id="rollout-1", input={}, start_time=0.0, metadata={"data_id": "sample-1"})

    completed = await daemon._build_completed_rollout(rollout)

    assert daemon.store.query_count == 2
    assert len(completed.triplets) == 1
    assert completed.final_reward == 1.0

# Copyright (c) Microsoft. All rights reserved.

"""Tests for LightningGEPAAdapter — evaluate and make_reflective_dataset."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

import pytest

from agentlightning.algorithm.gepa.resources import PromptResourceCodec
from agentlightning.algorithm.gepa.rollout_adapter import LightningGEPAAdapter
from agentlightning.algorithm.gepa.trajectories import RolloutOutput, RolloutTrajectory
from agentlightning.types import PromptTemplate, ResourcesUpdate, Rollout, RolloutConfig, Span
from agentlightning.types.tracer import OtelResource, TraceStatus


def _make_span(
    rollout_id: str = "r1",
    attempt_id: str = "a1",
    name: str = "test_span",
    attributes: Optional[Dict[str, Any]] = None,
) -> Span:
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=1,
        span_id="span-1",
        parent_id=None,
        trace_id="trace-1",
        name=name,
        start_time=0.0,
        end_time=1.0,
        attributes=attributes or {},
        status=TraceStatus(status_code="UNSET"),
        events=[],
        links=[],
        context=None,
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


def _make_rollout(rollout_id: str = "r1", status: str = "succeeded") -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        input={"task": "test"},
        start_time=0.0,
        status=status,  # type: ignore[arg-type]
        config=RolloutConfig(),
    )


def _make_resources_update(resources_id: str = "v0") -> ResourcesUpdate:
    return ResourcesUpdate(
        resources_id=resources_id,
        create_time=0.0,
        update_time=0.0,
        version=1,
        resources={"prompt": PromptTemplate(template="test", engine="f-string")},
    )


class _MockStore:
    """Minimal mock store for adapter tests."""

    def __init__(
        self,
        rollouts: Optional[List[Rollout]] = None,
        spans: Optional[Sequence[Span]] = None,
    ) -> None:
        self._rollouts = rollouts or []
        self._spans: Sequence[Span] = spans or []
        self._enqueued: List[Dict[str, Any]] = []
        self._update_resources_calls: List[Any] = []

    async def update_resources(self, resources_id: str, resources: Any) -> ResourcesUpdate:
        self._update_resources_calls.append((resources_id, resources))
        return _make_resources_update(resources_id)

    async def enqueue_rollout(self, **kwargs: Any) -> Rollout:
        rollout_id = f"r{len(self._enqueued)}"
        self._enqueued.append(kwargs)
        return _make_rollout(rollout_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        # Return matching rollouts from our pre-built list
        by_id = {r.rollout_id: r for r in self._rollouts}
        return [by_id[rid] for rid in rollout_ids if rid in by_id]

    async def query_spans(self, rollout_id: str, attempt_id: Any = None) -> Sequence[Span]:
        return [s for s in self._spans if s.rollout_id == rollout_id]


@pytest.mark.gepa
class TestEvaluate:
    def test_evaluate_returns_evaluation_batch(self):
        """evaluate() should return an EvaluationBatch with correct structure."""
        from gepa import EvaluationBatch

        rollouts = [_make_rollout("r0", "succeeded")]
        spans = [_make_span("r0")]
        store = _MockStore(rollouts=rollouts, spans=spans)
        codec = PromptResourceCodec("prompt", "f-string")

        loop = asyncio.new_event_loop()
        try:
            adapter = LightningGEPAAdapter(
                store=store,  # type: ignore[arg-type]
                codec=codec,
                loop=loop,
                version_counter=[0],
                rollout_batch_timeout=5.0,
                rollout_poll_interval=0.1,
            )

            # Run evaluate in a thread that uses our loop
            import threading

            result_holder: List[Any] = []

            def run_in_thread():
                result_holder.append(
                    adapter.evaluate(
                        batch=[{"task": "test"}],
                        candidate={"prompt": "Hello"},
                        capture_traces=False,
                    )
                )

            thread = threading.Thread(target=run_in_thread)

            async def pump():
                thread.start()
                # Let the loop process coroutines
                while thread.is_alive():
                    await asyncio.sleep(0.01)

            loop.run_until_complete(pump())
            thread.join()

            batch = result_holder[0]
            assert isinstance(batch, EvaluationBatch)
            assert len(batch.outputs) == 1
            assert len(batch.scores) == 1
            assert batch.trajectories is None  # capture_traces=False
        finally:
            loop.close()

    def test_evaluate_timeout_scores_zero(self):
        """Timed-out rollouts should score 0.0."""
        from gepa import EvaluationBatch

        # Store returns no finished rollouts
        store = _MockStore(rollouts=[], spans=[])
        codec = PromptResourceCodec("prompt", "f-string")

        loop = asyncio.new_event_loop()
        try:
            adapter = LightningGEPAAdapter(
                store=store,  # type: ignore[arg-type]
                codec=codec,
                loop=loop,
                version_counter=[0],
                rollout_batch_timeout=0.1,
                rollout_poll_interval=0.05,
            )

            import threading

            result_holder: List[Any] = []

            def run_in_thread():
                result_holder.append(
                    adapter.evaluate(
                        batch=[{"task": "test"}],
                        candidate={"prompt": "Hello"},
                        capture_traces=True,
                    )
                )

            thread = threading.Thread(target=run_in_thread)

            async def pump():
                thread.start()
                while thread.is_alive():
                    await asyncio.sleep(0.01)

            loop.run_until_complete(pump())
            thread.join()

            batch = result_holder[0]
            assert batch.scores == [0.0]
            assert batch.trajectories is not None
            assert len(batch.trajectories) == 1
            assert batch.trajectories[0].status == "cancelled"
        finally:
            loop.close()


class TestMakeReflectiveDataset:
    def test_returns_per_component_records(self):
        """make_reflective_dataset should produce records for each component."""
        codec = PromptResourceCodec("prompt", "f-string")
        loop = asyncio.new_event_loop()
        try:
            adapter = LightningGEPAAdapter(
                store=MagicMock(),
                codec=codec,
                loop=loop,
                version_counter=[0],
            )

            traj = RolloutTrajectory(
                rollout_id="r0",
                status="succeeded",
                spans=[_make_span("r0")],
                final_reward=0.8,
                input={"task": "test"},
            )

            mock_eval_batch = MagicMock()
            mock_eval_batch.trajectories = [traj]

            result = adapter.make_reflective_dataset(
                candidate={"prompt": "Hello world"},
                eval_batch=mock_eval_batch,
                components_to_update=["prompt"],
            )

            assert "prompt" in result
            assert len(result["prompt"]) == 1
            record = result["prompt"][0]
            assert "Inputs" in record
            assert "Generated Outputs" in record
            assert "Feedback" in record
            assert "Component Text" in record
            assert record["Component Text"] == "Hello world"
        finally:
            loop.close()

    def test_empty_when_no_trajectories(self):
        """Returns empty records when trajectories are None."""
        codec = PromptResourceCodec("prompt", "f-string")
        loop = asyncio.new_event_loop()
        try:
            adapter = LightningGEPAAdapter(
                store=MagicMock(),
                codec=codec,
                loop=loop,
                version_counter=[0],
            )

            mock_eval_batch = MagicMock()
            mock_eval_batch.trajectories = None

            result = adapter.make_reflective_dataset(
                candidate={"prompt": "Hello"},
                eval_batch=mock_eval_batch,
                components_to_update=["prompt"],
            )

            assert result == {"prompt": []}
        finally:
            loop.close()

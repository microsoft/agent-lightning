# Copyright (c) Microsoft. All rights reserved.

"""Bridge between GEPA's synchronous adapter protocol and AGL's async store."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agentlightning.adapter.messages import TraceToMessages
from agentlightning.emitter.reward import find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, Rollout, Span

from .resources import PromptResourceCodec
from .trajectories import RolloutOutput, RolloutTrajectory

logger = logging.getLogger(__name__)


class LightningGEPAAdapter:
    """GEPA adapter that evaluates candidates by running AGL store-backed rollouts.

    This class structurally satisfies the ``GEPAAdapter[Any, RolloutTrajectory,
    RolloutOutput]`` protocol without importing ``gepa`` at module level. It
    bridges GEPA's synchronous call convention with the asynchronous
    `LightningStore` API using `asyncio.run_coroutine_threadsafe`.

    Args:
        store: The `LightningStore` instance for enqueuing rollouts and querying spans.
        codec: Codec for converting between GEPA candidates and AGL resources.
        loop: The event loop on which async store methods should execute.
        version_counter: Shared mutable ``list[int]`` for generating monotonic
            resource version identifiers across calls.
        rollout_batch_timeout: Maximum seconds to wait for rollout completion.
        rollout_poll_interval: Seconds between completion polls.
    """

    propose_new_texts = None
    """Let GEPA use its default LLM-based proposer."""

    def __init__(
        self,
        store: LightningStore,
        codec: PromptResourceCodec,
        loop: asyncio.AbstractEventLoop,
        version_counter: List[int],
        rollout_batch_timeout: float = 3600.0,
        rollout_poll_interval: float = 2.0,
    ) -> None:
        self._store = store
        self._codec = codec
        self._loop = loop
        self._version_counter = version_counter
        self._rollout_batch_timeout = rollout_batch_timeout
        self._rollout_poll_interval = rollout_poll_interval

    # ------------------------------------------------------------------
    # Sync → async bridge
    # ------------------------------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Execute an async coroutine on the algorithm's event loop from a worker thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def _next_version(self) -> str:
        version = self._version_counter[0]
        self._version_counter[0] = version + 1
        return f"gepa-v{version}"

    # ------------------------------------------------------------------
    # GEPAAdapter.evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: List[Any],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Evaluate a candidate prompt on a batch of inputs via AGL rollouts.

        Publishes the candidate as AGL resources, enqueues one rollout per
        batch item, polls for completion, and returns an ``EvaluationBatch``
        with per-example scores, outputs, and optional trajectories.

        Per-example failures are scored as 0.0 — the run is never crashed.

        Args:
            batch: List of task inputs.
            candidate: GEPA candidate mapping component names to text.
            capture_traces: Whether to populate trajectory data for reflection.

        Returns:
            A ``gepa.EvaluationBatch`` with outputs, scores, and trajectories.
        """
        from gepa import EvaluationBatch  # type: ignore[reportAttributeAccessIssue]

        resources: NamedResources = self._codec.candidate_to_resources(candidate)
        version_id = self._next_version()
        resource_update = self._run_async(self._store.update_resources(version_id, resources))

        # Enqueue rollouts
        rollout_ids: List[str] = []
        for task_input in batch:
            rollout = self._run_async(
                self._store.enqueue_rollout(
                    input=task_input,
                    mode="train",
                    resources_id=resource_update.resources_id,
                )
            )
            rollout_ids.append(rollout.rollout_id)

        # Poll for completion
        finished = self._wait_for_rollouts(rollout_ids)

        # Build per-example results
        outputs: List[RolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[RolloutTrajectory]] = [] if capture_traces else None

        finished_by_id: Dict[str, Rollout] = {r.rollout_id: r for r in finished}

        for idx, rollout_id in enumerate(rollout_ids):
            rollout = finished_by_id.get(rollout_id)
            if rollout is None:
                # Timed out — score as 0.0
                logger.warning("Rollout %s did not complete within timeout, scoring 0.0", rollout_id)
                outputs.append(RolloutOutput(rollout_id=rollout_id, status="cancelled", final_reward=None))
                scores.append(0.0)
                if trajectories is not None:
                    trajectories.append(
                        RolloutTrajectory(
                            rollout_id=rollout_id,
                            status="cancelled",
                            spans=[],
                            final_reward=None,
                            input=batch[idx],
                        )
                    )
                continue

            try:
                spans: Sequence[Span] = self._run_async(
                    self._store.query_spans(rollout.rollout_id, attempt_id="latest")
                )
            except Exception:
                logger.exception("Failed to query spans for rollout %s", rollout.rollout_id)
                spans = []

            reward = find_final_reward(list(spans))
            score = reward if reward is not None else 0.0

            outputs.append(
                RolloutOutput(
                    rollout_id=rollout.rollout_id,
                    status=rollout.status,
                    final_reward=reward,
                )
            )
            scores.append(score)

            if trajectories is not None:
                messages = self._try_trace_to_messages(list(spans))
                trajectories.append(
                    RolloutTrajectory(
                        rollout_id=rollout.rollout_id,
                        status=rollout.status,
                        spans=list(spans),
                        final_reward=reward,
                        input=batch[idx],
                        messages=messages,
                    )
                )

        return EvaluationBatch(  # type: ignore[reportUnknownVariableType]
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    # ------------------------------------------------------------------
    # GEPAAdapter.make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: Any,
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build per-component reflective records from evaluation trajectories.

        Each record includes the task input, candidate text, reward, status,
        a span summary, and optional reconstructed messages for the component.

        Args:
            candidate: The evaluated GEPA candidate.
            eval_batch: The ``EvaluationBatch`` returned by ``evaluate()``.
            components_to_update: Component names that GEPA wants to refine.

        Returns:
            Mapping from component name to a list of reflective dataset records.
        """
        result: Dict[str, List[Dict[str, Any]]] = {comp: [] for comp in components_to_update}

        trajectories: Optional[List[RolloutTrajectory]] = eval_batch.trajectories
        if trajectories is None:
            logger.warning("No trajectories available for reflective dataset; returning empty records")
            return result

        for traj in trajectories:
            span_summary = self._summarize_spans(traj.spans)
            record: Dict[str, Any] = {
                "Inputs": repr(traj.input),
                "Generated Outputs": span_summary,
                "Feedback": f"Reward: {traj.final_reward}, Status: {traj.status}",
            }
            if traj.messages:
                record["Messages"] = traj.messages

            for comp in components_to_update:
                comp_record = {**record, "Component": comp, "Component Text": candidate.get(comp, "")}
                result[comp].append(comp_record)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_rollouts(self, rollout_ids: List[str]) -> List[Rollout]:
        """Poll the store for rollout completion with timeout."""
        deadline = time.time() + self._rollout_batch_timeout
        while time.time() < deadline:
            finished: List[Rollout] = self._run_async(
                self._store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0)
            )
            if len(finished) >= len(rollout_ids):
                logger.info("All %d rollouts finished within timeout", len(rollout_ids))
                return finished
            logger.debug(
                "%d / %d rollouts finished, polling again in %.1fs",
                len(finished),
                len(rollout_ids),
                self._rollout_poll_interval,
            )
            time.sleep(self._rollout_poll_interval)

        # Deadline passed — return whatever finished
        finished = self._run_async(self._store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0))
        logger.warning(
            "Rollout batch timed out after %.0fs: %d / %d finished",
            self._rollout_batch_timeout,
            len(finished),
            len(rollout_ids),
        )
        return finished

    @staticmethod
    def _try_trace_to_messages(spans: List[Span]) -> Optional[List[Any]]:
        """Attempt to reconstruct OpenAI messages from spans, returning None on failure."""
        try:
            adapter = TraceToMessages()
            return adapter.adapt(spans)
        except Exception:
            logger.debug("TraceToMessages failed, falling back to None", exc_info=True)
            return None

    @staticmethod
    def _summarize_spans(spans: List[Span]) -> str:
        """Build a concise textual summary of span names and rewards."""
        if not spans:
            return "(no spans)"
        parts: List[str] = []
        for span in spans:
            reward = find_final_reward([span])
            if reward is not None:
                parts.append(f"{span.name} [reward={reward}]")
            else:
                parts.append(span.name)
        return " -> ".join(parts)

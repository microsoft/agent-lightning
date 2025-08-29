import asyncio
import json
import logging
import os
import time
from contextlib import nullcontext
from typing import List, Optional, Union, Dict, Any
from collections import defaultdict

import agentops

from opentelemetry.sdk.trace import ReadableSpan
from .client import AgentLightningClient
from .litagent import LitAgent
from .types import Rollout, Task, Triplet, RolloutRawResult, TaskInput, NamedResources
from .types import ParallelWorkerBase
from .tracer.base import BaseTracer
from .tracer import TripletExporter

logger = logging.getLogger(__name__)


class AgentRunner(ParallelWorkerBase):
    """Manages the agent's execution loop and integrates with AgentOps.

    This class orchestrates the interaction between the agent (`LitAgent`) and
    the server (`AgentLightningClient`). It handles polling for tasks, executing
    the agent's logic, and reporting results back to the server. If enabled,
    it will also automatically trace each rollout using AgentOps.

    Attributes:
        agent: The `LitAgent` instance containing the agent's logic.
        client: The `AgentLightningClient` for server communication.
        tracer: The tracer instance for this runner/worker.
        worker_id: An optional identifier for the worker process.
        max_tasks: The maximum number of tasks to process before stopping.
    """

    def __init__(
        self,
        agent: LitAgent,
        client: AgentLightningClient,
        tracer: BaseTracer,
        triplet_exporter: TripletExporter,
        worker_id: Optional[int] = None,
        max_tasks: Optional[int] = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.agent = agent
        self.client = client
        self.tracer = tracer
        self.triplet_exporter = triplet_exporter

        # Worker-specific attributes
        self.worker_id = worker_id
        self.max_tasks = max_tasks
        self.batch_size = batch_size

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generates a standardized log prefix for the current worker."""
        if self.worker_id is not None:
            if rollout_id:
                return f"[Worker {self.worker_id} | Rollout {rollout_id}]"
            else:
                return f"[Worker {self.worker_id}]"
        if rollout_id:
            return f"[Rollout {rollout_id}]"
        return "[Default Worker]"

    def _to_rollout_object(
        self,
        result: RolloutRawResult,
        rollout_id: str,
    ) -> Rollout:
        """Standardizes the agent's return value into a Rollout object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.

        Returns:
            A standardized `Rollout` object for reporting to the server.
        """
        trace: Any = None
        final_reward: Optional[float] = None
        triplets: Optional[List[Triplet]] = None
        trace_spans: Optional[List[ReadableSpan]] = None

        # Handle different types of results from the agent
        # Case 1: result is a float (final reward)
        if isinstance(result, float):
            final_reward = result
        # Case 2: result is a list of Triplets
        if isinstance(result, list) and all(isinstance(t, Triplet) for t in result):
            triplets = result  # type: ignore
        # Case 3: result is a list of ReadableSpan (OpenTelemetry spans)
        if isinstance(result, list) and all(isinstance(t, ReadableSpan) for t in result):
            trace_spans = result  # type: ignore
            trace = [json.loads(readable_span.to_json()) for readable_span in trace_spans]  # type: ignore
        # Case 4: result is a list of dict (trace JSON)
        if isinstance(result, list) and all(isinstance(t, dict) for t in result):
            trace = result
        # Case 5: result is a Rollout object
        if isinstance(result, Rollout):
            final_reward = result.final_reward
            triplets = result.triplets
            trace = result.trace

        # If the agent has tracing enabled, use the tracer's last trace if not already set
        if self.tracer and (trace is None or trace_spans is None):
            spans = self.tracer.get_last_trace()
            if spans:
                trace = [json.loads(readable_span.to_json()) for readable_span in spans]
                trace_spans = spans

        # Always extract triplets from the trace using TripletExporter
        if trace_spans:
            triplets = self.triplet_exporter.export(trace_spans)

        # If the agent has triplets, use the last one for final reward if not set
        if triplets and triplets[-1].reward is not None and final_reward is None:
            final_reward = triplets[-1].reward

        # Create the Rollout object with standardized fields
        result_dict: Dict[str, Any] = {
            "rollout_id": rollout_id,
        }
        if final_reward is not None:
            result_dict["final_reward"] = final_reward
        if triplets is not None:
            result_dict["triplets"] = triplets
        if trace is not None:
            result_dict["trace"] = trace

        if isinstance(result, Rollout):
            return result.model_copy(update=result_dict)
        return Rollout(**result_dict)

    def run(self) -> int:
        """Poll tasks and execute rollouts synchronously.

        Returns the number of tasks processed."""
        self.agent.set_runner(self)

        tasks: List[Task] = []
        for _ in range(self.batch_size):
            task = self.client.poll_next_task()
            if task is None:
                break
            tasks.append(task)

        if not tasks:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return 0

        inputs: List[TaskInput] = []
        rollout_ids: List[str] = []
        resources_list: List[NamedResources] = []
        modes: List[Optional[str]] = []

        for task in tasks:
            resources_id = task.resources_id
            resources_update = None
            if resources_id:
                resources_update = self.client.get_resources_by_id(resources_id)
            else:
                logger.debug(f"{self._log_prefix(task.rollout_id)} No 'resources_id'. Fetching latest resources.")
                resources_update = self.client.get_latest_resources()
            if not resources_update:
                logger.error(f"{self._log_prefix(task.rollout_id)} Failed to fetch resources. Skipping.")
                continue
            inputs.append(task.input)
            rollout_ids.append(task.rollout_id)
            resources_list.append(resources_update.resources)
            modes.append(task.mode)

        if not inputs:
            return 0

        results: List[RolloutRawResult] = [None] * len(inputs)
        mode_groups: Dict[Optional[str], List[int]] = defaultdict(list)
        for idx, mode in enumerate(modes):
            mode_groups[mode].append(idx)

        for mode, indices in mode_groups.items():
            sub_tasks = [inputs[i] for i in indices]
            sub_ids = [rollout_ids[i] for i in indices]
            sub_res = [resources_list[i] for i in indices]
            if mode == "train":
                sub_results = self.agent.training_rollout_batch(sub_tasks, sub_ids, sub_res)
            else:
                sub_results = self.agent.validation_rollout_batch(sub_tasks, sub_ids, sub_res)
            for idx, res in zip(indices, sub_results):
                results[idx] = res

        for rid, res in zip(rollout_ids, results):
            rollout_obj = self._to_rollout_object(res, rid)
            try:
                self.client.post_rollout(rollout_obj)
            except Exception:
                logger.exception(f"{self._log_prefix(rid)} Exception during rollout.")

        return len(rollout_ids)

    def iter(self) -> int:
        """Executes the synchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started sync rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            processed = self.run()
            if processed:
                num_tasks_processed += processed
            else:
                break

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished sync rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def run_async(self) -> int:
        """Poll tasks and execute rollouts asynchronously.

        Returns the number of tasks processed."""
        self.agent.set_runner(self)

        tasks: List[Task] = []
        for _ in range(self.batch_size):
            task = await self.client.poll_next_task_async()
            if task is None:
                break
            tasks.append(task)

        if not tasks:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return 0

        inputs: List[TaskInput] = []
        rollout_ids: List[str] = []
        resources_list: List[NamedResources] = []
        modes: List[Optional[str]] = []

        for task in tasks:
            resources_id = task.resources_id
            resources_update = None
            if resources_id:
                resources_update = await self.client.get_resources_by_id_async(resources_id)
            else:
                logger.debug(f"{self._log_prefix(task.rollout_id)} No 'resources_id'. Fetching latest resources.")
                resources_update = await self.client.get_latest_resources_async()
            if not resources_update:
                logger.error(f"{self._log_prefix(task.rollout_id)} Failed to fetch resources. Skipping.")
                continue
            inputs.append(task.input)
            rollout_ids.append(task.rollout_id)
            resources_list.append(resources_update.resources)
            modes.append(task.mode)

        if not inputs:
            return 0

        results: List[RolloutRawResult] = [None] * len(inputs)
        mode_groups: Dict[Optional[str], List[int]] = defaultdict(list)
        for idx, mode in enumerate(modes):
            mode_groups[mode].append(idx)

        for mode, indices in mode_groups.items():
            sub_tasks = [inputs[i] for i in indices]
            sub_ids = [rollout_ids[i] for i in indices]
            sub_res = [resources_list[i] for i in indices]
            if mode == "train":
                sub_results = await self.agent.training_rollout_batch_async(sub_tasks, sub_ids, sub_res)
            else:
                sub_results = await self.agent.validation_rollout_batch_async(sub_tasks, sub_ids, sub_res)
            for idx, res in zip(indices, sub_results):
                results[idx] = res

        for rid, res in zip(rollout_ids, results):
            rollout_obj = self._to_rollout_object(res, rid)
            try:
                await self.client.post_rollout_async(rollout_obj)
            except Exception:
                logger.exception(f"{self._log_prefix(rid)} Exception during rollout.")

        return len(rollout_ids)

    async def iter_async(self) -> int:
        """Executes the asynchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            processed = await self.run_async()
            if processed:
                num_tasks_processed += processed
            else:
                break

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")
        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

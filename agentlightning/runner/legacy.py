# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import time
from typing import Any, Dict, List, Optional, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.adapter import TracerTraceToTriplet
from agentlightning.client import AgentLightningClient
from agentlightning.litagent import LitAgent
from agentlightning.litagent.litagent import is_v0_1_rollout_api
from agentlightning.tracer.base import Tracer
from agentlightning.types import RolloutLegacy, RolloutRawResultLegacy, Triplet

from .base import Runner

logger = logging.getLogger(__name__)

__all__ = [
    "LegacyAgentRunner",
]


class LegacyAgentRunner(Runner[Any]):
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
        agent: LitAgent[Any],
        client: AgentLightningClient,
        tracer: Tracer,
        triplet_exporter: TracerTraceToTriplet,
        worker_id: Optional[int] = None,
        max_tasks: Optional[int] = None,
    ):
        super().__init__()
        self.agent = agent
        self.client = client
        self.tracer = tracer
        self.triplet_exporter = triplet_exporter

        # Worker-specific attributes
        self.worker_id = worker_id
        self.max_tasks = max_tasks

    # These methods are overridden by Runner, getting them back to old behavior.
    def init(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.worker_id = worker_id

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        pass

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generates a standardized log prefix for the current worker."""
        if self.worker_id is not None:
            if rollout_id:
                return f"[Worker {self.worker_id} | RolloutLegacy {rollout_id}]"
            else:
                return f"[Worker {self.worker_id}]"
        if rollout_id:
            return f"[RolloutLegacy {rollout_id}]"
        return "[Default Worker]"

    def _to_rollout_object(
        self,
        result: RolloutRawResultLegacy,
        rollout_id: str,
    ) -> RolloutLegacy:
        """Standardizes the agent's return value into a RolloutLegacy object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.

        Returns:
            A standardized `RolloutLegacy` object for reporting to the server.
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
        # Case 5: result is a RolloutLegacy object
        if isinstance(result, RolloutLegacy):
            final_reward = result.final_reward
            triplets = result.triplets
            trace = result.trace

        # If the agent has tracing enabled, use the tracer's last trace if not already set
        if self.tracer and (trace is None or trace_spans is None):
            spans = self.tracer.get_last_trace()
            if spans:
                trace = [json.loads(readable_span.to_json()) for readable_span in spans]
                trace_spans = spans

        # Always extract triplets from the trace using TracerTraceToTriplet
        if trace_spans:
            triplets = self.triplet_exporter(trace_spans)  # type: ignore

        # If the agent has triplets, use the last one for final reward if not set
        if triplets and triplets[-1].reward is not None and final_reward is None:
            final_reward = triplets[-1].reward

        # Create the RolloutLegacy object with standardized fields
        result_dict: Dict[str, Any] = {
            "rollout_id": rollout_id,
        }
        if final_reward is not None:
            result_dict["final_reward"] = final_reward
        if triplets is not None:
            result_dict["triplets"] = triplets
        if trace is not None:
            result_dict["trace"] = trace

        if isinstance(result, RolloutLegacy):
            return result.model_copy(update=result_dict)
        return RolloutLegacy(**result_dict)

    def run(self) -> bool:  # type: ignore
        """Poll the task and rollout once synchronously."""
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner

        task = self.client.poll_next_task()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id

        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = self.client.get_resources_by_id(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = self.client.get_latest_resources()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return False

        rollout_obj = RolloutLegacy(rollout_id=task.rollout_id, task=task)  # Default empty rollout

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            with self.tracer._trace_context_sync(name=f"rollout_{rollout_id}"):  # pyright: ignore[reportPrivateUsage]
                start_time = time.time()
                rollout_method = self.agent.training_rollout if task.mode == "train" else self.agent.validation_rollout
                # Pass the task input, not the whole task object
                if is_v0_1_rollout_api(rollout_method):
                    result = cast(
                        RolloutRawResultLegacy,
                        rollout_method(
                            task.input, rollout_id=rollout_obj.rollout_id, resources=resources_update.resources  # type: ignore
                        ),
                    )  # type: ignore
                else:
                    result = rollout_method(task.input, resources=resources_update.resources, rollout=rollout_obj)  # type: ignore
                rollout_obj = self._to_rollout_object(result, task.rollout_id)  # type: ignore
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Triplet length: "
                    f"{len(rollout_obj.triplets) if rollout_obj.triplets is not None else 'N/A'}. "
                    f"Reward: {rollout_obj.final_reward}"
                )

        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)  # type: ignore
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            self.client.post_rollout(rollout_obj)

        return True

    def iter(self) -> int:  # type: ignore
        """Executes the synchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started sync rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if self.run():
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished sync rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def run_async(self) -> bool:
        """Poll the task and rollout once."""
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner

        task = await self.client.poll_next_task_async()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id

        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = await self.client.get_resources_by_id_async(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = await self.client.get_latest_resources_async()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return False

        rollout_obj = RolloutLegacy(rollout_id=task.rollout_id, task=task)  # Default empty rollout

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            async with self.tracer.trace_context(name=f"rollout_{rollout_id}"):
                start_time = time.time()
                rollout_method = (
                    self.agent.training_rollout_async if task.mode == "train" else self.agent.validation_rollout_async
                )
                # Pass the task input, not the whole task object
                if is_v0_1_rollout_api(rollout_method):
                    result = cast(
                        RolloutRawResultLegacy,
                        await rollout_method(
                            task.input, rollout_id=rollout_obj.rollout_id, resources=resources_update.resources  # type: ignore
                        ),
                    )  # type: ignore
                else:
                    result = await rollout_method(task.input, resources=resources_update.resources, rollout=rollout_obj)  # type: ignore
                rollout_obj = self._to_rollout_object(result, task.rollout_id)  # type: ignore
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Triplet length: "
                    f"{len(rollout_obj.triplets) if rollout_obj.triplets is not None else 'N/A'}. "
                    f"Reward: {rollout_obj.final_reward}"
                )
        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)  # type: ignore
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            await self.client.post_rollout_async(rollout_obj)

        return True

    async def iter_async(self) -> int:
        """Executes the asynchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if await self.run_async():
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")
        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

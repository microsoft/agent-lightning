import asyncio
import logging
import os
import time
from contextlib import nullcontext
from typing import List, Optional, Union, Any

import agentops

from agentlightning.instrumentation import instrument_all
from agentlightning.trace import LightningSpanProcessor, lightning_span_processor
from .client import AgentLightningClient
from .litagent import LitAgent
from .types import Rollout, Task, Triplet

logger = logging.getLogger(__name__)


class AgentRunner:
    """Manages the agent's execution loop and integrates with AgentOps.

    This class orchestrates the interaction between the agent (`LitAgent`) and
    the server (`AgentLightningClient`). It handles polling for tasks, executing
    the agent's logic, and reporting results back to the server. If enabled,
    it will also automatically trace each rollout using AgentOps.

    Attributes:
        agent: The `LitAgent` instance containing the agent's logic.
        client: The `AgentLightningClient` for server communication.
        worker_id: An optional identifier for the worker process.
        max_tasks: The maximum number of tasks to process before stopping.
        agentops_managed: If True, automatically trace rollouts with AgentOps.
        agentops_server_port: A port number for the AgentOps server, if managed by AgentOps.
        instrument_managed: If True, automatically apply instrumentation to the agent.
    """

    def __init__(
        self,
        agent: LitAgent,
        client: AgentLightningClient,
        worker_id: Optional[int] = None,
        max_tasks: Optional[int] = None,
        agentops_managed: bool = True,
        agentops_server_port: Optional[int] = None,
        instrument_managed: bool = True,
    ):
        self.agent = agent
        self.client = client
        self.worker_id = worker_id
        self.max_tasks = max_tasks
        self.agentops_managed = agentops_managed
        self.agentops_server_port = agentops_server_port
        self.instrument_managed = instrument_managed

    def _init_runner_env(self):
        logger.info(f"{self._log_prefix()} Setting up environment...")  # worker_id included in process name
        if self.agentops_managed:
            if self.agentops_server_port:
                base_url = f"http://localhost:{self.agentops_server_port}"
                env_vars_to_set = {
                    "AGENTOPS_API_KEY": "dummy",
                    "AGENTOPS_API_ENDPOINT": base_url,
                    "AGENTOPS_APP_URL": f"{base_url}/notavailable",
                    "AGENTOPS_EXPORTER_ENDPOINT": f"{base_url}/traces",
                }
                for key, value in env_vars_to_set.items():
                    os.environ[key] = value
                    logger.info(f"{self._log_prefix()} Env var set: {key}={value}")
            else:
                logger.warning(
                    f"{self._log_prefix()} AgentOps managed, but local server port is not available. Client may not connect as expected."
                )

            if not agentops.get_client().initialized:
                agentops.init()
                logger.info(f"{self._log_prefix()} AgentOps client initialized.")
            else:
                logger.warning(f"{self._log_prefix()} AgentOps client was already initialized.")

        if self.instrument_managed:
            instrument_all()
            logger.info(f"{self._log_prefix()} Instrumentation applied.")

    def _teardown_runner_env(self):
        logger.info(f"{self._log_prefix()} Cleaning up environment...")
        # Do nothing for now.
        logger.info(f"{self._log_prefix()} Environment cleanup complete.")

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
        self, result: Union[None, float, List[Triplet], Rollout], rollout_id: str,
        lightning_span_processor: Optional[LightningSpanProcessor] = None
    ) -> Rollout:
        """Standardizes the agent's return value into a Rollout object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.
            lightning_span_processor: Optional span processor for tracing.

        Returns:
            A standardized `Rollout` object for reporting to the server.
        """
        trace_json: Any = None
        final_reward: Optional[float] = None
        triplets: Optional[List[Triplet]] = None

        if isinstance(result, float):
            final_reward = result
        if isinstance(result, Rollout):
            final_reward = result.final_reward

        if lightning_span_processor:
            trace = lightning_span_processor.last_trace()
            if trace:
                trace_json = trace.to_json()
                trajectory = trace.to_trajectory(agent_match=self.agent.trained_agents, final_reward=final_reward)
                triplets = [Triplet(prompt=step.state, response=step.action, reward=step.reward) for step in trajectory]

        if isinstance(result, list) and all(isinstance(t, Triplet) for t in result):
            triplets = result

        if triplets and triplets[-1].reward is not None and final_reward is None:
            final_reward = triplets[-1].reward

        if result is None:
            # Agent wants runner to handle tracing; return minimal Rollout
            return Rollout(rollout_id=rollout_id)
        if isinstance(result, Rollout):
            result.rollout_id = rollout_id
            return result
        elif isinstance(result, float):
            return Rollout(rollout_id=rollout_id, final_reward=result)
        elif isinstance(result, list):
            return Rollout(rollout_id=rollout_id, triplets=result)
        else:
            logger.warning(f"Unexpected return type: {type(result)}. Reporting empty Rollout.")
            return Rollout(rollout_id=rollout_id)

    def run(self, worker_id: int) -> int:
        """Executes the synchronous polling and rollout loop."""
        # ... (implementation is analogous to run_async, but with sync calls)
        # This is omitted for brevity but would follow the same logic as run_async
        # using client.poll_next_task() and client.post_rollout().
        raise NotImplementedError("Sync runner not implemented for this example. Use async.")

    async def run_async(self, worker_id: int) -> int:
        """Executes the asynchronous polling and rollout loop with AgentOps."""
        num_tasks_processed = 0
        logger.info(f"[Worker {worker_id}] Started async rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            task = await self.client.poll_next_task_async()
            if task is None:
                logger.info(f"[Worker {worker_id}] No more tasks available. Exiting.")
                break

            resources_id = task.metadata.resources_id
            resources_update = None
            if resources_id:
                resources_update = await self.client.get_resources_by_id_async(resources_id)
            else:
                logger.debug(f"[Worker {worker_id}] Task {task.rollout_id} has no 'resources_id'. Fetching latest resources.")
                resources_update = await self.client.get_latest_resources_async()
            if not resources_update:
                logger.error(f"[Worker {worker_id}] Task {task.rollout_id} failed to fetch resources. Skipping.")
                continue

            exception_occurred = False
            rollout_obj = Rollout(rollout_id=task.rollout_id) # Default empty rollout
            try:
                # Use AgentOps record as a context manager if enabled
                context = agentops.record_async(Event(name=f"rollout_{task.rollout_id}")) if self.agentops_enabled else nullcontext()
                async with context:
                    start_time = time.time()
                    rollout_method = (
                        self.agent.training_rollout_async
                        if task.metadata.mode == "train"
                        else self.agent.validation_rollout_async
                    )
                    # Pass the task input, not the whole task object
                    result = await rollout_method(task.input, task.rollout_id, resources_update.resources)
                    rollout_obj = self._to_rollout_object(result, task.rollout_id)
                    end_time = time.time()
                    logger.info(
                        f"[Worker {worker_id}] (Rollout {task.rollout_id}) Completed in "
                        f"{end_time - start_time:.2f}s. Reward: {rollout_obj.final_reward}"
                    )

            except Exception:
                exception_occurred = True
                logger.exception(f"[Worker {worker_id}] (Rollout {task.rollout_id}) Exception during rollout.")
            finally:
                # End AgentOps session and report to the Agent Lightning Server
                self._end_agentops_session(rollout_obj, exception_occurred)
                await self.client.post_rollout_async(rollout_obj)
                num_tasks_processed += 1

        logger.info(f"[Worker {worker_id}] Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed
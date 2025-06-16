import asyncio
import logging
import time
from typing import List, Optional, Union

from .client import AgentLightningClient
from .litagent import LitAgent
from .types import Rollout, Task, Triplet

logger = logging.getLogger(__name__)


# WIP
class Loop:
    """Manages the agent's execution loop.

    This class orchestrates the interaction between the agent (`LitAgent`) and
    the server (`AgentLightningClient`). It handles polling for tasks, fetching
    resources, executing the agent's rollout logic, processing the results,
    and reporting back to the server.

    Attributes:
        agent: The `LitAgent` instance containing the agent's logic.
        client: The `AgentLightningClient` for server communication.
        max_tasks: The maximum number of tasks to process before stopping.
    """

    def __init__(self, agent: LitAgent, client: AgentLightningClient, max_tasks: Optional[int] = None):
        """Initializes the Loop.

        Args:
            agent: The `LitAgent` instance to execute.
            client: The client to communicate with the Agent Lightning server.
            max_tasks: Optional limit on the number of tasks to process.
        """
        self.agent = agent
        self.client = client
        self.max_tasks = max_tasks

    def _process_rollout_result(self, result: Union[float, List[Triplet], Rollout], rollout_id: str) -> Rollout:
        """Standardizes the agent's return value into a Rollout object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.

        Returns:
            A standardized `Rollout` object ready to be sent to the server.
        """
        if isinstance(result, Rollout):
            # Ensure the rollout_id is correctly set
            result.rollout_id = rollout_id
            return result
        elif isinstance(result, float):
            return Rollout(rollout_id=rollout_id, final_reward=result)
        elif isinstance(result, list):
            return Rollout(rollout_id=rollout_id, triplets=result)
        else:
            logger.warning(
                f"Unexpected return type from rollout: {type(result)}. " "Returning empty Rollout with no reward."
            )
            return Rollout(rollout_id=rollout_id)

    def run(self, worker_id: int) -> int:
        """Executes the synchronous polling and rollout loop.

        This loop continuously polls for tasks, executes the agent's synchronous
        rollout method, and reports the results until the task limit is
        reached or no more tasks are available.

        Args:
            worker_id: The identifier for the worker running this loop.

        Returns:
            The total number of tasks processed.
        """
        num_tasks_processed = 0
        logger.info(f"[Worker {worker_id}] Started synchronous rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            task = self.client.poll_next_task()
            if task is None:
                logger.info(f"[Worker {worker_id}] No more tasks available from server. Exiting.")
                break

            resources_id = task.metadata.resources_id
            resources_update = None
            if resources_id:
                resources_update = self.client.get_resources_by_id(resources_id)
            else:
                logger.debug(f"[Worker {worker_id}] Task {task.rollout_id} has no 'resources_id'. Fetching latest resources.")
                resources_update = self.client.get_latest_resources()
            if not resources_update:
                logger.error(f"[Worker {worker_id}] Task {task.rollout_id} failed to fetch resources. Skipping.")
                continue

            try:
                start_time = time.time()
                rollout_method = (
                    self.agent.training_rollout if task.metadata.mode == "train" else self.agent.validation_rollout
                )
                result = rollout_method(task.input, task.rollout_id, resources_update.resources)
                rollout_obj = self._process_rollout_result(result, task.rollout_id)
                self.client.post_rollout(rollout_obj)

                end_time = time.time()
                logger.info(
                    f"[Worker {worker_id}] (Rollout {task.rollout_id}) Completed in "
                    f"{end_time - start_time:.2f}s. Reward: {rollout_obj.final_reward}"
                )

            except Exception:
                logger.exception(f"[Worker {worker_id}] (Rollout {task.rollout_id}) Exception during rollout.")

            num_tasks_processed += 1

        logger.info(f"[Worker {worker_id}] Finished rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def run_async(self, worker_id: int) -> int:
        """Executes the asynchronous polling and rollout loop.

        This loop continuously polls for tasks, executes the agent's
        asynchronous rollout method, and reports the results until the task
        limit is reached or no more tasks are available.

        Args:
            worker_id: The identifier for the worker running this loop.

        Returns:
            The total number of tasks processed.
        """
        num_tasks_processed = 0
        logger.info(f"[Worker {worker_id}] Started asynchronous rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            task = await self.client.poll_next_task_async()
            if task is None:
                logger.info(f"[Worker {worker_id}] No more tasks available from server. Exiting.")
                break

            resources_id = task.metadata.resources_id
            if not resources_id:
                logger.error(f"[Worker {worker_id}] Task {task.rollout_id} missing required 'resources_id'. Skipping.")
                continue

            resources_update = await self.client.get_resources_by_id_async(resources_id)
            if not resources_update:
                logger.error(f"[Worker {worker_id}] Could not fetch resources for ID '{resources_id}'. Skipping.")
                continue

            try:
                start_time = time.time()
                rollout_method = (
                    self.agent.training_rollout_async
                    if task.metadata.mode == "train"
                    else self.agent.validation_rollout_async
                )
                result = await rollout_method(task.input, task.rollout_id, resources_update.resources)
                rollout_obj = self._process_rollout_result(result, task.rollout_id)
                await self.client.post_rollout_async(rollout_obj)

                end_time = time.time()
                logger.info(
                    f"[Worker {worker_id}] (Rollout {task.rollout_id}) Completed in "
                    f"{end_time - start_time:.2f}s. Reward: {rollout_obj.final_reward}"
                )

            except Exception:
                logger.exception(f"[Worker {worker_id}] (Rollout {task.rollout_id}) Exception during async rollout.")

            num_tasks_processed += 1

        logger.info(f"[Worker {worker_id}] Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

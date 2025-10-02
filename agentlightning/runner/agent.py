# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar

from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward
from agentlightning.store.base import LightningStore
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import Hook, RolloutRawResultV2, RolloutV2

if TYPE_CHECKING:
    from agentlightning.execution.events import Event

from .base import BaseRunner

T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


def rollout_to_rollout_v2_and_spans(rollout: Rollout) -> Tuple[RolloutV2, List[ReadableSpan]]:
    """Convert a rollout to a rollout v2."""


def rollout_v2_to_rollout(rollout_v2: RolloutV2) -> Rollout:
    """Convert a rollout v2 to a rollout."""


class AgentRunnerV2(BaseRunner[T_task]):
    """The runner for the agent."""

    def __init__(self, tracer: BaseTracer, max_tasks: Optional[int] = None, poll_interval: float = 5.0) -> None:
        super().__init__()
        self._tracer = tracer
        self._max_tasks = max_tasks
        self._poll_interval = poll_interval

        # Set later
        self._agent: Optional[LitAgent[T_task]] = None
        self._hooks: Sequence[Hook] = []
        self._store: Optional[LightningStore] = None
        self._worker_id: Optional[int] = None

    def init(self, agent: LitAgent[T_task], *, hooks: Sequence[Hook], **kwargs: Any) -> None:
        """Initialize the runner with the agent."""
        self._agent = agent
        self._agent.set_runner(self)
        self._hooks = [*hooks]

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store."""
        self._store = store
        self._worker_id = worker_id

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner."""
        self._agent = None
        self._store = None
        self._worker_id = None

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner for each worker."""
        self._worker_id = None

    def get_agent(self) -> LitAgent[T_task]:
        """Get the agent."""
        if self._agent is None:
            raise ValueError("Agent not initialized. Call init() first.")
        return self._agent

    def get_store(self) -> LightningStore:
        """Get the store."""
        if self._store is None:
            raise ValueError("Store not initialized. Call init_worker() first.")
        return self._store

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generates a standardized log prefix for the current worker."""
        if self._worker_id is not None:
            if rollout_id:
                return f"[Worker {self.worker_id} | Rollout {rollout_id}]"
            else:
                return f"[Worker {self.worker_id}]"
        if rollout_id:
            return f"[Rollout {rollout_id}]"
        return "[Default Worker]"

    def _post_process_rollout_result(self, rollout: RolloutV2, raw_result: RolloutRawResultV2) -> RolloutV2:
        """Standardizes the agent's return value into a Rollout object.

        Args:
            rollout: The rollout object for the current task.
            raw_result: The output from the agent's rollout method.

        Returns:
            A standardized `RolloutV2` object for reporting to the server.
        """

        # Case 1: result is a float (final reward)
        if isinstance(raw_result, float):
            # This will actually reward a span to the tracer, but tracer has already stopped so it doesn't matter
            reward_span = emit_reward(raw_result)
        # Case 2: result is a list of ReadableSpan (OpenTelemetry spans)
        if isinstance(raw_result, list) and all(isinstance(t, ReadableSpan) for t in raw_result):
            return RolloutV2(trace=raw_result)
        # Case 3: result is a list of dict (trace JSON)

    async def _sleep_until_next_poll(self, event: Event) -> None:
        """Sleep until the next poll interval."""
        current_time = time.time()
        next_time = current_time + self._poll_interval
        while time.time() < next_time:
            await asyncio.sleep(0.1)
            if event.is_set():
                return

    async def _iter_step(self, event: Event) -> None:
        store = self.get_store()
        agent = self.get_agent()

        next_rollout: Optional[RolloutV2] = None
        while not event.is_set():
            logger.debug(f"{self._log_prefix()} Try to poll for next rollout.")
            next_rollout = await store.dequeue_rollout()
            if next_rollout is None:
                logger.debug(f"{self._log_prefix()} No rollout to poll. Waiting for {self._poll_interval} seconds.")
                await self._sleep_until_next_poll(event)

        if next_rollout is None:
            return

        rollout_id = next_rollout.rollout_id

        resources_id = next_rollout.resources_id
        resources_update = None
        if resources_id:
            resources_update = await store.get_resources_by_id(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = await store.get_latest_resources()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return

        try:
            for hook in self._hooks:
                try:
                    await hook.on_rollout_start(next_rollout, self, self._tracer)
                except Exception:
                    logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook {hook}.")

            start_time = time.time()
            with self._tracer.trace_context(name=rollout_id):
                rollout_method = agent.training_rollout if next_rollout.mode == "train" else agent.validation_rollout
            result = rollout_method(next_rollout.input, resources=resources_update.resources, rollout=next_rollout)
            rollout_obj = self._to_rollout_object(result, task.rollout_id)
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
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            self.client.post_rollout(rollout_obj)

        return True

    async def iter(self, store: LightningStore, event: Event) -> None:
        """Run the runner, iterate over the tasks in the store. Abort if the event is set."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self._max_tasks or 'unlimited'}).")

        while self._max_tasks is None or num_tasks_processed < self._max_tasks:

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")
        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def step(self, input: T_task, store: LightningStore, event: Event) -> None:
        """Step the runner, execute one task."""
        store = self.get_store()
        agent = self.get_agent()

        attempted_rollout = await store.start_rollout(input=input)

        if agent.is_async():
            await agent.roll(input=input, resources=attempted_rollout.resources)
        else:
            await agent.training_rollout(input=input, resources=attempted_rollout.resources)

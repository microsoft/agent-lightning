# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, TypeVar, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward, get_last_reward
from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import AttemptedRollout, Hook, RolloutRawResultV2, RolloutV2, Span

if TYPE_CHECKING:
    from agentlightning.execution.events import Event

from .base import BaseRunner

T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


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

    def get_worker_id(self) -> str:
        """Get the worker id."""
        return f"Worker-{self._worker_id}" if self._worker_id is not None else "Worker-Unknown"

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

    async def _trigger_hooks(
        self,
        hook_type: Literal["on_trace_start", "on_trace_end", "on_rollout_start", "on_rollout_end"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Trigger the hooks."""
        for hook in self._hooks:
            try:
                await getattr(hook, hook_type)(*args, **kwargs)
            except Exception:
                logger.exception(f"{self._log_prefix()} Exception during {hook_type} hook {hook}.")

    async def _post_process_rollout_result(
        self, rollout: AttemptedRollout, raw_result: RolloutRawResultV2
    ) -> List[ReadableSpan] | List[Span]:
        """Standardizes the agent's return value and report what's needed to report to the store.

        Args:
            rollout: The rollout object for the current task.
            raw_result: The output from the agent's rollout method.

        Returns:
            The spans that are assumed to be added to the store.
            This only serves as an estimation for logging purposes. For precise tracking, use the store directly.
        """
        store = self.get_store()

        trace_spans: list[ReadableSpan] | list[Span] = []

        # Case 0: result is None
        if raw_result is None:
            trace_spans = self._tracer.get_last_trace()

        # Case 1: result is a float (final reward)
        if isinstance(raw_result, float):
            # This will emit another span to the tracer
            reward_span = emit_reward(raw_result)
            await store.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, reward_span)
            trace_spans = self._tracer.get_last_trace() + [reward_span]

        if isinstance(raw_result, list):
            # For rollout methods that return a list, we assume that the returned spans
            # are the complete span set from the whole rollout
            trace_spans = raw_result

            # Case 2: result is a list of ReadableSpan (OpenTelemetry spans)
            if len(raw_result) > 0 and all(isinstance(t, ReadableSpan) for t in raw_result):

                if not isinstance(
                    self._tracer, AgentOpsTracer
                ):  # TODO: this should be replaced with general OpenTelemetry tracer in next version
                    for span in raw_result:
                        await store.add_otel_span(
                            rollout.rollout_id, rollout.attempt.attempt_id, cast(ReadableSpan, span)
                        )
                else:
                    logger.warning(
                        f"{self._log_prefix(rollout.rollout_id)} Tracer is already an OpenTelemetry tracer. "
                        "The traces should have already been added to the store. "
                        "No need to return anything from rollout."
                    )

            # Case 3: result is a list of Span (agentlightning spans)
            elif len(raw_result) > 0 and all(isinstance(t, Span) for t in raw_result):
                # Add the spans directly to the store
                for span in raw_result:
                    await store.add_span(cast(Span, span))
                trace_spans = raw_result

            # Left over cases for list
            elif len(raw_result) == 0:
                logger.warning(
                    f"{self._log_prefix(rollout.rollout_id)} The rollout returns an empty list. "
                    "Please check your rollout implementation."
                )
                trace_spans = raw_result

            else:
                types = [type(t).__name__ for t in raw_result][:10]
                raise ValueError(
                    f"Invalid raw result type. It's expected to be a list of ReadableSpan or Span, "
                    f"but got: {', '.join(types)}..."
                )

        return trace_spans

    async def _sleep_until_next_poll(self, event: Event) -> None:
        """Sleep until the next poll interval."""
        current_time = time.time()
        next_time = current_time + self._poll_interval
        while time.time() < next_time:
            await asyncio.sleep(0.1)
            if event.is_set():
                return

    async def _step_impl(self, next_rollout: AttemptedRollout, event: Event) -> None:
        store = self.get_store()
        agent = self.get_agent()

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

        trace_spans: List[ReadableSpan] | List[Span] = []
        has_exception: bool = False

        try:
            await self._trigger_hooks(hook_type="on_rollout_start", agent=agent, runner=self, rollout=next_rollout)

            start_time = time.time()
            with self._tracer.trace_context(
                name=rollout_id, store=store, rollout_id=rollout_id, attempt_id=next_rollout.attempt.attempt_id
            ):
                await self._trigger_hooks(
                    hook_type="on_trace_start", agent=agent, runner=self, tracer=self._tracer, rollout=next_rollout
                )

                # NOTE: This is the most costly step in the whole function
                # If the rollout method becomes unresponsive or timeouts, there is nothing we can do within the runner.
                # We might need some mechanisms in execution strategy to restart the runner. But that's a future work.
                if agent.is_async:
                    rollout_method = (
                        agent.training_rollout_async if next_rollout.mode == "train" else agent.validation_rollout_async
                    )
                    result = await rollout_method(
                        next_rollout.input, resources=resources_update.resources, rollout=next_rollout
                    )
                else:
                    rollout_method = (
                        agent.training_rollout if next_rollout.mode == "train" else agent.validation_rollout
                    )
                    result = rollout_method(
                        next_rollout.input, resources=resources_update.resources, rollout=next_rollout
                    )

                await self._trigger_hooks(
                    hook_type="on_trace_end", agent=agent, runner=self, tracer=self._tracer, rollout=next_rollout
                )

            trace_spans = await self._post_process_rollout_result(next_rollout, result)
            last_reward = get_last_reward(trace_spans)

            end_time = time.time()
            logger.info(
                f"{self._log_prefix(rollout_id)} Completed in "
                f"{end_time - start_time:.2f}s. Triplet length: {len(trace_spans)}. "
                f"Final reward: {last_reward}"
            )

        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
            has_exception = True

        finally:
            try:
                await self._trigger_hooks(
                    hook_type="on_rollout_end", agent=agent, runner=self, rollout=next_rollout, spans=trace_spans
                )
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")

            if has_exception:
                # possibly timed out and cancelled?
                await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="failed")
            else:
                await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="succeeded")

    async def iter(self, event: Event) -> None:
        """Run the runner, iterate over the tasks in the store. Abort if the event is set."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self._max_tasks or 'unlimited'}).")
        store = self.get_store()

        while not event.is_set() and (self._max_tasks is None or num_tasks_processed < self._max_tasks):
            # Retrieve the next rollout
            next_rollout: Optional[RolloutV2] = None
            while not event.is_set():
                logger.debug(f"{self._log_prefix()} Try to poll for next rollout.")
                next_rollout = await store.dequeue_rollout()
                if next_rollout is None:
                    logger.debug(f"{self._log_prefix()} No rollout to poll. Waiting for {self._poll_interval} seconds.")
                    await self._sleep_until_next_poll(event)

            if next_rollout is None:
                return

            # Claim the rollout but updating the current worker id
            await store.update_attempt(
                next_rollout.rollout_id, next_rollout.attempt.attempt_id, worker_id=self.get_worker_id()
            )

            # Execute the step
            await self._step_impl(next_rollout, event)

            num_tasks_processed += 1
            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self._max_tasks or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")

    async def step(self, input: T_task, event: Event) -> None:
        """Step the runner, execute one task."""
        attempted_rollout = await self.get_store().start_rollout(input=input)
        await self._step_impl(attempted_rollout, event)

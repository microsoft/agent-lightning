# Copyright (c) Microsoft. All rights reserved.

"""Agent runner implementation for executing agent rollouts.

This module provides the concrete implementation of the runner interface,
handling the execution of agent rollouts with support for tracing, hooks,
and distributed worker coordination.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, TypeVar, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward, get_last_reward
from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import (
    AttemptedRollout,
    Hook,
    NamedResources,
    RolloutMode,
    RolloutRawResultV2,
    RolloutV2,
    Span,
)

if TYPE_CHECKING:
    from agentlightning.execution.events import Event

from .base import BaseRunner

T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


class AgentRunnerV2(BaseRunner[T_task]):
    """Runner implementation for executing agent tasks with distributed support.

    This runner manages the complete lifecycle of agent rollout execution,
    including task polling, resource management, tracing, and hooks. It supports
    both continuous iteration over tasks from the store and single-step execution.

    Attributes:
        worker_id: The unique identifier for this worker process.
    """

    def __init__(self, tracer: BaseTracer, max_rollouts: Optional[int] = None, poll_interval: float = 5.0) -> None:
        """Initialize the agent runner.

        Args:
            tracer: The tracer instance for recording execution traces and spans.
            max_rollouts: Maximum number of tasks to process in iter() mode. If None,
                the runner will continue indefinitely until interrupted.
            poll_interval: Time in seconds to wait between polling attempts when
                no tasks are available in the store.
        """
        super().__init__()
        self._tracer = tracer
        self._max_rollouts = max_rollouts
        self._poll_interval = poll_interval

        # Set later
        self._agent: Optional[LitAgent[T_task]] = None
        self._hooks: Sequence[Hook] = []
        self._store: Optional[LightningStore] = None
        self.worker_id: Optional[int] = None

    def init(self, agent: LitAgent[T_task], *, hooks: Optional[Sequence[Hook]] = None, **kwargs: Any) -> None:
        """Initialize the runner with the agent.

        This sets up the agent-runner relationship, registers hooks, and
        initializes the tracer.

        Args:
            agent: The LitAgent instance to be managed by this runner.
            hooks: Optional sequence of Hook objects to be called at various
                lifecycle stages (on_trace_start, on_trace_end, on_rollout_start,
                on_rollout_end).
            **kwargs: Additional initialization arguments (currently unused).
        """
        self._agent = agent
        self._agent.set_runner(self)
        self._hooks = [*hooks] if hooks is not None else []

        self._tracer.init()

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store.

        This method is called once per worker in a distributed setup to provide
        the worker with its ID and store connection.

        Args:
            worker_id: Unique identifier for this worker process.
            store: The LightningStore instance for task coordination and data persistence.
            **kwargs: Additional worker-specific initialization arguments (currently unused).
        """
        self._store = store
        self.worker_id = worker_id

        self._tracer.init_worker(worker_id)

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner and clean up all resources.

        This method resets all internal state including the agent, store,
        hooks, and worker ID, and calls the tracer's teardown method.

        Args:
            *args: Additional teardown arguments (currently unused).
            **kwargs: Additional teardown keyword arguments (currently unused).
        """
        self._agent = None
        self._store = None
        self.worker_id = None
        self._hooks = []

        self._tracer.teardown()

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        """Teardown the runner for a specific worker.

        This method cleans up worker-specific resources and resets the worker ID.

        Args:
            worker_id: The unique identifier of the worker being torn down.
            *args: Additional teardown arguments (currently unused).
            **kwargs: Additional teardown keyword arguments (currently unused).
        """
        self.worker_id = None

        self._tracer.teardown_worker(worker_id)

    def get_agent(self) -> LitAgent[T_task]:
        """Get the agent instance.

        Returns:
            The LitAgent instance managed by this runner.

        Raises:
            ValueError: If the agent has not been initialized via init().
        """
        if self._agent is None:
            raise ValueError("Agent not initialized. Call init() first.")
        return self._agent

    def get_store(self) -> LightningStore:
        """Get the store instance.

        Returns:
            The LightningStore instance for this worker.

        Raises:
            ValueError: If the store has not been initialized via init_worker().
        """
        if self._store is None:
            raise ValueError("Store not initialized. Call init_worker() first.")
        return self._store

    def get_worker_id(self) -> str:
        """Get the formatted worker ID string.

        Returns:
            A formatted string like "Worker-0" if initialized, or "Worker-Unknown"
            if the worker ID has not been set.
        """
        return f"Worker-{self.worker_id}" if self.worker_id is not None else "Worker-Unknown"

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generate a standardized log prefix for the current worker.

        This creates a consistent prefix format for log messages to identify
        which worker and rollout the message is associated with.

        Args:
            rollout_id: Optional rollout ID to include in the prefix.

        Returns:
            A formatted log prefix string like "[Worker 0 | Rollout xyz]",
            "[Worker 0]", "[Rollout xyz]", or "[Default Worker]".
        """
        if self.worker_id is not None:
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
        """Trigger all registered hooks of a specific type.

        This method calls the specified hook method on all registered hooks,
        catching and logging any exceptions that occur during hook execution
        to prevent them from disrupting the main execution flow.

        Args:
            hook_type: The type of hook to trigger. Valid values are:
                "on_trace_start", "on_trace_end", "on_rollout_start", "on_rollout_end".
            *args: Positional arguments to pass to the hook methods.
            **kwargs: Keyword arguments to pass to the hook methods.
        """
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
            # Preserve the existing spans before another span is emitted
            trace_spans = list(self._tracer.get_last_trace())
            # This will emit another span to the tracer
            reward_span = emit_reward(raw_result)
            await store.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, reward_span)
            trace_spans.append(reward_span)

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

    async def _sleep_until_next_poll(self, event: Optional[Event] = None) -> None:
        """Sleep until the next poll interval, with optional event-based interruption.

        If an event is provided, the method will check it periodically (every 0.1s)
        and return early if the event is set.

        Args:
            event: Optional Event object that can be used to interrupt the sleep.
                If set during the sleep period, the method returns immediately.
        """
        if event is None:
            await asyncio.sleep(self._poll_interval)
            return
        current_time = time.time()
        next_time = current_time + self._poll_interval
        while time.time() < next_time:
            await asyncio.sleep(0.1)
            if event.is_set():
                return

    async def _step_impl(self, next_rollout: AttemptedRollout, raise_on_exception: bool = False) -> None:
        """Execute a single rollout implementation.

        This is the core method that handles the execution of a single rollout,
        including resource fetching, hook triggering, agent invocation, tracing,
        and result processing.

        Args:
            next_rollout: The rollout to execute, containing input data, mode,
                and resources information.
            raise_on_exception: If True, exceptions during rollout execution will
                be re-raised. If False, exceptions are logged but not propagated.
        """
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
            if raise_on_exception:
                raise RuntimeError(f"{self._log_prefix(rollout_id)} Failed to fetch resources")
            else:
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

            # Possible exceptions in post_process will be caught in the overall exception handler
            trace_spans = await self._post_process_rollout_result(next_rollout, result)
            last_reward = get_last_reward(trace_spans)

            end_time = time.time()
            logger.info(
                f"{self._log_prefix(rollout_id)} Completed in "
                f"{end_time - start_time:.2f}s. Collected {len(trace_spans)} span(s). "
                f"Final reward: {last_reward}"
            )

        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
            has_exception = True

            if raise_on_exception:
                raise
        finally:
            try:
                await self._trigger_hooks(
                    hook_type="on_rollout_end", agent=agent, runner=self, rollout=next_rollout, spans=trace_spans
                )
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")

            try:
                if has_exception:
                    # possibly timed out and cancelled?
                    await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="failed")
                else:
                    await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="succeeded")
            except Exception:
                logger.exception(
                    f"{self._log_prefix(rollout_id)} Exception during update_attempt. Giving up the update."
                )

    async def iter(self, *, event: Optional[Event] = None) -> None:
        """Run the runner, continuously iterating over tasks in the store.

        This method polls the store for new rollouts and executes them until:
        - The event is set (if provided)
        - The max_rollouts limit is reached (if configured)
        - No more tasks are available

        All exceptions during rollout execution are caught and logged but not
        propagated, allowing the runner to continue processing subsequent tasks.

        Args:
            event: Optional Event object to signal the runner to stop. The runner
                will check this event periodically and stop gracefully when set.
        """
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self._max_rollouts or 'unlimited'}).")
        store = self.get_store()

        while not (event is not None and event.is_set()) and (
            self._max_rollouts is None or num_tasks_processed < self._max_rollouts
        ):
            # Retrieve the next rollout
            next_rollout: Optional[RolloutV2] = None
            while not (event is not None and event.is_set()):
                logger.debug(f"{self._log_prefix()} Try to poll for next rollout.")
                next_rollout = await store.dequeue_rollout()
                if next_rollout is None:
                    logger.debug(f"{self._log_prefix()} No rollout to poll. Waiting for {self._poll_interval} seconds.")
                    await self._sleep_until_next_poll(event)
                else:
                    break

            if next_rollout is None:
                return

            try:
                # Claim the rollout but updating the current worker id
                await store.update_attempt(
                    next_rollout.rollout_id, next_rollout.attempt.attempt_id, worker_id=self.get_worker_id()
                )
            except Exception:
                logger.exception(f"{self._log_prefix()} Exception during update_attempt, giving up the rollout.")
                continue

            # Execute the step
            await self._step_impl(next_rollout)

            num_tasks_processed += 1
            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self._max_rollouts or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[NamedResources] = None,
        mode: Optional[RolloutMode] = None,
        event: Optional[Event] = None,
    ) -> None:
        """Execute a single task directly, bypassing the task queue.

        This method creates a new rollout for the given input and executes it
        immediately. Unlike iter(), exceptions are propagated to the caller.

        Args:
            input: The task input to be processed by the agent.
            resources: Optional named resources to be used for this specific task.
                If provided, a new resources entry will be created in the store.
                If not provided, the latest resources from the store will be used.
            mode: Optional rollout mode ("train" or "validation"). If not provided,
                the agent's default mode will be used.
            event: Optional Event object to signal interruption (currently unused
                but included for interface consistency).

        Raises:
            Exception: Any exception that occurs during rollout execution will be
                re-raised to the caller.
        """
        store = self.get_store()

        if resources is not None:
            # TODO: move this to store.add_resources()
            resources_id = "resource-" + str(uuid.uuid4())
            await store.update_resources(resources_id=resources_id, resources=resources)
        else:
            resources_id = None

        attempted_rollout = await self.get_store().start_rollout(input=input, mode=mode, resources_id=resources_id)
        await self._step_impl(attempted_rollout, raise_on_exception=True)

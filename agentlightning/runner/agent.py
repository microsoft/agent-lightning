# Copyright (c) Microsoft. All rights reserved.

"""Agent runner implementation for executing agent rollouts.

This module provides the concrete implementation of the runner interface,
handling the execution of agent rollouts with support for tracing, hooks,
and distributed worker coordination.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward, find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import Tracer
from agentlightning.types import (
    AttemptedRollout,
    Hook,
    NamedResources,
    Rollout,
    RolloutMode,
    RolloutRawResult,
    Span,
)
from agentlightning.utils.system_snapshot import system_snapshot

if TYPE_CHECKING:
    from agentlightning.execution.events import ExecutionEvent

from .base import Runner

T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


class LitAgentRunner(Runner[T_task]):
    """Execute [`LitAgent`][agentlightning.LitAgent] tasks with tracing support.

    This runner manages the complete lifecycle of agent rollout execution,
    including task polling, resource management, tracing, and hooks. It supports
    both continuous iteration over tasks from the store and single-step execution.

    Attributes:
        worker_id: Identifier for the active worker process, if any.
    """

    def __init__(
        self,
        tracer: Tracer,
        max_rollouts: Optional[int] = None,
        poll_interval: float = 5.0,
        heartbeat_interval: float = 10.0,
        interval_jitter: float = 0.1,
        heartbeat_launch_mode: Literal["asyncio", "thread"] = "asyncio",
    ) -> None:
        """Initialize the agent runner.

        Args:
            tracer: [`Tracer`][agentlightning.Tracer] used for rollout spans.
            max_rollouts: Optional cap on iterations processed by
                [`iter`][agentlightning.LitAgentRunner.iter].
            poll_interval: Seconds to wait between store polls when no work is available.
            heartbeat_interval: Seconds to wait between sending heartbeats to the store.
            interval_jitter: Jitter factor for the poll interval. The actual interval will be between
                poll_interval - interval_jitter and poll_interval + interval_jitter.
                This is to avoid the overload caused by the synchronization of the runners.
            heartbeat_launch_mode: Launch mode for the heartbeat loop. Can be "asyncio" or "thread".
                "asyncio" is the default and recommended mode. Use "thread" if you are experiencing blocking coroutines.
        """
        super().__init__()
        self._tracer = tracer
        self._max_rollouts = max_rollouts
        self._poll_interval = poll_interval
        self._heartbeat_interval = heartbeat_interval
        self._interval_jitter = interval_jitter
        self._heartbeat_launch_mode = heartbeat_launch_mode
        self._random_state = random.Random()

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
            agent: [`LitAgent`][agentlightning.LitAgent] instance executed by the runner.
            hooks: Optional sequence of [`Hook`][agentlightning.Hook]
                callbacks invoked around tracing and rollout boundaries.
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
            store: [`LightningStore`][agentlightning.LightningStore]
                used for task coordination and persistence.
            **kwargs: Additional worker-specific initialization arguments (currently unused).
        """
        self._store = store
        self.worker_id = worker_id

        self._tracer.init_worker(worker_id, store)

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
            worker_id: Unique identifier of the worker being torn down.
            *args: Additional teardown arguments (currently unused).
            **kwargs: Additional teardown keyword arguments (currently unused).
        """
        self.worker_id = None

        self._tracer.teardown_worker(worker_id)

    @property
    def tracer(self) -> Tracer:
        """Get the tracer instance.

        Returns:
            The Tracer instance used by this runner.
        """
        return self._tracer

    def get_agent(self) -> LitAgent[T_task]:
        """Get the agent instance.

        Returns:
            The LitAgent instance managed by this runner.

        Raises:
            ValueError: If the agent has not been initialized via [`init`][agentlightning.LitAgentRunner.init].
        """
        if self._agent is None:
            raise ValueError("Agent not initialized. Call init() first.")
        return self._agent

    def get_store(self) -> LightningStore:
        """Get the store instance.

        Returns:
            The LightningStore instance for this worker.

        Raises:
            ValueError: If the store has not been initialized via [`init_worker`][agentlightning.LitAgentRunner.init_worker].
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
        self, rollout: AttemptedRollout, raw_result: RolloutRawResult
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
            # This will NOT emit another span to the tracer
            reward_span = emit_reward(raw_result, propagate=False)
            # We add it to the store manually
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

    async def _emit_heartbeat(self, store: LightningStore) -> None:
        """Send a heartbeat tick to the store."""
        worker_id = self.get_worker_id()

        try:
            await store.update_worker(worker_id, system_snapshot())
        except asyncio.CancelledError:
            # bypass the exception
            raise
        except Exception:
            logger.exception("%s Unable to update worker heartbeat.", self._log_prefix())

    def _start_heartbeat_loop(self, store: LightningStore) -> Optional[Callable[[], Awaitable[None]]]:
        """Start a background heartbeat loop and return an async stopper."""

        if self._heartbeat_interval <= 0:
            return None

        if self.worker_id is None:
            logger.warning("%s Cannot start heartbeat loop without worker_id.", self._log_prefix())
            return None

        if self._heartbeat_launch_mode == "asyncio":
            stop_event = asyncio.Event()

            async def heartbeat_loop() -> None:
                while not stop_event.is_set():
                    await self._emit_heartbeat(store)
                    with suppress(asyncio.TimeoutError):
                        interval = self._heartbeat_interval + self._random_state.uniform(
                            -self._interval_jitter, self._interval_jitter
                        )
                        interval = max(interval, 0.01)
                        await asyncio.wait_for(stop_event.wait(), timeout=interval)

            task = asyncio.create_task(heartbeat_loop(), name=f"{self.get_worker_id()}-heartbeat")

            async def stop() -> None:
                stop_event.set()
                with suppress(asyncio.CancelledError):
                    await task

            return stop

        if self._heartbeat_launch_mode == "thread":
            stop_evt = threading.Event()

            def thread_worker() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                while not stop_evt.is_set():
                    loop.run_until_complete(self._emit_heartbeat(store))
                    interval = self._heartbeat_interval + self._random_state.uniform(
                        -self._interval_jitter, self._interval_jitter
                    )
                    interval = max(interval, 0.01)
                    stop_evt.wait(interval)

            thread = threading.Thread(target=thread_worker, name=f"{self.get_worker_id()}-heartbeat", daemon=True)
            thread.start()

            async def stop() -> None:
                stop_evt.set()
                await asyncio.to_thread(thread.join)

            return stop

        raise ValueError(f"Unsupported heartbeat launch mode: {self._heartbeat_launch_mode}")

    async def _sleep_until_next_poll(self, event: Optional[ExecutionEvent] = None) -> None:
        """Sleep until the next poll interval, with optional event-based interruption.

        If an event is provided, the method will check it periodically (every 0.1s)
        and return early if the event is set.

        Args:
            event: Optional [`ExecutionEvent`][agentlightning.ExecutionEvent] object that can be used to interrupt the sleep.
                If set during the sleep period, the method returns immediately.
        """
        interval = self._poll_interval + self._random_state.uniform(-self._interval_jitter, self._interval_jitter)
        interval = max(interval, 0.01)
        if event is None:
            await asyncio.sleep(interval)
            return
        current_time = time.time()
        next_time = current_time + interval
        while time.time() < next_time:
            await asyncio.sleep(0.1)
            if event.is_set():
                return

    async def _step_impl(self, next_rollout: AttemptedRollout, raise_on_exception: bool = False) -> str:
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
                return rollout_id

        trace_spans: List[ReadableSpan] | List[Span] = []
        has_exception: bool = False

        try:
            await self._trigger_hooks(hook_type="on_rollout_start", agent=agent, runner=self, rollout=next_rollout)

            start_time = time.time()
            async with self._tracer.trace_context(
                name=rollout_id, rollout_id=rollout_id, attempt_id=next_rollout.attempt.attempt_id
            ):
                await self._trigger_hooks(
                    hook_type="on_trace_start", agent=agent, runner=self, tracer=self._tracer, rollout=next_rollout
                )

                # NOTE: This is the most costly step in the whole function
                # If the rollout method becomes unresponsive or timeouts, there is nothing we can do within the runner.
                # We might need some mechanisms in execution strategy to restart the runner. But that's a future work.
                if agent.is_async():
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
            last_reward = find_final_reward(trace_spans)

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

        return rollout_id

    async def iter(self, *, event: Optional[ExecutionEvent] = None) -> None:
        """Run the runner, continuously iterating over tasks in the store.

        This method polls the store for new rollouts and executes them until:

        - The event is set (if provided)
        - The max_rollouts limit is reached (if configured)
        - No more tasks are available

        All exceptions during rollout execution are caught and logged but not
        propagated, allowing the runner to continue processing subsequent tasks.

        Args:
            event: Optional ExecutionEvent object to signal the runner to stop. The runner
                will check this event periodically and stop gracefully when set.
        """
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self._max_rollouts or 'unlimited'}).")
        store = self.get_store()

        stop_heartbeat = self._start_heartbeat_loop(store)

        try:
            while not (event is not None and event.is_set()) and (
                self._max_rollouts is None or num_tasks_processed < self._max_rollouts
            ):
                # Retrieve the next rollout
                next_rollout: Optional[Rollout] = None
                while not (event is not None and event.is_set()):
                    logger.debug(f"{self._log_prefix()} Try to poll for next rollout.")
                    next_rollout = await store.dequeue_rollout(worker_id=self.get_worker_id())
                    if next_rollout is None:
                        logger.debug(
                            f"{self._log_prefix()} No rollout to poll. Waiting for {self._poll_interval} seconds."
                        )
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
                    # This exception could happen if the rollout is dequeued and the other end died for some reason
                    logger.exception(f"{self._log_prefix()} Exception during update_attempt, giving up the rollout.")
                    continue

                # Execute the step
                await self._step_impl(next_rollout)

                num_tasks_processed += 1
                if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                    logger.info(
                        f"{self._log_prefix()} Progress: {num_tasks_processed}/{self._max_rollouts or 'unlimited'}"
                    )
        finally:
            if stop_heartbeat is not None:
                await stop_heartbeat()

        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[NamedResources] = None,
        mode: Optional[RolloutMode] = None,
        event: Optional[ExecutionEvent] = None,
    ) -> Rollout:
        """Execute a single task directly, bypassing the task queue.

        This method creates a new rollout for the given input and executes it
        immediately. Unlike [`iter()`][agentlightning.LitAgentRunner.iter],
        exceptions are propagated to the caller.

        Args:
            input: The task input to be processed by the agent.
            resources: Optional named resources to be used for this specific task.
                If provided, a new resources entry will be created in the store.
                If not provided, the latest resources from the store will be used.
            mode: Optional rollout mode ("train" or "validation"). If not provided,
                the agent's default mode will be used.
            event: Optional ExecutionEvent object to signal interruption (currently unused
                but included for interface consistency).

        Returns:
            The completed rollout.

        Raises:
            Exception: Any exception that occurs during rollout execution will be
                re-raised to the caller.
        """
        store = self.get_store()

        if resources is not None:
            resources_update = await store.add_resources(resources)
            resources_id = resources_update.resources_id
        else:
            resources_id = None

        attempted_rollout = await self.get_store().start_rollout(input=input, mode=mode, resources_id=resources_id)
        # Register the attempt as running by the current worker
        await self.get_store().update_attempt(
            attempted_rollout.rollout_id,
            attempted_rollout.attempt.attempt_id,
            worker_id=self.get_worker_id(),
        )
        rollout_id = await self._step_impl(attempted_rollout, raise_on_exception=True)

        completed_rollout = await store.get_rollout_by_id(rollout_id)
        if completed_rollout is None:
            raise RuntimeError(f"{self._log_prefix()} Failed to fetch completed rollout by id after step: {rollout_id}")
        return completed_rollout

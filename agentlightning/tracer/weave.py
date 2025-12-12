# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import logging
import re
import weakref
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

import weave
from weave.trace_server import trace_server_interface as tsi

from agentlightning.instrumentation.weave import InMemoryWeaveTraceServer, instrument_weave, uninstrument_weave
from agentlightning.semconv import LightningResourceAttributes, WeaveSpanAttributes
from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus
from agentlightning.utils.otel import flatten_attributes, sanitize_attributes

from .base import Tracer

logger = logging.getLogger(__name__)


def op_name_to_func_name(op_name: str) -> str:
    """Convert a Weave operation name to a function name.

    Weave operation names look like this: `weave:///xxx/agentlightning.tracer.weave/op/openai.chat.completions.create:019b10be-...-44d74272569c`
    """
    match = re.match(r"/([^/:]+):", op_name)
    if match:
        return match.group(1)
    else:
        return op_name


class WeaveTracerManagedTraceServer(InMemoryWeaveTraceServer):
    """A managed trace server for WeaveTracer."""

    def __init__(
        self, call_start_callback: Callable[[tsi.CallSchema], None], call_end_callback: Callable[[tsi.CallSchema], None]
    ):
        super().__init__()
        self.call_start_callback = call_start_callback
        self.call_end_callback = call_end_callback

    def call_start(self, req: tsi.CallStartReq) -> tsi.CallStartRes:
        ret = super().call_start(req)
        if req.start.id in self.calls:
            self.call_start_callback(self.calls[req.start.id])
        return ret

    def call_end(self, req: tsi.CallEndReq) -> tsi.CallEndRes:
        ret = super().call_end(req)
        if req.end.id in self.calls:
            self.call_end_callback(self.calls[req.end.id])
        return ret


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:

    - Function/method calls
    - Input/Output data
    - Exceptions

    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.
    """

    def __init__(
        self, *, project_name: str | None = None, wandb_api_key: str | None = None, instrument_managed: bool = True
    ):
        """
        Initialize a WeaveTracer instance.

        Args:
            project_name: Optional project name for Weave; defaults to the current module name.
            wandb_api_key: Optional W&B API key.
            instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
        """
        super().__init__()
        self.project_name = project_name or __name__
        self.sequence_id = 0
        self._store: Optional[LightningStore] = None
        self.instrument_managed = instrument_managed
        self._server = WeaveTracerManagedTraceServer(
            call_start_callback=self.call_start_callback, call_end_callback=self.call_end_callback
        )

        self._default_sequence_counter: int = 0
        self._calls: Dict[str, tsi.CallSchema] = {}  # call_id -> call
        self._spans: List[Span] = []  # spans in the current trace
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None
        self._call_start_futures: Dict[str, asyncio.Future[int] | futures.Future[int]] = {}
        self._call_end_futures: List[asyncio.Future[None] | futures.Future[None]] = []
        self._loop: weakref.ReferenceType[asyncio.AbstractEventLoop] | None = None

    def instrument(self, worker_id: int):
        instrument_weave(self._server)

    def uninstrument(self, worker_id: int):
        uninstrument_weave()

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None):
        """
        Initialize the tracer for a worker thread/process.

        Args:
            worker_id: Identifier of the worker.
            store: Optional LightningStore for storing spans.
        """
        super().init_worker(worker_id, store)
        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")
        self._store = store

        try:
            import weave
        except ImportError as exc:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.") from exc

        # Optionally patch network calls to bypass real Weave/W&B endpoints
        if self.instrument_managed:
            self.instrument(worker_id)

        # Initialize the Weave client if not already initialized
        if weave.get_client() is None:  # type: ignore
            try:
                weave.init(project_name=self.project_name)  # type: ignore
                logger.info(f"[Worker {worker_id}] Weave client initialized.")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}'") from exc
        else:
            # FIXME
            pass

    def teardown_worker(self, worker_id: int):
        """
        Clean up tracer resources for the worker.

        Args:
            worker_id: Identifier of the worker.
        """
        super().teardown_worker(worker_id)

        if self.instrument_managed:
            self.uninstrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation removed.")

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Synchronous implementation of the tracing context.

        Args:
            name: Optional operation name.
            store: Optional LightningStore instance.
            rollout_id: Optional rollout ID.
            attempt_id: Optional attempt ID.

        Raises:
            ValueError: If store, rollout_id, and attempt_id are inconsistently provided.
            RuntimeError: If Weave is not installed or client is uninitialized.
        """

        if rollout_id is not None and attempt_id is not None:
            self._rollout_id = rollout_id
            self._attempt_id = attempt_id
        elif rollout_id is None and attempt_id is None:
            logger.warning("No rollout_id or attempt_id provided. Skipping writing to store.")
        else:
            raise ValueError("rollout_id and attempt_id must be either both provided or both None")

        await self._init_trace_context()

        try:
            arg_op = name or self.project_name
            arg_inputs: dict[str, str] = {}
            if rollout_id is not None:
                arg_inputs[LightningResourceAttributes.ROLLOUT_ID.value] = rollout_id
            if attempt_id is not None:
                arg_inputs[LightningResourceAttributes.ATTEMPT_ID.value] = attempt_id

            weave_client = weave.get_client()
            if not weave_client:
                raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

            # Create a new trace call object in Weave
            trace_call = weave_client.create_call(  # pyright: ignore[reportUnknownMemberType]
                op=arg_op, inputs=arg_inputs
            )

            try:
                yield trace_call
                # Finish trace even if no exception
                weave_client.finish_call(trace_call)  # pyright: ignore[reportUnknownMemberType]
            except Exception as exc:
                # Finish trace and log any exception
                weave_client.finish_call(trace_call, exception=exc)  # pyright: ignore[reportUnknownMemberType]
                logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={exc}")

            weave_client.flush()

            # It's possible that the call end futures are from a dedicated Weave thread pool,
            await asyncio.gather(*[asyncio.wrap_future(future) for future in self._call_end_futures])
            print("final call_end_futures", self._call_end_futures)

        finally:
            # Mandatory cleanup
            self._rollout_id = None
            self._attempt_id = None

    async def _init_trace_context(self) -> None:
        """Initialize the trace context."""
        self._spans.clear()
        self._calls.clear()
        self._rollout_id = None
        self._attempt_id = None
        self._call_start_futures.clear()
        self._call_end_futures.clear()
        print(id(asyncio.get_running_loop()))
        import threading

        print(threading.current_thread())
        self._loop = weakref.ref(asyncio.get_running_loop())

    def _ensure_loop(self) -> tuple[asyncio.AbstractEventLoop, bool]:
        """Returns a usable event loop and a boolean indicating whether it's the current running loop.

        Prefer using the main loop if it's possible. Otherwise, use the current running loop.
        """
        # Get the current running loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        # Get the main loop, which can be a different loop
        if self._loop is not None:
            main_loop = self._loop()
        else:
            main_loop = None

        if main_loop is not None:
            return main_loop, id(main_loop) == id(running_loop)
        elif running_loop is not None:
            return running_loop, True
        else:
            raise RuntimeError("No running event loop found. This should not happen.")

    def get_last_trace(self) -> List[Span]:
        return self._spans

    def call_start_callback(self, call: tsi.CallSchema) -> None:
        if call.id in self._call_start_futures:
            raise ValueError(f"Call {call.id} already has a start future")
        self._calls[call.id] = call

        # The callback must possibly be called from a dedicated Weave thread pool,
        # but it should be executed on the main event loop.
        try:
            loop, is_current_loop = self._ensure_loop()
            import threading

            print("call start", id(loop), threading.current_thread(), is_current_loop)
            if is_current_loop:
                task = loop.create_task(self.call_start_handler(call))
                self._call_start_futures[call.id] = task
            else:
                # Schedule the task on the dedicated loop
                task = asyncio.run_coroutine_threadsafe(self.call_start_handler(call), loop)
                self._call_start_futures[call.id] = task
            print("call_start_futures", self._call_start_futures)
        except Exception as exc:
            logger.exception(f"Error creating call start task: {exc}", exc_info=True)

    def call_end_callback(self, call: tsi.CallSchema) -> None:
        try:
            loop, is_current_loop = self._ensure_loop()
            import threading

            print("call end", id(loop), threading.current_thread(), is_current_loop)
            if is_current_loop:
                task = loop.create_task(self.call_finish_handler(call))
            else:
                task = asyncio.run_coroutine_threadsafe(self.call_finish_handler(call), loop)
            self._call_end_futures.append(task)
            print("call_end_futures", self._call_end_futures)
        except Exception as exc:
            logger.exception(f"Error creating call finish task: {exc}", exc_info=True)

    async def _get_next_sequence_id(self) -> int:
        if self._rollout_id and self._attempt_id and self._store:
            return await self._store.get_next_span_sequence_id(self._rollout_id, self._attempt_id)
        else:
            self._default_sequence_counter += 1
            return self._default_sequence_counter

    async def call_start_handler(self, call: tsi.CallSchema) -> int:
        """Handler called when a Weave Call starts.

        Args:
            call: The Weave Call object.

        Returns:
            The sequence ID for the call.
        """
        sequence_id = await self._get_next_sequence_id()
        return sequence_id

    async def call_finish_handler(self, call: tsi.CallSchema) -> None:
        """Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        # Make sure the corresponding call_start_future is complete
        if call.id in self._call_start_futures:
            sequence_id = await asyncio.wrap_future(self._call_start_futures[call.id])
            del self._call_start_futures[call.id]
        else:
            # Fetch a new sequence ID as the call_start is somehow missing
            logger.warning(f"Call {call.id} has no start future. Fetching a new sequence ID.")
            sequence_id = await self._get_next_sequence_id()

        self._calls[call.id] = call
        print("call_finish_handler", call)

        span = await self.convert_call_to_span(call, self._rollout_id, self._attempt_id, sequence_id)
        self._spans.append(span)
        if self._store and self._rollout_id and self._attempt_id:
            try:
                await self._store.add_span(span)
            except Exception as exc:
                logger.exception(f"Error adding span to store: {exc}")

    async def convert_call_to_span(
        self,
        call: tsi.CallSchema,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        sequence_id: Optional[int] = None,
    ) -> Span:
        """Convert a Weave Call (with nested children) into a Agent-lightning Span.

        `rollout_id` and `attempt_id` are required to attach the spans to the store.

        Args:
            call: The Weave Call object.
            rollout_id: Optional rollout ID to attach to spans.
            attempt_id: Optional attempt ID to attach to spans.
            sequence_id: Optional sequence ID to attach to spans.

        Returns:
            List of converted spans.
        """
        rollout_id = rollout_id or "rollout-dummy"
        attempt_id = attempt_id or "attempt-dummy"
        sequence_id = sequence_id or 0

        start_ts: float = call.started_at.timestamp()
        end_ts: Optional[float] = call.ended_at.timestamp() if call.ended_at else None

        if call.exception:
            status = TraceStatus(status_code="ERROR", description=call.exception)
        else:
            status = TraceStatus(status_code="OK")

        attributes: Dict[str, Any] = {
            WeaveSpanAttributes.WEAVE_OP_NAME.value: call.op_name,
        }
        if call.inputs:
            attributes[WeaveSpanAttributes.WEAVE_INPUT.value] = call.inputs
        if call.output:
            attributes[WeaveSpanAttributes.WEAVE_OUTPUT.value] = call.output
        if call.summary:
            attributes[WeaveSpanAttributes.WEAVE_SUMMARY.value] = call.summary
        if call.attributes:
            attributes[WeaveSpanAttributes.WEAVE_ATTRIBUTES.value] = call.attributes
        if call.exception:
            attributes[WeaveSpanAttributes.WEAVE_EXCEPTION.value] = call.exception

        sanitized_attributes = sanitize_attributes(flatten_attributes(attributes))

        context = SpanContext(
            trace_id=call.trace_id,
            span_id=call.id,
            is_remote=False,
            trace_state={},
        )

        # Get context for parent
        if call.parent_id:
            parent_call = self._calls.get(call.parent_id)
            if parent_call:
                parent_context = SpanContext(
                    trace_id=parent_call.trace_id,
                    span_id=parent_call.id,
                    is_remote=False,
                    trace_state={},
                )
            else:
                parent_context = None
        else:
            parent_context = None

        # Build the Span object
        return Span(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
            trace_id=call.trace_id,
            span_id=call.id,
            parent_id=call.parent_id,
            name=op_name_to_func_name(call.op_name),
            status=status,
            attributes=sanitized_attributes,
            events=[],  # Weave calls do not generate events
            links=[],  # Weave calls do not generate links
            start_time=start_ts,
            end_time=end_ts,
            context=context,
            parent=parent_context,
            resource=OtelResource(
                attributes={
                    LightningResourceAttributes.ROLLOUT_ID.value: rollout_id,
                    LightningResourceAttributes.ATTEMPT_ID.value: attempt_id,
                    LightningResourceAttributes.SPAN_SEQUENCE_ID.value: sequence_id,
                    LightningResourceAttributes.TRACER_NAME.value: "weave",
                },
                schema_url="",
            ),
        )

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import logging
import os
import re
import weakref
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    cast,
)

import weave
from opentelemetry.semconv.attributes import exception_attributes
from weave.trace.call import Call
from weave.trace.settings import UserSettings
from weave.trace.weave_client import WeaveClient
from weave.trace_server import trace_server_interface as tsi
from weave.wandb_interface.context import set_wandb_api_context

from agentlightning.instrumentation.weave import InMemoryWeaveTraceServer, instrument_weave, uninstrument_weave
from agentlightning.semconv import LightningResourceAttributes, LightningSpanAttributes
from agentlightning.store.base import LightningStore
from agentlightning.types import (
    Attributes,
    OtelResource,
    Span,
    SpanContext,
    SpanCoreFields,
    SpanRecordingContext,
    StatusCode,
    TraceStatus,
)
from agentlightning.utils.id import generate_id
from agentlightning.utils.otel import (
    filter_and_unflatten_attributes,
    flatten_attributes,
    format_exception_attributes,
    sanitize_attributes,
)

from .base import Tracer, with_active_tracer_context

logger = logging.getLogger(__name__)


def op_name_to_func_name(op_name: str) -> str:
    """Convert a Weave operation name to a function name.

    Weave operation names look like this: `weave:///xxx/agentlightning.tracer.weave/op/openai.chat.completions.create:019b10be-...-44d74272569c`
    """
    match = re.search(r"/([^/:]+):", op_name)
    if match:
        return match.group(1)
    else:
        return op_name


def random_project_name() -> str:
    return "agl/weave-" + generate_id(12)


def get_timestamp_or_throw(date: Optional[datetime], field_name: str) -> float:
    if date is None:
        raise ValueError(f"{field_name} is required but not set")
    return date.timestamp()


class WeaveSpanRecordingContext(SpanRecordingContext):
    """Universal interface for recording operations on a Weave call."""

    def __init__(self, call: Call) -> None:
        self._call = call

    def record_exception(self, exception: BaseException) -> None:
        self._call.exception = str(exception)
        self.record_status("ERROR", str(exception))
        self.record_attributes(format_exception_attributes(exception))

    def _get_input_from_attributes(self, attributes: Attributes) -> Dict[str, Any]:
        if LightningSpanAttributes.OPERATION_INPUT.value in attributes:
            # This can be a very rare case. If it happens, we can just let it throw.
            return cast(Dict[str, Any], attributes[LightningSpanAttributes.OPERATION_INPUT.value])
        else:
            filtered_attributes = filter_and_unflatten_attributes(
                attributes, LightningSpanAttributes.OPERATION_INPUT.value
            )
            if isinstance(filtered_attributes, list):
                return {str(i): v for i, v in enumerate(filtered_attributes)}
            else:
                return filtered_attributes

    def _get_output_from_attributes(self, attributes: Attributes) -> Any:
        if LightningSpanAttributes.OPERATION_OUTPUT.value in attributes:
            return attributes[LightningSpanAttributes.OPERATION_OUTPUT.value]
        else:
            return filter_and_unflatten_attributes(attributes, LightningSpanAttributes.OPERATION_OUTPUT.value)

    def record_attributes(self, attributes: Attributes) -> None:
        input_attributes = self._get_input_from_attributes(attributes)
        if input_attributes:
            self._call.inputs.update(input_attributes)

        output_attributes = self._get_output_from_attributes(attributes)
        if output_attributes:
            if self._call.output is not None:
                logger.warning(f"Output is already set. It will be overridden: {self._call.output}")
            self._call.output = output_attributes

        if LightningSpanAttributes.OPERATION_NAME.value in attributes:
            logger.error(
                f"Cannot record operation name as an attribute. It will be skipped: {attributes[LightningSpanAttributes.OPERATION_NAME.value]}"
            )

        # The rest of the attributes are recorded as summary.
        for key, value in attributes.items():
            if (
                not key == LightningSpanAttributes.OPERATION_INPUT.value
                and not key.startswith(LightningSpanAttributes.OPERATION_INPUT.value + ".")
                and not key == LightningSpanAttributes.OPERATION_OUTPUT.value
                and not key.startswith(LightningSpanAttributes.OPERATION_OUTPUT.value + ".")
                and not key == LightningSpanAttributes.OPERATION_NAME.value
            ):
                if self._call.summary is None:
                    self._call.summary = {}
                self._call.summary[key] = value

    def record_status(self, status_code: StatusCode, description: Optional[str] = None) -> None:
        if status_code == "ERROR":
            if not description:
                raise ValueError("Description is required when status code is ERROR")
            self._call.exception = description
        elif status_code == "OK":
            self._call.exception = None
        # Do nothing for other status codes.

    def finalize(self) -> None:
        # Do nothing
        pass

    def get_recorded_span(self) -> SpanCoreFields:
        return SpanCoreFields(
            name=self._call.op_name,
            attributes=flatten_attributes(self._call.attributes or {}),
            start_time=self._call.started_at.timestamp() if self._call.started_at else None,
            end_time=self._call.ended_at.timestamp() if self._call.ended_at else None,
            status=TraceStatus(
                status_code="OK" if self._call.exception is None else "ERROR", description=self._call.exception
            ),
        )


class WeaveTracerManagedTraceServer(InMemoryWeaveTraceServer):
    """A managed trace server for WeaveTracer."""

    def __init__(
        self,
        partial_call_callback: Callable[[Dict[str, Any]], None],
        complete_call_callback: Callable[[tsi.CallSchema], None],
    ):
        super().__init__()
        self.partial_call_callback = partial_call_callback
        self.complete_call_callback = complete_call_callback

    def trigger_callbacks(self, call_id: str) -> None:
        with self._call_threading_lock:
            if call_id in self.calls:
                self.complete_call_callback(self.calls[call_id])
            elif call_id in self.partial_calls:
                self.partial_call_callback(self.partial_calls[call_id])
            else:
                logger.error(f"Call {call_id} not found in partial_calls or calls")

    def call_start(self, req: tsi.CallStartReq) -> tsi.CallStartRes:
        try:
            ret = super().call_start(req)
            self.trigger_callbacks(ret.id)
            return ret
        except Exception:
            logger.exception(f"Error calling call_start: {req}", exc_info=True)
            raise

    def call_end(self, req: tsi.CallEndReq) -> tsi.CallEndRes:
        try:
            ret = super().call_end(req)
            self.trigger_callbacks(req.end.id)
            return ret
        except Exception:
            logger.exception(f"Error calling call_end: {req}", exc_info=True)
            raise


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:

    - Function/method calls
    - Input/Output data
    - Exceptions

    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.
    """

    def __init__(
        self,
        *,
        project_name: str | None = None,
        weave_user_settings: UserSettings | None = None,
        instrument_managed: bool = True,
    ):
        """Initialize a WeaveTracer instance.

        Args:
            project_name: Optional project name for Weave; defaults to the current module name.
            weave_user_settings: Optional UserSettings for Weave.
            instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
        """
        super().__init__()
        self.project_name = project_name
        self.instrument_managed = instrument_managed
        self.weave_user_settings = weave_user_settings or UserSettings(use_server_cache=False)

        self._store: Optional[LightningStore] = None
        self._server = WeaveTracerManagedTraceServer(
            partial_call_callback=self.partial_call_callback, complete_call_callback=self.complete_call_callback
        )

        self._default_sequence_counter: int = 0
        self._calls: Dict[str, tsi.CallSchema] = {}  # call_id -> call
        self._spans: List[Span] = []  # spans in the current trace
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None
        self._partial_call_futures: Dict[str, asyncio.Future[int] | futures.Future[int]] = {}
        self._complete_call_futures: List[asyncio.Future[None] | futures.Future[None]] = []
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

        # Optionally patch network calls to bypass real Weave/W&B endpoints
        if self.instrument_managed:
            self.instrument(worker_id)

        # If WANDB_API_KEY is not set, we need to initialize Weave with a hack
        if not os.getenv("WANDB_API_KEY"):
            logger.info("WANDB_API_KEY is not set. Initializing Weave a mock context.")
            set_wandb_api_context("agl", api_key=None, headers=None, cookies=None)
        else:
            logger.debug("WANDB_API_KEY is set. Weave will be initialized automatically.")

        weave_client = weave.get_client()
        if self.project_name is None:
            self.project_name = random_project_name()

        if weave_client is not None:
            logger.warning("Weave client was already initialized. Reentrant calls are at your own risk.")
            if weave_client.project == self.project_name:
                logger.error(
                    f"Weave client was already initialized for the same project '{self.project_name}'. It's very likely that weave won't work correctly."
                )

        # Init no matter what
        try:
            weave.init(project_name=self.project_name, settings=self.weave_user_settings)
            logger.info(f"[Worker {worker_id}] Weave client initialized.")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}'") from exc

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

    @with_active_tracer_context
    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronous implementation of the tracing context.

        Args:
            name: Optional operation name.
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
            logger.info("No rollout_id or attempt_id provided. Skipping writing to store.")
            self._rollout_id = self._attempt_id = None
        else:
            raise ValueError("rollout_id and attempt_id must be either both provided or both None")

        await self._init_trace_context()

        weave_client = self._get_weave_client()

        if weave_client.server is not self._server:
            logger.error(
                "Weave client is not using the correct trace server. You might have multiple WeaveTracer instances running in the same process. "
                f"Expected {self._server}, got {weave_client.server}"
            )

        arg_op = name or weave_client.project
        arg_inputs: dict[str, str] = {}
        if rollout_id is not None:
            arg_inputs[LightningResourceAttributes.ROLLOUT_ID.value] = rollout_id
        if attempt_id is not None:
            arg_inputs[LightningResourceAttributes.ATTEMPT_ID.value] = attempt_id

        try:
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
                raise

        finally:
            try:
                weave_client.flush()
                # It's possible that the call end futures are from a dedicated Weave thread pool,
                await asyncio.gather(*[asyncio.wrap_future(future) for future in self._complete_call_futures])

            finally:
                # Mandatory cleanup
                self._rollout_id = None
                self._attempt_id = None

    def create_span(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        if timestamp is not None:
            logger.warning("Weave doesn't support customizing the start time of a call. Timestamp is ignored.")
        weave_client = self._get_weave_client()
        trace_call = weave_client.create_call(  # pyright: ignore[reportUnknownMemberType]
            op=name,
            attributes=attributes,
            inputs={},
        )
        # Immediately finish the call
        weave_client.finish_call(trace_call)  # pyright: ignore[reportUnknownMemberType]
        # We don't wait for the call to be propagated to the server.
        start_time = trace_call.started_at.timestamp() if trace_call.started_at else None
        end_time = trace_call.ended_at.timestamp() if trace_call.ended_at else None
        trace_status = (
            TraceStatus(status_code="OK")
            if trace_call.exception is None
            else TraceStatus(status_code="ERROR", description=trace_call.exception)
        )
        return SpanCoreFields(
            name=name,
            attributes=flatten_attributes(trace_call.attributes or {}),
            start_time=start_time,
            end_time=end_time,
            status=trace_status,
        )

    @contextmanager
    def operation_context(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[SpanRecordingContext]:
        if start_time is not None:
            logger.warning("Weave doesn't support customizing the start time of a call. Timestamp is ignored.")
        if end_time is not None:
            logger.warning("Weave doesn't support customizing the end time of a call. Timestamp is ignored.")
        weave_client = self._get_weave_client()
        trace_call = weave_client.create_call(  # pyright: ignore[reportUnknownMemberType]
            op=name,
            attributes=attributes,
            inputs={},
        )
        recording_context = WeaveSpanRecordingContext(trace_call)
        try:
            yield recording_context
        except Exception as exc:
            recording_context.record_exception(exc)
            raise
        finally:
            weave_client.finish_call(trace_call)  # pyright: ignore[reportUnknownMemberType]

    async def _init_trace_context(self) -> None:
        """Initialize the trace context."""
        self._spans.clear()
        self._calls.clear()
        self._partial_call_futures.clear()
        self._complete_call_futures.clear()
        self._loop = weakref.ref(asyncio.get_running_loop())

    def _get_weave_client(self) -> WeaveClient:
        """Get the Weave client."""
        weave_client = weave.get_client()
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")
        return weave_client

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

    def partial_call_callback(self, request_content: Dict[str, Any]) -> None:
        call_id = request_content.get("id")
        if call_id is None:
            raise ValueError("Call ID is required even for partial calls")

        if call_id in self._partial_call_futures:
            raise ValueError(f"Call {call_id} already has a start future")

        # The callback must possibly be called from a dedicated Weave thread pool,
        # but it should be executed on the main event loop.
        try:
            loop, is_current_loop = self._ensure_loop()
            if is_current_loop:
                task = loop.create_task(self.partial_call_handler(request_content))
            else:
                # Schedule the task on the dedicated loop
                task = asyncio.run_coroutine_threadsafe(self.partial_call_handler(request_content), loop)
            self._partial_call_futures[call_id] = task
        except Exception as exc:
            logger.exception(f"Error creating call start task: {exc}", exc_info=True)

    def complete_call_callback(self, call: tsi.CallSchema) -> None:
        try:
            loop, is_current_loop = self._ensure_loop()
            if is_current_loop:
                task = loop.create_task(self.complete_call_handler(call))
            else:
                # Schedule the task on the dedicated loop
                task = asyncio.run_coroutine_threadsafe(self.complete_call_handler(call), loop)
            self._complete_call_futures.append(task)
        except Exception as exc:
            logger.exception(f"Error creating call finish task: {exc}", exc_info=True)

    async def _get_next_sequence_id(self) -> int:
        """Get the next sequence ID for a span.

        Use store to get the next sequence ID if available, otherwise use a default counter.
        """
        if self._rollout_id and self._attempt_id and self._store:
            return await self._store.get_next_span_sequence_id(self._rollout_id, self._attempt_id)
        else:
            self._default_sequence_counter += 1
            return self._default_sequence_counter

    async def partial_call_handler(self, request_content: Dict[str, Any]) -> int:
        """Handler called when a Weave Call starts.

        Args:
            request_content: The partial Weave Call object.

        Returns:
            The sequence ID for the call.
        """
        sequence_id = await self._get_next_sequence_id()
        return sequence_id

    async def complete_call_handler(self, call: tsi.CallSchema) -> None:
        """Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        # Make sure the corresponding call_start_future is complete
        if call.id in self._partial_call_futures:
            sequence_id = await asyncio.wrap_future(self._partial_call_futures[call.id])
            del self._partial_call_futures[call.id]
        else:
            # Fetch a new sequence ID as the call_start is somehow missing
            logger.warning(f"Call {call.id} has no start future. Fetching a new sequence ID.")
            sequence_id = await self._get_next_sequence_id()

        self._calls[call.id] = call

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
            LightningSpanAttributes.OPERATION_NAME.value: call.op_name,
            # op_name can be possibly overridden by the attributes.
            **call.attributes,
        }
        if call.inputs:
            attributes[LightningSpanAttributes.OPERATION_INPUT.value] = call.inputs
        if call.output:
            attributes[LightningSpanAttributes.OPERATION_OUTPUT.value] = call.output
        if call.summary:
            # attributes can be possibly overridden by the summary.
            attributes.update(call.summary)
        if call.exception:
            attributes[exception_attributes.EXCEPTION_MESSAGE] = call.exception

        sanitized_attributes = sanitize_attributes(flatten_attributes(attributes, expand_leaf_lists=False))

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

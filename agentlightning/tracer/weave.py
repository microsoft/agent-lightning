# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from agentlightning.instrumentation.weave import instrument_weave, uninstrument_weave
from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus

from .base import Tracer

if TYPE_CHECKING:
    from weave.trace.call import Call

JSONPrimitive = Union[str, int, float, bool, None]

logger = logging.getLogger(__name__)


def generate_span_id() -> str:
    return "sp-" + hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:

    - Function/method calls
    - Input/Output data
    - Exceptions

    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.

    Attributes:
        project_name: Name of the Weave project. Used to initialize the Weave client.
        instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
    """

    def __init__(
        self, *, project_name: str | None = None, wandb_api_key: str | None = None, instrument_managed: bool = True
    ):
        """
        Initialize a WeaveTracer instance.

        Args:
            project_name: Optional project name for Weave; defaults to the current module name.
            wandb_api_key: Optional W&B API key; sets environment variable if provided.
            instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
        """
        super().__init__()
        self.project_name = project_name or __name__
        self.sequence_id = 0
        self._store: Optional[LightningStore] = None
        self.instrument_managed = instrument_managed

        self._default_sequence_counter: int = 0
        self._spans: List[Span] = []
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None

        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

    def instrument(self, worker_id: int):
        instrument_weave()

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
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

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
        arg_op = name or self.project_name
        arg_inputs: dict[str, str] | None = {"rollout_id": rollout_id or "", "attempt_id": attempt_id or ""}

        self._spans.clear()

        try:
            import weave
        except ImportError as exc:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.") from exc

        weave_client = weave.get_client()
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        if rollout_id is not None and attempt_id is not None:
            self._rollout_id = rollout_id
            self._attempt_id = attempt_id
        elif rollout_id is None and attempt_id is None:
            logger.warning("No rollout_id or attempt_id provided. Skipping writing to store.")
        else:
            raise ValueError("rollout_id and attempt_id must be either both provided or both None")

        # Create a new trace call object in Weave
        trace_call = weave_client.create_call(op=arg_op, inputs=arg_inputs)  # pyright: ignore[reportUnknownMemberType]
        trace_call.started_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            yield trace_call
        except Exception as e:
            # Finish trace and log any exception
            weave_client.finish_call(trace_call, exception=e)  # pyright: ignore[reportUnknownMemberType]
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            # Finish trace even if no exception
            weave_client.finish_call(trace_call)  # pyright: ignore[reportUnknownMemberType]
            await self._on_finish_handler(trace_call)

            self._rollout_id = None
            self._attempt_id = None

    def get_last_trace(self) -> List[Span]:
        return self._spans

    async def _on_finish_handler(self, call: Call, *args: Any, **kwargs: Any) -> None:
        """
        Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        spans = await self.convert_call_to_spans(call, self._rollout_id, self._attempt_id)
        self._spans.extend(spans)

        if self._store and self._rollout_id and self._attempt_id:
            try:
                await self._store.add_many_spans(spans)
            except Exception as e:
                logger.exception(f"Error adding span to store: {e}")

    async def _sequence_generator(self, request_count: int) -> Sequence[int]:
        if self._rollout_id and self._attempt_id and self._store:
            return await self._store.get_many_span_sequence_ids(
                [(self._rollout_id, self._attempt_id) for _ in range(request_count)]
            )
        else:
            ret = [self._default_sequence_counter + i for i in range(request_count)]
            self._default_sequence_counter += request_count
            return ret

    async def convert_call_to_spans(
        self,
        call: Call,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> List[Span]:
        """
        Recursively convert a Weave Call (with nested children) into a flat list of Agent Lightning Spans.

        Args:
            call: The Weave Call object.
            rollout_id: Optional rollout ID to attach to spans.
            attempt_id: Optional attempt ID to attach to spans.
            sequence_generator: Callable to generate sequence IDs.

        Returns:
            List of converted spans.
        """
        spans: List[Span] = []
        sequence_id = await self._sequence_generator(1)

        rollout_id = rollout_id or ""
        attempt_id = attempt_id or ""

        start_dt = getattr(call, "started_at", None)
        start_ts: Optional[float] = start_dt.timestamp() if start_dt else None

        end_dt = getattr(call, "ended_at", None)
        end_ts: Optional[float] = end_dt.timestamp() if end_dt else None

        trace_id = call.trace_id
        span_id = call.id or generate_span_id()
        parent_id = call.parent_id

        exception = getattr(call, "exception", None)
        status_code = "ERROR" if exception else "OK"

        def sanitize(
            inputs: Dict[str, Any],
            output: Dict[str, Any],
        ) -> Dict[str, str | JSONPrimitive]:
            stack: List[Tuple[Any, str]] = [
                (inputs or {}, "input"),
                (output or {}, "output"),
            ]

            attributes: Dict[str, str | JSONPrimitive] = {}

            while stack:
                value, key = stack.pop()

                if isinstance(value, dict):
                    for k, v in cast(Dict[str, Any], value).items():
                        stack.append((v, f"{key}.{k}"))
                elif isinstance(value, (list, tuple)):
                    for i, v in enumerate(cast(List[Any], value)):
                        stack.append((v, f"{key}.{i}"))
                else:
                    if value is None:
                        attributes[key] = "None"
                    elif isinstance(value, (str, int, float, bool)):
                        attributes[key] = value
                    else:
                        try:
                            attributes[key] = str(value)
                        except Exception:
                            attributes[key] = "None"

            return attributes

        inputs = call.inputs
        output = call.output
        attributes = sanitize(inputs, output)

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_state={},
        )

        parent_context = (
            SpanContext(
                trace_id=trace_id,
                span_id=parent_id,
                is_remote=False,
                trace_state={},
            )
            if parent_id
            else None
        )

        # Build the Span object
        span = Span(
            rollout_id=rollout_id or "",
            attempt_id=attempt_id or "",
            sequence_id=sequence_id[0],
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=call.func_name,
            status=TraceStatus(status_code=status_code),
            attributes=attributes,  # type: ignore
            events=[],  # Weave calls do not generate events
            links=[],  # Weave calls do not generate links
            start_time=start_ts,
            end_time=end_ts,
            context=context,
            parent=parent_context,
            resource=OtelResource(attributes={}, schema_url=""),
        )

        spans.append(span)

        children: List[Call] = call._children  # pyright: ignore[reportPrivateUsage]
        # Recursively process child calls
        for child in children:
            child_spans = await self.convert_call_to_spans(
                child,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
            )
            spans.extend(child_spans)

        return spans

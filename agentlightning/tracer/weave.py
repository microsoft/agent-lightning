# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from agentlightning.instrumentation import instrument_weave, uninstrument_weave
from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus

from .base import Tracer

if TYPE_CHECKING:
    from weave.trace.call import Call  # type: ignore

JSONPrimitive = Union[str, int, float, bool, None]

logger = logging.getLogger(__name__)


class WeaveTracer(Tracer):
    """
    Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:
    - Function/method calls
    - Input/Output data
    - Exceptions
    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.

    Attributes:
        project_name: Name of the Weave project. Used to initialize the Weave client.
        _store: Optional LightningStore instance for storing collected spans.
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
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}': {e}")

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
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
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

        if store is not None and rollout_id is not None and attempt_id is not None:
            self._rollout_id = rollout_id
            self._attempt_id = attempt_id
            self._store = store
        else:
            raise ValueError("store, rollout_id, and attempt_id must be either all provided")

        try:
            import datetime

            import weave
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

        weave_client = weave.get_client()  # type: ignore
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        # Create a new trace call object in Weave
        trace_call = weave_client.create_call(op=arg_op, inputs=arg_inputs)  # type: ignore
        trace_call.started_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            yield trace_call
        except Exception as e:
            # Finish trace and log any exception
            weave_client.finish_call(trace_call, exception=e)  # type: ignore
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            # Finish trace even if no exception
            weave_client.finish_call(trace_call)  # type: ignore
            await self._on_finish_handler(trace_call)  # type: ignore

    async def _on_finish_handler(self, call: "Call", *args: Any, **kwargs: Any) -> None:  # type: ignore
        """
        Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        spans, self.sequence_id = self.convert_call_to_spans(call, self._rollout_id, self._attempt_id, self.sequence_id)  # type: ignore

        if self._store and self._rollout_id and self._attempt_id:
            try:
                await self._store.add_many_spans(spans)
            except Exception as e:
                logger.exception(f"Error adding span to store: {e}")

    def convert_call_to_spans(
        self,
        call: "Call",  # type: ignore
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        seq_start: int = 0,
    ) -> tuple[List[Span], int]:
        """
        Recursively convert a Weave Call (with nested children) into a flat list of Agent Lightning Spans.

        Args:
            call: The Weave Call object.
            rollout_id: Optional rollout ID to attach to spans.
            attempt_id: Optional attempt ID to attach to spans.
            seq_start: Sequence number to start from.

        Returns:
            Tuple of (list_of_spans, next_sequence_id).
        """
        spans: List[Span] = []
        sequence_id = seq_start

        rollout_id = rollout_id or ""  # type: ignore
        attempt_id = attempt_id or ""  # type: ignore

        start_dt = getattr(call, "started_at", None)  # type: ignore
        start_ts: Optional[float] = start_dt.timestamp() if start_dt else None

        end_dt = getattr(call, "ended_at", None)  # type: ignore
        end_ts: Optional[float] = end_dt.timestamp() if end_dt else None

        trace_id = str(getattr(call, "trace_id", None))  # type: ignore
        span_id = str(getattr(call, "id", None))  # type: ignore
        parent_id = str(getattr(call, "parent_id", None)) if getattr(call, "parent_id", None) else None  # type: ignore

        exception = getattr(call, "exception", None)  # type: ignore
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
                    for k, v in value.items():  # type: ignore
                        stack.append((v, f"{key}.{k}"))  # type: ignore
                elif isinstance(value, (list, tuple)):
                    for i, v in enumerate(value):  # type: ignore
                        stack.append((v, f"{key}.{i}"))  # type: ignore
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

        inputs = getattr(call, "inputs", {})  # type: ignore
        output = getattr(call, "output", {})  # type: ignore
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
            sequence_id=sequence_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=getattr(call, "func_name", "unknown"),  # type: ignore
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
        sequence_id += 1

        children: List["Call"] = getattr(call, "_children", [])  # type: ignore
        # Recursively process child calls
        for child in children:  # type: ignore
            child_spans, sequence_id = self.convert_call_to_spans(  # type: ignore
                child,  # type: ignore
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                seq_start=sequence_id,
            )
            spans.extend(child_spans)

        return spans, sequence_id

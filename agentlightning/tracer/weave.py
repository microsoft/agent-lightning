from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Optional

import weave
from opentelemetry.sdk.trace import ReadableSpan, Resource
from opentelemetry.trace import (
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)

from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import LightningSpanProcessor

from .base import Tracer

if TYPE_CHECKING:
    from weave.client import WeaveClient

logger = logging.getLogger(__name__)


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context.
    It logs function calls, input/output, and exceptions to Weave Cloud (W&B backend).
    """

    def __init__(self, *, project_name: str | None = None):
        super().__init__()
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self.project_name = project_name or __name__
        self._client: Optional[WeaveClient] = None
        self._initialized = False

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        if self._initialized:
            logger.warning(f"[Worker {worker_id}] Weave client was already initialized.")
            return

        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")
        self._lightning_span_processor = LightningSpanProcessor()

        weave.init(project_name=self.project_name)
        self._client = weave.get_client()
        self._initialized = True

    def _convert_weave_call_to_readable_span(self, call) -> ReadableSpan:
        def _datetime_to_ns(dt: Optional[datetime]) -> int:
            return int((dt or datetime.utcnow()).timestamp() * 1e9)

        def _make_span_context(call) -> SpanContext:
            trace_id_int = uuid.UUID(call.trace_id).int & ((1 << 128) - 1)
            span_id_int = uuid.UUID(call.id).int & ((1 << 64) - 1)
            return SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )

        # Prepare context and timestamps
        span_context = _make_span_context(call)
        start_time = _datetime_to_ns(call.started_at)
        end_time = _datetime_to_ns(call.ended_at)

        # Flatten attributes to primitives only
        attributes = {
            "weave.project_id": call.project_id,
            "weave.op_name": str(getattr(call, "_op_name", None)),
            "weave.rollout_id": call.inputs.get("rollout_id") if call.inputs else None,
            "weave.status": "error" if call.exception else "success",
            "weave.output": json.dumps(call.output, ensure_ascii=False) if call.output else "",
            "weave.attributes_raw": json.dumps(dict(call.attributes), ensure_ascii=False) if call.attributes else "{}",
        }

        status = Status(StatusCode.ERROR, str(call.exception)) if call.exception else Status(StatusCode.OK)

        span = ReadableSpan(
            name=str(getattr(call, "_display_name", None) or call.id),
            context=span_context,
            parent=(_make_span_context(call) if getattr(call, "parent_id", None) else None),
            kind=0,
            resource=Resource.create({}),
            attributes=attributes,
            events=[],
            links=[],
            status=status,
            start_time=start_time,
            end_time=end_time,
            instrumentation_info=None,
        )

        # Recursively process children
        span._children = [self._convert_weave_call_to_readable_span(child) for child in getattr(call, "_children", [])]

        return span

    def teardown_worker(self, worker_id: int):
        super().teardown_worker(worker_id)
        if self._lightning_span_processor is not None:
            self._lightning_span_processor.shutdown()
            self._lightning_span_processor = None
        self._initialized = False

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[LightningSpanProcessor, None]:
        """Async version of the tracing context."""
        with self._trace_context_sync(
            name=name, store=store, rollout_id=rollout_id, attempt_id=attempt_id
        ) as processor:
            yield processor

    @contextmanager
    def _trace_context_sync(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Iterator[LightningSpanProcessor]:
        """Manual tracing context using Weave for synchronous execution."""
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")

        arg_op = name if name is not None else "weave_trace"
        arg_inputs = {
            "rollout_id": rollout_id if rollout_id is not None else "",
        }

        trace = self._client.create_call(op=arg_op, inputs=arg_inputs)
        try:
            if store and rollout_id and attempt_id:
                ctx = self._lightning_span_processor.with_context(
                    store=store, rollout_id=rollout_id, attempt_id=attempt_id
                )
                with ctx as processor:
                    yield processor
            elif store is None and rollout_id is None and attempt_id is None:
                with self._lightning_span_processor:
                    yield self._lightning_span_processor
            else:
                raise ValueError("store, rollout_id, and attempt_id must be either all provided or all None")
        except Exception as e:
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
            self._client.fail_call(trace, e)
        else:
            self._client.finish_call(trace)
        finally:
            self._client.finish()
            self._lightning_span_processor.on_end(self._convert_weave_call_to_readable_span(trace))

    def get_last_trace(self):
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

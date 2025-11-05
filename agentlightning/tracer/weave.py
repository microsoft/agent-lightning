from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, AsyncGenerator, Iterator, Optional

import weave
from opentelemetry.sdk.trace import ReadableSpan, Resource
from opentelemetry.trace import (
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)
from opentelemetry.trace.status import StatusCode

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

    def __init__(self, *, weave_managed: bool = True, instrument_managed: bool = True):
        super().__init__()
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self.weave_managed = weave_managed
        self.instrument_managed = instrument_managed
        self._client: Optional[WeaveClient] = None

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")

        self._lightning_span_processor = LightningSpanProcessor()

        try:
            weave.init(
                project_name="agent-weave-demo",
            )
            self._client = weave.get_client()
        except AttributeError:
            print("[WeaveTracer] Warning: Weave client initialization failed. Ensure Weave is properly installed.")

    def _datetime_to_ns(self, dt: Optional[datetime]) -> int:
        if dt is None:
            return int(time.time() * 1e9)
        return int(dt.timestamp() * 1e9)

    def _make_span_context(self, call) -> SpanContext:
        trace_id_int = uuid.UUID(call.trace_id).int & ((1 << 128) - 1)
        span_id_int = uuid.UUID(call.id).int & ((1 << 64) - 1)

        return SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

    def convert_weave_call_to_readable_span(self, call) -> ReadableSpan:
        """
        Convert a Weave Call object to an OpenTelemetry ReadableSpan.
        """

        span_context = self._make_span_context(call)

        attributes = {
            "weave.project_id": call.project_id,
            "weave.op_name": str(getattr(call, "_op_name", None)),
            "weave.rollout_id": call.inputs.get("rollout_id") if call.inputs else None,
            "weave.status": "error" if call.exception else "success",
            "weave.output": str(call.output),
            "weave.attributes": dict(call.attributes) if call.attributes else {},
        }

        start_time = self._datetime_to_ns(call.started_at)
        end_time = self._datetime_to_ns(call.ended_at)

        status = Status(StatusCode.ERROR, str(call.exception)) if call.exception else Status(StatusCode.OK)

        span = ReadableSpan(
            name=str(getattr(call, "_display_name", None) or call.id),
            context=span_context,
            parent=(self._make_span_context(call) if getattr(call, "parent_id", None) else None),
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

        span._children = [self.convert_weave_call_to_readable_span(child) for child in getattr(call, "_children", [])]

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
        if not self._client:
            raise RuntimeError("Weave client not initialized. Call init_worker() first.")

        trace_name = name or rollout_id or "weave_trace"
        # Start manual call trace
        root_call = self._client.create_call(trace_name, inputs={"rollout_id": rollout_id})

        try:
            logger.debug(f"[WeaveTracer] Started trace {trace_name}")

            if store and rollout_id and attempt_id:
                ctx = self._lightning_span_processor.with_context(
                    store=store, rollout_id=rollout_id, attempt_id=attempt_id
                )
                with ctx as processor:
                    yield processor
            else:
                with self._lightning_span_processor:
                    yield self._lightning_span_processor

        except Exception as e:
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
            self._client.fail_call(root_call, e)
        else:
            self._client.finish_call(root_call)
        finally:
            logger.debug(f"[WeaveTracer] Ended trace {trace_name}")
            self._client.finish()
            self._lightning_span_processor.on_end(self.convert_weave_call_to_readable_span(root_call))

    def get_last_trace_url(self) -> Optional[str]:
        """Return URL of the last weave trace session if available."""
        return NotImplementedError()

    def get_last_trace(self):
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

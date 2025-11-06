from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Optional

import weave
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Resource, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import (
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)

from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import LightningSpanProcessor

from .base import Tracer

logger = logging.getLogger(__name__)


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

        weave.init(project_name=self.project_name)
        self._client = weave.get_client()

        self._wandb_api_key = os.getenv("WANDB_API_KEY")

        self._lightning_span_processor = LightningSpanProcessor()
        provider = TracerProvider()
        provider.add_span_processor(self._lightning_span_processor)

        # ✅ 让 global tracer provider 生效
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("agent-lightning.weave")

        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")
        self._initialized = True

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
        arg_inputs = {"rollout_id": rollout_id or ""}

        trace_call = self._client.create_call(op=arg_op, inputs=arg_inputs)

        tracer = trace.get_tracer("agent-lightning.weave")

        try:
            with tracer.start_as_current_span(arg_op) as span:
                span.set_attribute("rollout_id", rollout_id or "")
                span.set_attribute("attempt_id", attempt_id or "")
                span.set_attribute("source", "WeaveTracer")

                if store and rollout_id and attempt_id:
                    ctx = self._lightning_span_processor.with_context(
                        store=store, rollout_id=rollout_id, attempt_id=attempt_id
                    )
                    with ctx as processor:
                        yield processor
                else:
                    with self._lightning_span_processor:
                        yield self._lightning_span_processor

                # ✅ span will automatically end here
                span.set_status(Status(StatusCode.OK))
                self._client.finish_call(trace_call, {"result": "ok"})

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            self._client.fail_call(trace_call, e)
            raise

        finally:
            self._client.finish()

    def get_last_trace(self):
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

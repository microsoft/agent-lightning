# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Iterator, List, Optional

import weave

# from weave.client import WeaveClient
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import LightningSpanProcessor

from .base import Tracer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from weave.trace.weave_client import WeaveClient


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context.
    It logs function calls, input/output, and exceptions to Weave Cloud (W&B backend).
    """

    def __init__(self, *, project_name: str | None = None, wandb_api_key: str | None = None):
        super().__init__()
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self.project_name = project_name or __name__
        self._client: Optional[WeaveClient] = None
        self._wandb_api_key = wandb_api_key or os.getenv("WANDB_API_KEY")
        self.otel_trace: Optional[otel_trace.Tracer] = None
        self._initialized = False

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        if self._initialized:
            logger.warning(f"[Worker {worker_id}] Weave client was already initialized.")
            return

        if weave.get_client() is None:
            try:
                weave.init(project_name=self.project_name)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}': {e}")

        self._client = weave.get_client()
        if not self._client:
            raise RuntimeError(f"Failed to initialize Weave client for project '{self.project_name}'")

        self._lightning_span_processor = LightningSpanProcessor()
        provider = TracerProvider()
        provider.add_span_processor(self._lightning_span_processor)

        otel_trace.set_tracer_provider(provider)
        self._otel_tracer = otel_trace.get_tracer("agent-lightning.weave")

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
        if not self._client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        arg_op = name or "weave_trace"
        arg_inputs = {"rollout_id": rollout_id or ""}
        trace_call = self._client.create_call(op=arg_op, inputs=arg_inputs)  # type: ignore

        try:
            with self._otel_tracer.start_as_current_span(arg_op):
                if all(x is None for x in (store, rollout_id, attempt_id)):
                    processor_ctx = self._lightning_span_processor
                elif all(x is not None for x in (store, rollout_id, attempt_id)):
                    processor_ctx = self._lightning_span_processor.with_context(
                        store=store, rollout_id=rollout_id, attempt_id=attempt_id  # type: ignore
                    )
                else:
                    raise ValueError("store, rollout_id, and attempt_id must be either all provided or all None")

                with processor_ctx as processor:
                    yield processor
        except Exception as e:
            self._client.fail_call(trace_call, exception=e)
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            self._client.finish_call(trace_call)  # type: ignore

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

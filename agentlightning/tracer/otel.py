# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from agentlightning.store.base import LightningStore

from .agentops import LightningSpanProcessor  # FIXME: This import should be from otel to agentops
from .base import BaseTracer

logger = logging.getLogger(__name__)


class OtelTracer(BaseTracer):
    """Tracer that provides a basic OpenTelemetry tracer provider.

    You should be able to collect agent-lightning signals like rewards with this tracer,
    but no other function instrumentations like `openai.chat.completion`.
    """

    def __init__(self):
        super().__init__()
        # This provider is only initialized when the worker is initialized.
        self._tracer_provider: Optional[TracerProvider] = None
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self._initialized: bool = False

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up OpenTelemetry tracer...")

        if self._initialized:
            logger.error("Tracer provider is already initialized. OpenTelemetry may not work as expected.")

        tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        self._lightning_span_processor = LightningSpanProcessor()
        tracer_provider.add_span_processor(self._lightning_span_processor)
        self._initialized = True

    def teardown_worker(self, worker_id: int):
        super().teardown_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Tearing down OpenTelemetry tracer...")
        self._tracer_provider = None

    @contextmanager
    def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Iterator[LightningSpanProcessor]:
        """
        Starts a new tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.
            store: Optional store to add the spans to.
            rollout_id: Optional rollout ID to add the spans to.
            attempt_id: Optional attempt ID to add the spans to.

        Yields:
            The LightningSpanProcessor instance to collect spans.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")

        if store is not None and rollout_id is not None and attempt_id is not None:
            ctx = self._lightning_span_processor.with_context(store=store, rollout_id=rollout_id, attempt_id=attempt_id)
            with ctx as processor:
                yield processor
        elif store is None and rollout_id is None and attempt_id is None:
            with self._lightning_span_processor:
                yield self._lightning_span_processor
        else:
            raise ValueError("store, rollout_id, and attempt_id must be either all provided or all None")

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Iterator, Optional

import opentelemetry.trace as trace_api

from agentlightning.store.base import LightningStore

from .otel import OtelTracer

logger = logging.getLogger(__name__)


class WeaveTracer(OtelTracer):
    """Tracer implementation using Weave for trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context.
    It logs function calls, input/output, and exceptions to Weave Cloud (W&B backend).
    """

    def __init__(self, *, project_name: str | None = None, wandb_api_key: str | None = None):
        super().__init__()
        self.project_name = project_name or __name__
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

    def _initialize_tracer_provider(self, worker_id: int):
        super()._initialize_tracer_provider(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")

        try:
            import weave
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

        if weave.get_client() is None:  # type: ignore
            try:
                weave.init(project_name=self.project_name)  # type: ignore
                logger.info(f"[Worker {worker_id}] Weave client initialized.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}': {e}")

    def teardown_worker(self, worker_id: int):
        super().teardown_worker(worker_id)

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[trace_api.Tracer, None]:
        """
        Starts a new tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.
            store: Optional store to add the spans to.
            rollout_id: Optional rollout ID to add the spans to.
            attempt_id: Optional attempt ID to add the spans to.

        Yields:
            The OpenTelemetry tracer instance to collect spans.
        """
        with self._trace_context_sync(name=name, store=store, rollout_id=rollout_id, attempt_id=attempt_id) as tracer:
            yield tracer

    @contextmanager
    def _trace_context_sync(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Iterator[trace_api.Tracer]:
        """Implementation of `trace_context` for synchronous execution."""
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")

        tracer_provider = self._get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)

        arg_op = name or "weave_trace"
        arg_inputs: dict[str, str] | None = {"rollout_id": rollout_id or "", "attempt_id": attempt_id or ""}

        all_provided = store is not None and rollout_id is not None and attempt_id is not None
        all_none = store is None and rollout_id is None and attempt_id is None

        if not (all_provided or all_none):
            raise ValueError("store, rollout_id, and attempt_id must be either all provided or all None")

        if all_provided and getattr(store, "capabilities", {}).get("otlp_traces", False):  # type: ignore
            logger.info(f"Tracing to LightningStore rollout_id={rollout_id}, attempt_id={attempt_id}")
            self._enable_native_otlp_exporter(store, rollout_id, attempt_id)  # type: ignore
        else:
            self._disable_native_otlp_exporter()

        ctx = (
            self._lightning_span_processor.with_context(store=store, rollout_id=rollout_id, attempt_id=attempt_id)  # type: ignore
            if all_provided
            else self._lightning_span_processor
        )

        with ctx:
            with self._weave_trace_context(rollout_id or "", attempt_id or "", arg_op, arg_inputs):
                # Since Weave does not natively support OTEL, tracing needs to be enabled manually.
                with tracer.start_as_current_span(arg_op):
                    yield tracer

    @contextmanager
    def _weave_trace_context(
        self,
        rollout_id: Optional[str],
        attempt_id: Optional[str],
        arg_op: Optional[str],
        arg_inputs: Optional[dict[str, str]],
    ):
        try:
            import weave
        except ImportError:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.")

        weave_client = weave.get_client()  # type: ignore
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        trace_call = weave_client.create_call(op=arg_op, inputs=arg_inputs)  # type: ignore
        try:
            yield
        except Exception as e:
            weave_client.finish_call(trace_call, exception=e)  # type: ignore
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            weave_client.finish_call(trace_call)  # type: ignore

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, List, Optional

import opentelemetry.trace as trace_api
from agentops.sdk.core import BatchSpanProcessor
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import SpanNames
from agentlightning.utils.otlp import LightningStoreOTLPExporter

from .base import Tracer

logger = logging.getLogger(__name__)


class OtelTracer(Tracer):
    """Tracer that provides a basic OpenTelemetry tracer provider.

    You should be able to collect agent-lightning signals like rewards with this tracer,
    but no other function instrumentations like `openai.chat.completion`.
    """

    def __init__(self):
        super().__init__()
        # This provider is only initialized when the worker is initialized.
        self._tracer_provider: Optional[TracerProvider] = None
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self._simple_span_processor: Optional[SimpleSpanProcessor] = None
        self._otlp_span_exporter: Optional[LightningStoreOTLPExporter] = None
        self._initialized: bool = False

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None):
        super().init_worker(worker_id, store)
        self._initialize_tracer_provider(worker_id)

    def _initialize_tracer_provider(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up OpenTelemetry tracer...")

        if self._initialized:
            logger.error("Tracer provider is already initialized. OpenTelemetry may not work as expected.")

        self._tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(self._tracer_provider)
        self._lightning_span_processor = LightningSpanProcessor()
        self._tracer_provider.add_span_processor(self._lightning_span_processor)
        self._otlp_span_exporter = LightningStoreOTLPExporter()
        self._simple_span_processor = SimpleSpanProcessor(self._otlp_span_exporter)
        self._tracer_provider.add_span_processor(self._simple_span_processor)
        self._initialized = True

        logger.info(f"[Worker {worker_id}] OpenTelemetry tracer provider initialized.")

    def teardown_worker(self, worker_id: int):
        super().teardown_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Tearing down OpenTelemetry tracer...")
        self._tracer_provider = None

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
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")

        if store is not None:
            warnings.warn(
                "store is deprecated in favor of init_worker(). It will be removed in the future.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            store = self._store

        if rollout_id is not None and attempt_id is not None:
            if store is None:
                raise ValueError("store is required to be initialized when rollout_id and attempt_id are provided")
            if store.capabilities.get("otlp_traces", False) is True:
                logger.debug(f"Tracing to LightningStore rollout_id={rollout_id}, attempt_id={attempt_id}")
                self._enable_native_otlp_exporter(store, rollout_id, attempt_id)
            else:
                self._disable_native_otlp_exporter()
            ctx = self._lightning_span_processor.with_context(store=store, rollout_id=rollout_id, attempt_id=attempt_id)
            with ctx:
                yield trace_api.get_tracer(__name__, tracer_provider=self._tracer_provider)
        elif rollout_id is None and attempt_id is None:
            self._disable_native_otlp_exporter()
            with self._lightning_span_processor:
                yield trace_api.get_tracer(__name__, tracer_provider=self._tracer_provider)
        else:
            raise ValueError("rollout_id and attempt_id must be either all provided or all None")

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

    def _get_tracer_provider(self) -> TracerProviderImpl:
        if self._tracer_provider is None:
            raise RuntimeError("TracerProvider is not initialized. Call init_worker() first.")
        return self._tracer_provider

    def _enable_native_otlp_exporter(self, store: LightningStore, rollout_id: str, attempt_id: str):
        tracer_provider = self._get_tracer_provider()
        active_span_processor = tracer_provider._active_span_processor  # pyright: ignore[reportPrivateUsage]

        # Override the resources so that the server knows where the request comes from.
        tracer_provider._resource = tracer_provider._resource.merge(  # pyright: ignore[reportPrivateUsage]
            Resource.create(
                {
                    SpanNames.ROLLOUT_ID: rollout_id,
                    SpanNames.ATTEMPT_ID: attempt_id,
                }
            )
        )
        instrumented = False
        candidates: List[str] = []
        for processor in active_span_processor._span_processors:  # pyright: ignore[reportPrivateUsage]
            if isinstance(processor, LightningSpanProcessor):
                # We don't need the LightningSpanProcessor any more.
                logger.debug("LightningSpanProcessor already present in TracerProvider, disabling it.")
                processor.disable_store_submission = True
            elif isinstance(processor, (SimpleSpanProcessor, BatchSpanProcessor)):
                # Instead, we rely on the OTLPSpanExporter to send spans to the store.
                if isinstance(processor.span_exporter, LightningStoreOTLPExporter):
                    processor.span_exporter.enable_store_otlp(store.otlp_traces_endpoint(), rollout_id, attempt_id)
                    logger.debug(f"Set LightningStoreOTLPExporter endpoint to {store.otlp_traces_endpoint()}")
                    instrumented = True
                else:
                    candidates.append(
                        f"{processor.__class__.__name__} with {processor.span_exporter.__class__.__name__}"
                    )
            else:
                candidates.append(f"{processor.__class__.__name__}")

        if not instrumented:
            raise RuntimeError(
                "Failed to enable native OTLP exporter: no BatchSpanProcessor or SimpleSpanProcessor with "
                "LightningStoreOTLPExporter found in TracerProvider. Please try using a non-OTLP store."
                "Candidates are: " + ", ".join(candidates)
            )

    def _disable_native_otlp_exporter(self):
        tracer_provider = self._get_tracer_provider()
        active_span_processor = tracer_provider._active_span_processor  # pyright: ignore[reportPrivateUsage]
        tracer_provider._resource = tracer_provider._resource.merge(  # pyright: ignore[reportPrivateUsage]
            Resource.create(
                {
                    SpanNames.ROLLOUT_ID: "",
                    SpanNames.ATTEMPT_ID: "",
                }
            )
        )  # reset resource
        for processor in active_span_processor._span_processors:  # pyright: ignore[reportPrivateUsage]
            if isinstance(processor, LightningSpanProcessor):
                # We will be in need of the LightningSpanProcessor again.
                logger.debug("Enabling LightningSpanProcessor in TracerProvider.")
                processor.disable_store_submission = False


class LightningSpanProcessor(SpanProcessor):
    """Span processor that subclasses OpenTelemetry's `SpanProcessor` and adds support to dump traces
    to a [`LightningStore`][agentlightning.LightningStore].

    It serves two purposes:

    1. Records all the spans in a local buffer.
    2. Submits the spans to the event loop to be added to the store.
    """

    def __init__(self, disable_store_submission: bool = False):
        self._disable_store_submission: bool = disable_store_submission
        self._spans: List[ReadableSpan] = []

        # Store related context and states
        self._store: Optional[LightningStore] = None
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None
        self._lock = threading.Lock()

        # private asyncio loop running in a daemon thread
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    @property
    def disable_store_submission(self) -> bool:
        """Whether to disable submitting spans to the store."""
        return self._disable_store_submission

    @disable_store_submission.setter
    def disable_store_submission(self, value: bool) -> None:
        self._disable_store_submission = value

    def _ensure_loop(self) -> None:
        if self._loop_thread is None or self._loop is None:
            self._loop_ready.clear()
            self._loop_thread = threading.Thread(target=self._loop_runner, name="otel-loop", daemon=True)
            self._loop_thread.start()
            self._loop_ready.wait()  # loop is ready

    def _loop_runner(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        loop.run_forever()
        loop.close()

    def __enter__(self):
        self._last_trace = None
        self._spans = []
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self._store = None
        self._rollout_id = None
        self._attempt_id = None

    def _await_in_loop(self, coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        # submit to the dedicated loop and wait synchronously
        self._ensure_loop()
        if self._loop is None:
            raise RuntimeError("Loop is not initialized. This should not happen.")

        # If already on the exporter loop thread, schedule and return immediately.
        # ---------------------------------------------------------------------------
        # WHY THIS CONDITIONAL EXISTS:
        # In rare cases, span.end() is triggered from a LangchainCallbackHandler.__del__
        # (or another finalizer) while the Python garbage collector is running on the
        # *same thread* that owns our exporter event loop ("otel-loop").
        #
        # When that happens, on_end() executes on the exporter loop thread itself.
        # If we were to call `asyncio.run_coroutine_threadsafe(...).result()` here,
        # it would deadlock immediately — because the loop cannot both wait on and run
        # the same coroutine. The Future stays pending forever and the loop stops
        # processing scheduled callbacks.
        #
        # To avoid that self-deadlock, we detect when on_end() runs on the exporter
        # loop thread. If so, we *schedule* the coroutine on the loop (fire-and-forget)
        # instead of blocking with .result().
        #
        # This situation can occur because Python calls __del__ in whatever thread
        # releases the last reference, which can easily be our loop thread if the
        # object is dereferenced during loop._run_once().
        # ---------------------------------------------------------------------------
        if threading.current_thread() is self._loop_thread:
            self._loop.call_soon_threadsafe(asyncio.create_task, coro)  # type: ignore
            return None

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore
        return fut.result(timeout=timeout)  # raises on error  # type: ignore

    def shutdown(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def spans(self) -> List[ReadableSpan]:
        """
        Get the list of spans collected by this processor.
        This is useful for debugging and testing purposes.

        Returns:
            List of ReadableSpan objects collected during tracing.
        """
        return self._spans

    def with_context(self, store: LightningStore, rollout_id: str, attempt_id: str):
        # simple context manager without nesting into asyncio
        class _Ctx:
            def __enter__(_):  # type: ignore
                # Use _ instead of self to avoid shadowing the instance method.
                with self._lock:
                    self._store, self._rollout_id, self._attempt_id = store, rollout_id, attempt_id
                    self._last_trace = None
                    self._spans = []
                return self

            def __exit__(_, exc_type, exc, tb):  # type: ignore
                with self._lock:
                    self._store = self._rollout_id = self._attempt_id = None

        return _Ctx()

    def on_end(self, span: ReadableSpan) -> None:
        """
        Process a span when it ends.

        Args:
            span: The span that has ended.
        """
        # Skip if span is not sampled
        if not span.context or not span.context.trace_flags.sampled:
            return

        if not self._disable_store_submission and self._store and self._rollout_id and self._attempt_id:
            try:
                # Submit add_otel_span to the event loop and wait for it to complete
                with suppress_instrumentation():
                    self._ensure_loop()
                    self._await_in_loop(
                        self._store.add_otel_span(self._rollout_id, self._attempt_id, span),
                        timeout=60.0,
                    )
            except Exception:
                # log; on_end MUST NOT raise
                logger.exception(f"Error adding span to store: {span.name}")

        self._spans.append(span)

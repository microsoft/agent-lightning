# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Awaitable, Iterator, List, Optional

import opentelemetry.trace as trace_api
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from agentlightning.semconv import LightningResourceAttributes
from agentlightning.store.base import LightningStore
from agentlightning.types import Attributes, Span, SpanCoreFields, SpanRecordingContext, StatusCode, TraceStatus
from agentlightning.types.tracer import convert_timestamp
from agentlightning.utils.otel import get_tracer_provider
from agentlightning.utils.otlp import LightningStoreOTLPExporter

from .base import Tracer

logger = logging.getLogger(__name__)


def to_otel_status_code(status_code: StatusCode) -> trace_api.StatusCode:
    if status_code == "UNSET":
        return trace_api.StatusCode.UNSET
    elif status_code == "ERROR":
        return trace_api.StatusCode.ERROR
    else:
        return trace_api.StatusCode.OK


class OtelSpanRecordingContext(SpanRecordingContext):
    def __init__(self, span: trace_api.Span) -> None:
        self._span = span

    def record_exception(self, exception: BaseException) -> None:
        self._span.record_exception(exception)
        self.record_status("ERROR", str(exception))

    def record_attributes(self, attributes: Attributes) -> None:
        self._span.set_attributes(attributes)

    def record_status(self, status_code: StatusCode, description: Optional[str] = None) -> None:
        otel_status_code = to_otel_status_code(status_code)
        self._span.set_status(otel_status_code, description)

    def get_otel_span(self) -> trace_api.Span:
        return self._span

    def get_recorded_span(self) -> SpanCoreFields:
        if isinstance(self._span, ReadableSpan):
            return SpanCoreFields(
                name=self._span.name,
                attributes=self._span.attributes,  # pyright: ignore[reportArgumentType]
                start_time=convert_timestamp(self._span.start_time),  # pyright: ignore[reportArgumentType]
                end_time=convert_timestamp(self._span.end_time),  # pyright: ignore[reportArgumentType]
                status=TraceStatus.from_opentelemetry(self._span.status),
            )
        else:
            raise ValueError(f"Span is not a ReadableSpan: {self._span}")


class OtelTracer(Tracer):
    """Tracer that provides a basic OpenTelemetry tracer provider.

    You should be able to collect agent-lightning signals like rewards with this tracer,
    but no other function instrumentations like `openai.chat.completion`.
    """

    def __init__(self):
        super().__init__()
        # This provider is only initialized when the worker is initialized.
        self._tracer_provider: Optional[trace_api.TracerProvider] = None
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
            logger.info(f"[Worker {worker_id}] Tracer provider is already initialized. Skipping initialization.")
            return

        try:
            get_tracer_provider()
            logger.error(
                f"[Worker {worker_id}] Tracer provider is already initialized but not by OtelTracer. OpenTelemetry may not work as expected."
            )
        except RuntimeError:
            logger.debug(f"[Worker {worker_id}] Tracer provider is not initialized by OtelTracer. Initializing it now.")

        self._tracer_provider = TracerProviderImpl()
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
        logger.info(f"[Worker {worker_id}] Tearing down OpenTelemetry tracer does NOT remove the tracer provider.")

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

    def create_span(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        # Fire the span to the current active tracer provider.
        tracer_provider = self._get_tracer_provider()
        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span(
            name, attributes=attributes, start_time=int(timestamp * 1_000_000_000) if timestamp else None
        )
        if status is not None:
            span.set_status(to_otel_status_code(status.status_code), status.description)
        span.end(int(timestamp * 1_000_000_000) if timestamp else None)

        # The span should have been auto-created by now.
        # Return the core fields of the span.
        if isinstance(span, ReadableSpan):
            return SpanCoreFields(
                name=name,
                attributes=span.attributes,  # pyright: ignore[reportArgumentType]
                start_time=convert_timestamp(span.start_time),  # pyright: ignore[reportArgumentType]
                end_time=convert_timestamp(span.end_time),  # pyright: ignore[reportArgumentType]
                status=TraceStatus.from_opentelemetry(span.status),
            )
        else:
            raise ValueError(f"Span is not a ReadableSpan: {span}")

    @contextmanager
    def operation_context(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[SpanRecordingContext]:
        if end_time is not None:
            logger.warning("OpenTelemetry doesn't support customizing the end time of a span. End time is ignored.")
        # Record the span to the current active tracer provider.
        tracer_provider = self._get_tracer_provider()
        tracer = tracer_provider.get_tracer(__name__)

        # Activate the span as the current span within otel.
        with tracer.start_as_current_span(
            name, attributes=attributes, start_time=int(start_time * 1_000_000_000) if start_time else None
        ) as span:
            recording_context = OtelSpanRecordingContext(span)
            try:
                yield recording_context
            except Exception as exc:
                recording_context.record_exception(exc)
                raise

        # No need to retrieve the span here. It's already been sent to otel processor.

    def get_last_trace(self) -> List[Span]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of `Span` objects.
        """
        if not self._lightning_span_processor:
            raise RuntimeError("LightningSpanProcessor is not initialized. Call init_worker() first.")
        return self._lightning_span_processor.spans()

    def _get_tracer_provider(self) -> TracerProviderImpl:
        if self._tracer_provider is None:
            raise RuntimeError("TracerProvider is not initialized. Call init_worker() first.")
        if not isinstance(self._tracer_provider, TracerProviderImpl):
            raise TypeError(f"TracerProvider is not a opentelemetry.sdk.trace.TracerProvider: {self._tracer_provider}")
        return self._tracer_provider

    def _enable_native_otlp_exporter(self, store: LightningStore, rollout_id: str, attempt_id: str):
        tracer_provider = self._get_tracer_provider()
        active_span_processor = tracer_provider._active_span_processor  # pyright: ignore[reportPrivateUsage]

        # Override the resources so that the server knows where the request comes from.
        tracer_provider._resource = tracer_provider._resource.merge(  # pyright: ignore[reportPrivateUsage]
            Resource.create(
                {
                    LightningResourceAttributes.ROLLOUT_ID.value: rollout_id,
                    LightningResourceAttributes.ATTEMPT_ID.value: attempt_id,
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
                    LightningResourceAttributes.ROLLOUT_ID.value: "",
                    LightningResourceAttributes.ATTEMPT_ID.value: "",
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
        self._spans: List[Span] = []

        # Store related context and states
        self._store: Optional[LightningStore] = None
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None
        self._local_sequence_id: int = 0
        self._lock = threading.Lock()

        # private asyncio loop running in a daemon thread
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"disable_store_submission={self.disable_store_submission}, "
            + f"store={self.store!r}, "
            + f"rollout_id={self.rollout_id!r}, "
            + f"attempt_id={self.attempt_id!r})"
        )

    @property
    def store(self) -> Optional[LightningStore]:
        """The store to submit the spans to."""
        return self._store

    @property
    def rollout_id(self) -> Optional[str]:
        """The rollout ID to submit the spans to."""
        return self._rollout_id

    @property
    def attempt_id(self) -> Optional[str]:
        """The attempt ID to submit the spans to."""
        return self._attempt_id

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

    def spans(self) -> List[Span]:
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
                    uploaded_span = self._await_in_loop(
                        self._store.add_otel_span(self._rollout_id, self._attempt_id, span),
                        timeout=60.0,
                    )
                    if uploaded_span is not None:
                        self._spans.append(uploaded_span)
            except Exception:
                # log; on_end MUST NOT raise
                logger.exception(f"Error adding span to store: {span.name}")

        else:
            # Fallback path
            created_span = Span.from_opentelemetry(
                span,
                rollout_id=self._rollout_id or "rollout-dummy",
                attempt_id=self._attempt_id or "attempt-dummy",
                sequence_id=self._local_sequence_id,
            )
            self._local_sequence_id += 1
            self._spans.append(created_span)

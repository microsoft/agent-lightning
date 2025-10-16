# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Iterator, List, Optional

import agentops
import agentops.sdk.core
from agentops.sdk.core import TracingCore
from agentops.sdk.processors import SpanProcessor
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.instrumentation import instrument_all, uninstrument_all
from agentlightning.instrumentation.agentops import AgentOpsServerManager
from agentlightning.store.base import LightningStore

from .base import Tracer

if TYPE_CHECKING:
    from agentops.integration.callbacks.langchain import LangchainCallbackHandler


logger = logging.getLogger(__name__)


class AgentOpsTracer(Tracer):
    """Traces agent execution using AgentOps.

    This tracer provides functionality to capture execution details using the
    AgentOps library. It manages the AgentOps client initialization, server setup,
    and integration with the OpenTelemetry tracing ecosystem.

    Attributes:
        agentops_managed: Whether to automatically manage `agentops`.
                          When set to true, tracer calls `agentops.init()`
                          automatically and launches an agentops endpoint locally.
                          If not, you are responsible for calling and using it
                          before using the tracer.
        instrument_managed: Whether to automatically manage instrumentation.
                            When set to false, you will manage the instrumentation
                            yourself and the tracer might not work as expected.
        daemon: Whether the AgentOps server runs as a daemon process.
                Only applicable if `agentops_managed` is True.
    """

    def __init__(self, *, agentops_managed: bool = True, instrument_managed: bool = True, daemon: bool = True):
        super().__init__()
        self._lightning_span_processor: Optional[LightningSpanProcessor] = None
        self.agentops_managed = agentops_managed
        self.instrument_managed = instrument_managed
        self.daemon = daemon

        self._agentops_server_manager = AgentOpsServerManager(self.daemon)
        self._agentops_server_port_val: Optional[int] = None

        if not self.agentops_managed:
            logger.warning("agentops_managed=False. You are responsible for AgentOps setup.")
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_agentops_server_manager"] = None  # Exclude the unpicklable server manager
        # _agentops_server_port_val (int) is inherently picklable and will be included.
        logger.debug(f"Getting state for pickling Trainer (PID {os.getpid()}). _agentops_server_manager excluded.")
        return state

    def __setstate__(self, state: Any):
        self.__dict__.update(state)
        # In child process, self._agentops_server_manager will be None.
        logger.debug(f"Setting state for unpickled Trainer (PID {os.getpid()}). _agentops_server_manager is None.")

    def init(self, *args: Any, **kwargs: Any):
        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.start()
            self._agentops_server_port_val = self._agentops_server_manager.get_port()
            if self._agentops_server_port_val is None:
                if (
                    self._agentops_server_manager.server_process is not None
                    and self._agentops_server_manager.server_process.is_alive()
                ):
                    raise RuntimeError("AgentOps server started but port is None. Check server manager logic.")
                elif (
                    self._agentops_server_port_val is None and self._agentops_server_manager.server_process is None
                ):  # Server failed to start
                    raise RuntimeError("AgentOps server manager indicates server is not running and port is None.")

    def teardown(self):
        if self.agentops_managed:
            self._agentops_server_manager.stop()
            logger.info("AgentOps server stopped.")

    def instrument(self, worker_id: int):
        instrument_all()

    def uninstrument(self, worker_id: int):
        uninstrument_all()

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up tracer...")  # worker_id included in process name

        if self.instrument_managed:
            self.instrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

        if self.agentops_managed:
            if self._agentops_server_port_val:  # Use the stored, picklable port value
                base_url = f"http://localhost:{self._agentops_server_port_val}"
                env_vars_to_set = {
                    "AGENTOPS_API_KEY": "dummy",
                    "AGENTOPS_API_ENDPOINT": base_url,
                    "AGENTOPS_APP_URL": f"{base_url}/notavailable",
                    "AGENTOPS_EXPORTER_ENDPOINT": f"{base_url}/traces",
                }
                for key, value in env_vars_to_set.items():
                    os.environ[key] = value
                    logger.info(f"[Worker {worker_id}] Env var set: {key}={value}")
            else:
                logger.warning(
                    f"[Worker {worker_id}] AgentOps managed, but local server port is not available. Client may not connect as expected."
                )

            if not agentops.get_client().initialized:
                agentops.init()  # type: ignore
                logger.info(f"[Worker {worker_id}] AgentOps client initialized.")
            else:
                logger.warning(f"[Worker {worker_id}] AgentOps client was already initialized.")

        self._lightning_span_processor = LightningSpanProcessor()

        try:
            # new versions
            instance = agentops.sdk.core.tracer
            # TODO: The span processor cannot be deleted once added.
            # This might be a problem if the tracer is entered and exited multiple times.
            instance.provider.add_span_processor(self._lightning_span_processor)  # type: ignore
        except AttributeError:
            # old versions
            instance = TracingCore.get_instance()  # type: ignore
            instance._provider.add_span_processor(self._lightning_span_processor)  # type: ignore

    def teardown_worker(self, worker_id: int) -> None:
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
    ) -> AsyncGenerator[LightningSpanProcessor, None]:
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
        """Implementation of `trace_context` for synchronous execution."""
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

    def get_langchain_handler(self, tags: List[str] | None = None) -> LangchainCallbackHandler:
        """
        Get the Langchain callback handler for integrating with Langchain.

        Args:
            tags: Optional list of tags to apply to the Langchain callback handler.

        Returns:
            An instance of the Langchain callback handler.
        """
        import agentops
        from agentops.integration.callbacks.langchain import LangchainCallbackHandler

        tags = tags or []
        client_instance = agentops.get_client()
        api_key = None
        if client_instance.initialized:
            api_key = client_instance.config.api_key
        else:
            logger.warning(
                "AgentOps client not initialized when creating LangchainCallbackHandler. API key may be missing."
            )
        return LangchainCallbackHandler(api_key=api_key, tags=tags)

    get_langchain_callback_handler = get_langchain_handler  # alias


class LightningSpanProcessor(SpanProcessor):
    def __init__(self):
        self._spans: List[ReadableSpan] = []

        # Store related context and states
        self._store: Optional[LightningStore] = None
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None
        self._lock = threading.Lock()

        # private asyncio loop running in a daemon thread
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
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
            self._loop_thread.join(timeout=5)
            self._loop = None

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

        if self._store and self._rollout_id and self._attempt_id:
            try:
                # Submit add_otel_span to the event loop and wait for it to complete
                with suppress_instrumentation():
                    self._await_in_loop(
                        self._store.add_otel_span(self._rollout_id, self._attempt_id, span),
                        timeout=60.0,
                    )
            except Exception:
                # log; on_end MUST NOT raise
                logger.exception(f"Error adding span to store: {span.name}")

        self._spans.append(span)

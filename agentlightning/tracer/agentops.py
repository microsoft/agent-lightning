# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
import warnings
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, List, Optional

import agentops
import agentops.sdk.core
import opentelemetry.trace as trace_api
from agentops.sdk.core import TracingCore
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.trace import get_tracer_provider
from opentelemetry.trace.status import StatusCode

from agentlightning.instrumentation import instrument_all, uninstrument_all
from agentlightning.store.base import LightningStore

from .otel import LightningSpanProcessor, OtelTracer

if TYPE_CHECKING:
    from agentops.integration.callbacks.langchain import LangchainCallbackHandler


logger = logging.getLogger(__name__)


class AgentOpsTracer(OtelTracer):
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

        if not self.agentops_managed:
            logger.warning("agentops_managed=False. You are responsible for AgentOps setup.")
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")

    def instrument(self, worker_id: int):
        instrument_all()

    def uninstrument(self, worker_id: int):
        uninstrument_all()

    def _initialize_tracer_provider(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up AgentOps tracer...")  # worker_id included in process name

        if self.instrument_managed:
            self.instrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

        if self.agentops_managed:
            os.environ.setdefault("AGENTOPS_API_KEY", "dummy")
            if not agentops.get_client().initialized:
                agentops.init(auto_start_session=False)  # type: ignore
                logger.info(f"[Worker {worker_id}] AgentOps client initialized.")
            else:
                logger.warning(f"[Worker {worker_id}] AgentOps client was already initialized.")

        self._lightning_span_processor = LightningSpanProcessor()

        # TODO: The span processor cannot be deleted once added.
        # This might be a problem if the tracer is entered and exited multiple times.
        self._get_tracer_provider().add_span_processor(self._lightning_span_processor)  # type: ignore

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
        if store is not None:
            warnings.warn(
                "store is deprecated in favor of init_worker(). It will be removed in the future.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            store = self._store
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

        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["trace_name"] = name
        elif rollout_id is not None:
            kwargs["trace_name"] = rollout_id
        if store is not None and rollout_id is not None and attempt_id is not None:
            if store.capabilities.get("otlp_traces", False) is True:
                logger.debug(f"Tracing to LightningStore rollout_id={rollout_id}, attempt_id={attempt_id}")
                self._enable_native_otlp_exporter(store, rollout_id, attempt_id)
            else:
                self._disable_native_otlp_exporter()
            ctx = self._lightning_span_processor.with_context(store=store, rollout_id=rollout_id, attempt_id=attempt_id)
            with ctx:
                # AgentOps end_trace and start_trace must live inside the lightning span processor context.
                # Otherwise some traces might not be recorded.
                with self._agentops_trace_context(rollout_id, attempt_id, kwargs):
                    yield trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        elif store is None and rollout_id is None and attempt_id is None:
            # TODO: Add tests to cover both paths
            self._disable_native_otlp_exporter()
            with self._lightning_span_processor:
                with self._agentops_trace_context(None, None, kwargs):
                    yield trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        else:
            raise ValueError("store, rollout_id, and attempt_id must be either all provided or all None")

    @contextmanager
    def _agentops_trace_context(self, rollout_id: Optional[str], attempt_id: Optional[str], kwargs: dict[str, Any]):
        trace = agentops.start_trace(**kwargs)
        status = StatusCode.OK  # type: ignore
        try:
            yield
        except Exception as e:
            # This will catch errors in user code.
            status = StatusCode.ERROR  # type: ignore
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}: {e}")
            raise  # should reraise the error here so that runner can handle it
        finally:
            agentops.end_trace(trace, end_state=status)  # type: ignore

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

    def _get_tracer_provider(self) -> TracerProviderImpl:
        try:
            # new versions
            instance = agentops.sdk.core.tracer
            if instance.provider is None:
                raise RuntimeError("AgentOps TracerProvider is not initialized.")

            if get_tracer_provider() is not instance.provider:
                logger.error(
                    "Mismatch between global singleton TracerProvider and AgentOps TracerProvider. "
                    "AgentOps might not work properly."
                )

            if not isinstance(instance.provider, TracerProviderImpl):  # type: ignore
                raise RuntimeError("Unsupported TracerProvider type for AgentOps instrumentation.")

            self._tracer_provider = instance.provider
            return self._tracer_provider
        except AttributeError:
            # old versions
            instance = TracingCore.get_instance()  # type: ignore
            self._tracer_provider = instance._provider  # type: ignore
            return self._tracer_provider  # type: ignore

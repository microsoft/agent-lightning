"""Phoenix-backed tracer integration for Agent Lightning.

This tracer bridges Agent Lightning's tracing interface with Arize Phoenix by
leveraging the ``arize-phoenix-otel`` package. It registers a Phoenix-aware
``TracerProvider`` for each worker process and reuses the built-in
``LightningSpanProcessor`` to capture spans so that they can be stored or
inspected inside Agent Lightning.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from agentlightning.store.base import LightningStore
from agentlightning.tracer.agentops import LightningSpanProcessor
from agentlightning.tracer.base import Tracer
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from phoenix.otel import register as phoenix_register

logger = logging.getLogger(__name__)


class PhoenixTracer(Tracer):
    """Tracer implementation that sends spans to Arize Phoenix.

    Parameters are primarily thin wrappers around ``phoenix.otel.register``. By
    default, configuration is read from the standard Phoenix environment
    variables so that existing deployments keep working without code changes.

    Note: This tracer will set its own global OpenTelemetry TracerProvider.
    If you have already called ``setup_otel_tracing()`` from this module,
    there may be conflicts. Choose one approach:

    - For Agent Lightning training: Use PhoenixTracer with Trainer
    - For general agent tracing: Use setup_otel_tracing()
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        project_name: str | None = None,
        api_key: str | None = None,
        auto_instrument: bool = True,
        use_batch_processor: bool = False,
        headers: dict[str, str] | None = None,
        register_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint or os.getenv("PHOENIX_ENDPOINT")
        self.project_name = project_name or os.getenv("PHOENIX_PROJECT_NAME")
        self.api_key = api_key or os.getenv("PHOENIX_API_KEY")
        self.auto_instrument = auto_instrument
        self.use_batch_processor = use_batch_processor
        self.headers = headers
        self.register_kwargs = register_kwargs.copy() if register_kwargs else {}

        self._tracer_provider: TracerProvider | None = None
        self._lightning_span_processor: LightningSpanProcessor | None = None
        self._initialized = False

    def init(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - hook required by interface
        """Main-process initialization hook (no-op for Phoenix)."""
        logger.debug("PhoenixTracer main-process init invoked.")

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("PhoenixTracer main-process teardown invoked.")

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        super().init_worker(worker_id, *args, **kwargs)
        if self._initialized:
            logger.warning(
                "PhoenixTracer already initialized in worker %s; skipping re-registration.",
                worker_id,
            )
            return

        logger.info("[Worker %s] Configuring Phoenix tracer provider...", worker_id)

        register_options: dict[str, Any] = {
            "endpoint": self.endpoint,
            "project_name": self.project_name,
            "headers": self.headers,
            "batch": self.use_batch_processor,
            "set_global_tracer_provider": False,  # Don't override existing global provider
            "auto_instrument": self.auto_instrument,
        }
        if self.api_key:
            register_options["api_key"] = self.api_key
        register_options.update(self.register_kwargs)

        tracer_provider = phoenix_register(**register_options)
        self._tracer_provider = tracer_provider

        # Set as global tracer provider (will override if already set)
        trace_api.set_tracer_provider(tracer_provider)
        logger.info("[Worker %s] Phoenix tracer provider set as global.", worker_id)

        self._lightning_span_processor = LightningSpanProcessor()
        span_processor_kwargs: dict[str, Any] = {}
        parameters = inspect.signature(tracer_provider.add_span_processor).parameters
        if "replace_default_processor" in parameters:
            span_processor_kwargs["replace_default_processor"] = False
        tracer_provider.add_span_processor(
            self._lightning_span_processor, **span_processor_kwargs
        )  # type: ignore[misc]

        self._initialized = True
        logger.info("[Worker %s] Phoenix tracer provider ready.", worker_id)

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        super().teardown_worker(worker_id, *args, **kwargs)
        logger.info("[Worker %s] Tearing down Phoenix tracer provider...", worker_id)
        if self._lightning_span_processor is not None:
            self._lightning_span_processor.shutdown()
            self._lightning_span_processor = None
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
            self._tracer_provider = None
        self._initialized = False

    @asynccontextmanager
    async def trace_context(
        self,
        name: str | None = None,
        *,
        store: LightningStore | None = None,
        rollout_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncGenerator[LightningSpanProcessor, None]:
        if not self._lightning_span_processor:
            raise RuntimeError(
                "LightningSpanProcessor is not initialized. Call init_worker() first."
            )

        with self._trace_context_sync(
            name=name,
            store=store,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
        ) as processor:
            yield processor

    @contextmanager
    def _trace_context_sync(
        self,
        name: str | None = None,
        *,
        store: LightningStore | None = None,
        rollout_id: str | None = None,
        attempt_id: str | None = None,
    ) -> Iterator[LightningSpanProcessor]:
        if not self._lightning_span_processor:
            raise RuntimeError(
                "LightningSpanProcessor is not initialized. Call init_worker() first."
            )

        if store is not None and rollout_id is not None and attempt_id is not None:
            ctx = self._lightning_span_processor.with_context(
                store=store, rollout_id=rollout_id, attempt_id=attempt_id
            )
            with ctx as processor:
                yield processor
        elif store is None and rollout_id is None and attempt_id is None:
            with self._lightning_span_processor:
                yield self._lightning_span_processor
        else:
            raise ValueError(
                "store, rollout_id, and attempt_id must be either all provided or all None"
            )

    def get_last_trace(self) -> list[ReadableSpan]:
        if not self._lightning_span_processor:
            raise RuntimeError(
                "LightningSpanProcessor is not initialized. Call init_worker() first."
            )
        return self._lightning_span_processor.spans()

    def get_config(self) -> dict[str, Any]:
        """Expose current Phoenix configuration for debugging or tests."""
        return {
            "endpoint": self.endpoint,
            "project_name": self.project_name,
            "api_key": bool(self.api_key),
            "auto_instrument": self.auto_instrument,
            "use_batch_processor": self.use_batch_processor,
            "headers": self.headers,
            "register_kwargs": self.register_kwargs,
        }

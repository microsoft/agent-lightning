# Copyright (c) Microsoft. All rights reserved.

"""Utilities shared across emitter implementations."""

import logging
from typing import List, cast
from warnings import filterwarnings

import opentelemetry.trace as trace_api
from agentops.sdk.exporters import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanLimits, SynchronousMultiSpanProcessor, Tracer
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.util.instrumentation import InstrumentationInfo, InstrumentationScope
from opentelemetry.trace import get_tracer_provider as otel_get_tracer_provider

from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
from agentlightning.tracer.otel import LightningSpanProcessor
from agentlightning.utils.otlp import LightningStoreOTLPExporter

logger = logging.getLogger(__name__)


def _full_qualified_name(obj: type) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def get_tracer_provider() -> TracerProviderImpl:
    """Get the OpenTelemetry tracer provider configured for Agent Lightning."""
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")
    tracer_provider = otel_get_tracer_provider()
    if not isinstance(tracer_provider, TracerProviderImpl):
        logger.error(
            "Tracer provider is expected to be an instance of opentelemetry.sdk.trace.TracerProvider, found: %s",
            _full_qualified_name(type(tracer_provider)),
        )
        return cast(TracerProviderImpl, tracer_provider)

    emitter_debug = resolve_bool_env_var(LightningEnvVar.AGL_EMITTER_DEBUG, fallback=None)
    logger_effective_level = logger.getEffectiveLevel()
    if emitter_debug is True and logger_effective_level > logging.DEBUG:
        logger.warning(
            "Emitter debug logging is enabled but logging level is not set to DEBUG. Nothing will be logged."
        )

    if emitter_debug is None:
        # Set to true by default if the logging level is lower than DEBUG
        emitter_debug = logging.DEBUG >= logger_effective_level

    if emitter_debug:
        active_span_processor = tracer_provider._active_span_processor  # pyright: ignore[reportPrivateUsage]
        processors: List[str] = []
        active_span_processor_cls = active_span_processor.__class__.__name__
        for processor in active_span_processor._span_processors:  # pyright: ignore[reportPrivateUsage]
            if isinstance(processor, LightningSpanProcessor):
                # The legacy case for tracers without OTLP support.
                lsp_arguments: List[str] = []
                if processor.disable_store_submission:
                    lsp_arguments.append("disable_store_submission=True")
                if processor.store is not None:
                    lsp_arguments.append(f"store={processor.store}")
                if processor.rollout_id is not None:
                    lsp_arguments.append(f"rollout_id={processor.rollout_id}")
                if processor.attempt_id is not None:
                    lsp_arguments.append(f"attempt_id={processor.attempt_id}")
                processors.append(
                    f"{active_span_processor_cls} - {processor.__class__.__name__}({', '.join(lsp_arguments)})"
                )
            elif isinstance(processor, (SimpleSpanProcessor, BatchSpanProcessor)):
                processor_cls = processor.__class__.__name__
                if isinstance(processor.span_exporter, LightningStoreOTLPExporter):
                    # This should be the main path now.
                    otlp_arguments: List[str] = []
                    if processor.span_exporter.endpoint is not None:
                        otlp_arguments.append(f"endpoint={processor.span_exporter.endpoint}")
                    if processor.span_exporter.rollout_id is not None:
                        otlp_arguments.append(f"rollout_id={processor.span_exporter.rollout_id}")
                    if processor.span_exporter.attempt_id is not None:
                        otlp_arguments.append(f"attempt_id={processor.span_exporter.attempt_id}")
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - "
                        f"{processor.span_exporter.__class__.__name__}({', '.join(otlp_arguments)})"
                    )
                elif isinstance(processor.span_exporter, OTLPSpanExporter):
                    # You need to be careful if the code goes into this path.
                    endpoint = processor.span_exporter._endpoint  # pyright: ignore[reportPrivateUsage]
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - "
                        f"{processor.span_exporter.__class__.__name__}(endpoint={endpoint})"
                    )
                else:
                    # Other cases like Console Span Exporter.
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - {processor.span_exporter.__class__.__name__}"
                    )
            else:
                processors.append(f"{active_span_processor_cls} - {processor.__class__.__name__}")

        logger.debug("Active span processors:")
        for processor in processors:
            logger.debug("  * " + processor)

    return tracer_provider


def get_tracer(use_active_span_processor: bool = True) -> trace_api.Tracer:
    """Resolve the OpenTelemetry tracer configured for Agent Lightning.

    Args:
        use_active_span_processor: Whether to use the active span processor.

    Returns:
        OpenTelemetry tracer tagged with the `agentlightning` instrumentation name.

    Raises:
        RuntimeError: If OpenTelemetry was not initialized before calling this helper.
    """
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()

    if use_active_span_processor:
        return tracer_provider.get_tracer("agentlightning")

    else:
        filterwarnings(
            "ignore",
            message=r"You should use InstrumentationScope. Deprecated since version 1.11.1.",
            category=DeprecationWarning,
            module="opentelemetry.sdk.trace",
        )

        return Tracer(
            tracer_provider.sampler,
            tracer_provider.resource,
            # We use an empty span processor to avoid emitting spans to the tracer
            SynchronousMultiSpanProcessor(),
            tracer_provider.id_generator,
            InstrumentationInfo("agentlightning", "", ""),  # type: ignore
            SpanLimits(),
            InstrumentationScope(
                "agentlightning",
                "",
                "",
                {},
            ),
        )

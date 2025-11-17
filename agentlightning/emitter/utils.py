# Copyright (c) Microsoft. All rights reserved.

"""Utilities shared across emitter implementations."""

from typing import cast
from warnings import filterwarnings

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import SpanLimits, SynchronousMultiSpanProcessor, Tracer
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.sdk.util.instrumentation import InstrumentationInfo, InstrumentationScope
from opentelemetry.trace import get_tracer_provider


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

    tracer_provider = cast(TracerProviderImpl, get_tracer_provider())

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

# Copyright (c) Microsoft. All rights reserved.

"""Utilities shared across emitter implementations."""

import logging
from typing import Any, Dict, List, Union, cast
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


def get_tracer_provider(inspect: bool = True) -> TracerProviderImpl:
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

    if not inspect:
        return tracer_provider

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
                lsp_arguments.append(f"disable_store_submission={processor.disable_store_submission}")
                if processor.store is not None:
                    lsp_arguments.append(f"store={processor.store!r}")
                if processor.rollout_id is not None:
                    lsp_arguments.append(f"rollout_id={processor.rollout_id!r}")
                if processor.attempt_id is not None:
                    lsp_arguments.append(f"attempt_id={processor.attempt_id!r}")
                processors.append(
                    f"{active_span_processor_cls} - {processor.__class__.__name__}({', '.join(lsp_arguments)})"
                )
            elif isinstance(processor, (SimpleSpanProcessor, BatchSpanProcessor)):
                processor_cls = processor.__class__.__name__
                if isinstance(processor.span_exporter, LightningStoreOTLPExporter):
                    # This should be the main path now.
                    otlp_arguments: List[str] = []
                    if processor.span_exporter.endpoint is not None:
                        otlp_arguments.append(f"endpoint={processor.span_exporter.endpoint!r}")
                    if processor.span_exporter.rollout_id is not None:
                        otlp_arguments.append(f"rollout_id={processor.span_exporter.rollout_id!r}")
                    if processor.span_exporter.attempt_id is not None:
                        otlp_arguments.append(f"attempt_id={processor.span_exporter.attempt_id!r}")
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - "
                        f"{processor.span_exporter.__class__.__name__}({', '.join(otlp_arguments)})"
                    )
                elif isinstance(processor.span_exporter, OTLPSpanExporter):
                    # You need to be careful if the code goes into this path.
                    endpoint = processor.span_exporter._endpoint  # pyright: ignore[reportPrivateUsage]
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - "
                        f"{processor.span_exporter.__class__.__name__}(endpoint={endpoint!r})"
                    )
                else:
                    # Other cases like Console Span Exporter.
                    processors.append(
                        f"{active_span_processor_cls} - {processor_cls} - {processor.span_exporter.__class__.__name__}"
                    )
            else:
                processors.append(f"{active_span_processor_cls} - {processor.__class__.__name__}")

        logger.debug(f"Tracer provider: {tracer_provider!r}. Active span processors:")
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


def flatten_attributes(nested_data: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """Flatten a nested dictionary or list into a flat dictionary with dotted keys.

    This function recursively traverses dictionaries and lists, producing a flat
    key-value mapping where nested paths are represented via dot-separated keys.
    Lists are indexed numerically.

    Example:
        >>> flatten_attributes({"a": {"b": 1, "c": [2, 3]}})
        {"a.b": 1, "a.c.0": 2, "a.c.1": 3}

    Args:
        nested_data: A nested structure composed of dictionaries, lists, or
            primitive values.

    Returns:
        A flat dictionary mapping dotted-string paths to primitive values.
    """

    flat: Dict[str, Any] = {}

    def _walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for k, v in cast(Dict[Any, Any], value).items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"Only string keys are supported in dictionaries, got '{k}' of type {type(k)} in {prefix}"
                    )
                new_prefix = f"{prefix}.{k}" if prefix else k
                _walk(v, new_prefix)
        elif isinstance(value, list):
            for idx, item in enumerate(cast(List[Any], value)):
                new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                _walk(item, new_prefix)
        else:
            flat[prefix] = value

    _walk(nested_data)
    return flat


def unflatten_attributes(flat_data: Dict[str, Any]) -> Union[Dict[str, Any], List[Any]]:
    """Reconstruct a nested dictionary/list structure from a flat dictionary.

    Keys are dot-separated paths. Segments that are digit strings will only
    become list indices if *all* keys in that dict form a consecutive
    0..n-1 range. Otherwise they remain dict keys.

    Example:
        >>> unflatten_attributes({"a.b": 1, "a.c.0": 2, "a.c.1": 3})
        {"a": {"b": 1, "c": [2, 3]}}

    Args:
        flat_data: A dictionary whose keys are dot-separated paths and whose
            values are primitive data elements.

    Returns:
        A nested dictionary (and lists where appropriate) corresponding to
        the flattened structure.
    """
    # 1) Build a pure dict tree first (no lists yet)
    root: Dict[str, Any] = {}

    for flat_key, value in flat_data.items():
        parts = flat_key.split(".")
        curr: Dict[str, Any] = root

        for part in parts[:-1]:
            # Ensure intermediate node is a dict
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]  # type: ignore[assignment]

        curr[parts[-1]] = value

    # 2) Recursively convert dicts-with-consecutive-numeric-keys into lists
    def convert(node: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        if isinstance(node, dict):
            # First convert children
            for k, v in list(node.items()):
                node[k] = convert(v)

            if not node:
                # empty dict stays dict
                return node

            # Check if keys are all numeric strings
            keys = list(node.keys())
            if all(isinstance(k, str) and k.isdigit() for k in keys):  # pyright: ignore[reportUnnecessaryIsInstance]
                indices = sorted(int(k) for k in keys)
                # Must be exactly 0..n-1
                if indices == list(range(len(indices))):
                    return [node[str(i)] for i in range(len(indices))]

            return node

        if isinstance(node, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return [convert(v) for v in node]

        # Keep as is
        return node

    return convert(root)

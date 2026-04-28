# Copyright (c) Microsoft. All rights reserved.

"""Utilities shared for OpenTelemetry span (attributes) support."""

import logging
from typing import Any, Dict, List, Sequence, Union, cast
from warnings import filterwarnings

import opentelemetry.trace as trace_api
from agentops.sdk.exporters import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, SynchronousMultiSpanProcessor, Tracer
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.util.instrumentation import InstrumentationInfo, InstrumentationScope
from opentelemetry.trace import get_tracer_provider as otel_get_tracer_provider
from pydantic import TypeAdapter

from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
from agentlightning.semconv import LightningSpanAttributes, LinkAttributes, LinkPydanticModel
from agentlightning.types import SpanLike
from agentlightning.utils.otlp import LightningStoreOTLPExporter

logger = logging.getLogger(__name__)

__all__ = [
    "full_qualified_name",
    "get_tracer_provider",
    "get_tracer",
    "make_tag_attributes",
    "extract_tags_from_attributes",
    "make_link_attributes",
    "query_linked_spans",
    "extract_links_from_attributes",
    "filter_attributes",
    "filter_and_unflatten_attributes",
    "flatten_attributes",
    "unflatten_attributes",
]


def full_qualified_name(obj: type) -> str:
    if str(obj.__module__) == "builtins":
        return obj.__qualname__
    return f"{obj.__module__}.{obj.__qualname__}"


def get_tracer_provider(inspect: bool = True) -> TracerProviderImpl:
    """Get the OpenTelemetry tracer provider configured for Agent Lightning.

    Args:
        inspect: Whether to inspect the tracer provider and log its configuration.
            When it's on, make sure you also set the logger level to DEBUG to see the logs.
    """
    from agentlightning.tracer.otel import LightningSpanProcessor

    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore[attr-defined]
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")
    tracer_provider = otel_get_tracer_provider()
    if not isinstance(tracer_provider, TracerProviderImpl):
        logger.error(
            "Tracer provider is expected to be an instance of opentelemetry.sdk.trace.TracerProvider, found: %s",
            full_qualified_name(type(tracer_provider)),
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
                processors.append(f"{active_span_processor_cls} - {processor!r}")
            elif isinstance(processor, (SimpleSpanProcessor, BatchSpanProcessor)):
                processor_cls = processor.__class__.__name__
                if isinstance(processor.span_exporter, LightningStoreOTLPExporter):
                    # This should be the main path now.
                    processors.append(f"{active_span_processor_cls} - {processor_cls} - {processor.span_exporter!r}")
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

    tracer_provider = get_tracer_provider(inspect=True)  # inspection is on by default

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


def make_tag_attributes(tags: List[str]) -> Dict[str, Any]:
    """Convert a list of tags into flattened attributes for span tagging.

    There is no syntax enforced for tags, they are just strings. For example:

    ```python
    ["gen_ai.model:gpt-4", "reward.extrinsic"]
    ```
    """
    return flatten_attributes({LightningSpanAttributes.TAG.value: tags})


def extract_tags_from_attributes(attributes: Dict[str, Any]) -> List[str]:
    """Extract tag attributes from flattened span attributes.

    Args:
        attributes: A dictionary of flattened span attributes.
    """
    maybe_tag_list = filter_and_unflatten_attributes(attributes, LightningSpanAttributes.TAG.value)
    return TypeAdapter(List[str]).validate_python(maybe_tag_list)


def make_link_attributes(links: Dict[str, str]) -> Dict[str, Any]:
    """Convert a dictionary of links into flattened attributes for span linking.

    Links example:

    ```python
    {
        "gen_ai.response.id": "response-123",
        "span_id": "abcd-efgh-ijkl",
    }
    ```
    """
    link_list: List[Dict[str, str]] = []
    for key, value in links.items():
        if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"Link value must be a string, got {type(value)} for key '{key}'")
        link_list.append({LinkAttributes.KEY_MATCH.value: key, LinkAttributes.VALUE_MATCH.value: value})
    return flatten_attributes({LightningSpanAttributes.LINK.value: link_list})


def query_linked_spans(spans: Sequence[SpanLike], links: List[LinkPydanticModel]) -> List[SpanLike]:
    """Query spans that are linked by the given link attributes.

    Args:
        spans: A sequence of spans to search.
        links: A list of link attributes to match.

    Returns:
        A list of spans that match the given link attributes.
    """
    matched_spans: List[SpanLike] = []

    for span in spans:
        span_attributes = span.attributes or {}
        is_match = True
        for link in links:
            # trace_id and span_id must be full match.
            if link.key_match == "trace_id":
                if isinstance(span, ReadableSpan):
                    trace_id = trace_api.format_trace_id(span.context.trace_id) if span.context else None
                else:
                    trace_id = span.trace_id
                if trace_id != link.value_match:
                    is_match = False
                    break

            elif link.key_match == "span_id":
                if isinstance(span, ReadableSpan):
                    span_id = trace_api.format_span_id(span.context.span_id) if span.context else None
                else:
                    span_id = span.span_id
                if span_id != link.value_match:
                    is_match = False
                    break

            else:
                attribute = span_attributes.get(link.key_match)
                # attributes must also be a full match currently.
                if attribute != link.value_match:
                    is_match = False
                    break

        if is_match:
            matched_spans.append(span)

    return matched_spans


def extract_links_from_attributes(attributes: Dict[str, Any]) -> List[LinkPydanticModel]:
    """Extract link attributes from flattened span attributes.

    Args:
        attributes: A dictionary of flattened span attributes.
    """
    maybe_link_list = filter_and_unflatten_attributes(attributes, LightningSpanAttributes.LINK.value)
    return TypeAdapter(List[LinkPydanticModel]).validate_python(maybe_link_list)


def filter_attributes(attributes: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Filter attributes that start with the given prefix.

    The attribute must start with `prefix.` or be exactly `prefix` to be included.

    Args:
        attributes: A dictionary of span attributes.
        prefix: The prefix to filter by.

    Returns:
        A dictionary of attributes that start with the given prefix.
    """
    return {k: v for k, v in attributes.items() if k.startswith(prefix + ".") or k == prefix}


def filter_and_unflatten_attributes(attributes: Dict[str, Any], prefix: str) -> Union[Dict[str, Any], List[Any]]:
    """Filter attributes that start with the given prefix and unflatten them.
    The prefix will be removed during unflattening.

    Args:
        attributes: A dictionary of span attributes.
        prefix: The prefix to filter by.

    Returns:
        A nested dictionary or list of attributes that start with the given prefix.
    """
    filtered_attributes = filter_attributes(attributes, prefix)
    stripped_attributes: Dict[str, Any] = {}
    for k, v in filtered_attributes.items():
        if k == prefix:
            raise ValueError(f"Cannot unflatten attribute with key exactly equal to prefix: {prefix}")
        else:
            stripped_key = k[len(prefix) + 1 :]  # +1 to remove the dot
        stripped_attributes[stripped_key] = v
    return unflatten_attributes(stripped_attributes)


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

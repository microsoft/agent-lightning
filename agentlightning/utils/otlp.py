# Copyright (c) Microsoft. All rights reserved.

import gzip
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from fastapi import Request, Response
from google.protobuf import json_format
from google.rpc.status_pb2 import Status
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource as ProtoResource
from opentelemetry.proto.trace.v1.trace_pb2 import Span as ProtoSpan
from opentelemetry.proto.trace.v1.trace_pb2 import Status as ProtoStatus
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.util.types import AttributeValue

from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import (
    Attributes,
    Event,
    Link,
    OtelResource,
    Span,
    SpanContext,
    SpanNames,
    TraceStatus,
    convert_timestamp,
)

PROTOBUF_CT = "application/x-protobuf"

logger = logging.getLogger(__name__)


T_request = TypeVar("T_request", ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest)
T_response = TypeVar("T_response", ExportLogsServiceResponse, ExportMetricsServiceResponse, ExportTraceServiceResponse)


async def handle_otlp_export(
    request: Request,
    request_message_cls: Type[T_request],
    response_message_cls: Type[T_response],
    message_callback: Optional[Callable[[T_request], Awaitable[None]]],
    signal_name: str,
) -> Response:
    """
    Generic handler for /v1/traces, /v1/metrics, /v1/logs.

    Convert the OTLP Protobuf request to a JSON-like object.
    """
    content_type = request.headers.get("Content-Type", "").split(";")[0].strip()

    if content_type != PROTOBUF_CT:
        # For brevity we only support binary protobuf here.
        return _bad_request_response(
            request,
            f"Unsupported Content-Type '{content_type}', expected '{PROTOBUF_CT}'",
            content_type=PROTOBUF_CT,
        )

    raw_body = await request.body()
    body = _read_body_maybe_gzip(request, raw_body)

    # Empty request is allowed and should still succeed.
    if not body:
        req_msg = request_message_cls()
    else:
        req_msg = request_message_cls()
        try:
            req_msg.ParseFromString(body)
        except Exception as exc:
            return _bad_request_response(request, f"Unable to parse OTLP {signal_name} payload: {exc}")

    if message_callback is not None:
        await message_callback(req_msg)

    # Build success response. Partial success field is left unset.
    resp_msg = response_message_cls()

    # Encode response in the same Content-Type as request.
    if content_type == PROTOBUF_CT:
        resp_bytes = resp_msg.SerializeToString()
    else:
        resp_bytes = json_format.MessageToJson(resp_msg).encode("utf-8")

    resp_bytes, headers = _maybe_gzip_response(request, resp_bytes)

    return Response(
        content=resp_bytes,
        media_type=content_type,
        status_code=200,
        headers=headers,
    )


async def spans_from_proto(request: ExportTraceServiceRequest, store: LightningStore) -> List[Span]:
    """Parse an OTLP proto payload into List[Span].

    A store is needed here for generating a sequence ID for each span.
    """
    output_spans: List[Span] = []

    for resource_spans in request.resource_spans:
        # Resource-level attributes & IDs
        resource_attrs = _kv_list_to_dict(resource_spans.resource.attributes)
        # rollout_id, attempt_id from resource attributes when present.
        rollout_id_resource = resource_attrs.get(SpanNames.ROLLOUT_ID)
        attempt_id_resource = resource_attrs.get(SpanNames.ATTEMPT_ID)
        # If sequence id is provided, all the spans will share the same sequence ID.
        # unless otherwise overridden by span-level attributes.
        sequence_id_resource = resource_attrs.get(SpanNames.SPAN_SEQUENCE_ID)

        otel_resource = _resource_from_proto(resource_spans.resource, getattr(resource_spans, "schema_url", ""))

        # Each ScopeSpans contains multiple spans
        for scope_spans in resource_spans.scope_spans:
            for proto_span in scope_spans.spans:
                trace_id_hex = _bytes_to_trace_id_hex(proto_span.trace_id)
                span_id_hex = _bytes_to_span_id_hex(proto_span.span_id)
                parent_id_hex = _bytes_to_span_id_hex(proto_span.parent_span_id) if proto_span.parent_span_id else None

                # Status
                status_code_str = _STATUS_CODE_MAP.get(proto_span.status.code, "UNSET")
                status = TraceStatus(
                    status_code=status_code_str,
                    description=proto_span.status.message or None,
                )

                # Attributes
                span_attrs = _kv_list_to_dict(proto_span.attributes)

                # Context
                context = SpanContext(
                    trace_id=trace_id_hex,
                    span_id=span_id_hex,
                    is_remote=False,
                    trace_state={},
                )

                # Try to get if span attributes contain something like rollout_id or attempt_id
                # Override the resource-level attributes with the span-level attributes if present.
                rollout_id_span = span_attrs.get(SpanNames.ROLLOUT_ID)
                attempt_id_span = span_attrs.get(SpanNames.ATTEMPT_ID)
                sequence_id_span = span_attrs.get(SpanNames.SPAN_SEQUENCE_ID)

                # Normalize to regular strings and ints
                rollout_id_raw = rollout_id_span if rollout_id_span is not None else rollout_id_resource
                attempt_id_raw = attempt_id_span if attempt_id_span is not None else attempt_id_resource
                sequence_id_raw = sequence_id_span if sequence_id_span is not None else sequence_id_resource

                rollout_id, attempt_id = _normalize_rollout_attempt_id(rollout_id_raw, attempt_id_raw)
                sequence_id = _normalize_sequence_id(sequence_id_raw)

                if rollout_id is None or attempt_id is None:
                    logger.warning(
                        "Both rollout_id and attempt_id must be present in resource attributes. "
                        "Spans will not be able to log to the store because of missing IDs: rollout_id=%s, attempt_id=%s, sequence_id=%s",
                        rollout_id,
                        attempt_id,
                        sequence_id,
                    )
                    continue

                # Generate a new sequence ID if not provided
                if sequence_id is None:
                    current_sequence_id = await store.get_next_span_sequence_id(
                        rollout_id=rollout_id, attempt_id=attempt_id
                    )
                else:
                    current_sequence_id = sequence_id

                # Build Span
                span = Span(
                    rollout_id=rollout_id,
                    attempt_id=attempt_id,
                    sequence_id=current_sequence_id,
                    trace_id=trace_id_hex,
                    span_id=span_id_hex,
                    parent_id=parent_id_hex,
                    name=proto_span.name,
                    status=status,
                    attributes=span_attrs,
                    events=_events_from_proto(proto_span),
                    links=_links_from_proto(proto_span),
                    start_time=convert_timestamp(proto_span.start_time_unix_nano),
                    end_time=convert_timestamp(proto_span.end_time_unix_nano),
                    context=context,
                    parent=None,  # OTLP only has parent_span_id; we don't have full SpanContext
                    resource=otel_resource,
                )

                output_spans.append(span)

    return output_spans


class LightningStoreOTLPExporter(OTLPSpanExporter):
    """OTLP Exporter that write to a LightningStore-compatible backend.

    The backend requires two special attributes on each span:

    - `agentlightning.rollout_id`: The rollout ID to associate the span with.
    - `agentlightning.attempt_id`: The attempt ID to associate the span with.

    It can optionally use the following attribute to sequence spans:

    - `agentlightning.span_sequence_id`: A decimal string representing the sequence ID of the span.
    """

    _default_endpoint: Optional[str] = None
    _rollout_id: Optional[str] = None
    _attempt_id: Optional[str] = None

    def enable_store_otlp(self, endpoint: str, rollout_id: str, attempt_id: str) -> None:
        """Enable storing OTLP data to a specific LightningStore rollout/attempt."""
        self._rollout_id = rollout_id
        self._attempt_id = attempt_id

        self._default_endpoint = self._endpoint
        self._endpoint = endpoint

    def disable_store_otlp(self) -> None:
        """Disable storing OTLP data to LightningStore."""
        self._rollout_id = None
        self._attempt_id = None
        if self._default_endpoint is not None:
            self._endpoint = self._default_endpoint

    def should_bypass(self) -> bool:
        """Check if the exporter should bypass the default export if rollout_id and attempt_id are not set."""
        return True

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._rollout_id is not None and self._attempt_id is not None:
            # rollout_id and attempt_id are present in resource attributes
            # It means that the server supports OTLP endpoint.
            for span in spans:
                # Override the resources so that the server knows where the request comes from.
                span._resource = span._resource.merge(  # pyright: ignore[reportPrivateUsage]
                    Resource.create(
                        {
                            SpanNames.ROLLOUT_ID: self._rollout_id,
                            SpanNames.ATTEMPT_ID: self._attempt_id,
                        }
                    )
                )
            return super().export(spans)
        elif not self.should_bypass():
            logger.debug("Rollout ID and Attempt ID not set; using default OTLP exporter behavior.")
            return super().export(spans)
        else:
            logger.debug("Rollout ID and Attempt ID not set; bypassing export.")
            return SpanExportResult.SUCCESS


def _read_body_maybe_gzip(request: Request, raw_body: bytes) -> bytes:
    """
    Decompress body if Content-Encoding: gzip; otherwise return as is.
    """
    encoding = request.headers.get("Content-Encoding", "").lower()
    if encoding == "gzip":
        return gzip.decompress(raw_body)
    return raw_body


def _maybe_gzip_response(request: Request, payload: bytes) -> Tuple[bytes, Dict[str, str]]:
    """
    If Accept-Encoding includes gzip, gzip the payload and set Content-Encoding header.
    """
    ae = request.headers.get("Accept-Encoding", "")
    tokens = [token.split(";")[0].strip().lower() for token in ae.split(",") if token.strip()]
    headers: Dict[str, str] = {}
    if "gzip" in tokens:
        payload = gzip.compress(payload)
        headers["Content-Encoding"] = "gzip"
    return payload, headers


def _bad_request_response(request: Request, message: str, content_type: str = PROTOBUF_CT) -> Response:
    """
    Build a 400 response whose body is a protobuf Status message, encoded
    in the same Content-Type as the request (OTLP/HTTP requirement).
    """
    status_msg = Status(message=message)

    if content_type == PROTOBUF_CT:
        body = status_msg.SerializeToString()
    else:
        # Fallback: JSON representation of Status.
        body = json_format.MessageToJson(status_msg).encode("utf-8")

    body, headers = _maybe_gzip_response(request, body)

    return Response(
        content=body,
        status_code=400,
        media_type=content_type,
        headers=headers,
    )


def _normalize_rollout_attempt_id(
    rollout_id: Optional[AttributeValue], attempt_id: Optional[AttributeValue]
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize a rollout or attempt ID to a string."""
    rollout_id_str = str(rollout_id) if rollout_id is not None else None
    attempt_id_str = str(attempt_id) if attempt_id is not None else None
    return rollout_id_str, attempt_id_str


def _normalize_sequence_id(sequence_id: Optional[AttributeValue]) -> Optional[int]:
    """Normalize a sequence ID to an integer."""
    if sequence_id is None:
        return None
    try:
        sequence_id_int = int(str(sequence_id))
    except (ValueError, TypeError):
        logger.warning(
            "Invalid sequence_id value in resource attributes: %r. Must be an integer or string representing an integer. Assuming None.",
            sequence_id,
        )
        sequence_id_int = None
    return sequence_id_int


def _any_value_to_python(value: AnyValue) -> Any:
    """Convert OTLP AnyValue -> plain Python value."""
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return int(value.int_value)
    if kind == "double_value":
        return float(value.double_value)
    if kind == "array_value":
        return [_any_value_to_python(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        # Map<string, AnyValue> -> dict
        return {kv.key: _any_value_to_python(kv.value) for kv in value.kvlist_value.values}
    if kind == "bytes_value":
        # Serialize bytes as hex string to stay JSON-friendly
        return value.bytes_value.hex()
    return None


def _kv_list_to_dict(kvs: Sequence[KeyValue]) -> Attributes:
    """Convert repeated KeyValue -> Attributes dict."""
    return {kv.key: _any_value_to_python(kv.value) for kv in kvs}


_STATUS_CODE_MAP = {
    ProtoStatus.STATUS_CODE_UNSET: "UNSET",
    ProtoStatus.STATUS_CODE_OK: "OK",
    ProtoStatus.STATUS_CODE_ERROR: "ERROR",
}


def _bytes_to_trace_id_hex(b: bytes) -> str:
    # OTLP uses 16-byte trace IDs; format as 32-char hex
    if not b:
        return "0" * 32
    return b.hex().rjust(32, "0")


def _bytes_to_span_id_hex(b: bytes) -> str:
    # OTLP uses 8-byte span IDs; format as 16-char hex
    if not b:
        return "0" * 16
    return b.hex().rjust(16, "0")


def _events_from_proto(span: ProtoSpan) -> List[Event]:
    """Event converter from OTLP ProtoSpan to List[Event]."""
    return [
        Event(
            name=e.name,
            attributes=_kv_list_to_dict(e.attributes),
            timestamp=convert_timestamp(e.time_unix_nano),
        )
        for e in span.events
    ]


def _links_from_proto(span: ProtoSpan) -> List[Link]:
    """Link converter from OTLP ProtoSpan to List[Link]."""
    links: List[Link] = []
    for link in span.links:
        trace_id_hex = _bytes_to_trace_id_hex(link.trace_id)
        span_id_hex = _bytes_to_span_id_hex(link.span_id)
        ctx = SpanContext(
            trace_id=trace_id_hex,
            span_id=span_id_hex,
            is_remote=False,
            trace_state={},  # OTLP trace_state is currently a string; you can parse if needed
        )
        links.append(
            Link(
                context=ctx,
                attributes=_kv_list_to_dict(link.attributes) or None,
            )
        )
    return links


def _resource_from_proto(resource: ProtoResource, schema_url: str = "") -> OtelResource:
    return OtelResource(
        attributes=_kv_list_to_dict(resource.attributes),
        schema_url=schema_url or "",
    )

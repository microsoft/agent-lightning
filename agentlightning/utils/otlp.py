# Copyright (c) Microsoft. All rights reserved.

import gzip
from typing import Awaitable, Callable, Dict, Optional, Tuple, Type, TypeVar

from fastapi import FastAPI, Request, Response
from google.protobuf import json_format
from google.rpc.status_pb2 import Status
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

app = FastAPI(title="Simple OTLP/HTTP Protobuf Collector")

PROTOBUF_CT = "application/x-protobuf"


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
    headers: Dict[str, str] = {}
    if "gzip" in ae.replace(" ", "").split(","):
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
        except Exception as exc:  # noqa: BLE001
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

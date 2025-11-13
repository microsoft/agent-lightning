import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from google.protobuf.json_format import MessageToDict
from google.rpc.status_pb2 import Status as PbStatus
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest as PbExportTraceServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse as PbExportTraceServiceResponse,
)

app = FastAPI()

SPAN_FILE = "spans.jsonl"
file_lock = asyncio.Lock()


def _get_scope_span_list(resource_span: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    OTLP used to use 'instrumentationLibrarySpans' and now uses 'scopeSpans'.
    Support both, just in case.
    """
    if "scopeSpans" in resource_span:
        return resource_span.get("scopeSpans", [])
    if "instrumentationLibrarySpans" in resource_span:
        return resource_span.get("instrumentationLibrarySpans", [])
    return []


def _extract_scope(scope_span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Newer name: scope, older: instrumentationLibrary
    return scope_span.get("scope") or scope_span.get("instrumentationLibrary")


async def write_spans_jsonl(body: Dict[str, Any]) -> None:
    """
    Take an OTLP ExportTraceServiceRequest (JSON-ish dict) and append each span
    as a JSON object in spans.jsonl (one span per line).
    """
    print(body)
    resource_spans: List[Dict[str, Any]] = body.get("resourceSpans", [])

    async with file_lock:
        with open(SPAN_FILE, "a", encoding="utf-8") as f:
            for rs in resource_spans:
                resource = rs.get("resource", {})
                scope_spans = _get_scope_span_list(rs)

                for ss in scope_spans:
                    scope = _extract_scope(ss)
                    spans = ss.get("spans", [])
                    for span in spans:
                        record = {
                            "resource": resource,
                            "scope": scope,
                            "span": span,
                        }
                        f.write(json.dumps(record))
                        f.write("\n")
                        f.flush()


def _make_status(
    code: int,
    message: str,
) -> PbStatus:
    """
    Build a google.rpc.Status message.

    NOTE: `code` is the gRPC status code (NOT HTTP status).
    For a quick demo we just stuff something appropriate:
      - e.g. INVALID_ARGUMENT (3) for 400 Bad Request.
    """
    status = PbStatus()
    status.code = code
    status.message = message
    return status


def _encode_status_response(
    status: PbStatus,
    content_type: str,
    http_status: int,
) -> Response:
    """
    Encode a google.rpc.Status with the same encoding as the request.
    """
    if "application/json" in content_type:
        body = MessageToDict(status, preserving_proto_field_name=True)
        return Response(
            content=json.dumps(body),
            media_type="application/json",
            status_code=http_status,
        )
    else:
        # Default to protobuf for non-JSON.
        body = status.SerializeToString()
        # Mirror the request Content-Type (x-protobuf or octet-stream)
        return Response(
            content=body,
            media_type=content_type,
            status_code=http_status,
        )


def _encode_export_response(
    content_type: str,
) -> Response:
    """
    Encode an (empty) ExportTraceServiceResponse.

    Spec: on full success, return HTTP 200 and an ExportTraceServiceResponse
    in the same encoding as the request. `partial_success` is unset.
    """
    resp = PbExportTraceServiceResponse()

    if "application/json" in content_type:
        body = MessageToDict(resp, preserving_proto_field_name=True)
        # Typically this will just be {} since partial_success is unset
        return Response(
            content=json.dumps(body),
            media_type="application/json",
            status_code=200,
        )
    else:
        body = resp.SerializeToString()
        return Response(
            content=body,
            media_type=content_type,
            status_code=200,
        )


@app.post("/v1/traces")
async def otlp_traces(request: Request):
    """
    OTLP/HTTP traces endpoint.

    Supports:
    - JSON: Content-Type: application/json
    - Protobuf: Content-Type: application/x-protobuf or application/octet-stream

    Response behaviour (per OTLP spec):
    - 200 OK on success (response is ExportTraceServiceResponse)
    - 400 Bad Request + google.rpc.Status if payload cannot be parsed/decoded
    - 415 Unsupported Media Type + google.rpc.Status if Content-Type unsupported
    """
    content_type = request.headers.get("content-type", "")

    # --- Unsupported content type → 415 + Status ---------------------------
    if not any(
        ct in content_type
        for ct in (
            "application/json",
            "application/x-protobuf",
            "application/octet-stream",
        )
    ):
        status = _make_status(
            code=3,  # INVALID_ARGUMENT
            message=f"Unsupported Content-Type: {content_type}",
        )
        return _encode_status_response(status, content_type or "application/x-protobuf", 415)

    # --- JSON path ---------------------------------------------------------
    if "application/json" in content_type:
        try:
            body_json = await request.json()
            print(body_json)
        except Exception as exc:
            status = _make_status(
                code=3,  # INVALID_ARGUMENT
                message=f"Invalid JSON payload: {exc}",
            )
            return _encode_status_response(status, "application/json", 400)

        # At this point the request is valid per our server, so we write spans.
        await write_spans_jsonl(body_json)
        return _encode_export_response("application/json")

    # --- Protobuf path -----------------------------------------------------
    if "application/x-protobuf" in content_type or "application/octet-stream" in content_type:
        raw = await request.body()

        pb_request = PbExportTraceServiceRequest()
        try:
            pb_request.ParseFromString(raw)
            print(pb_request)
        except Exception as exc:
            # Malformed protobuf → 400 + Status
            status = _make_status(
                code=3,  # INVALID_ARGUMENT
                message=f"Invalid protobuf payload: {exc}",
            )
            return _encode_status_response(status, content_type, 400)

        # Convert to dict and reuse JSON-based writing logic
        body_dict = MessageToDict(
            pb_request,
            preserving_proto_field_name=True,
        )

        await write_spans_jsonl(body_dict)
        return _encode_export_response(content_type)

    # Fallback (should not reach because of the earlier content-type checks)
    status = _make_status(
        code=3,
        message=f"Unsupported Content-Type: {content_type}",
    )
    return _encode_status_response(status, content_type or "application/x-protobuf", 415)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

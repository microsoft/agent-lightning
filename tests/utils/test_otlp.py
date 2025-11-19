# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

from __future__ import annotations

import gzip
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, cast

import pytest
from fastapi import Request
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
from starlette.types import Message, Scope

from agentlightning.store import LightningStore
from agentlightning.types.tracer import SpanNames
from agentlightning.utils import otlp

BASE_TIME_NANOS = 1_700_000_000_000_000_000
EVENT_TIME_OFFSET = 3_000_000_000
EVENT_TIME_SECONDS = (BASE_TIME_NANOS + EVENT_TIME_OFFSET) / 1_000_000_000
EXTRA_EVENT_TIME_OFFSET = 4_000_000_000
EXTRA_EVENT_TIME_SECONDS = (BASE_TIME_NANOS + EXTRA_EVENT_TIME_OFFSET) / 1_000_000_000


class _StubStore(LightningStore):
    def __init__(self) -> None:
        self.sequence_calls: List[tuple[str, str]] = []
        self.next_value = 1

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        self.sequence_calls.append((rollout_id, attempt_id))
        value = self.next_value
        self.next_value += 1
        return value


def _make_request(
    body: bytes,
    *,
    content_type: str = otlp.PROTOBUF_CT,
    content_encoding: Optional[str] = None,
    accept_encoding: Optional[str] = None,
) -> Request:
    headers: List[tuple[bytes, bytes]] = [(b"content-type", content_type.encode())]
    if content_encoding:
        headers.append((b"content-encoding", content_encoding.encode()))
    if accept_encoding:
        headers.append((b"accept-encoding", accept_encoding.encode()))

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "path": "/v1/test",
        "headers": headers,
    }

    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if body_sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _set_any_value(av: AnyValue, value: object) -> None:
    if isinstance(value, bool):
        av.bool_value = value
    elif isinstance(value, int):
        av.int_value = value
    elif isinstance(value, float):
        av.double_value = value
    elif isinstance(value, bytes):
        av.bytes_value = value
    elif isinstance(value, list):
        for item in cast(List[Any], value):
            _set_any_value(av.array_value.values.add(), item)
    elif isinstance(value, dict):
        for key, item in cast(Dict[str, Any], value).items():
            kv = av.kvlist_value.values.add()
            kv.key = key
            _set_any_value(kv.value, item)
    else:
        av.string_value = str(value)


def _add_attribute(attrs: Iterable[KeyValue], key: str, value: object) -> None:
    kv = attrs.add()  # type: ignore
    kv.key = key
    _set_any_value(kv.value, value)  # type: ignore


def _build_span_request() -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    _add_attribute(resource_spans.resource.attributes, SpanNames.ROLLOUT_ID, "resource-rollout")
    _add_attribute(resource_spans.resource.attributes, SpanNames.ATTEMPT_ID, "resource-attempt")
    _add_attribute(resource_spans.resource.attributes, SpanNames.SPAN_SEQUENCE_ID, "5")
    resource_spans.schema_url = "https://example/schema"

    scope_spans = resource_spans.scope_spans.add()
    span = scope_spans.spans.add()
    span.trace_id = bytes.fromhex("01" * 16)
    span.span_id = bytes.fromhex("02" * 8)
    span.parent_span_id = bytes.fromhex("03" * 8)
    span.name = "test-span"
    span.start_time_unix_nano = BASE_TIME_NANOS
    span.end_time_unix_nano = BASE_TIME_NANOS + 2_000_000_000
    span.status.code = ProtoStatus.STATUS_CODE_ERROR
    span.status.message = "boom"

    _add_attribute(span.attributes, "foo", "bar")
    _add_attribute(span.attributes, SpanNames.ROLLOUT_ID, "span-rollout")
    _add_attribute(span.attributes, SpanNames.ATTEMPT_ID, "span-attempt")
    _add_attribute(span.attributes, SpanNames.SPAN_SEQUENCE_ID, "7")

    event = span.events.add()
    event.name = "event"
    event.time_unix_nano = BASE_TIME_NANOS + EVENT_TIME_OFFSET
    _add_attribute(event.attributes, "event-attr", 9)

    link = span.links.add()
    link.trace_id = bytes.fromhex("04" * 16)
    link.span_id = bytes.fromhex("05" * 8)
    _add_attribute(link.attributes, "link-attr", True)

    return request


@pytest.mark.asyncio
async def test_handle_otlp_export_success_with_gzip_response() -> None:
    request_msg = _build_span_request()
    body = request_msg.SerializeToString()
    request = _make_request(
        body,
        accept_encoding="gzip;q=0.9,br",
    )

    received: List[ExportTraceServiceRequest] = []

    async def callback(message: ExportTraceServiceRequest) -> None:
        received.append(message)

    response = await otlp.handle_otlp_export(
        request,
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
        callback,
        signal_name="traces",
    )

    assert response.status_code == 200
    assert received and received[0].SerializeToString() == body
    assert response.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(response.body) == ExportTraceServiceResponse().SerializeToString()


@pytest.mark.asyncio
async def test_handle_otlp_export_rejects_invalid_content_type() -> None:
    request = _make_request(b"{}", content_type="application/json")

    response = await otlp.handle_otlp_export(
        request,
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
        None,
        signal_name="traces",
    )

    assert response.status_code == 400
    status = otlp.Status()  # type: ignore[attr-defined]
    status.ParseFromString(response.body)  # type: ignore
    assert "Unsupported Content-Type" in status.message


@pytest.mark.asyncio
async def test_handle_otlp_export_rejects_bad_payload() -> None:
    request = _make_request(b"not-a-proto")

    response = await otlp.handle_otlp_export(
        request,
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
        None,
        signal_name="traces",
    )

    assert response.status_code == 400
    status = otlp.Status()  # type: ignore[attr-defined]
    status.ParseFromString(response.body)  # type: ignore
    assert "Unable to parse" in status.message


@pytest.mark.asyncio
async def test_handle_otlp_export_accepts_gzip_body() -> None:
    request_msg = ExportTraceServiceRequest()
    request_msg.resource_spans.add()
    gz_body = gzip.compress(request_msg.SerializeToString())
    request = _make_request(gz_body, content_encoding="gzip")

    response = await otlp.handle_otlp_export(
        request,
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
        None,
        signal_name="traces",
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_spans_from_proto_prefers_span_level_metadata() -> None:
    store = _StubStore()
    request = _build_span_request()

    spans = await otlp.spans_from_proto(request, store)

    assert len(spans) == 1
    span = spans[0]
    assert span.rollout_id == "span-rollout"
    assert span.attempt_id == "span-attempt"
    assert span.sequence_id == 7
    assert span.status.status_code == "ERROR"
    assert span.events[0].timestamp == pytest.approx(EVENT_TIME_SECONDS)  # type: ignore
    assert span.links[0].context.trace_id == "0404" * 8
    assert span.links[0].attributes == {"link-attr": True}
    assert span.resource.attributes[SpanNames.ROLLOUT_ID] == "resource-rollout"
    assert span.resource.schema_url == "https://example/schema"
    assert not store.sequence_calls


@pytest.mark.asyncio
async def test_spans_from_proto_requests_sequence_ids_when_missing() -> None:
    store = _StubStore()
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    _add_attribute(resource_spans.resource.attributes, SpanNames.ROLLOUT_ID, "r1")
    _add_attribute(resource_spans.resource.attributes, SpanNames.ATTEMPT_ID, "a1")

    scope_span = resource_spans.scope_spans.add()
    span = scope_span.spans.add()
    span.trace_id = b""  # exercise default ids
    span.span_id = b""
    span.name = "needs-seq"

    spans = await otlp.spans_from_proto(request, store)

    assert len(spans) == 1
    assert spans[0].sequence_id == 1
    assert store.sequence_calls == [("r1", "a1")]


@pytest.mark.asyncio
async def test_spans_from_proto_skips_spans_without_ids() -> None:
    store = _StubStore()
    request = ExportTraceServiceRequest()
    request.resource_spans.add()  # missing rollout and attempt

    spans = await otlp.spans_from_proto(request, store)

    assert spans == []
    assert store.sequence_calls == []


def test_normalize_sequence_id_handles_bad_values(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    assert otlp._normalize_sequence_id("not-int") is None
    assert any("Invalid sequence_id" in record.message for record in caplog.records)


def test_any_value_to_python_full_roundtrip() -> None:
    av = AnyValue()
    _set_any_value(
        av,
        {
            "s": "hello",
            "b": True,
            "i": 5,
            "d": 1.5,
            "arr": ["x", 2],
            "nested": {"k": b"\x01"},
        },
    )

    result = otlp._any_value_to_python(av)
    assert result == {
        "s": "hello",
        "b": True,
        "i": 5,
        "d": 1.5,
        "arr": ["x", 2],
        "nested": {"k": "01"},
    }


def test_kv_list_to_dict_converts_values() -> None:
    resource = ProtoResource()
    _add_attribute(resource.attributes, "num", 10)
    _add_attribute(resource.attributes, "flag", False)

    converted = otlp._kv_list_to_dict(resource.attributes)
    assert converted == {"num": 10, "flag": False}


def test_bytes_to_hex_helpers() -> None:
    assert otlp._bytes_to_trace_id_hex(b"") == "0" * 32
    assert otlp._bytes_to_span_id_hex(b"") == "0" * 16
    assert otlp._bytes_to_trace_id_hex(b"\xff") == "ff".rjust(32, "0")
    assert otlp._bytes_to_span_id_hex(b"\xaa") == "aa".rjust(16, "0")


def test_events_and_links_from_proto() -> None:
    span = ProtoSpan()
    event = span.events.add()
    event.name = "evt"
    event.time_unix_nano = BASE_TIME_NANOS + EXTRA_EVENT_TIME_OFFSET
    _add_attribute(event.attributes, "alpha", "beta")

    link = span.links.add()
    link.trace_id = bytes.fromhex("06" * 16)
    link.span_id = bytes.fromhex("07" * 8)
    _add_attribute(link.attributes, "delta", 1)

    events = otlp._events_from_proto(span)
    links = otlp._links_from_proto(span)

    assert events[0].timestamp == pytest.approx(EXTRA_EVENT_TIME_SECONDS)  # type: ignore
    assert events[0].attributes == {"alpha": "beta"}
    assert links[0].context.trace_id == "0606" * 8
    assert links[0].attributes == {"delta": 1}


def test_resource_from_proto() -> None:
    resource = ProtoResource()
    _add_attribute(resource.attributes, "key", "value")
    result = otlp._resource_from_proto(resource, schema_url="https://example/schema")
    assert result.attributes == {"key": "value"}
    assert result.schema_url == "https://example/schema"


def test_maybe_gzip_response_parses_quality_values() -> None:
    request = SimpleNamespace(headers={"Accept-Encoding": "br, gzip;q=0.1"})
    payload = b"payload"
    compressed, headers = otlp._maybe_gzip_response(cast(Request, request), payload)

    assert headers == {"Content-Encoding": "gzip"}
    assert gzip.decompress(compressed) == payload


def test_bad_request_response_matches_request_encoding() -> None:
    request = SimpleNamespace(headers={"Accept-Encoding": "gzip"})
    response = otlp._bad_request_response(cast(Request, request), "error")

    assert response.status_code == 400
    assert response.media_type == otlp.PROTOBUF_CT
    status = otlp.Status()  # type: ignore[attr-defined]
    status.ParseFromString(gzip.decompress(response.body))
    assert status.message == "error"


class _DummyReadableSpan:
    def __init__(self) -> None:
        self._resource = Resource.create({"existing": "value"})


def test_lightning_store_otlp_exporter_overrides_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = otlp.LightningStoreOTLPExporter(endpoint="http://collector")

    captured_spans: List[List[_DummyReadableSpan]] = []

    def fake_export(self: otlp.LightningStoreOTLPExporter, spans: List[_DummyReadableSpan]) -> SpanExportResult:
        captured_spans.append(spans)
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(otlp.OTLPSpanExporter, "export", fake_export, raising=False)

    exporter.enable_store_otlp("http://store", "rollout", "attempt")
    span = _DummyReadableSpan()

    result = exporter.export([cast(ReadableSpan, span)])

    assert result == SpanExportResult.SUCCESS
    assert captured_spans
    attributes = captured_spans[0][0]._resource.attributes  # type: ignore[attr-defined]
    assert attributes[SpanNames.ROLLOUT_ID] == "rollout"
    assert attributes[SpanNames.ATTEMPT_ID] == "attempt"

    exporter.disable_store_otlp()
    assert exporter._rollout_id is None
    assert exporter._attempt_id is None

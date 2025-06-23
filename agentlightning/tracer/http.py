from contextlib import contextmanager
from typing import Iterator, List, Optional, Any, Dict
import logging
import uuid
from urllib.parse import urlparse

from .base import BaseTracer

from httpdbg.hooks.all import httprecord
from httpdbg.records import HTTPRecords
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode, SpanKind, Status
from opentelemetry.trace.span import (
    SpanContext,
    TraceFlags,
    TraceState,
)


logger = logging.getLogger(__name__)


class HttpTracer(BaseTracer):
    """
    A tracer implementation that captures HTTP requests using httpdbg.

    This tracer hooks into the Python HTTP libraries and captures all
    HTTP requests and responses made during the traced code execution.
    The captured requests are converted to OpenTelemetry spans for
    compatibility with the rest of the tracing ecosystem.

    Attributes:
        include_headers: Whether to include HTTP headers in the spans.
            Headers may contain sensitive information. Use with caution.
        include_body: Whether to include HTTP request and response bodies in the spans.
            Bodies may be large and contain sensitive information. Use with caution.
        include_agentlightning_requests: Whether to include requests initiated by AgentLightning itself.
    """

    AGENTLIGHTNING_HEADERS = {"x-agentlightning-client"}

    def __init__(
        self, include_headers: bool = False, include_body: bool = False, include_agentlightning_requests: bool = False
    ):
        super().__init__()
        self._last_records = None
        self.include_headers = include_headers
        self.include_body = include_body
        self.include_agentlightning_requests = include_agentlightning_requests

    def init_worker(self, worker_id: int):
        """
        Initialize the tracer in a worker process.

        Args:
            worker_id: The ID of the worker process.
        """
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] HttpTracer initialized.")

    @contextmanager
    def trace_context(self, name: Optional[str] = None) -> Iterator[HTTPRecords]:
        """
        Starts a new HTTP tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.

        Yields:
            The HTTPRecords instance containing traced HTTP activities.
        """
        records = HTTPRecords()
        with httprecord(records):
            self._last_records = records
            yield records

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects converted from HTTP records.
        """
        if self._last_records is None:
            return []

        return self._convert_to_spans(self._last_records)

    def _convert_to_spans(self, records: HTTPRecords) -> List[ReadableSpan]:
        """
        Convert HTTPRecords to OpenTelemetry spans.

        Args:
            records: The HTTPRecords instance containing HTTP traces.

        Returns:
            A list of ReadableSpan objects representing the HTTP activities.
        """
        spans = []

        # Create a trace ID that will be shared by all spans in this trace
        trace_id = int(uuid.uuid4().hex[:16], 16)

        for record in records.requests.values():
            # Skip AgentLightning requests if include_agentlightning_requests is False
            should_skip = False
            if not self.include_agentlightning_requests and record.request and record.request.headers:
                for header in record.request.headers:
                    if header.name.lower() in self.AGENTLIGHTNING_HEADERS and header.value.lower() == "true":
                        should_skip = True
                        break

            if should_skip:
                continue

            # Create a span ID for this specific HTTP request
            span_id = int(uuid.uuid4().hex[:8], 16)

            # Create a span context
            span_context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )

            # Extract important information from the HTTP record
            method = record.method
            url = record.url
            parsed_url = urlparse(url)
            status_code = record.status_code

            # Create attributes dictionary
            attributes: Dict[str, Any] = {
                "http.method": method,
                "http.url": url,
                "http.target": parsed_url.path,
                "http.host": parsed_url.netloc,
            }

            if status_code is not None and status_code > 0:
                attributes["http.status_code"] = status_code

            # Calculate duration - from begin time to last update
            duration = None
            if hasattr(record, "last_update") and record.last_update and record.tbegin:
                duration = (record.last_update - record.tbegin).total_seconds()
                attributes["http.duration_ms"] = duration * 1000  # Convert to ms

            # Optionally include headers
            if self.include_headers and record.request and record.request.headers:
                for header in record.request.headers:
                    header_name = header.name.lower()
                    attributes[f"http.request.header.{header_name}"] = header.value

            if self.include_headers and record.response and record.response.headers:
                for header in record.response.headers:
                    header_name = header.name.lower()
                    attributes[f"http.response.header.{header_name}"] = header.value

            # Optionally include body - preserve complete content for analysis
            if self.include_body and record.request:
                body_content = record.request.content
                if body_content:
                    # Store raw body content for later parsing/analysis
                    attributes["http.request.body"] = body_content

            if self.include_body and record.response:
                body_content = record.response.content
                if body_content:
                    # Store raw body content for later parsing/analysis 
                    attributes["http.response.body"] = body_content

            # Determine span status
            span_status = StatusCode.OK
            if status_code and status_code >= 400 or record.exception:
                span_status = StatusCode.ERROR

            # Create start and end timestamps in nanoseconds
            # If we have duration, use it, otherwise default to current time - 1ms
            start_time_ns = int(record.tbegin.timestamp() * 1e9)
            if duration:
                end_time_ns = int((record.tbegin.timestamp() + duration) * 1e9)
            else:
                end_time_ns = int(record.last_update.timestamp() * 1e9)

            span = ReadableSpan(
                name=f"HTTP {method} {url}",
                context=span_context,
                parent=None,
                kind=SpanKind.CLIENT,
                status=Status(span_status),
                start_time=start_time_ns,
                end_time=end_time_ns,
                attributes=attributes,
                events=[],
                links=[],
                resource=None,
            )

            spans.append(span)

        return spans

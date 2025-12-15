# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import (
    Iterator,
    Optional,
)

from agentlightning.types import (
    Attributes,
    SpanCoreFields,
    SpanRecordingContext,
    StatusCode,
    TraceStatus,
)
from agentlightning.utils.otel import format_exception_attributes

from .base import Tracer

logger = logging.getLogger(__name__)


class DummySpanRecordingContext(SpanRecordingContext):
    """Context for recording operations on a dummy span, not dependent on any backend tracer."""

    def __init__(self, name: str, attributes: Optional[Attributes] = None, start_time: Optional[float] = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.start_time = start_time or time.time()
        self.end_time = None
        self.status = TraceStatus(status_code="OK")

    def record_exception(self, exception: BaseException) -> None:
        self.record_status("ERROR", str(exception))
        self.record_attributes(format_exception_attributes(exception))

    def record_attributes(self, attributes: Attributes) -> None:
        self.attributes.update(attributes)

    def record_status(self, status_code: StatusCode, description: Optional[str] = None) -> None:
        self.status = TraceStatus(status_code=status_code, description=description)

    def finalize(self, end_time: Optional[float] = None) -> None:
        self.end_time = end_time or time.time()

    def get_recorded_span(self) -> SpanCoreFields:
        if self.end_time is None:
            raise ValueError("End time is not set. Call finalize() first.")
        return SpanCoreFields(
            name=self.name,
            attributes=self.attributes,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
        )


class DummyTracer(Tracer):
    """A dummy tracer that does not trace anything, but it is compatible with the emitter API.

    It doesn't rely on any backend tracer, and also doesn't use any stores.
    """

    def create_span(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        if attributes is None:
            attributes = {}
        if timestamp is None:
            timestamp = time.time()
        if status is None:
            status = TraceStatus(status_code="OK")
        return SpanCoreFields(
            name=name,
            attributes=attributes,
            start_time=timestamp,
            end_time=timestamp,
            status=status,
        )

    @contextmanager
    def operation_context(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[DummySpanRecordingContext]:
        start_time = start_time or time.time()
        recording_context = DummySpanRecordingContext(name, attributes, start_time)
        try:
            yield recording_context
        except Exception as exc:
            recording_context.record_exception(exc)
            recording_context.record_status("ERROR", str(exc))
            raise
        finally:
            recording_context.finalize(end_time)

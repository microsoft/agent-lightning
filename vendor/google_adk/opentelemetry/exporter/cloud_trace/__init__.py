# Copyright 2021 The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud Trace Span Exporter for OpenTelemetry. Uses Cloud Trace Client's REST
API to export traces and spans for viewing in Cloud Trace.

Usage
-----

.. code-block:: python

    from opentelemetry import trace
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.sdk.trace import TracerProvider

    # For debugging
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    # Otherwise
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    trace.set_tracer_provider(TracerProvider())

    cloud_trace_exporter = CloudTraceSpanExporter()
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(cloud_trace_exporter)
    )
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("foo"):
        print("Hello world!")


When not debugging, make sure to use
:class:`opentelemetry.sdk.trace.export.BatchSpanProcessor` with the
default parameters for performance reasons.

Auto-instrumentation
--------------------

This exporter can also be used with `OpenTelemetry auto-instrumentation
<https://opentelemetry.io/docs/instrumentation/python/automatic/>`_:

.. code-block:: sh

    opentelemetry-instrument --traces_exporter gcp_trace <command> <args>

Configuration is supported through environment variables
(:mod:`opentelemetry.exporter.cloud_trace.environment_variables`) or the corresponding command
line arguments to ``opentelemetry-instrument``:

.. code-block:: sh

    opentelemetry-instrument --traces_exporter gcp_trace \\
        --exporter_gcp_trace_project_id my-project \\
        <command> <args>

See ``opentelemetry-instrument --help`` for all configuration options.

API
---
"""

import logging
import re
from collections.abc import Sequence as SequenceABC
from os import environ
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    overload,
)

import google.auth
import opentelemetry.trace as trace_api
from google.cloud.trace_v2 import BatchWriteSpansRequest, TraceServiceClient
from google.cloud.trace_v2 import types as trace_types
from google.cloud.trace_v2.services.trace_service.transports import (
    TraceServiceGrpcTransport,
)
from google.protobuf.timestamp_pb2 import (  # pylint: disable=no-name-in-module
    Timestamp,
)
from google.rpc import code_pb2, status_pb2
from opentelemetry.exporter.cloud_trace.environment_variables import (
    OTEL_EXPORTER_GCP_TRACE_PROJECT_ID,
    OTEL_EXPORTER_GCP_TRACE_RESOURCE_REGEX,
)
from opentelemetry.exporter.cloud_trace.version import __version__
from opentelemetry.resourcedetector.gcp_resource_detector import (
    _constants as _resource_constants,
)
from opentelemetry.resourcedetector.gcp_resource_detector._mapping import (
    get_monitored_resource,
)
from opentelemetry.sdk import version as opentelemetry_sdk_version
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event
from opentelemetry.sdk.trace.export import (
    ReadableSpan,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.util import BoundedDict
from opentelemetry.trace import format_span_id, format_trace_id
from opentelemetry.trace.status import StatusCode
from opentelemetry.util import types

logger = logging.getLogger(__name__)

_OTEL_SDK_VERSION = opentelemetry_sdk_version.__version__
_USER_AGENT = f"opentelemetry-python {_OTEL_SDK_VERSION}; google-cloud-trace-exporter {__version__}"

# Set user-agent metadata, see https://github.com/grpc/grpc/issues/23644 and default options
# from
# https://github.com/googleapis/python-trace/blob/v1.7.3/google/cloud/trace_v1/services/trace_service/transports/grpc.py#L177-L180
_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.primary_user_agent", _USER_AGENT),
]


MAX_NUM_LINKS = 128
MAX_NUM_EVENTS = 32
MAX_EVENT_ATTRS = 4
MAX_LINK_ATTRS = 32
MAX_SPAN_ATTRS = 32
MAX_ATTR_KEY_BYTES = 128
MAX_ATTR_VAL_BYTES = 16 * 1024  # 16 kilobytes


def _create_default_client() -> TraceServiceClient:
    return TraceServiceClient(
        transport=TraceServiceGrpcTransport(
            channel=TraceServiceGrpcTransport.create_channel(options=_OPTIONS)
        )
    )


class CloudTraceSpanExporter(SpanExporter):
    """Cloud Trace span exporter for OpenTelemetry.

    Args:
        project_id: GCP project ID for the project to send spans to. Alternatively, can be
            configured with :envvar:`OTEL_EXPORTER_GCP_TRACE_PROJECT_ID`.
        client: Cloud Trace client. If not given, will be taken from gcloud
            default credentials
        resource_regex: Resource attributes with keys matching this regex will be added to
            exported spans as labels (default: None). Alternatively, can be configured with
            :envvar:`OTEL_EXPORTER_GCP_TRACE_RESOURCE_REGEX`.
    """

    def __init__(
        self,
        project_id=None,
        client=None,
        resource_regex=None,
    ):
        self.client: TraceServiceClient = client or _create_default_client()

        if not project_id:
            project_id = environ.get(OTEL_EXPORTER_GCP_TRACE_PROJECT_ID)
        if not project_id:
            _, project_id = google.auth.default()
        self.project_id = project_id

        if not resource_regex:
            resource_regex = environ.get(
                OTEL_EXPORTER_GCP_TRACE_RESOURCE_REGEX
            )
        self.resource_regex = (
            re.compile(resource_regex) if resource_regex else None
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export the spans to Cloud Trace.

        See: https://cloud.google.com/trace/docs/reference/v2/rest/v2/projects.traces/batchWrite

        Args:
            spans: Sequence of spans to export
        """
        try:
            self.client.batch_write_spans(
                request=BatchWriteSpansRequest(
                    name="projects/{}".format(self.project_id),
                    spans=self._translate_to_cloud_trace(spans),
                )
            )
        # pylint: disable=broad-except
        except Exception as ex:
            logger.error("Error while writing to Cloud Trace", exc_info=ex)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def _translate_to_cloud_trace(
        self, spans: Sequence[ReadableSpan]
    ) -> List[trace_types.Span]:
        """Translate the spans to Cloud Trace format.

        Args:
            spans: Sequence of spans to convert
        """

        cloud_trace_spans: List[trace_types.Span] = []

        for span in spans:
            ctx = span.get_span_context()
            trace_id = format_trace_id(ctx.trace_id)
            span_id = format_span_id(ctx.span_id)
            span_name = "projects/{}/traces/{}/spans/{}".format(
                self.project_id, trace_id, span_id
            )

            parent_id = None
            if span.parent:
                parent_id = format_span_id(span.parent.span_id)

            start_time = _get_time_from_ns(span.start_time)
            end_time = _get_time_from_ns(span.end_time)

            if span.attributes and len(span.attributes) > MAX_SPAN_ATTRS:
                logger.warning(
                    "Span has more then %s attributes, some will be truncated",
                    MAX_SPAN_ATTRS,
                )

            # Span does not support a MonitoredResource object. We put the
            # information into attributes instead.
            resources_and_attrs = {
                **(span.attributes or {}),
                **_extract_resources(span.resource, self.resource_regex),
            }

            cloud_trace_spans.append(
                trace_types.Span(
                    name=span_name,
                    span_id=span_id,
                    display_name=_get_truncatable_str_object(span.name, 128),
                    start_time=start_time,
                    end_time=end_time,
                    parent_span_id=parent_id,
                    attributes=_extract_attributes(
                        resources_and_attrs,
                        MAX_SPAN_ATTRS,
                        add_agent_attr=True,
                    ),
                    links=_extract_links(span.links),
                    status=_extract_status(span.status),
                    time_events=_extract_events(span.events),
                    span_kind=_extract_span_kind(span.kind),
                )
            )
            # TODO: Leverage more of the Cloud Trace API, e.g.
            #  same_process_as_parent_span and child_span_count

        return cloud_trace_spans

    def shutdown(self):
        pass


def _get_time_from_ns(nanoseconds: Optional[int]) -> Optional[Timestamp]:
    """Given epoch nanoseconds, split into epoch milliseconds and remaining
    nanoseconds"""
    if not nanoseconds:
        return None
    ts = Timestamp()
    # pylint: disable=no-member
    ts.FromNanoseconds(nanoseconds)
    return ts


def _get_truncatable_str_object(str_to_convert: str, max_length: int):
    """Truncate the string if it exceeds the length limit and record the
    truncated bytes count."""
    truncated, truncated_byte_count = _truncate_str(str_to_convert, max_length)

    return trace_types.TruncatableString(
        value=truncated, truncated_byte_count=truncated_byte_count
    )


def _truncate_str(str_to_check: str, limit: int) -> Tuple[str, int]:
    """Check the length of a string. If exceeds limit, then truncate it."""
    encoded = str_to_check.encode("utf-8")
    truncated_str = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated_str, len(encoded) - len(truncated_str.encode("utf-8"))


def _extract_status(status: trace_api.Status) -> Optional[status_pb2.Status]:
    """Convert a OTel Status to protobuf Status."""
    if status.status_code is StatusCode.UNSET:
        status_proto = None
    elif status.status_code is StatusCode.OK:
        status_proto = status_pb2.Status(code=code_pb2.OK)
    elif status.status_code is StatusCode.ERROR:
        status_proto = status_pb2.Status(
            code=code_pb2.UNKNOWN, message=status.description or ""
        )
    # future added value
    else:
        logger.info(
            "Couldn't handle OTel status code %s, assuming error",
            status.status_code,
        )
        status_proto = status_pb2.Status(
            code=code_pb2.UNKNOWN, message=status.description or ""
        )

    return status_proto


def _extract_links(
    links: Sequence[trace_api.Link],
) -> Optional[trace_types.Span.Links]:
    """Convert span.links"""
    if not links:
        return None
    extracted_links: List[trace_types.Span.Link] = []
    dropped_links = 0
    if len(links) > MAX_NUM_LINKS:
        logger.warning(
            "Exporting more then %s links, some will be truncated",
            MAX_NUM_LINKS,
        )
        dropped_links = len(links) - MAX_NUM_LINKS
        links = links[:MAX_NUM_LINKS]
    for link in links:
        link_attributes = link.attributes or {}
        if len(link_attributes) > MAX_LINK_ATTRS:
            logger.warning(
                "Link has more then %s attributes, some will be truncated",
                MAX_LINK_ATTRS,
            )
        trace_id = format_trace_id(link.context.trace_id)
        span_id = format_span_id(link.context.span_id)
        extracted_links.append(
            trace_types.Span.Link(
                trace_id=trace_id,
                span_id=span_id,
                type="TYPE_UNSPECIFIED",
                attributes=_extract_attributes(
                    link_attributes, MAX_LINK_ATTRS
                ),
            )
        )
    return trace_types.Span.Links(
        link=extracted_links, dropped_links_count=dropped_links
    )


def _extract_events(
    events: Sequence[Event],
) -> Optional[trace_types.Span.TimeEvents]:
    """Convert span.events to dict."""
    if not events:
        return None
    time_events: List[trace_types.Span.TimeEvent] = []
    dropped_annontations = 0
    if len(events) > MAX_NUM_EVENTS:
        logger.warning(
            "Exporting more then %s annotations, some will be truncated",
            MAX_NUM_EVENTS,
        )
        dropped_annontations = len(events) - MAX_NUM_EVENTS
        events = events[:MAX_NUM_EVENTS]
    for event in events:
        if event.attributes and len(event.attributes) > MAX_EVENT_ATTRS:
            logger.warning(
                "Event %s has more then %s attributes, some will be truncated",
                event.name,
                MAX_EVENT_ATTRS,
            )
        time_events.append(
            trace_types.Span.TimeEvent(
                time=_get_time_from_ns(event.timestamp),
                annotation=trace_types.Span.TimeEvent.Annotation(
                    description=_get_truncatable_str_object(event.name, 256),
                    attributes=_extract_attributes(
                        event.attributes, MAX_EVENT_ATTRS
                    ),
                ),
            )
        )
    return trace_types.Span.TimeEvents(
        time_event=time_events,
        dropped_annotations_count=dropped_annontations,
        dropped_message_events_count=0,
    )


# pylint: disable=no-member
SPAN_KIND_MAPPING = {
    trace_api.SpanKind.INTERNAL: trace_types.Span.SpanKind.INTERNAL,
    trace_api.SpanKind.CLIENT: trace_types.Span.SpanKind.CLIENT,
    trace_api.SpanKind.SERVER: trace_types.Span.SpanKind.SERVER,
    trace_api.SpanKind.PRODUCER: trace_types.Span.SpanKind.PRODUCER,
    trace_api.SpanKind.CONSUMER: trace_types.Span.SpanKind.CONSUMER,
}


# pylint: disable=no-member
def _extract_span_kind(
    span_kind: trace_api.SpanKind,
) -> int:
    return SPAN_KIND_MAPPING.get(
        span_kind, trace_types.Span.SpanKind.SPAN_KIND_UNSPECIFIED
    )


def _strip_characters(ot_version):
    return "".join(filter(lambda x: x.isdigit() or x == ".", ot_version))


def _extract_resources(
    resource: Resource, resource_regex: Optional[Pattern] = None
) -> Dict[str, str]:
    extracted_attributes = {}
    resource_attributes = resource.attributes
    if resource_regex:
        extracted_attributes.update(
            {
                k: str(v)
                for k, v in resource_attributes.items()
                if resource_regex.match(k)
            }
        )
    monitored_resource = get_monitored_resource(resource)
    # Do not map generic_task and generic_node to g.co/r/... span labels.
    if monitored_resource and monitored_resource.type not in (
        _resource_constants.GENERIC_NODE,
        _resource_constants.GENERIC_TASK,
    ):
        extracted_attributes.update(
            {
                "g.co/r/{}/{}".format(monitored_resource.type, k): v
                for k, v in monitored_resource.labels.items()
            }
        )
    return extracted_attributes


LABELS_MAPPING = {
    # this one might be http.flavor? I'm not sure
    "http.scheme": "/http/client_protocol",
    "http.host": "/http/host",
    "http.method": "/http/method",
    # https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/semantic_conventions/http.md#common-attributes
    "http.request_content_length": "/http/request/size",
    "http.response_content_length": "/http/response/size",
    "http.route": "/http/route",
    "http.status_code": "/http/status_code",
    "http.url": "/http/url",
    "http.user_agent": "/http/user_agent",
}


def _extract_attributes(
    attrs: types.Attributes,
    num_attrs_limit: int,
    add_agent_attr: bool = False,
) -> trace_types.Span.Attributes:
    """Convert span.attributes to dict."""
    attributes_dict: BoundedDict[
        str, trace_types.AttributeValue
    ] = BoundedDict(num_attrs_limit)
    invalid_value_dropped_count = 0
    for ot_key, ot_value in attrs.items() if attrs else []:
        key = _truncate_str(ot_key, MAX_ATTR_KEY_BYTES)[0]
        if key in LABELS_MAPPING:  # pylint: disable=consider-using-get
            key = LABELS_MAPPING[key]
        value = _format_attribute_value(ot_value)

        if value is not None:
            attributes_dict[key] = value
        else:
            invalid_value_dropped_count += 1
    if add_agent_attr:
        attributes_dict["g.co/agent"] = _format_attribute_value(
            "opentelemetry-python {}; google-cloud-trace-exporter {}".format(
                _strip_characters(_OTEL_SDK_VERSION),
                _strip_characters(__version__),
            )
        )
    return trace_types.Span.Attributes(
        attribute_map=dict(attributes_dict),
        dropped_attributes_count=attributes_dict.dropped
        + invalid_value_dropped_count,
    )


@overload
def _format_attribute_value(
    value: types.AttributeValue,
) -> trace_types.AttributeValue:
    ...


@overload
def _format_attribute_value(
    value: Any,
) -> Optional[trace_types.AttributeValue]:
    ...


def _format_attribute_value(
    value,
) -> Optional[trace_types.AttributeValue]:
    if isinstance(value, bool):
        value_type = "bool_value"
    elif isinstance(value, int):
        value_type = "int_value"
    elif isinstance(value, str):
        value_type = "string_value"
        value = _get_truncatable_str_object(value, MAX_ATTR_VAL_BYTES)
    elif isinstance(value, float):
        value_type = "string_value"
        value = _get_truncatable_str_object(
            "{:0.4f}".format(value), MAX_ATTR_VAL_BYTES
        )
    elif isinstance(value, SequenceABC):
        value_type = "string_value"
        value = _get_truncatable_str_object(
            ",".join(str(x) for x in value), MAX_ATTR_VAL_BYTES
        )
    else:
        logger.warning(
            "ignoring attribute value %s of type %s. Values type must be one "
            "of bool, int, string or float, or a sequence of these",
            value,
            type(value),
        )
        return None

    return trace_types.AttributeValue(**{value_type: value})

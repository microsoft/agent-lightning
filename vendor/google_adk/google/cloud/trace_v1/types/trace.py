# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.devtools.cloudtrace.v1",
    manifest={
        "Trace",
        "Traces",
        "TraceSpan",
        "ListTracesRequest",
        "ListTracesResponse",
        "GetTraceRequest",
        "PatchTracesRequest",
    },
)


class Trace(proto.Message):
    r"""A trace describes how long it takes for an application to
    perform an operation. It consists of a set of spans, each of
    which represent a single timed event within the operation.

    Attributes:
        project_id (str):
            Project ID of the Cloud project where the
            trace data is stored.
        trace_id (str):
            Globally unique identifier for the trace. This identifier is
            a 128-bit numeric value formatted as a 32-byte hex string.
            For example, ``382d4f4c6b7bb2f4a972559d9085001d``.
        spans (MutableSequence[google.cloud.trace_v1.types.TraceSpan]):
            Collection of spans in the trace.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    trace_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    spans: MutableSequence["TraceSpan"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="TraceSpan",
    )


class Traces(proto.Message):
    r"""List of new or updated traces.

    Attributes:
        traces (MutableSequence[google.cloud.trace_v1.types.Trace]):
            List of traces.
    """

    traces: MutableSequence["Trace"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Trace",
    )


class TraceSpan(proto.Message):
    r"""A span represents a single timed event within a trace. Spans
    can be nested and form a trace tree. Often, a trace contains a
    root span that describes the end-to-end latency of an operation
    and, optionally, one or more subspans for its suboperations.
    Spans do not need to be contiguous. There may be gaps between
    spans in a trace.

    Attributes:
        span_id (int):
            Identifier for the span. Must be a 64-bit integer other than
            0 and unique within a trace. For example,
            ``2205310701640571284``.
        kind (google.cloud.trace_v1.types.TraceSpan.SpanKind):
            Distinguishes between spans generated in a particular
            context. For example, two spans with the same name may be
            distinguished using ``RPC_CLIENT`` and ``RPC_SERVER`` to
            identify queueing latency associated with the span.
        name (str):
            Name of the span. Must be less than 128
            bytes. The span name is sanitized and displayed
            in the Stackdriver Trace tool in the Google
            Cloud Platform Console.
            The name may be a method name or some other
            per-call site name. For the same executable and
            the same call point, a best practice is to use a
            consistent name, which makes it easier to
            correlate cross-trace spans.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Start time of the span in nanoseconds from
            the UNIX epoch.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            End time of the span in nanoseconds from the
            UNIX epoch.
        parent_span_id (int):
            Optional. ID of the parent span, if any.
        labels (MutableMapping[str, str]):
            Collection of labels associated with the span. Label keys
            must be less than 128 bytes. Label values must be less than
            16 kilobytes (10MB for ``/stacktrace`` values).

            Some predefined label keys exist, or you may create your
            own. When creating your own, we recommend the following
            formats:

            - ``/category/product/key`` for agents of well-known
              products (e.g. ``/db/mongodb/read_size``).
            - ``short_host/path/key`` for domain-specific keys (e.g.
              ``foo.com/myproduct/bar``)

            Predefined labels include:

            - ``/agent``
            - ``/component``
            - ``/error/message``
            - ``/error/name``
            - ``/http/client_city``
            - ``/http/client_country``
            - ``/http/client_protocol``
            - ``/http/client_region``
            - ``/http/host``
            - ``/http/method``
            - ``/http/path``
            - ``/http/redirected_url``
            - ``/http/request/size``
            - ``/http/response/size``
            - ``/http/route``
            - ``/http/status_code``
            - ``/http/url``
            - ``/http/user_agent``
            - ``/pid``
            - ``/stacktrace``
            - ``/tid``
    """

    class SpanKind(proto.Enum):
        r"""Type of span. Can be used to specify additional relationships
        between spans in addition to a parent/child relationship.

        Values:
            SPAN_KIND_UNSPECIFIED (0):
                Unspecified.
            RPC_SERVER (1):
                Indicates that the span covers server-side
                handling of an RPC or other remote network
                request.
            RPC_CLIENT (2):
                Indicates that the span covers the
                client-side wrapper around an RPC or other
                remote request.
        """
        SPAN_KIND_UNSPECIFIED = 0
        RPC_SERVER = 1
        RPC_CLIENT = 2

    span_id: int = proto.Field(
        proto.FIXED64,
        number=1,
    )
    kind: SpanKind = proto.Field(
        proto.ENUM,
        number=2,
        enum=SpanKind,
    )
    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    parent_span_id: int = proto.Field(
        proto.FIXED64,
        number=6,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=7,
    )


class ListTracesRequest(proto.Message):
    r"""The request message for the ``ListTraces`` method. All fields are
    required unless specified.

    Attributes:
        project_id (str):
            Required. ID of the Cloud project where the
            trace data is stored.
        view (google.cloud.trace_v1.types.ListTracesRequest.ViewType):
            Optional. Type of data returned for traces in the list.
            Default is ``MINIMAL``.
        page_size (int):
            Optional. Maximum number of traces to return.
            If not specified or <= 0, the implementation
            selects a reasonable value.  The implementation
            may return fewer traces than the requested page
            size.
        page_token (str):
            Token identifying the page of results to return. If
            provided, use the value of the ``next_page_token`` field
            from a previous request.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Start of the time interval (inclusive) during
            which the trace data was collected from the
            application.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            End of the time interval (inclusive) during
            which the trace data was collected from the
            application.
        filter (str):
            Optional. A filter against labels for the request.

            By default, searches use prefix matching. To specify exact
            match, prepend a plus symbol (``+``) to the search term.
            Multiple terms are ANDed. Syntax:

            - ``root:NAME_PREFIX`` or ``NAME_PREFIX``: Return traces
              where any root span starts with ``NAME_PREFIX``.
            - ``+root:NAME`` or ``+NAME``: Return traces where any root
              span's name is exactly ``NAME``.
            - ``span:NAME_PREFIX``: Return traces where any span starts
              with ``NAME_PREFIX``.
            - ``+span:NAME``: Return traces where any span's name is
              exactly ``NAME``.
            - ``latency:DURATION``: Return traces whose overall latency
              is greater or equal to than ``DURATION``. Accepted units
              are nanoseconds (``ns``), milliseconds (``ms``), and
              seconds (``s``). Default is ``ms``. For example,
              ``latency:24ms`` returns traces whose overall latency is
              greater than or equal to 24 milliseconds.
            - ``label:LABEL_KEY``: Return all traces containing the
              specified label key (exact match, case-sensitive)
              regardless of the key:value pair's value (including empty
              values).
            - ``LABEL_KEY:VALUE_PREFIX``: Return all traces containing
              the specified label key (exact match, case-sensitive)
              whose value starts with ``VALUE_PREFIX``. Both a key and a
              value must be specified.
            - ``+LABEL_KEY:VALUE``: Return all traces containing a
              key:value pair exactly matching the specified text. Both a
              key and a value must be specified.
            - ``method:VALUE``: Equivalent to ``/http/method:VALUE``.
            - ``url:VALUE``: Equivalent to ``/http/url:VALUE``.
        order_by (str):
            Optional. Field used to sort the returned traces. Can be one
            of the following:

            - ``trace_id``
            - ``name`` (``name`` field of root span in the trace)
            - ``duration`` (difference between ``end_time`` and
              ``start_time`` fields of the root span)
            - ``start`` (``start_time`` field of the root span)

            Descending order can be specified by appending ``desc`` to
            the sort field (for example, ``name desc``).

            Only one sort field is permitted.
    """

    class ViewType(proto.Enum):
        r"""Type of data returned for traces in the list.

        Values:
            VIEW_TYPE_UNSPECIFIED (0):
                Default is ``MINIMAL`` if unspecified.
            MINIMAL (1):
                Minimal view of the trace record that
                contains only the project and trace IDs.
            ROOTSPAN (2):
                Root span view of the trace record that
                returns the root spans along with the minimal
                trace data.
            COMPLETE (3):
                Complete view of the trace record that contains the actual
                trace data. This is equivalent to calling the REST ``get``
                or RPC ``GetTrace`` method using the ID of each listed
                trace.
        """
        VIEW_TYPE_UNSPECIFIED = 0
        MINIMAL = 1
        ROOTSPAN = 2
        COMPLETE = 3

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    view: ViewType = proto.Field(
        proto.ENUM,
        number=2,
        enum=ViewType,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=7,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=8,
    )


class ListTracesResponse(proto.Message):
    r"""The response message for the ``ListTraces`` method.

    Attributes:
        traces (MutableSequence[google.cloud.trace_v1.types.Trace]):
            List of trace records as specified by the
            view parameter.
        next_page_token (str):
            If defined, indicates that there are more
            traces that match the request and that this
            value should be passed to the next request to
            continue retrieving additional traces.
    """

    @property
    def raw_page(self):
        return self

    traces: MutableSequence["Trace"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Trace",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetTraceRequest(proto.Message):
    r"""The request message for the ``GetTrace`` method.

    Attributes:
        project_id (str):
            Required. ID of the Cloud project where the
            trace data is stored.
        trace_id (str):
            Required. ID of the trace to return.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    trace_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class PatchTracesRequest(proto.Message):
    r"""The request message for the ``PatchTraces`` method.

    Attributes:
        project_id (str):
            Required. ID of the Cloud project where the
            trace data is stored.
        traces (google.cloud.trace_v1.types.Traces):
            Required. The body of the message.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    traces: "Traces" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Traces",
    )


__all__ = tuple(sorted(__protobuf__.manifest))

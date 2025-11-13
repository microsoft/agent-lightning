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

from google.logging.type import log_severity_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.appengine.logging.v1",
    manifest={
        "LogLine",
        "SourceLocation",
        "SourceReference",
        "RequestLog",
    },
)


class LogLine(proto.Message):
    r"""Application log line emitted while processing a request.

    Attributes:
        time (google.protobuf.timestamp_pb2.Timestamp):
            Approximate time when this log entry was
            made.
        severity (google.logging.type.log_severity_pb2.LogSeverity):
            Severity of this log entry.
        log_message (str):
            App-provided log message.
        source_location (google.cloud.appengine_logging_v1.types.SourceLocation):
            Where in the source code this log message was
            written.
    """

    time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    severity: log_severity_pb2.LogSeverity = proto.Field(
        proto.ENUM,
        number=2,
        enum=log_severity_pb2.LogSeverity,
    )
    log_message: str = proto.Field(
        proto.STRING,
        number=3,
    )
    source_location: "SourceLocation" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="SourceLocation",
    )


class SourceLocation(proto.Message):
    r"""Specifies a location in a source code file.

    Attributes:
        file (str):
            Source file name. Depending on the runtime
            environment, this might be a simple name or a
            fully-qualified name.
        line (int):
            Line within the source file.
        function_name (str):
            Human-readable name of the function or method being invoked,
            with optional context such as the class or package name.
            This information is used in contexts such as the logs
            viewer, where a file and line number are less meaningful.
            The format can vary by language. For example:
            ``qual.if.ied.Class.method`` (Java), ``dir/package.func``
            (Go), ``function`` (Python).
    """

    file: str = proto.Field(
        proto.STRING,
        number=1,
    )
    line: int = proto.Field(
        proto.INT64,
        number=2,
    )
    function_name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class SourceReference(proto.Message):
    r"""A reference to a particular snapshot of the source tree used
    to build and deploy an application.

    Attributes:
        repository (str):
            Optional. A URI string identifying the
            repository. Example:
            "https://github.com/GoogleCloudPlatform/kubernetes.git".
        revision_id (str):
            The canonical and persistent identifier of
            the deployed revision. Example (git):
            "0035781c50ec7aa23385dc841529ce8a4b70db1b".
    """

    repository: str = proto.Field(
        proto.STRING,
        number=1,
    )
    revision_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class RequestLog(proto.Message):
    r"""Complete log information about a single HTTP request to an
    App Engine application.

    Attributes:
        app_id (str):
            Application that handled this request.
        module_id (str):
            Module of the application that handled this
            request.
        version_id (str):
            Version of the application that handled this
            request.
        request_id (str):
            Globally unique identifier for a request,
            which is based on the request start time.
            Request IDs for requests which started later
            will compare greater as strings than those for
            requests which started earlier.
        ip (str):
            Origin IP address.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Time when the request started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Time when the request finished.
        latency (google.protobuf.duration_pb2.Duration):
            Latency of the request.
        mega_cycles (int):
            Number of CPU megacycles used to process
            request.
        method (str):
            Request method. Example: ``"GET"``, ``"HEAD"``, ``"PUT"``,
            ``"POST"``, ``"DELETE"``.
        resource (str):
            Contains the path and query portion of the URL that was
            requested. For example, if the URL was
            "http://example.com/app?name=val", the resource would be
            "/app?name=val". The fragment identifier, which is
            identified by the ``#`` character, is not included.
        http_version (str):
            HTTP version of request. Example: ``"HTTP/1.1"``.
        status (int):
            HTTP response status code. Example: 200, 404.
        response_size (int):
            Size in bytes sent back to client by request.
        referrer (str):
            Referrer URL of request.
        user_agent (str):
            User agent that made the request.
        nickname (str):
            The logged-in user who made the request.

            Most likely, this is the part of the user's email before the
            ``@`` sign. The field value is the same for different
            requests from the same user, but different users can have
            similar names. This information is also available to the
            application via the App Engine Users API.

            This field will be populated starting with App Engine
            1.9.21.
        url_map_entry (str):
            File or class that handled the request.
        host (str):
            Internet host and port number of the resource
            being requested.
        cost (float):
            An indication of the relative cost of serving
            this request.
        task_queue_name (str):
            Queue name of the request, in the case of an
            offline request.
        task_name (str):
            Task name of the request, in the case of an
            offline request.
        was_loading_request (bool):
            Whether this was a loading request for the
            instance.
        pending_time (google.protobuf.duration_pb2.Duration):
            Time this request spent in the pending
            request queue.
        instance_index (int):
            If the instance processing this request
            belongs to a manually scaled module, then this
            is the 0-based index of the instance. Otherwise,
            this value is -1.
        finished (bool):
            Whether this request is finished or active.
        first (bool):
            Whether this is the first ``RequestLog`` entry for this
            request. If an active request has several ``RequestLog``
            entries written to Stackdriver Logging, then this field will
            be set for one of them.
        instance_id (str):
            An identifier for the instance that handled
            the request.
        line (MutableSequence[google.cloud.appengine_logging_v1.types.LogLine]):
            A list of log lines emitted by the
            application while serving this request.
        app_engine_release (str):
            App Engine release version.
        trace_id (str):
            Stackdriver Trace identifier for this
            request.
        trace_sampled (bool):
            If true, the value in the 'trace_id' field was sampled for
            storage in a trace backend.
        source_reference (MutableSequence[google.cloud.appengine_logging_v1.types.SourceReference]):
            Source code for the application that handled
            this request. There can be more than one source
            reference per deployed application if source
            code is distributed among multiple repositories.
    """

    app_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    module_id: str = proto.Field(
        proto.STRING,
        number=37,
    )
    version_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    request_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    ip: str = proto.Field(
        proto.STRING,
        number=4,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    latency: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=8,
        message=duration_pb2.Duration,
    )
    mega_cycles: int = proto.Field(
        proto.INT64,
        number=9,
    )
    method: str = proto.Field(
        proto.STRING,
        number=10,
    )
    resource: str = proto.Field(
        proto.STRING,
        number=11,
    )
    http_version: str = proto.Field(
        proto.STRING,
        number=12,
    )
    status: int = proto.Field(
        proto.INT32,
        number=13,
    )
    response_size: int = proto.Field(
        proto.INT64,
        number=14,
    )
    referrer: str = proto.Field(
        proto.STRING,
        number=15,
    )
    user_agent: str = proto.Field(
        proto.STRING,
        number=16,
    )
    nickname: str = proto.Field(
        proto.STRING,
        number=40,
    )
    url_map_entry: str = proto.Field(
        proto.STRING,
        number=17,
    )
    host: str = proto.Field(
        proto.STRING,
        number=20,
    )
    cost: float = proto.Field(
        proto.DOUBLE,
        number=21,
    )
    task_queue_name: str = proto.Field(
        proto.STRING,
        number=22,
    )
    task_name: str = proto.Field(
        proto.STRING,
        number=23,
    )
    was_loading_request: bool = proto.Field(
        proto.BOOL,
        number=24,
    )
    pending_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=25,
        message=duration_pb2.Duration,
    )
    instance_index: int = proto.Field(
        proto.INT32,
        number=26,
    )
    finished: bool = proto.Field(
        proto.BOOL,
        number=27,
    )
    first: bool = proto.Field(
        proto.BOOL,
        number=42,
    )
    instance_id: str = proto.Field(
        proto.STRING,
        number=28,
    )
    line: MutableSequence["LogLine"] = proto.RepeatedField(
        proto.MESSAGE,
        number=29,
        message="LogLine",
    )
    app_engine_release: str = proto.Field(
        proto.STRING,
        number=38,
    )
    trace_id: str = proto.Field(
        proto.STRING,
        number=39,
    )
    trace_sampled: bool = proto.Field(
        proto.BOOL,
        number=43,
    )
    source_reference: MutableSequence["SourceReference"] = proto.RepeatedField(
        proto.MESSAGE,
        number=41,
        message="SourceReference",
    )


__all__ = tuple(sorted(__protobuf__.manifest))

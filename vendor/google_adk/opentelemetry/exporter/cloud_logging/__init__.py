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
from __future__ import annotations

import base64
import datetime
import json
import logging
import re
from base64 import b64encode
from functools import partial
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TextIO,
    cast,
)

import google.auth
from google.api.monitored_resource_pb2 import (  # pylint: disable = no-name-in-module
    MonitoredResource,
)
from google.cloud.logging_v2.services.logging_service_v2 import (
    LoggingServiceV2Client,
)
from google.cloud.logging_v2.services.logging_service_v2.transports.grpc import (
    LoggingServiceV2GrpcTransport,
)
from google.cloud.logging_v2.types.log_entry import LogEntry
from google.cloud.logging_v2.types.logging import WriteLogEntriesRequest
from google.logging.type.log_severity_pb2 import (  # pylint: disable = no-name-in-module
    LogSeverity,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import (  # pylint: disable = no-name-in-module
    Struct,
)
from google.protobuf.timestamp_pb2 import (  # pylint: disable = no-name-in-module
    Timestamp,
)
from opentelemetry.exporter.cloud_logging.version import __version__
from opentelemetry.resourcedetector.gcp_resource_detector._mapping import (
    get_monitored_resource,
)
from opentelemetry.sdk import version as opentelemetry_sdk_version
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import format_span_id, format_trace_id
from opentelemetry.util.types import AnyValue
from proto.datetime_helpers import (  # type: ignore[import]
    DatetimeWithNanoseconds,
)

DEFAULT_MAX_ENTRY_SIZE = 256000  # 256 KB
DEFAULT_MAX_REQUEST_SIZE = 10000000  # 10 MB

HTTP_REQUEST_ATTRIBUTE_KEY = "gcp.http_request"
LOG_NAME_ATTRIBUTE_KEY = "gcp.log_name"
SOURCE_LOCATION_ATTRIBUTE_KEY = "gcp.source_location"
TRACE_SAMPLED_ATTRIBUTE_KEY = "gcp.trace_sampled"
PROJECT_ID_ATTRIBUTE_KEY = "gcp.project_id"
_OTEL_SDK_VERSION = opentelemetry_sdk_version.__version__
_USER_AGENT = f"opentelemetry-python {_OTEL_SDK_VERSION}; google-cloud-logging-exporter {__version__}"

# Set user-agent metadata, see https://github.com/grpc/grpc/issues/23644 and default options
# from
# https://github.com/googleapis/python-logging/blob/5309478c054d0f2b9301817fd835f2098f51dc3a/google/cloud/logging_v2/services/logging_service_v2/transports/grpc.py#L179-L182
_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.primary_user_agent", _USER_AGENT),
]

# severityMapping maps the integer severity level values from OTel [0-24]
# to matching Cloud Logging severity levels.
SEVERITY_MAPPING: dict[int, int] = {
    0: LogSeverity.DEFAULT,
    1: LogSeverity.DEBUG,
    2: LogSeverity.DEBUG,
    3: LogSeverity.DEBUG,
    4: LogSeverity.DEBUG,
    5: LogSeverity.DEBUG,
    6: LogSeverity.DEBUG,
    7: LogSeverity.DEBUG,
    8: LogSeverity.DEBUG,
    9: LogSeverity.INFO,
    10: LogSeverity.INFO,
    11: LogSeverity.NOTICE,
    12: LogSeverity.NOTICE,
    13: LogSeverity.WARNING,
    14: LogSeverity.WARNING,
    15: LogSeverity.WARNING,
    16: LogSeverity.WARNING,
    17: LogSeverity.ERROR,
    18: LogSeverity.ERROR,
    19: LogSeverity.ERROR,
    20: LogSeverity.ERROR,
    21: LogSeverity.CRITICAL,
    22: LogSeverity.CRITICAL,
    23: LogSeverity.ALERT,
    24: LogSeverity.EMERGENCY,
}

INVALID_LOG_NAME_MESSAGE = "%s is not a valid log name. log name must be <512 characters and only contain characters: A-Za-z0-9/-_."


class _GenAiJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return super().default(o)


def _convert_any_value_to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (list, tuple, Mapping)):
        return json.dumps(value, separators=(",", ":"), cls=_GenAiJsonEncoder)
    try:
        return str(value)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(
            "Error mapping AnyValue to string, this field will not be added to the LogEntry: %s",
            exc,
        )
        return ""


# Be careful not to mutate original body. Make copies of anything that needs to change.
def _sanitized_body(
    body: Mapping[str, AnyValue],
) -> MutableMapping[str, AnyValue]:
    new_body: MutableMapping[str, AnyValue] = {}
    for key, value in body.items():
        if (
            isinstance(value, Sequence)
            and len(value) > 0
            and isinstance(value[0], bytes)
        ):
            # Should not be possible for a non-bytes value to be present. AnyValue requires Sequence be of one type, and above
            # we verified the first value is type bytes.
            new_body[key] = [
                base64.b64encode(v).decode() if isinstance(v, bytes) else v
                for v in value
            ]
        elif (
            isinstance(value, Sequence)
            and len(value) > 0
            and isinstance(value[0], Mapping)
        ):
            # Should not be possible for a non-mapping value to be present. AnyValue requires Sequence be of one type, and above
            # we verified the first value is type mapping.
            new_body[key] = [
                _sanitized_body(x) if isinstance(x, Mapping) else x
                for x in value
            ]
        elif isinstance(value, bytes):
            new_body[key] = base64.b64encode(value).decode()
        elif isinstance(value, Mapping):
            new_body[key] = _sanitized_body(value)
        else:
            new_body[key] = value
    return new_body


def _set_payload_in_log_entry(log_entry: LogEntry, body: AnyValue):
    struct = Struct()
    if isinstance(body, Mapping):
        sanitized = _sanitized_body(body)
        try:
            struct.update(sanitized)
            log_entry.json_payload = struct
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Error mapping LogRecord.body to Struct, will write log with empty body: %s",
                exc,
            )
    elif isinstance(body, bytes):
        json_str = body.decode("utf-8", errors="replace")
        json_dict = json.loads(json_str)
        if isinstance(json_dict, Mapping):
            struct.update(json_dict)
            log_entry.json_payload = struct
        else:
            log_entry.text_payload = base64.b64encode(body).decode()
    elif body is not None:
        log_entry.text_payload = _convert_any_value_to_string(body)


def is_log_id_valid(log_id: str) -> bool:
    return len(log_id) < 512 and not bool(
        re.search(r"[^A-Za-z0-9\-_/\.]", log_id)
    )


def _get_monitored_resource(
    resource: Optional[Resource],
) -> MonitoredResource | None:
    if not resource:
        return None

    # TODO: Remove temporary special case for Vertex Agent Engine
    # https://github.com/GoogleCloudPlatform/opentelemetry-operations-python/issues/444
    cloud_resource_id = resource.attributes.get("cloud.resource_id")
    if isinstance(cloud_resource_id, str) and (
        match := re.match(
            r"//aiplatform\.googleapis\.com/projects/(?P<project_id>[^/]+)"
            r"/locations/(?P<location>[^/]+)"
            r"/reasoningEngines/(?P<agent_engine_id>[^/]+)",
            cloud_resource_id,
        )
    ):
        location = match.group("location")
        agent_engine_id = match.group("agent_engine_id")
        # https://cloud.google.com/monitoring/api/resources#tag_aiplatform.googleapis.com/ReasoningEngine
        return MonitoredResource(
            type="aiplatform.googleapis.com/ReasoningEngine",
            labels={
                # Intentionally omit the project ID
                "location": location,
                "reasoning_engine_id": agent_engine_id,
            },
        )

    monitored_resource_data = get_monitored_resource(resource)
    if not monitored_resource_data:
        return None

    return MonitoredResource(
        type=monitored_resource_data.type,
        labels=monitored_resource_data.labels,
    )


class CloudLoggingExporter(LogExporter):
    def __init__(
        self,
        project_id: Optional[str] = None,
        default_log_name: Optional[str] = None,
        client: Optional[LoggingServiceV2Client] = None,
        *,
        structured_json_file: Optional[TextIO] = None,
    ) -> None:
        """Create a CloudLoggingExporter

        Args:
            project_id: The GCP project ID to which the logs will be sent. If not
                provided, the exporter will infer it from Application Default Credentials.
            default_log_name: The default log name to use for log entries.
                If not provided, a default name will be used.
            client: An optional `LoggingServiceV2Client` instance to use for
                sending logs. If not provided and ``structured_json_file`` is not provided, a
                new client will be created. Passing both ``client`` and
                ``structured_json_file`` is not supported.
            structured_json_file: An optional file-like object (like `sys.stdout`) to write
                logs to in Cloud Logging `structured JSON format
                <https://cloud.google.com/logging/docs/structured-logging>`_. If provided,
                ``client`` must not be provided and logs will only be written to the file-like
                object.
        """

        self.project_id: str
        if not project_id:
            _, default_project_id = google.auth.default()
            self.project_id = str(default_project_id)
        else:
            self.project_id = project_id

        if default_log_name:
            self.default_log_name = default_log_name
        else:
            self.default_log_name = "otel_python_inprocess_log_name_temp"

        if client and structured_json_file:
            raise ValueError(
                "Cannot specify both client and structured_json_file"
            )

        if structured_json_file:
            self._write_log_entries = partial(
                self._write_log_entries_to_file, structured_json_file
            )
        else:
            client = client or LoggingServiceV2Client(
                transport=LoggingServiceV2GrpcTransport(
                    channel=LoggingServiceV2GrpcTransport.create_channel(
                        options=_OPTIONS,
                    )
                )
            )
            self._write_log_entries = partial(
                self._write_log_entries_to_client, client
            )

    def pick_log_id(self, log_name_attr: Any, event_name: str | None) -> str:
        if log_name_attr and isinstance(log_name_attr, str):
            if is_log_id_valid(log_name_attr):
                return log_name_attr.replace("/", "%2F")
            logging.warning(INVALID_LOG_NAME_MESSAGE, log_name_attr)
        if event_name and is_log_id_valid(event_name):
            return event_name.replace("/", "%2F")
        return self.default_log_name

    def export(self, batch: Sequence[LogData]):
        now = datetime.datetime.now()
        log_entries = []
        for log_data in batch:
            log_entry = LogEntry()
            log_record = log_data.log_record
            attributes = log_record.attributes or {}
            project_id = str(
                attributes.get(PROJECT_ID_ATTRIBUTE_KEY, self.project_id)
            )
            log_entry.log_name = f"projects/{project_id}/logs/{self.pick_log_id(attributes.get(LOG_NAME_ATTRIBUTE_KEY), log_record.event_name)}"
            # If timestamp is unset fall back to observed_time_unix_nano as recommended,
            # see https://github.com/open-telemetry/opentelemetry-proto/blob/4abbb78/opentelemetry/proto/logs/v1/logs.proto#L176-L179
            ts = Timestamp()
            if log_record.timestamp or log_record.observed_timestamp:
                ts.FromNanoseconds(
                    log_record.timestamp or log_record.observed_timestamp
                )
            else:
                ts.FromDatetime(now)
            log_entry.timestamp = ts
            if monitored_resource := _get_monitored_resource(
                log_record.resource
            ):
                log_entry.resource = monitored_resource
            log_entry.trace_sampled = (
                log_record.trace_flags is not None
                and log_record.trace_flags.sampled
            )
            if log_record.trace_id:
                log_entry.trace = f"projects/{project_id}/traces/{format_trace_id(log_record.trace_id)}"
            if log_record.span_id:
                log_entry.span_id = format_span_id(log_record.span_id)
            if (
                log_record.severity_number
                and log_record.severity_number.value in SEVERITY_MAPPING
            ):
                log_entry.severity = SEVERITY_MAPPING[  # type: ignore[assignment]
                    log_record.severity_number.value  # type: ignore[index]
                ]
            log_entry.labels = {
                k: _convert_any_value_to_string(v)
                for k, v in attributes.items()
            }
            if log_record.event_name:
                log_entry.labels["event.name"] = log_record.event_name
            _set_payload_in_log_entry(log_entry, log_record.body)
            log_entries.append(log_entry)

        self._write_log_entries(log_entries)

    @staticmethod
    def _write_log_entries_to_file(file: TextIO, log_entries: list[LogEntry]):
        """Formats logs into the Cloud Logging structured log format, and writes them to the
        specified file-like object

        See https://cloud.google.com/logging/docs/structured-logging
        """
        # TODO: this is not resilient to exceptions which can cause recursion when using OTel's
        # logging handler. See
        # https://github.com/open-telemetry/opentelemetry-python/issues/4261 for outstanding
        # issue in OTel.

        for entry in log_entries:
            json_dict: dict[str, Any] = {}

            # These are not added in export() so not added to the JSON here.
            # - httpRequest
            # - logging.googleapis.com/sourceLocation
            # - logging.googleapis.com/operation
            # - logging.googleapis.com/insertId

            # https://cloud.google.com/logging/docs/agent/logging/configuration#timestamp-processing
            timestamp = cast(DatetimeWithNanoseconds, entry.timestamp)
            json_dict["time"] = timestamp.rfc3339()

            json_dict["severity"] = LogSeverity.Name(
                cast(LogSeverity.ValueType, entry.severity)
            )
            json_dict["logging.googleapis.com/labels"] = dict(entry.labels)
            json_dict["logging.googleapis.com/spanId"] = entry.span_id
            json_dict[
                "logging.googleapis.com/trace_sampled"
            ] = entry.trace_sampled
            json_dict["logging.googleapis.com/trace"] = entry.trace

            if entry.text_payload:
                json_dict["message"] = entry.text_payload
            if entry.json_payload:
                json_dict.update(
                    MessageToDict(LogEntry.pb(entry).json_payload)
                )

            # Use dumps to avoid invalid json written to the stream if serialization fails for any reason
            file.write(
                json.dumps(json_dict, separators=(",", ":"), sort_keys=True)
                + "\n"
            )

    @staticmethod
    def _write_log_entries_to_client(
        client: LoggingServiceV2Client, log_entries: list[LogEntry]
    ):
        batch: list[LogEntry] = []
        batch_byte_size = 0
        for entry in log_entries:
            msg_size = LogEntry.pb(entry).ByteSize()
            if msg_size > DEFAULT_MAX_ENTRY_SIZE:
                logging.warning(
                    "Cannot write log that is %s bytes which exceeds Cloud Logging's maximum limit of %s bytes.",
                    msg_size,
                    DEFAULT_MAX_ENTRY_SIZE,
                )
                continue
            if msg_size + batch_byte_size > DEFAULT_MAX_REQUEST_SIZE:
                try:
                    client.write_log_entries(
                        WriteLogEntriesRequest(
                            entries=batch, partial_success=True
                        )
                    )
                # pylint: disable=broad-except
                except Exception as ex:
                    logging.error(
                        "Error while writing to Cloud Logging", exc_info=ex
                    )
                batch = [entry]
                batch_byte_size = msg_size
            else:
                batch.append(entry)
                batch_byte_size += msg_size
        if batch:
            try:
                client.write_log_entries(
                    WriteLogEntriesRequest(entries=batch, partial_success=True)
                )
            # pylint: disable=broad-except
            except Exception as ex:
                logging.error(
                    "Error while writing to Cloud Logging", exc_info=ex
                )

    def shutdown(self):
        pass

# Copyright 2016 Google LLC
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

"""Log entries within the Google Cloud Logging API."""

import collections
import json
import re

from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import Parse
from google.protobuf.message import Message

from google.cloud.logging_v2.resource import Resource
from google.cloud._helpers import _name_from_project_path
from google.cloud._helpers import _rfc3339_nanos_to_datetime
from google.cloud._helpers import _datetime_to_rfc3339

# import officially supported proto definitions
import google.cloud.audit.audit_log_pb2  # noqa: F401
import google.cloud.appengine_logging  # noqa: F401
from google.iam.v1.logging import audit_data_pb2  # noqa: F401

_GLOBAL_RESOURCE = Resource(type="global", labels={})


_LOGGER_TEMPLATE = re.compile(
    r"""
    projects/            # static prefix
    (?P<project>[^/]+)   # initial letter, wordchars + hyphen
    /logs/               # static midfix
    (?P<name>[^/]+)      # initial letter, wordchars + allowed punc
""",
    re.VERBOSE,
)


def logger_name_from_path(path, project=None):
    """Validate a logger URI path and get the logger name.

    Args:
        path (str): URI path for a logger API request
        project (str): The project the path is expected to belong to

    Returns:
        str: Logger name parsed from ``path``.

    Raises:
        ValueError: If the ``path`` is ill-formed of if the project
            from ``path`` does not agree with the ``project`` passed in.
    """
    return _name_from_project_path(path, project, _LOGGER_TEMPLATE)


def _int_or_none(value):
    """Helper: return an integer or ``None``."""
    if value is not None:
        value = int(value)
    return value


_LOG_ENTRY_FIELDS = (  # (name, default)
    ("log_name", None),
    ("labels", None),
    ("insert_id", None),
    ("severity", None),
    ("http_request", None),
    ("timestamp", None),
    ("resource", _GLOBAL_RESOURCE),
    ("trace", None),
    ("span_id", None),
    ("trace_sampled", None),
    ("source_location", None),
    ("operation", None),
    ("logger", None),
    ("payload", None),
)


_LogEntryTuple = collections.namedtuple(
    "LogEntry", (field for field, _ in _LOG_ENTRY_FIELDS)
)

_LogEntryTuple.__new__.__defaults__ = tuple(default for _, default in _LOG_ENTRY_FIELDS)


_LOG_ENTRY_PARAM_DOCSTRING = """\

    Args:
        log_name (str): The name of the logger used to post the entry.
        labels (Optional[dict]): Mapping of labels for the entry
        insert_id (Optional[str]): The ID used to identify an entry
            uniquely.
        severity (Optional[str]): The severity of the event being logged.
        http_request (Optional[dict]): Info about HTTP request associated
            with the entry.
        timestamp (Optional[datetime.datetime]): Timestamp for the entry.
        resource (Optional[google.cloud.logging_v2.resource.Resource]):
            Monitored resource of the entry.
        trace (Optional[str]): Trace ID to apply to the entry.
        span_id (Optional[str]): Span ID within the trace for the log
            entry. Specify the trace parameter if ``span_id`` is set.
        trace_sampled (Optional[bool]): The sampling decision of the trace
            associated with the log entry.
        source_location (Optional[dict]): Location in source code from which
            the entry was emitted.
        operation (Optional[dict]): Additional information about a potentially
            long-running operation associated with the log entry.
        logger (logging_v2.logger.Logger): the logger used
            to write the entry.
"""

_LOG_ENTRY_SEE_ALSO_DOCSTRING = """\

    See:
    https://cloud.google.com/logging/docs/reference/v2/rest/v2/LogEntry
"""


class LogEntry(_LogEntryTuple):
    __doc__ = (
        """
    Log entry.

    """
        + _LOG_ENTRY_PARAM_DOCSTRING
        + _LOG_ENTRY_SEE_ALSO_DOCSTRING
    )

    received_timestamp = None

    @classmethod
    def _extract_payload(cls, resource):
        """Helper for :meth:`from_api_repr`"""
        return None

    @classmethod
    def from_api_repr(cls, resource, client, *, loggers=None):
        """Construct an entry given its API representation

        Args:
            resource (dict): text entry resource representation returned from
                the API
            client (~logging_v2.client.Client):
                Client which holds credentials and project configuration.
            loggers (Optional[dict]):
                A mapping of logger fullnames -> loggers.  If not
                passed, the entry will have a newly-created logger if possible,
                or an empty logger field if not.

        Returns:
            google.cloud.logging.entries.LogEntry: Log entry parsed from ``resource``.
        """
        if loggers is None:
            loggers = {}
        logger_fullname = resource["logName"]
        logger = loggers.get(logger_fullname)
        if logger is None:
            # attempt to create a logger if possible
            try:
                logger_name = logger_name_from_path(logger_fullname, client.project)
                logger = loggers[logger_fullname] = client.logger(logger_name)
            except ValueError:
                # log name is not scoped to a project. Leave logger as None
                pass
        payload = cls._extract_payload(resource)
        insert_id = resource.get("insertId")
        timestamp = resource.get("timestamp")
        if timestamp is not None:
            timestamp = _rfc3339_nanos_to_datetime(timestamp)
        labels = resource.get("labels")
        severity = resource.get("severity")
        http_request = resource.get("httpRequest")
        trace = resource.get("trace")
        span_id = resource.get("spanId")
        trace_sampled = resource.get("traceSampled")
        source_location = resource.get("sourceLocation")
        if source_location is not None:
            line = source_location.pop("line", None)
            source_location["line"] = _int_or_none(line)
        operation = resource.get("operation")

        monitored_resource_dict = resource.get("resource")
        monitored_resource = None
        if monitored_resource_dict is not None:
            monitored_resource = Resource._from_dict(monitored_resource_dict)

        inst = cls(
            log_name=logger_fullname,
            insert_id=insert_id,
            timestamp=timestamp,
            labels=labels,
            severity=severity,
            http_request=http_request,
            resource=monitored_resource,
            trace=trace,
            span_id=span_id,
            trace_sampled=trace_sampled,
            source_location=source_location,
            operation=operation,
            logger=logger,
            payload=payload,
        )
        received = resource.get("receiveTimestamp")
        if received is not None:
            inst.received_timestamp = _rfc3339_nanos_to_datetime(received)
        return inst

    def to_api_repr(self):
        """API repr (JSON format) for entry."""
        info = {}
        if self.log_name is not None:
            info["logName"] = self.log_name
        if self.resource is not None:
            info["resource"] = self.resource._to_dict()
        if self.labels is not None:
            info["labels"] = self.labels
        if self.insert_id is not None:
            info["insertId"] = self.insert_id
        if self.severity is not None:
            if isinstance(self.severity, str):
                info["severity"] = self.severity.upper()
            else:
                info["severity"] = self.severity
        if self.http_request is not None:
            info["httpRequest"] = self.http_request
        if self.timestamp is not None:
            info["timestamp"] = _datetime_to_rfc3339(self.timestamp)
        if self.trace is not None:
            info["trace"] = self.trace
        if self.span_id is not None:
            info["spanId"] = self.span_id
        if self.trace_sampled is not None:
            info["traceSampled"] = self.trace_sampled
        if self.source_location is not None:
            source_location = self.source_location.copy()
            source_location["line"] = str(source_location.pop("line", 0))
            info["sourceLocation"] = source_location
        if self.operation is not None:
            info["operation"] = self.operation
        return info


class TextEntry(LogEntry):
    __doc__ = (
        """
    Log entry with text payload.

    """
        + _LOG_ENTRY_PARAM_DOCSTRING
        + """

        payload (str): payload for the log entry.
    """
        + _LOG_ENTRY_SEE_ALSO_DOCSTRING
    )

    @classmethod
    def _extract_payload(cls, resource):
        """Helper for :meth:`from_api_repr`"""
        return resource["textPayload"]

    def to_api_repr(self):
        """API repr (JSON format) for entry."""
        info = super(TextEntry, self).to_api_repr()
        info["textPayload"] = self.payload
        return info


class StructEntry(LogEntry):
    __doc__ = (
        """
    Log entry with JSON payload.

    """
        + _LOG_ENTRY_PARAM_DOCSTRING
        + """

        payload (dict): payload for the log entry.
    """
        + _LOG_ENTRY_SEE_ALSO_DOCSTRING
    )

    @classmethod
    def _extract_payload(cls, resource):
        """Helper for :meth:`from_api_repr`"""
        return resource["jsonPayload"]

    def to_api_repr(self):
        """API repr (JSON format) for entry."""
        info = super(StructEntry, self).to_api_repr()
        info["jsonPayload"] = self.payload
        return info


class ProtobufEntry(LogEntry):
    __doc__ = (
        """
    Log entry with protobuf message payload.

    """
        + _LOG_ENTRY_PARAM_DOCSTRING
        + """

        payload (google.protobuf.Message): payload for the log entry.
    """
        + _LOG_ENTRY_SEE_ALSO_DOCSTRING
    )

    @classmethod
    def _extract_payload(cls, resource):
        """Helper for :meth:`from_api_repr`"""
        return resource["protoPayload"]

    @property
    def payload_pb(self):
        if isinstance(self.payload, Message):
            return self.payload

    @property
    def payload_json(self):
        if isinstance(self.payload, collections.abc.Mapping):
            return self.payload

    def to_api_repr(self):
        """API repr (JSON format) for entry."""
        info = super(ProtobufEntry, self).to_api_repr()
        proto_payload = None
        if self.payload_pb:
            proto_payload = MessageToDict(self.payload)
        elif self.payload_json:
            proto_payload = dict(self.payload)
        info["protoPayload"] = proto_payload
        return info

    def parse_message(self, message):
        """Parse payload into a protobuf message.

        Mutates the passed-in ``message`` in place.

        Args:
            message (google.protobuf.Message): the message to be logged
        """
        # NOTE: This assumes that ``payload`` is already a deserialized
        #       ``Any`` field and ``message`` has come from an imported
        #       ``pb2`` module with the relevant protobuf message type.
        Parse(json.dumps(self.payload), message)

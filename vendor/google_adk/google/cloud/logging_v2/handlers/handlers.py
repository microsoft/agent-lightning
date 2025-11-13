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

"""Python :mod:`logging` handlers for Cloud Logging."""

import collections
import json
import logging

from typing import Optional, IO, Type

from google.cloud.logging_v2.handlers.transports import (
    BackgroundThreadTransport,
    Transport,
)
from google.cloud.logging_v2.handlers._monitored_resources import (
    detect_resource,
    add_resource_labels,
)
from google.cloud.logging_v2.handlers._helpers import get_request_data
from google.cloud.logging_v2.resource import Resource


DEFAULT_LOGGER_NAME = "python"

"""Defaults for filtering out noisy loggers"""
EXCLUDED_LOGGER_DEFAULTS = (
    "google.api_core.bidi",
    "werkzeug",
)

"""Exclude internal logs from propagating through handlers"""
_INTERNAL_LOGGERS = (
    "google.cloud",
    "google.auth",
    "google_auth_httplib2",
)

"""These environments require us to remove extra handlers on setup"""
_CLEAR_HANDLER_RESOURCE_TYPES = ("gae_app", "cloud_function")


class CloudLoggingFilter(logging.Filter):
    """Python standard ``logging`` Filter class to add Cloud Logging
    information to each LogRecord.

    When attached to a LogHandler, each incoming log will be modified
    to include new Cloud Logging relevant data. This data can be manually
    overwritten using the `extras` argument when writing logs.
    """

    def __init__(self, project=None, default_labels=None):
        self.project = project
        self.default_labels = default_labels if default_labels else {}

    @staticmethod
    def _infer_source_location(record):
        """Helper function to infer source location data from a LogRecord.
        Will default to record.source_location if already set
        """
        if hasattr(record, "source_location"):
            return record.source_location
        else:
            name_map = [
                ("line", "lineno"),
                ("file", "pathname"),
                ("function", "funcName"),
            ]
            output = {}
            for gcp_name, std_lib_name in name_map:
                value = getattr(record, std_lib_name, None)
                if value is not None:
                    output[gcp_name] = value
            return output if output else None

    def filter(self, record):
        """
        Add new Cloud Logging data to each LogRecord as it comes in
        """
        user_labels = getattr(record, "labels", {})
        # infer request data from the environment
        (
            inferred_http,
            inferred_trace,
            inferred_span,
            inferred_sampled,
        ) = get_request_data()
        if inferred_trace is not None and self.project is not None:
            # add full path for detected trace
            inferred_trace = f"projects/{self.project}/traces/{inferred_trace}"
        # set new record values
        record._resource = getattr(record, "resource", None)
        record._trace = getattr(record, "trace", inferred_trace) or None
        record._span_id = getattr(record, "span_id", inferred_span) or None
        record._trace_sampled = bool(getattr(record, "trace_sampled", inferred_sampled))
        record._http_request = getattr(record, "http_request", inferred_http)
        record._source_location = CloudLoggingFilter._infer_source_location(record)
        # add logger name as a label if possible
        logger_label = {"python_logger": record.name} if record.name else {}
        record._labels = {**logger_label, **self.default_labels, **user_labels} or None
        # create string representations for structured logging
        record._trace_str = record._trace or ""
        record._span_id_str = record._span_id or ""
        record._trace_sampled_str = "true" if record._trace_sampled else "false"
        record._http_request_str = json.dumps(
            record._http_request or {}, ensure_ascii=False
        )
        record._source_location_str = json.dumps(
            record._source_location or {}, ensure_ascii=False
        )
        record._labels_str = json.dumps(record._labels or {}, ensure_ascii=False)
        return True


class CloudLoggingHandler(logging.StreamHandler):
    """Handler that directly makes Cloud Logging API calls.

    This is a Python standard ``logging`` handler using that can be used to
    route Python standard logging messages directly to the Stackdriver
    Logging API.

    This handler is used when not in GAE or GKE environment.

    This handler supports both an asynchronous and synchronous transport.

    Example:

    .. code-block:: python

        import logging
        import google.cloud.logging
        from google.cloud.logging_v2.handlers import CloudLoggingHandler

        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(client)

        cloud_logger = logging.getLogger('cloudLogger')
        cloud_logger.setLevel(logging.INFO)
        cloud_logger.addHandler(handler)

        cloud_logger.error('bad news')  # API call
    """

    def __init__(
        self,
        client,
        *,
        name: str = DEFAULT_LOGGER_NAME,
        transport: Type[Transport] = BackgroundThreadTransport,
        resource: Resource = None,
        labels: Optional[dict] = None,
        stream: Optional[IO] = None,
        **kwargs,
    ):
        """
        Args:
            client (~logging_v2.client.Client):
                The authenticated Google Cloud Logging client for this
                handler to use.
            name (str): the name of the custom log in Cloud Logging.
                Defaults to 'python'. The name of the Python logger will be represented
                in the ``python_logger`` field.
            transport (~logging_v2.transports.Transport):
                Class for creating new transport objects. It should
                extend from the base :class:`.Transport` type and
                implement :meth`.Transport.send`. Defaults to
                :class:`.BackgroundThreadTransport`. The other
                option is :class:`.SyncTransport`.
            resource (~logging_v2.resource.Resource):
                Resource for this Handler. If not given, will be inferred from the environment.
            labels (Optional[dict]): Additional labels to attach to logs.
            stream (Optional[IO]): Stream to be used by the handler.
        """
        super(CloudLoggingHandler, self).__init__(stream)
        if not resource:
            # infer the correct monitored resource from the local environment
            resource = detect_resource(client.project)
        self.name = name
        self.client = client
        client._handlers.add(self)
        self.transport = transport(client, name, resource=resource)
        self._transport_open = True
        self._transport_cls = transport
        self.project_id = client.project
        self.resource = resource
        self.labels = labels
        # add extra keys to log record
        log_filter = CloudLoggingFilter(project=self.project_id, default_labels=labels)
        self.addFilter(log_filter)

    def emit(self, record):
        """Actually log the specified logging record.

        Overrides the default emit behavior of ``StreamHandler``.

        See https://docs.python.org/2/library/logging.html#handler-objects

        Args:
            record (logging.LogRecord): The record to be logged.
        """
        resource = record._resource or self.resource
        labels = record._labels
        message = _format_and_parse_message(record, self)

        labels = {**add_resource_labels(resource, record), **(labels or {})} or None

        # send off request
        if not self._transport_open:
            self.transport = self._transport_cls(
                self.client, self.name, resource=self.resource
            )
            self._transport_open = True

        self.transport.send(
            record,
            message,
            resource=resource,
            labels=labels,
            trace=record._trace,
            span_id=record._span_id,
            trace_sampled=record._trace_sampled,
            http_request=record._http_request,
            source_location=record._source_location,
        )

    def flush(self):
        """Forces the Transport object to submit any pending log records.

        For SyncTransport, this is a no-op.
        """
        super(CloudLoggingHandler, self).flush()
        if self._transport_open:
            self.transport.flush()

    def close(self):
        """Closes the log handler and cleans up all Transport objects used."""
        if self._transport_open:
            self.transport.close()
            self.transport = None
            self._transport_open = False


def _format_and_parse_message(record, formatter_handler):
    """
    Helper function to apply formatting to a LogRecord message,
    and attempt to parse encoded JSON into a dictionary object.

    Resulting output will be of type (str | dict | None)

    Args:
        record (logging.LogRecord): The record object representing the log
        formatter_handler (logging.Handler): The handler used to format the log
    """
    passed_json_fields = getattr(record, "json_fields", {})
    # if message is a dictionary, use dictionary directly
    if isinstance(record.msg, collections.abc.Mapping):
        payload = record.msg
        # attach any extra json fields if present
        if passed_json_fields and isinstance(
            passed_json_fields, collections.abc.Mapping
        ):
            payload = {**payload, **passed_json_fields}
        return payload
    # format message string based on superclass
    message = formatter_handler.format(record)
    try:
        # attempt to parse encoded json into dictionary
        if message[0] == "{":
            json_message = json.loads(message)
            if isinstance(json_message, collections.abc.Mapping):
                message = json_message
    except (json.decoder.JSONDecodeError, IndexError):
        # log string is not valid json
        pass
    # if json_fields was set, create a dictionary using that
    if passed_json_fields and isinstance(passed_json_fields, collections.abc.Mapping):
        passed_json_fields = passed_json_fields.copy()
        if message != "None":
            passed_json_fields["message"] = message
        return passed_json_fields
    # if formatted message contains no content, return None
    return message if message != "None" else None


def setup_logging(
    handler, *, excluded_loggers=EXCLUDED_LOGGER_DEFAULTS, log_level=logging.INFO
):
    """Attach a logging handler to the Python root logger

    Excludes loggers that this library itself uses to avoid
    infinite recursion.

    Example:

    .. code-block:: python

        import logging
        import google.cloud.logging
        from google.cloud.logging_v2.handlers import CloudLoggingHandler

        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(client)
        google.cloud.logging.handlers.setup_logging(handler)
        logging.getLogger().setLevel(logging.DEBUG)

        logging.error('bad news')  # API call

    Args:
        handler (logging.handler): the handler to attach to the global handler
        excluded_loggers (Optional[Tuple[str]]): The loggers to not attach the handler
            to. This will always include the loggers in the
            path of the logging client itself.
        log_level (Optional[int]): The logging level threshold of the attached logger,
            as set by the :meth:`logging.Logger.setLevel` method. Defaults to
            :const:`logging.INFO`.
    """
    all_excluded_loggers = set(excluded_loggers + _INTERNAL_LOGGERS)
    logger = logging.getLogger()

    # remove built-in handlers on App Engine or Cloud Functions environments
    if detect_resource().type in _CLEAR_HANDLER_RESOURCE_TYPES:
        logger.handlers.clear()

    logger.setLevel(log_level)
    logger.addHandler(handler)
    for logger_name in all_excluded_loggers:
        # prevent excluded loggers from propagating logs to handler
        logger = logging.getLogger(logger_name)
        logger.propagate = False

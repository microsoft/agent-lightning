# Copyright 2016 Google LLC All Rights Reserved.
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

"""Helper functions for logging handlers."""

import math
import json
import re
import warnings

try:
    import flask
except ImportError:  # pragma: NO COVER
    flask = None

import opentelemetry.trace

from google.cloud.logging_v2.handlers.middleware.request import _get_django_request

_DJANGO_CONTENT_LENGTH = "CONTENT_LENGTH"
_DJANGO_XCLOUD_TRACE_HEADER = "HTTP_X_CLOUD_TRACE_CONTEXT"
_DJANGO_TRACEPARENT = "HTTP_TRACEPARENT"
_DJANGO_USERAGENT_HEADER = "HTTP_USER_AGENT"
_DJANGO_REMOTE_ADDR_HEADER = "REMOTE_ADDR"
_DJANGO_REFERER_HEADER = "HTTP_REFERER"
_FLASK_XCLOUD_TRACE_HEADER = "X_CLOUD_TRACE_CONTEXT"
_FLASK_TRACEPARENT = "TRACEPARENT"
_PROTOCOL_HEADER = "SERVER_PROTOCOL"


def format_stackdriver_json(record, message):
    """Helper to format a LogRecord in in Stackdriver fluentd format.

    Returns:
        str: JSON str to be written to the log file.

    DEPRECATED:  use StructuredLogHandler to write formatted logs to standard out instead.
    """
    subsecond, second = math.modf(record.created)

    payload = {
        "message": message,
        "timestamp": {"seconds": int(second), "nanos": int(subsecond * 1e9)},
        "thread": record.thread,
        "severity": record.levelname,
    }
    warnings.warn(
        "format_stackdriver_json is deprecated. Use StructuredLogHandler instead.",
        DeprecationWarning,
    )
    return json.dumps(payload, ensure_ascii=False)


def get_request_data_from_flask():
    """Get http_request and trace data from flask request headers.

    Returns:
        Tuple[Optional[dict], Optional[str], Optional[str], bool]:
            Data related to the current http request, trace_id, span_id and trace_sampled
            for the request. All fields will be None if a Flask request isn't found.
    """
    if flask is None or not flask.request:
        return None, None, None, False

    # build http_request
    http_request = {
        "requestMethod": flask.request.method,
        "requestUrl": flask.request.url,
        "userAgent": flask.request.user_agent.string,
        "protocol": flask.request.environ.get(_PROTOCOL_HEADER),
    }

    # find trace id and span id
    # first check for w3c traceparent header
    header = flask.request.headers.get(_FLASK_TRACEPARENT)
    trace_id, span_id, trace_sampled = _parse_trace_parent(header)
    if trace_id is None:
        # traceparent not found. look for xcloud_trace_context header
        header = flask.request.headers.get(_FLASK_XCLOUD_TRACE_HEADER)
        trace_id, span_id, trace_sampled = _parse_xcloud_trace(header)

    return http_request, trace_id, span_id, trace_sampled


def get_request_data_from_django():
    """Get http_request and trace data from django request headers.

    Returns:
        Tuple[Optional[dict], Optional[str], Optional[str], bool]:
            Data related to the current http request, trace_id, span_id, and trace_sampled
            for the request. All fields will be None if a django request isn't found.
    """
    request = _get_django_request()

    if request is None:
        return None, None, None, False

    # Django can raise django.core.exceptions.DisallowedHost here for a
    # malformed HTTP_HOST header. But we don't want to import Django modules.
    try:
        request_url = request.build_absolute_uri()
    except Exception:
        request_url = None

    # build http_request
    http_request = {
        "requestMethod": request.method,
        "requestUrl": request_url,
        "userAgent": request.META.get(_DJANGO_USERAGENT_HEADER),
        "protocol": request.META.get(_PROTOCOL_HEADER),
    }

    # find trace id and span id
    # first check for w3c traceparent header
    header = request.META.get(_DJANGO_TRACEPARENT)
    trace_id, span_id, trace_sampled = _parse_trace_parent(header)
    if trace_id is None:
        # traceparent not found. look for xcloud_trace_context header
        header = request.META.get(_DJANGO_XCLOUD_TRACE_HEADER)
        trace_id, span_id, trace_sampled = _parse_xcloud_trace(header)

    return http_request, trace_id, span_id, trace_sampled


def _parse_trace_parent(header):
    """Given a w3 traceparent header, extract the trace and span ids.
    For more information see https://www.w3.org/TR/trace-context/

    Args:
        header (str): the string extracted from the traceparent header
            example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    Returns:
        Tuple[Optional[dict], Optional[str], bool]:
            The trace_id, span_id and trace_sampled extracted from the header
            Each field will be None if header can't be parsed in expected format.
    """
    trace_id = span_id = None
    trace_sampled = False
    # see https://www.w3.org/TR/trace-context/ for W3C traceparent format
    if header:
        try:
            VERSION_PART = r"(?!ff)[a-f\d]{2}"
            TRACE_ID_PART = r"(?![0]{32})[a-f\d]{32}"
            PARENT_ID_PART = r"(?![0]{16})[a-f\d]{16}"
            FLAGS_PART = r"[a-f\d]{2}"
            regex = f"^\\s?({VERSION_PART})-({TRACE_ID_PART})-({PARENT_ID_PART})-({FLAGS_PART})(-.*)?\\s?$"
            match = re.match(regex, header)
            trace_id = match.group(2)
            span_id = match.group(3)
            # trace-flag component is an 8-bit bit field. Read as an int
            int_flag = int(match.group(4), 16)
            # trace sampled is set if the right-most bit in flag component is set
            trace_sampled = bool(int_flag & 1)
        except (IndexError, AttributeError):
            # could not parse header as expected. Return None
            pass
    return trace_id, span_id, trace_sampled


def _parse_xcloud_trace(header):
    """Given an X_CLOUD_TRACE header, extract the trace and span ids.

    Args:
        header (str): the string extracted from the X_CLOUD_TRACE header
    Returns:
        Tuple[Optional[str], Optional[str], bool]:
            The trace_id, span_id and trace_sampled extracted from the header
            Each field will be None if not found.
    """
    trace_id = span_id = None
    trace_sampled = False

    # As per the format described at https://cloud.google.com/trace/docs/trace-context#legacy-http-header
    #    "X-Cloud-Trace-Context: TRACE_ID[/SPAN_ID][;o=OPTIONS]"
    # for example:
    #    "X-Cloud-Trace-Context: 105445aa7843bc8bf206b12000100000/1;o=1"
    #
    # We expect:
    #   * trace_id (optional, 128-bit hex string):  "105445aa7843bc8bf206b12000100000"
    #   * span_id (optional, 16-bit hex string):   "0000000000000001" (needs to be converted into 16 bit hex string)
    #   * trace_sampled (optional, bool): 	       true
    if header:
        try:
            regex = r"([\w-]+)?(\/?([\w-]+))?(;?o=(\d))?"
            match = re.match(regex, header)
            trace_id = match.group(1)
            span_id = match.group(3)
            trace_sampled = match.group(5) == "1"

            # Convert the span ID to 16-bit hexadecimal instead of decimal
            try:
                span_id_int = int(span_id)
                if span_id_int > 0 and span_id_int < 2**64:
                    span_id = f"{span_id_int:016x}"
                else:
                    span_id = None
            except (ValueError, TypeError):
                span_id = None

        except IndexError:
            pass
    return trace_id, span_id, trace_sampled


def _retrieve_current_open_telemetry_span():
    """Helper to retrieve trace, span ID, and trace sampled information from the current
    OpenTelemetry span.

    Returns:
        Tuple[Optional[str], Optional[str], bool]:
            Data related to the current trace_id, span_id, and trace_sampled for the
            current OpenTelemetry span. If a span is not found, return None/False for all
            fields.
    """
    span = opentelemetry.trace.get_current_span()
    if span != opentelemetry.trace.span.INVALID_SPAN:
        context = span.get_span_context()
        trace_id = opentelemetry.trace.format_trace_id(context.trace_id)
        span_id = opentelemetry.trace.format_span_id(context.span_id)
        trace_sampled = context.trace_flags.sampled

        return trace_id, span_id, trace_sampled

    return None, None, False


def get_request_data():
    """Helper to get http_request and trace data from supported web
    frameworks (currently supported: Flask and Django), as well as OpenTelemetry. Attempts
    to retrieve trace/spanID from OpenTelemetry first, before going to Traceparent then XCTC.
    HTTP request data is taken from a supporting web framework (currently Flask or Django).
    Because HTTP request data is decoupled from OpenTelemetry, it is possible to get as a
    return value the HTTP request from the web framework of choice, and trace/span data from
    OpenTelemetry, even if trace data is present in the HTTP request headers.

    Returns:
        Tuple[Optional[dict], Optional[str], Optional[str], bool]:
            Data related to the current http request, trace_id, span_id, and trace_sampled
            for the request. All fields will be None if a http request isn't found.
    """

    (
        otel_trace_id,
        otel_span_id,
        otel_trace_sampled,
    ) = _retrieve_current_open_telemetry_span()

    # Get HTTP request data
    checkers = (
        get_request_data_from_django,
        get_request_data_from_flask,
    )

    http_request, http_trace_id, http_span_id, http_trace_sampled = (
        None,
        None,
        None,
        False,
    )

    for checker in checkers:
        http_request, http_trace_id, http_span_id, http_trace_sampled = checker()
        if http_request is None:
            http_trace_id, http_span_id, http_trace_sampled = None, None, False
        else:
            break

    # otel_trace_id existing means the other return values are non-null
    if otel_trace_id:
        return http_request, otel_trace_id, otel_span_id, otel_trace_sampled
    else:
        return http_request, http_trace_id, http_span_id, http_trace_sampled

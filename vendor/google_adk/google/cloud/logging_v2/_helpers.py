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

"""Common logging helpers."""

import logging

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import requests

from google.cloud.logging_v2.entries import LogEntry
from google.cloud.logging_v2.entries import ProtobufEntry
from google.cloud.logging_v2.entries import StructEntry
from google.cloud.logging_v2.entries import TextEntry

try:
    from google.cloud.logging_v2.types import LogSeverity
except ImportError:  # pragma: NO COVER

    class LogSeverity(object):
        """Map severities for non-GAPIC usage."""

        DEFAULT = 0
        DEBUG = 100
        INFO = 200
        NOTICE = 300
        WARNING = 400
        ERROR = 500
        CRITICAL = 600
        ALERT = 700
        EMERGENCY = 800


_NORMALIZED_SEVERITIES = {
    logging.CRITICAL: LogSeverity.CRITICAL,
    logging.ERROR: LogSeverity.ERROR,
    logging.WARNING: LogSeverity.WARNING,
    logging.INFO: LogSeverity.INFO,
    logging.DEBUG: LogSeverity.DEBUG,
    logging.NOTSET: LogSeverity.DEFAULT,
}

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
"""Time format for timestamps used in API"""

METADATA_URL = "http://metadata.google.internal./computeMetadata/v1/"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def entry_from_resource(resource, client, loggers):
    """Detect correct entry type from resource and instantiate.

    Args:
        resource (dict): One entry resource from API response.
        client (~logging_v2.client.Client):
            Client that owns the log entry.
        loggers (dict):
            A mapping of logger fullnames -> loggers.  If the logger
            that owns the entry is not in ``loggers``, the entry
            will have a newly-created logger.

    Returns:
        google.cloud.logging_v2.entries._BaseEntry:
            The entry instance, constructed via the resource
    """
    if "textPayload" in resource:
        return TextEntry.from_api_repr(resource, client, loggers=loggers)

    if "jsonPayload" in resource:
        return StructEntry.from_api_repr(resource, client, loggers=loggers)

    if "protoPayload" in resource:
        return ProtobufEntry.from_api_repr(resource, client, loggers=loggers)

    return LogEntry.from_api_repr(resource, client, loggers=loggers)


def retrieve_metadata_server(metadata_key, timeout=5):
    """Retrieve the metadata key in the metadata server.

    See: https://cloud.google.com/compute/docs/storing-retrieving-metadata

    Args:
        metadata_key (str):
            Key of the metadata which will form the url. You can
            also supply query parameters after the metadata key.
            e.g. "tags?alt=json"
        timeout (number): number of seconds to wait for the HTTP request

    Returns:
        str: The value of the metadata key returned by the metadata server.
    """
    url = METADATA_URL + metadata_key

    try:
        response = requests.get(url, headers=METADATA_HEADERS, timeout=timeout)

        if response.status_code == requests.codes.ok:
            return response.text

    except requests.exceptions.RequestException:
        # Ignore the exception, connection failed means the attribute does not
        # exist in the metadata server.
        pass

    return None


def _normalize_severity(stdlib_level):
    """Normalize a Python stdlib severity to LogSeverity enum.

    Args:
        stdlib_level (int): 'levelno' from a :class:`logging.LogRecord`

    Returns:
        int: Corresponding Stackdriver severity.
    """
    return _NORMALIZED_SEVERITIES.get(stdlib_level, stdlib_level)


def _add_defaults_to_filter(filter_):
    """Modify the input filter expression to add sensible defaults.

    Args:
        filter_ (str): The original filter expression

    Returns:
        str: sensible default filter string
    """

    # By default, requests should only return logs in the last 24 hours
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    time_filter = f'timestamp>="{yesterday.strftime(_TIME_FORMAT)}"'
    if filter_ is None:
        filter_ = time_filter
    elif "timestamp" not in filter_.lower():
        filter_ = f"{filter_} AND {time_filter}"
    return filter_

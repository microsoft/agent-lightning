# Copyright 2022 Google LLC
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

"""Add diagnostic instrumentation source information to logs"""
from google.cloud.logging_v2.entries import StructEntry
from google.cloud.logging_v2 import __version__

_DIAGNOSTIC_INFO_KEY = "logging.googleapis.com/diagnostic"
_INSTRUMENTATION_SOURCE_KEY = "instrumentation_source"
_PYTHON_LIBRARY_NAME = "python"

_LIBRARY_VERSION = __version__

_MAX_NAME_LENGTH = 14
_MAX_VERSION_LENGTH = 14
_MAX_INSTRUMENTATION_ENTRIES = 3


def _add_instrumentation(entries, **kw):
    """Add instrumentation information to a list of entries

        A new diagnostic entry is prepended to the list of
        entries.

    Args:
       entries (Sequence[Mapping[str, ...]]): sequence of mappings representing
            the log entry resources to log.

    Returns:
        Sequence[Mapping[str, ...]]: entries with instrumentation info added to
        the beginning of list.
    """

    diagnostic_entry = _create_diagnostic_entry(**kw)
    entries.insert(0, diagnostic_entry.to_api_repr())
    return entries


def _create_diagnostic_entry(name=_PYTHON_LIBRARY_NAME, version=_LIBRARY_VERSION, **kw):
    """Create a diagnostic log entry describing this library

        The diagnostic log consists of a list of library name and version objects
        that have handled a given log entry.  If this library is the originator
        of the log entry, it will look like:
        {logging.googleapis.com/diagnostic: {instrumentation_source: [{name: "python", version: "3.0.0"}]}}

    Args:
        name(str): The name of this library (e.g. 'python')
        version(str) The version of this library (e.g. '3.0.0')

    Returns:
        google.cloud.logging_v2.LogEntry: Log entry with library information
    """
    payload = {
        _DIAGNOSTIC_INFO_KEY: {
            _INSTRUMENTATION_SOURCE_KEY: [_get_instrumentation_source(name, version)]
        }
    }
    # only keep the log_name and resource from the parent log
    allow_list = ("log_name", "resource")
    active_kws = {k: v for k, v in kw.items() if k in allow_list}
    entry = StructEntry(payload=payload, **active_kws)
    return entry


def _get_instrumentation_source(name=_PYTHON_LIBRARY_NAME, version=_LIBRARY_VERSION):
    """Gets a JSON representation of the instrumentation_source

    Args:
        name(str): The name of this library (e.g. 'python')
        version(str) The version of this library (e.g. '3.0.0')
    Returns:
       obj: JSON object with library information
    """
    source = {"name": name, "version": version}
    # truncate strings to no more than _MAX_NAME_LENGTH characters
    for key, val in source.items():
        source[key] = (
            val if len(val) <= _MAX_NAME_LENGTH else f"{val[:_MAX_NAME_LENGTH]}*"
        )
    return source

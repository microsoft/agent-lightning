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

"""Logging handler for Google Container Engine (GKE).

Formats log messages in a JSON format, so that Kubernetes clusters with the
fluentd Google Cloud plugin installed can format their log messages so that
metadata such as log level is properly captured.
"""

import logging.handlers
import warnings

from google.cloud.logging_v2.handlers._helpers import format_stackdriver_json

_DEPRECATION_MSG = (
    "ContainerEngineHandler is deprecated. Use StructuredLogHandler instead."
)


class ContainerEngineHandler(logging.StreamHandler):
    """Handler to format log messages the format expected by GKE fluent.

    This handler is written to format messages for the Google Container Engine
    (GKE) fluentd plugin, so that metadata such as log level are properly set.

    DEPRECATED:  use StructuredLogHandler to write formatted logs to standard out instead.
    """

    def __init__(self, *, name=None, stream=None):
        """
        Args:
            name (Optional[str]): The name of the custom log in Cloud Logging.
            stream (Optional[IO]): Stream to be used by the handler.

        """
        super(ContainerEngineHandler, self).__init__(stream=stream)
        self.name = name
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning)

    def format(self, record):
        """Format the message into JSON expected by fluentd.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            str: A JSON string formatted for GKE fluentd.
        """
        message = super(ContainerEngineHandler, self).format(record)
        return format_stackdriver_json(record, message)

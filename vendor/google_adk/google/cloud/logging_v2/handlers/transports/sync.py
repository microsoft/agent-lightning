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

"""Transport for Python logging handler.

Logs directly to the Cloud Logging API with a synchronous call.
"""
from google.cloud.logging_v2 import _helpers
from google.cloud.logging_v2.handlers.transports.base import Transport
from google.cloud.logging_v2.logger import _GLOBAL_RESOURCE


class SyncTransport(Transport):
    """Basic sychronous transport.

    Uses this library's Logging client to directly make the API call.
    """

    def __init__(self, client, name, resource=_GLOBAL_RESOURCE, **kwargs):
        """
        Args:
            client (~logging_v2.client.Client):
                The Logging client.
            name (str): The name of the lgoger.
            resource (Optional[Resource|dict]): The default monitored resource to associate
                with logs when not specified
        """
        self.logger = client.logger(name, resource=resource)

    def send(self, record, message, **kwargs):
        """Overrides transport.send().

        Args:
            record (logging.LogRecord):
                Python log record that the handler was called with.
            message (str or dict): The message from the ``LogRecord`` after being
                formatted by the associated log formatters.
            kwargs: Additional optional arguments for the logger
        """
        # set python logger name as label if missing
        labels = kwargs.pop("labels", {})
        if record.name:
            labels["python_logger"] = labels.get("python_logger", record.name)
        # send log synchronously
        self.logger.log(
            message,
            severity=_helpers._normalize_severity(record.levelno),
            labels=labels,
            **kwargs,
        )

    def close(self):
        """Closes the transport and cleans up resources used by it.

        This call is usually followed up by cleaning up the reference to the transport.
        """
        self.logger = None

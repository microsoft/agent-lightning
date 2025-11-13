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

"""Module containing base class for logging transport."""

from google.cloud.logging_v2.logger import _GLOBAL_RESOURCE


class Transport(object):
    """Base class for Google Cloud Logging handler transports.

    Subclasses of :class:`Transport` must have constructors that accept a
    client and name object, and must override :meth:`send`.
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
        super().__init__()

    def send(self, record, message, **kwargs):
        """Transport send to be implemented by subclasses.

        Args:
            record (logging.LogRecord): Python log record that the handler was called with.
            message (str or dict): The message from the ``LogRecord`` after being
                formatted by the associated log formatters.
            kwargs: Additional optional arguments for the logger
        """
        raise NotImplementedError

    def flush(self):
        """Submit any pending log records.

        For blocking/sync transports, this is a no-op.
        """
        pass

    def close(self):
        """Closes the transport and cleans up resources used by it.

        This call should be followed up by disowning the transport.
        """
        pass

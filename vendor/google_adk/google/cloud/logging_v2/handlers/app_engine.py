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

"""Logging handler for App Engine Flexible

Sends logs to the Cloud Logging API with the appropriate resource
and labels for App Engine logs.
"""

import logging
import os
import warnings

from google.cloud.logging_v2.handlers._helpers import get_request_data
from google.cloud.logging_v2.handlers._monitored_resources import (
    _create_app_engine_resource,
)
from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport

_DEFAULT_GAE_LOGGER_NAME = "app"

_GAE_PROJECT_ENV_FLEX = "GCLOUD_PROJECT"
_GAE_PROJECT_ENV_STANDARD = "GOOGLE_CLOUD_PROJECT"
_GAE_SERVICE_ENV = "GAE_SERVICE"
_GAE_VERSION_ENV = "GAE_VERSION"

_TRACE_ID_LABEL = "appengine.googleapis.com/trace_id"

_DEPRECATION_MSG = "AppEngineHandler is deprecated. Use CloudLoggingHandler instead."


class AppEngineHandler(logging.StreamHandler):
    """A logging handler that sends App Engine-formatted logs to Stackdriver.

    DEPRECATED:  use CloudLoggingHandler instead.
    """

    def __init__(
        self,
        client,
        *,
        name=_DEFAULT_GAE_LOGGER_NAME,
        transport=BackgroundThreadTransport,
        stream=None,
    ):
        """
        Args:
            client (~logging_v2.client.Client): The authenticated
                Google Cloud Logging client for this handler to use.
            name (Optional[str]): Name for the logger.
            transport (Optional[~logging_v2.transports.Transport]):
                The transport class. It should be a subclass
                of :class:`.Transport`. If unspecified,
                :class:`.BackgroundThreadTransport` will be used.
            stream (Optional[IO]): Stream to be used by the handler.

        """
        super(AppEngineHandler, self).__init__(stream)
        self.name = name
        self.client = client
        self.transport = transport(client, name)
        self.project_id = os.environ.get(
            _GAE_PROJECT_ENV_FLEX, os.environ.get(_GAE_PROJECT_ENV_STANDARD, "")
        )
        self.module_id = os.environ.get(_GAE_SERVICE_ENV, "")
        self.version_id = os.environ.get(_GAE_VERSION_ENV, "")
        self.resource = self.get_gae_resource()

        warnings.warn(_DEPRECATION_MSG, DeprecationWarning)

    def get_gae_resource(self):
        """Return the GAE resource using the environment variables.

        Returns:
            google.cloud.logging_v2.resource.Resource: Monitored resource for GAE.
        """
        return _create_app_engine_resource()

    def get_gae_labels(self):
        """Return the labels for GAE app.

        If the trace ID can be detected, it will be included as a label.
        Currently, no other labels are included.

        Returns:
            dict: Labels for GAE app.
        """
        gae_labels = {}

        _, trace_id, _, _ = get_request_data()
        if trace_id is not None:
            gae_labels[_TRACE_ID_LABEL] = trace_id

        return gae_labels

    def emit(self, record):
        """Actually log the specified logging record.

        Overrides the default emit behavior of ``StreamHandler``.

        See https://docs.python.org/2/library/logging.html#handler-objects

        Args:
            record (logging.LogRecord): The record to be logged.
        """
        message = super(AppEngineHandler, self).format(record)
        inferred_http, inferred_trace, _, _ = get_request_data()
        if inferred_trace is not None:
            inferred_trace = f"projects/{self.project_id}/traces/{inferred_trace}"
        # allow user overrides
        trace = getattr(record, "trace", inferred_trace)
        span_id = getattr(record, "span_id", None)
        http_request = getattr(record, "http_request", inferred_http)
        resource = getattr(record, "resource", self.resource)
        user_labels = getattr(record, "labels", {})
        # merge labels
        gae_labels = self.get_gae_labels()
        gae_labels.update(user_labels)
        # send off request
        self.transport.send(
            record,
            message,
            resource=resource,
            labels=gae_labels,
            trace=trace,
            span_id=span_id,
            http_request=http_request,
        )

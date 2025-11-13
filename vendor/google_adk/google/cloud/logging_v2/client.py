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

"""Client for interacting with the Google Cloud Logging API."""

import logging
import os
import sys


import google.api_core.client_options
from google.cloud.client import ClientWithProject
from google.cloud.environment_vars import DISABLE_GRPC
from google.cloud.logging_v2._helpers import _add_defaults_to_filter
from google.cloud.logging_v2._http import Connection
from google.cloud.logging_v2._http import _LoggingAPI as JSONLoggingAPI
from google.cloud.logging_v2._http import _MetricsAPI as JSONMetricsAPI
from google.cloud.logging_v2._http import _SinksAPI as JSONSinksAPI
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import StructuredLogHandler
from google.cloud.logging_v2.handlers import setup_logging
from google.cloud.logging_v2.handlers.handlers import EXCLUDED_LOGGER_DEFAULTS
from google.cloud.logging_v2.resource import Resource
from google.cloud.logging_v2.handlers._monitored_resources import detect_resource


from google.cloud.logging_v2.logger import Logger
from google.cloud.logging_v2.metric import Metric
from google.cloud.logging_v2.sink import Sink


_DISABLE_GRPC = os.getenv(DISABLE_GRPC, False)
_HAVE_GRPC = False

try:
    if not _DISABLE_GRPC:
        # only import if DISABLE_GRPC is not set
        from google.cloud.logging_v2 import _gapic

        _HAVE_GRPC = True
except ImportError:  # pragma: NO COVER
    # could not import gapic library. Fall back to HTTP mode
    _HAVE_GRPC = False
    _gapic = None

_USE_GRPC = _HAVE_GRPC and not _DISABLE_GRPC

_GAE_RESOURCE_TYPE = "gae_app"
_GKE_RESOURCE_TYPE = "k8s_container"
_GCF_RESOURCE_TYPE = "cloud_function"
_RUN_RESOURCE_TYPE = "cloud_run_revision"


class Client(ClientWithProject):
    """Client to bundle configuration needed for API requests."""

    _logging_api = None
    _sinks_api = None
    _metrics_api = None

    SCOPE = (
        "https://www.googleapis.com/auth/logging.read",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/logging.admin",
        "https://www.googleapis.com/auth/cloud-platform",
    )
    """The scopes required for authenticating as a Logging consumer."""

    def __init__(
        self,
        *,
        project=None,
        credentials=None,
        _http=None,
        _use_grpc=None,
        client_info=None,
        client_options=None,
    ):
        """
        Args:
            project (Optional[str]): the project which the client acts on behalf of.
                If not passed, falls back to the default inferred
                from the environment.
            credentials (Optional[google.auth.credentials.Credentials]):
                Thehe OAuth2 Credentials to use for this
                client. If not passed (and if no ``_http`` object is
                passed), falls back to the default inferred from the
                environment.
            _http (Optional[requests.Session]):  HTTP object to make requests.
                Can be any object that defines ``request()`` with the same interface as
                :meth:`requests.Session.request`. If not passed, an
                ``_http`` object is created that is bound to the
                ``credentials`` for the current object.
                This parameter should be considered private, and could
                change in the future.
            _use_grpc (Optional[bool]): Explicitly specifies whether
                to use the gRPC transport or HTTP. If unset,
                falls back to the ``GOOGLE_CLOUD_DISABLE_GRPC``
                environment variable
                This parameter should be considered private, and could
                change in the future.
            client_info (Optional[Union[google.api_core.client_info.ClientInfo, google.api_core.gapic_v1.client_info.ClientInfo]]):
                The client info used to send a user-agent string along with API
                requests. If ``None``, then default info will be used. Generally,
                you only need to set this if you're developing your own library
                or partner tool.
            client_options (Optional[Union[dict, google.api_core.client_options.ClientOptions]]):
                Client options used to set user options
                on the client. API Endpoint should be set through client_options.
        """
        super(Client, self).__init__(
            project=project,
            credentials=credentials,
            _http=_http,
            client_options=client_options,
        )

        kw_args = {"client_info": client_info}
        if client_options:
            if isinstance(client_options, dict):
                client_options = google.api_core.client_options.from_dict(
                    client_options
                )
            if client_options.api_endpoint:
                api_endpoint = client_options.api_endpoint
                kw_args["api_endpoint"] = api_endpoint

        self._connection = Connection(self, **kw_args)
        if client_info is None:
            # if client info not passed in, use the discovered
            # client info from _connection object
            client_info = self._connection._client_info

        self._client_info = client_info
        self._client_options = client_options
        if _use_grpc is None:
            self._use_grpc = _USE_GRPC
        else:
            self._use_grpc = _use_grpc

        self._handlers = set()

    @property
    def logging_api(self):
        """Helper for logging-related API calls.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.logs
        """
        if self._logging_api is None:
            if self._use_grpc:
                self._logging_api = _gapic.make_logging_api(self)
            else:
                self._logging_api = JSONLoggingAPI(self)
        return self._logging_api

    @property
    def sinks_api(self):
        """Helper for log sink-related API calls.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks
        """
        if self._sinks_api is None:
            if self._use_grpc:
                self._sinks_api = _gapic.make_sinks_api(self)
            else:
                self._sinks_api = JSONSinksAPI(self)
        return self._sinks_api

    @property
    def metrics_api(self):
        """Helper for log metric-related API calls.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics
        """
        if self._metrics_api is None:
            if self._use_grpc:
                self._metrics_api = _gapic.make_metrics_api(self)
            else:
                self._metrics_api = JSONMetricsAPI(self)
        return self._metrics_api

    def logger(self, name, *, labels=None, resource=None):
        """Creates a logger bound to the current client.

        Args:
            name (str): The name of the logger to be constructed.
            resource (Optional[~logging_v2.Resource]): a monitored resource object
                representing the resource the code was run on. If not given, will
                be inferred from the environment.
            labels (Optional[dict]): Mapping of default labels for entries written
                via this logger.

        Returns:
            ~logging_v2.logger.Logger: Logger created with the current client.
        """
        return Logger(name, client=self, labels=labels, resource=resource)

    def list_entries(
        self,
        *,
        resource_names=None,
        filter_=None,
        order_by=None,
        max_results=None,
        page_size=None,
        page_token=None,
    ):
        """Return a generator of log entry resources.

        Args:
            resource_names (Sequence[str]): Names of one or more parent resources
                from which to retrieve log entries:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]"

                If not passed, defaults to the project bound to the API's client.

            filter_ (str): a filter expression. See
                https://cloud.google.com/logging/docs/view/advanced_filters
            order_by (str) One of :data:`~logging_v2.ASCENDING`
                or :data:`~logging_v2.DESCENDING`.
            max_results (Optional[int]):
                Optional. The maximum number of entries to return.
                Non-positive values are treated as 0. If None, uses API defaults.
            page_size (int): number of entries to fetch in each API call. Although
                requests are paged internally, logs are returned by the generator
                one at a time. If not passed, defaults to a value set by the API.
            page_token (str): opaque marker for the starting "page" of entries. If not
                passed, the API will return the first page of entries.

        Returns:
            Generator[~logging_v2.LogEntry]
        """
        if resource_names is None:
            resource_names = [f"projects/{self.project}"]
        filter_ = _add_defaults_to_filter(filter_)

        return self.logging_api.list_entries(
            resource_names=resource_names,
            filter_=filter_,
            order_by=order_by,
            max_results=max_results,
            page_size=page_size,
            page_token=page_token,
        )

    def sink(self, name, *, filter_=None, destination=None):
        """Creates a sink bound to the current client.

        Args:
            name (str): the name of the sink to be constructed.
            filter_ (Optional[str]): the advanced logs filter expression
                defining the entries exported by the sink.  If not
                passed, the instance should already exist, to be
                refreshed via :meth:`Sink.reload`.
            destination (str): destination URI for the entries exported by
                the sink.  If not passed, the instance should
                already exist, to be refreshed via
                :meth:`Sink.reload`.

        Returns:
            ~logging_v2.sink.Sink: Sink created with the current client.
        """
        return Sink(name, filter_=filter_, destination=destination, client=self)

    def list_sinks(
        self, *, parent=None, max_results=None, page_size=None, page_token=None
    ):
        """List sinks for the a parent resource.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/list

        Args:
            parent (Optional[str]): The parent resource whose sinks are to be listed:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]".

                If not passed, defaults to the project bound to the API's client.
            max_results (Optional[int]):
                Optional. The maximum number of entries to return.
                Non-positive values are treated as 0. If None, uses API defaults.
            page_size (int): number of entries to fetch in each API call. Although
                requests are paged internally, logs are returned by the generator
                one at a time. If not passed, defaults to a value set by the API.
            page_token (str): opaque marker for the starting "page" of entries. If not
                passed, the API will return the first page of entries.

        Returns:
            Generator[~logging_v2.Sink]
        """
        if parent is None:
            parent = f"projects/{self.project}"
        return self.sinks_api.list_sinks(
            parent=parent,
            max_results=max_results,
            page_size=page_size,
            page_token=page_token,
        )

    def metric(self, name, *, filter_=None, description=""):
        """Creates a metric bound to the current client.

        Args:
            name (str): The name of the metric to be constructed.
            filter_(Optional[str]): The advanced logs filter expression defining the
                entries tracked by the metric.  If not
                passed, the instance should already exist, to be
                refreshed via :meth:`Metric.reload`.
            description (Optional[str]): The description of the metric to be constructed.
                If not passed, the instance should already exist,
                to be refreshed via :meth:`Metric.reload`.

        Returns:
            ~logging_v2.metric.Metric: Metric created with the current client.
        """
        return Metric(name, filter_=filter_, client=self, description=description)

    def list_metrics(self, *, max_results=None, page_size=None, page_token=None):
        """List metrics for the project associated with this client.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/list

        Args:
            max_results (Optional[int]):
                Optional. The maximum number of entries to return.
                Non-positive values are treated as 0. If None, uses API defaults.
            page_size (int): number of entries to fetch in each API call. Although
                requests are paged internally, logs are returned by the generator
                one at a time. If not passed, defaults to a value set by the API.
            page_token (str): opaque marker for the starting "page" of entries. If not
                passed, the API will return the first page of entries.

        Returns:
            Generator[logging_v2.Metric]
        """
        return self.metrics_api.list_metrics(
            self.project,
            max_results=max_results,
            page_size=page_size,
            page_token=page_token,
        )

    def get_default_handler(self, **kw):
        """Return the default logging handler based on the local environment.

        Args:
            kw (dict): keyword args passed to handler constructor

        Returns:
            logging.Handler: The default log handler based on the environment
        """
        monitored_resource = kw.pop("resource", detect_resource(self.project))

        if isinstance(monitored_resource, Resource):
            if monitored_resource.type == _GAE_RESOURCE_TYPE:
                return CloudLoggingHandler(self, resource=monitored_resource, **kw)
            elif monitored_resource.type == _GKE_RESOURCE_TYPE:
                return StructuredLogHandler(**kw, project_id=self.project)
            elif monitored_resource.type == _GCF_RESOURCE_TYPE:
                # __stdout__ stream required to support structured logging on Python 3.7
                kw["stream"] = kw.get("stream", sys.__stdout__)
                return StructuredLogHandler(**kw, project_id=self.project)
            elif monitored_resource.type == _RUN_RESOURCE_TYPE:
                return StructuredLogHandler(**kw, project_id=self.project)
        return CloudLoggingHandler(self, resource=monitored_resource, **kw)

    def setup_logging(
        self, *, log_level=logging.INFO, excluded_loggers=EXCLUDED_LOGGER_DEFAULTS, **kw
    ):
        """Attach default Cloud Logging handler to the root logger.

        This method uses the default log handler, obtained by
        :meth:`~get_default_handler`, and attaches it to the root Python
        logger, so that a call such as ``logging.warn``, as well as all child
        loggers, will report to Cloud Logging.

        Args:
            log_level (Optional[int]): The logging level threshold of the attached logger,
                as set by the :meth:`logging.Logger.setLevel` method. Defaults to
                :const:`logging.INFO`.
            excluded_loggers (Optional[Tuple[str]]): The loggers to not attach the
                handler to. This will always include the
                loggers in the path of the logging client
                itself.
        Returns:
            dict: keyword args passed to handler constructor
        """
        handler = self.get_default_handler(**kw)
        self._handlers.add(handler)
        setup_logging(handler, log_level=log_level, excluded_loggers=excluded_loggers)

    def flush_handlers(self):
        """Flushes all Python log handlers associated with this Client."""

        for handler in self._handlers:
            handler.flush()

    def close(self):
        """Closes the Client and all handlers associated with this Client."""
        super(Client, self).close()
        for handler in self._handlers:
            handler.close()

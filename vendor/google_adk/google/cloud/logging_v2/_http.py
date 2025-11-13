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

"""Interact with Cloud Logging via JSON-over-HTTP."""

import functools

from google.api_core import page_iterator
from google.cloud import _http

from google.cloud.logging_v2 import __version__
from google.cloud.logging_v2._helpers import entry_from_resource
from google.cloud.logging_v2.sink import Sink
from google.cloud.logging_v2.metric import Metric


class Connection(_http.JSONConnection):
    DEFAULT_API_ENDPOINT = "https://logging.googleapis.com"

    def __init__(self, client, *, client_info=None, api_endpoint=DEFAULT_API_ENDPOINT):
        """A connection to Google Cloud Logging via the JSON REST API.

        Args:
            client (google.cloud.logging_v2.cliet.Client):
                The client that owns the current connection.
            client_info (Optional[google.api_core.client_info.ClientInfo]):
                Instance used to generate user agent.
            client_options (Optional[google.api_core.client_options.ClientOptions]):
                Client options used to set user options
                on the client. API Endpoint should be set through client_options.
        """
        super(Connection, self).__init__(client, client_info)
        self.API_BASE_URL = api_endpoint
        self._client_info.gapic_version = __version__
        self._client_info.client_library_version = __version__

    API_VERSION = "v2"
    """The version of the API, used in building the API call's URL."""

    API_URL_TEMPLATE = "{api_base_url}/{api_version}{path}"
    """A template for the URL of a particular API call."""


class _LoggingAPI(object):
    """Helper mapping logging-related APIs.

    See
    https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries
    https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.logs

    :type client: :class:`~google.cloud.logging.client.Client`
    :param client: The client used to make API requests.
    """

    def __init__(self, client):
        self._client = client
        self.api_request = client._connection.api_request

    def list_entries(
        self,
        resource_names,
        *,
        filter_=None,
        order_by=None,
        max_results=None,
        page_size=None,
        page_token=None,
    ):
        """Return a page of log entry resources.

        Args:
            resource_names (Sequence[str]): Names of one or more parent resources
                from which to retrieve log entries:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]"

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
        extra_params = {"resourceNames": resource_names}

        if filter_ is not None:
            extra_params["filter"] = filter_

        if order_by is not None:
            extra_params["orderBy"] = order_by

        if page_size is not None:
            extra_params["pageSize"] = page_size

        path = "/entries:list"
        # We attach a mutable loggers dictionary so that as Logger
        # objects are created by entry_from_resource, they can be
        # re-used by other log entries from the same logger.
        loggers = {}
        item_to_value = functools.partial(_item_to_entry, loggers=loggers)
        iterator = page_iterator.HTTPIterator(
            client=self._client,
            api_request=self._client._connection.api_request,
            path=path,
            item_to_value=item_to_value,
            items_key="entries",
            page_token=page_token,
            extra_params=extra_params,
        )
        # This method uses POST to make a read-only request.
        iterator._HTTP_METHOD = "POST"

        return _entries_pager(iterator, max_results)

    def write_entries(
        self,
        entries,
        *,
        logger_name=None,
        resource=None,
        labels=None,
        partial_success=True,
        dry_run=False,
    ):
        """Log an entry resource via a POST request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries/write

        Args:
            entries (Sequence[Mapping[str, ...]]): sequence of mappings representing
                the log entry resources to log.
            logger_name (Optional[str]): name of default logger to which to log the entries;
                individual entries may override.
            resource(Optional[Mapping[str, ...]]): default resource to associate with entries;
                individual entries may override.
            labels (Optional[Mapping[str, ...]]): default labels to associate with entries;
                individual entries may override.
            partial_success (Optional[bool]): Whether valid entries should be written even if
                some other entries fail due to INVALID_ARGUMENT or
                PERMISSION_DENIED errors. If any entry is not written, then
                the response status is the error associated with one of the
                failed entries and the response includes error details keyed
                by the entries' zero-based index in the ``entries.write``
                method.
            dry_run (Optional[bool]):
                If true, the request should expect normal response,
                but the entries won't be persisted nor exported.
                Useful for checking whether the logging API endpoints are working
                properly before sending valuable data.
        """
        data = {
            "entries": list(entries),
            "partialSuccess": partial_success,
            "dry_run": dry_run,
        }

        if logger_name is not None:
            data["logName"] = logger_name

        if resource is not None:
            data["resource"] = resource

        if labels is not None:
            data["labels"] = labels

        self.api_request(method="POST", path="/entries:write", data=data)

    def logger_delete(self, logger_name):
        """Delete all entries in a logger.

        Args:
            logger_name (str):  The resource name of the log to delete:

                ::

                    "projects/[PROJECT_ID]/logs/[LOG_ID]"
                    "organizations/[ORGANIZATION_ID]/logs/[LOG_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]"
                    "folders/[FOLDER_ID]/logs/[LOG_ID]"

                ``[LOG_ID]`` must be URL-encoded. For example,
                ``"projects/my-project-id/logs/syslog"``,
                ``"organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Factivity"``.
        """
        path = f"/{logger_name}"
        self.api_request(method="DELETE", path=path)


class _SinksAPI(object):
    """Helper mapping sink-related APIs.

    See
    https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks
    """

    def __init__(self, client):
        self._client = client
        self.api_request = client._connection.api_request

    def list_sinks(self, parent, *, max_results=None, page_size=None, page_token=None):
        """List sinks for the parent resource.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/list

        Args:
            parent (str): The parent resource whose sinks are to be listed:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]".
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
        extra_params = {}

        if page_size is not None:
            extra_params["pageSize"] = page_size

        path = f"/{parent}/sinks"
        iterator = page_iterator.HTTPIterator(
            client=self._client,
            api_request=self._client._connection.api_request,
            path=path,
            item_to_value=_item_to_sink,
            items_key="sinks",
            page_token=page_token,
            extra_params=extra_params,
        )

        return _entries_pager(iterator, max_results)

    def sink_create(
        self, parent, sink_name, filter_, destination, *, unique_writer_identity=False
    ):
        """Create a sink resource.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/create

        Args:
            parent(str): The resource in which to create the sink:

            ::

                "projects/[PROJECT_ID]"
                "organizations/[ORGANIZATION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]"
                "folders/[FOLDER_ID]".
            sink_name (str): The name of the sink.
            filter_ (str): The advanced logs filter expression defining the
                entries exported by the sink.
            destination (str): Destination URI for the entries exported by
                the sink.
            unique_writer_identity (Optional[bool]):  determines the kind of
                IAM identity returned as writer_identity in the new sink.

        Returns:
            dict: The sink resource returned from the API.
        """
        target = f"/{parent}/sinks"
        data = {"name": sink_name, "filter": filter_, "destination": destination}
        query_params = {"uniqueWriterIdentity": unique_writer_identity}
        return self.api_request(
            method="POST", path=target, data=data, query_params=query_params
        )

    def sink_get(self, sink_name):
        """Retrieve a sink resource.

        Args:
            sink_name (str): The resource name of the sink:

            ::

                "projects/[PROJECT_ID]/sinks/[SINK_ID]"
                "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
                "folders/[FOLDER_ID]/sinks/[SINK_ID]"

        Returns:
            dict: The JSON sink object returned from the API.
        """
        target = f"/{sink_name}"
        return self.api_request(method="GET", path=target)

    def sink_update(
        self, sink_name, filter_, destination, *, unique_writer_identity=False
    ):
        """Update a sink resource.

        Args:
            sink_name (str): Required. The resource name of the sink:

            ::

                "projects/[PROJECT_ID]/sinks/[SINK_ID]"
                "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
                "folders/[FOLDER_ID]/sinks/[SINK_ID]"
            filter_ (str): The advanced logs filter expression defining the
                entries exported by the sink.
            destination (str): destination URI for the entries exported by
                the sink.
            unique_writer_identity (Optional[bool]): determines the kind of
                IAM identity returned as writer_identity in the new sink.


        Returns:
            dict: The returned (updated) resource.
        """
        target = f"/{sink_name}"
        name = sink_name.split("/")[-1]  # parse name out of full resource name
        data = {"name": name, "filter": filter_, "destination": destination}
        query_params = {"uniqueWriterIdentity": unique_writer_identity}
        return self.api_request(
            method="PUT", path=target, query_params=query_params, data=data
        )

    def sink_delete(self, sink_name):
        """Delete a sink resource.

        Args:
            sink_name (str): Required. The full resource name of the sink to delete,
                including the parent resource and the sink identifier:

                ::

                    "projects/[PROJECT_ID]/sinks/[SINK_ID]"
                    "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
                    "folders/[FOLDER_ID]/sinks/[SINK_ID]"

                Example: ``"projects/my-project-id/sinks/my-sink-id"``.
        """
        target = f"/{sink_name}"
        self.api_request(method="DELETE", path=target)


class _MetricsAPI(object):
    """Helper mapping sink-related APIs."""

    def __init__(self, client):
        self._client = client
        self.api_request = client._connection.api_request

    def list_metrics(
        self, project, *, max_results=None, page_size=None, page_token=None
    ):
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
        extra_params = {}

        if page_size is not None:
            extra_params["pageSize"] = page_size

        path = f"/projects/{project}/metrics"
        iterator = page_iterator.HTTPIterator(
            client=self._client,
            api_request=self._client._connection.api_request,
            path=path,
            item_to_value=_item_to_metric,
            items_key="metrics",
            page_token=page_token,
            extra_params=extra_params,
        )
        return _entries_pager(iterator, max_results)

    def metric_create(self, project, metric_name, filter_, description):
        """Create a metric resource.

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/create

        Args:
            project (str): ID of the project in which to create the metric.
            metric_name (str): The name of the metric
            filter_ (str): The advanced logs filter expression defining the
                entries exported by the metric.
            description (str): description of the metric.
        """
        target = f"/projects/{project}/metrics"
        data = {"name": metric_name, "filter": filter_, "description": description}
        self.api_request(method="POST", path=target, data=data)

    def metric_get(self, project, metric_name):
        """Retrieve a metric resource.

        Args:
            project (str): ID of the project containing the metric.
            metric_name (str): The name of the metric

        Returns:
            dict: The JSON metric object returned from the API.
        """
        target = f"/projects/{project}/metrics/{metric_name}"
        return self.api_request(method="GET", path=target)

    def metric_update(self, project, metric_name, filter_, description):
        """Update a metric resource.

         See
         https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/update

        Args:
             project (str): ID of the project containing the metric.
             metric_name (str): the name of the metric
             filter_ (str): the advanced logs filter expression defining the
                 entries exported by the metric.
             description (str): description of the metric.

         Returns:
             dict: The returned (updated) resource.
        """
        target = f"/projects/{project}/metrics/{metric_name}"
        data = {"name": metric_name, "filter": filter_, "description": description}
        return self.api_request(method="PUT", path=target, data=data)

    def metric_delete(self, project, metric_name):
        """Delete a metric resource.

        Args:
            project (str): ID of the project containing the metric.
            metric_name (str): The name of the metric
        """
        target = f"/projects/{project}/metrics/{metric_name}"
        self.api_request(method="DELETE", path=target)


def _entries_pager(page_iter, max_results=None):
    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be positive")

    i = 0
    for page in page_iter:
        if max_results is not None and i >= max_results:
            break
        yield page
        i += 1


def _item_to_entry(iterator, resource, loggers):
    """Convert a log entry resource to the native object.

    .. note::

        This method does not have the correct signature to be used as
        the ``item_to_value`` argument to
        :class:`~google.api_core.page_iterator.Iterator`. It is intended to be
        patched with a mutable ``loggers`` argument that can be updated
        on subsequent calls. For an example, see how the method is
        used above in :meth:`_LoggingAPI.list_entries`.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that
            is currently in use.
        resource (dict): Log entry JSON resource returned from the API.
        loggers (Mapping[str, logging_v2.logger.Logger]):
            A mapping of logger fullnames -> loggers.  If the logger
            that owns the entry is not in ``loggers``, the entry
            will have a newly-created logger.

    Returns:
        ~logging_v2.entries._BaseEntry: The next log entry in the page.
    """
    return entry_from_resource(resource, iterator.client, loggers)


def _item_to_sink(iterator, resource):
    """Convert a sink resource to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that
            is currently in use.
        resource (dict): Sink JSON resource returned from the API.

    Returns:
        ~logging_v2.sink.Sink: The next sink in the page.
    """
    return Sink.from_api_repr(resource, iterator.client)


def _item_to_metric(iterator, resource):
    """Convert a metric resource to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that
            is currently in use.
        resource (dict): Sink JSON resource returned from the API.

    Returns:
        ~logging_v2.metric.Metric:
            The next metric in the page.
    """
    return Metric.from_api_repr(resource, iterator.client)

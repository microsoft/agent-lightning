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

"""Define Cloud Logging API Metrics."""

from google.cloud.exceptions import NotFound


class Metric(object):
    """Metrics represent named filters for log entries.

    See
    https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics
    """

    def __init__(self, name, *, filter_=None, client=None, description=""):
        """
        Args:
            name (str): The name of the metric.
            filter_ (str): the advanced logs filter expression defining the entries
                   tracked by the metric.  If not passed, the instance should
                   already exist, to be refreshed via :meth:`reload`.
            client (Optional[~logging_v2.client.Client]): A client which holds
                credentials and project configuration for the sink (which requires a project).
            description (Optional[str]): An optional description of the metric.

        """
        self.name = name
        self._client = client
        self.filter_ = filter_
        self.description = description

    @property
    def client(self):
        """Clent bound to the logger."""
        return self._client

    @property
    def project(self):
        """Project bound to the logger."""
        return self._client.project

    @property
    def full_name(self):
        """Fully-qualified name used in metric APIs"""
        return f"projects/{self.project}/metrics/{self.name}"

    @property
    def path(self):
        """URL path for the metric's APIs"""
        return f"/{self.full_name}"

    @classmethod
    def from_api_repr(cls, resource, client):
        """Construct a metric given its API representation

        Args:
            resource (dict): metric resource representation returned from the API
            client (~logging_v2.client.Client): Client which holds
                credentials and project configuration for the sink.

        Returns:
            google.cloud.logging_v2.metric.Metric
        """
        metric_name = resource["name"]
        filter_ = resource["filter"]
        description = resource.get("description", "")
        return cls(metric_name, filter_=filter_, client=client, description=description)

    def _require_client(self, client):
        """Check client or verify over-ride. Also sets ``parent``.

        Args:
            client (Union[None, ~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.

        Returns:
            google.cloud.logging_v2.client.Client: The client passed in
                or the currently bound client.
        """
        if client is None:
            client = self._client
        return client

    def create(self, *, client=None):
        """Create the metric via a PUT request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/create

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        client.metrics_api.metric_create(
            self.project, self.name, self.filter_, self.description
        )

    def exists(self, *, client=None):
        """Test for the existence of the metric via a GET request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/get

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.

        Returns:
            bool: Boolean indicating existence of the metric.
        """
        client = self._require_client(client)

        try:
            client.metrics_api.metric_get(self.project, self.name)
        except NotFound:
            return False
        else:
            return True

    def reload(self, *, client=None):
        """API call:  sync local metric configuration via a GET request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/get

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        data = client.metrics_api.metric_get(self.project, self.name)
        self.description = data.get("description", "")
        self.filter_ = data["filter"]

    def update(self, *, client=None):
        """API call:  update metric configuration via a PUT request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/update

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        client.metrics_api.metric_update(
            self.project, self.name, self.filter_, self.description
        )

    def delete(self, *, client=None):
        """API call:  delete a metric via a DELETE request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics/delete

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        client.metrics_api.metric_delete(self.project, self.name)

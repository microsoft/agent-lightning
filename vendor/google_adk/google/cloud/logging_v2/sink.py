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

"""Define Cloud Logging API Sinks."""

from google.cloud.exceptions import NotFound


class Sink(object):
    """Sinks represent filtered exports for log entries.

    See https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks
    """

    def __init__(
        self, name, *, filter_=None, parent=None, destination=None, client=None
    ):
        """
        Args:
            name (str): The name of the sink.
            parent(Optional[str]): The resource in which to create the sink:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]".

                Defaults to the project stored on the client.
            filter_ (Optional[str]): The advanced logs filter expression defining
                the entries exported by the sink.
            destination (Optional[str]): Destination URI for the entries exported by the sink.
                If not passed, the instance should already exist, to
                be refreshed via :meth:`reload`.
            client (Optional[~logging_v2.client.Client]): A client which holds
                credentials and project configuration for the sink (which requires a project).
        """
        self.name = name
        self.filter_ = filter_
        self.destination = destination
        self._client = client
        self._parent = parent
        self._writer_identity = None

    @property
    def client(self):
        """Client bound to the sink."""
        return self._client

    @property
    def parent(self):
        """Parent resource of the sink (project, organization, billingAccount, or folder)."""
        if self._parent is None:
            self._parent = f"projects/{self.client.project}"
        return self._parent

    @property
    def full_name(self):
        """Fully-qualified name used in sink APIs"""
        return f"{self.parent}/sinks/{self.name}"

    @property
    def path(self):
        """URL path for the sink's APIs"""
        return f"/{self.full_name}"

    @property
    def writer_identity(self):
        """Identity used for exports via the sink"""
        return self._writer_identity

    def _update_from_api_repr(self, resource):
        """Helper for API methods returning sink resources."""
        self.destination = resource["destination"]
        self.filter_ = resource.get("filter")
        self._writer_identity = resource.get("writerIdentity")

    @classmethod
    def from_api_repr(cls, resource, client, *, parent=None):
        """Construct a sink given its API representation

        Args:
            resource (dict): sink resource representation returned from the API
            client (~logging_v2.client.Client): Client which holds
                credentials and project configuration for the sink.
            parent(Optional[str]): The resource in which to create the sink:

                ::

                    "projects/[PROJECT_ID]"
                    "organizations/[ORGANIZATION_ID]"
                    "billingAccounts/[BILLING_ACCOUNT_ID]"
                    "folders/[FOLDER_ID]".

                Defaults to the project stored on the client.

        Returns:
            ~logging_v2.sink.Sink: Sink parsed from ``resource``.

        Raises:
            ValueError: if ``client`` is not ``None`` and the
                project from the resource does not agree with the project
                from the client.
        """
        sink_name = resource["name"]
        instance = cls(sink_name, client=client, parent=parent)
        instance._update_from_api_repr(resource)
        return instance

    def _require_client(self, client):
        """Check client or verify over-ride. Also sets ``parent``.

        Args:
            client (Union[None, ~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.

        Returns:
            ~logging_v2.client.Client: The client passed in
                or the currently bound client.
        """
        if client is None:
            client = self._client
        return client

    def create(self, *, client=None, unique_writer_identity=False):
        """Create the sink via a PUT request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/create

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
            unique_writer_identity (Optional[bool]): Determines the kind of
                IAM identity returned as writer_identity in the new sink.
        """
        client = self._require_client(client)
        resource = client.sinks_api.sink_create(
            self.parent,
            self.name,
            self.filter_,
            self.destination,
            unique_writer_identity=unique_writer_identity,
        )
        self._update_from_api_repr(resource)

    def exists(self, *, client=None):
        """Test for the existence of the sink via a GET request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/get

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.

        Returns:
            bool: Boolean indicating existence of the sink.
        """
        client = self._require_client(client)

        try:
            client.sinks_api.sink_get(self.full_name)
        except NotFound:
            return False
        else:
            return True

    def reload(self, *, client=None):
        """Sync local sink configuration via a GET request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/get

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        resource = client.sinks_api.sink_get(self.full_name)
        self._update_from_api_repr(resource)

    def update(self, *, client=None, unique_writer_identity=False):
        """Update sink configuration via a PUT request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/update

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
            unique_writer_identity (Optional[bool]): Determines the kind of
                IAM identity returned as writer_identity in the new sink.
        """
        client = self._require_client(client)
        resource = client.sinks_api.sink_update(
            self.full_name,
            self.filter_,
            self.destination,
            unique_writer_identity=unique_writer_identity,
        )
        self._update_from_api_repr(resource)

    def delete(self, *, client=None):
        """Delete a sink via a DELETE request

        See
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks/delete

        Args:
            client (Optional[~logging_v2.client.Client]):
                The client to use.  If not passed, falls back to the
                ``client`` stored on the current sink.
        """
        client = self._require_client(client)
        client.sinks_api.sink_delete(self.full_name)

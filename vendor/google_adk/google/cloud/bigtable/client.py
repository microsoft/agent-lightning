# Copyright 2015 Google LLC
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

"""Parent client for calling the Google Cloud Bigtable API.

This is the base from which all interactions with the API occur.

In the hierarchy of API concepts

* a :class:`~google.cloud.bigtable.client.Client` owns an
  :class:`~google.cloud.bigtable.instance.Instance`
* an :class:`~google.cloud.bigtable.instance.Instance` owns a
  :class:`~google.cloud.bigtable.table.Table`
* a :class:`~google.cloud.bigtable.table.Table` owns a
  :class:`~.column_family.ColumnFamily`
* a :class:`~google.cloud.bigtable.table.Table` owns a
  :class:`~google.cloud.bigtable.row.Row` (and all the cells in the row)
"""
import os
import warnings
import grpc  # type: ignore

from google.api_core.gapic_v1 import client_info as client_info_lib
from google.auth.credentials import AnonymousCredentials  # type: ignore

from google.cloud import bigtable_v2
from google.cloud import bigtable_admin_v2
from google.cloud.bigtable_v2.services.bigtable.transports import BigtableGrpcTransport
from google.cloud.bigtable_admin_v2.services.bigtable_instance_admin.transports import (
    BigtableInstanceAdminGrpcTransport,
)
from google.cloud.bigtable_admin_v2.services.bigtable_table_admin.transports import (
    BigtableTableAdminGrpcTransport,
)

from google.cloud import bigtable
from google.cloud.bigtable.instance import Instance
from google.cloud.bigtable.cluster import Cluster

from google.cloud.client import ClientWithProject  # type: ignore

from google.cloud.bigtable_admin_v2.types import instance
from google.cloud.bigtable.cluster import _CLUSTER_NAME_RE
from google.cloud.environment_vars import BIGTABLE_EMULATOR  # type: ignore


INSTANCE_TYPE_PRODUCTION = instance.Instance.Type.PRODUCTION
INSTANCE_TYPE_DEVELOPMENT = instance.Instance.Type.DEVELOPMENT
INSTANCE_TYPE_UNSPECIFIED = instance.Instance.Type.TYPE_UNSPECIFIED
SPANNER_ADMIN_SCOPE = "https://www.googleapis.com/auth/spanner.admin"
ADMIN_SCOPE = "https://www.googleapis.com/auth/bigtable.admin"
"""Scope for interacting with the Cluster Admin and Table Admin APIs."""
DATA_SCOPE = "https://www.googleapis.com/auth/bigtable.data"
"""Scope for reading and writing table data."""
READ_ONLY_SCOPE = "https://www.googleapis.com/auth/bigtable.data.readonly"
"""Scope for reading table data."""

_DEFAULT_BIGTABLE_EMULATOR_CLIENT = "google-cloud-bigtable-emulator"
_GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
)


def _create_gapic_client(client_class, client_options=None, transport=None):
    def inner(self):
        return client_class(
            credentials=None,
            client_info=self._client_info,
            client_options=client_options,
            transport=transport,
        )

    return inner


class Client(ClientWithProject):
    """Client for interacting with Google Cloud Bigtable API.

    .. note::

        Since the Cloud Bigtable API requires the gRPC transport, no
        ``_http`` argument is accepted by this class.

    :type project: :class:`str` or :func:`unicode <unicode>`
    :param project: (Optional) The ID of the project which owns the
                    instances, tables and data. If not provided, will
                    attempt to determine from the environment.

    :type credentials: :class:`~google.auth.credentials.Credentials`
    :param credentials: (Optional) The OAuth2 Credentials to use for this
                        client. If not passed, falls back to the default
                        inferred from the environment.

    :type read_only: bool
    :param read_only: (Optional) Boolean indicating if the data scope should be
                      for reading only (or for writing as well). Defaults to
                      :data:`False`.

    :type admin: bool
    :param admin: (Optional) Boolean indicating if the client will be used to
                  interact with the Instance Admin or Table Admin APIs. This
                  requires the :const:`ADMIN_SCOPE`. Defaults to :data:`False`.

    :type: client_info: :class:`google.api_core.gapic_v1.client_info.ClientInfo`
    :param client_info:
        The client info used to send a user-agent string along with API
        requests. If ``None``, then default info will be used. Generally,
        you only need to set this if you're developing your own library
        or partner tool.

    :type client_options: :class:`~google.api_core.client_options.ClientOptions`
        or :class:`dict`
    :param client_options: (Optional) Client options used to set user options
        on the client. API Endpoint should be set through client_options.

    :type admin_client_options:
        :class:`~google.api_core.client_options.ClientOptions` or :class:`dict`
    :param admin_client_options: (Optional) Client options used to set user
        options on the client. API Endpoint for admin operations should be set
        through admin_client_options.

    :type channel: :instance: grpc.Channel
    :param channel (grpc.Channel): (Optional) DEPRECATED:
            A ``Channel`` instance through which to make calls.
            This argument is mutually exclusive with ``credentials``;
            providing both will raise an exception. No longer used.

    :raises: :class:`ValueError <exceptions.ValueError>` if both ``read_only``
             and ``admin`` are :data:`True`
    """

    _table_data_client = None
    _table_admin_client = None
    _instance_admin_client = None

    def __init__(
        self,
        project=None,
        credentials=None,
        read_only=False,
        admin=False,
        client_info=None,
        client_options=None,
        admin_client_options=None,
        channel=None,
    ):
        if client_info is None:
            client_info = client_info_lib.ClientInfo(
                client_library_version=bigtable.__version__,
            )
        if read_only and admin:
            raise ValueError(
                "A read-only client cannot also perform" "administrative actions."
            )

        # NOTE: We set the scopes **before** calling the parent constructor.
        #       It **may** use those scopes in ``with_scopes_if_required``.
        self._read_only = bool(read_only)
        self._admin = bool(admin)
        self._client_info = client_info
        self._emulator_host = os.getenv(BIGTABLE_EMULATOR)

        if self._emulator_host is not None:
            if credentials is None:
                credentials = AnonymousCredentials()
            if project is None:
                project = _DEFAULT_BIGTABLE_EMULATOR_CLIENT

        if channel is not None:
            warnings.warn(
                "'channel' is deprecated and no longer used.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._client_options = client_options
        self._admin_client_options = admin_client_options
        self._channel = channel
        self.SCOPE = self._get_scopes()
        super(Client, self).__init__(
            project=project,
            credentials=credentials,
            client_options=client_options,
        )

    def _get_scopes(self):
        """Get the scopes corresponding to admin / read-only state.

        Returns:
            Tuple[str, ...]: The tuple of scopes.
        """
        if self._read_only:
            scopes = (READ_ONLY_SCOPE,)
        else:
            scopes = (DATA_SCOPE,)

        if self._admin:
            scopes += (ADMIN_SCOPE,)

        return scopes

    def _emulator_channel(self, transport, options):
        """Create a channel for use with the Bigtable emulator.

        Insecure channels are used for the emulator as secure channels
        cannot be used to communicate on some environments.
        https://github.com/googleapis/python-firestore/issues/359

        Returns:
            grpc.Channel or grpc.aio.Channel
        """
        # Note: this code also exists in the firestore client.
        if "GrpcAsyncIOTransport" in str(transport.__name__):
            channel_fn = grpc.aio.insecure_channel
        else:
            channel_fn = grpc.insecure_channel
        return channel_fn(self._emulator_host, options=options)

    def _create_gapic_client_channel(self, client_class, grpc_transport):
        if self._emulator_host is not None:
            api_endpoint = self._emulator_host
        elif self._client_options and self._client_options.api_endpoint:
            api_endpoint = self._client_options.api_endpoint
        else:
            api_endpoint = client_class.DEFAULT_ENDPOINT

        if self._emulator_host is not None:
            channel = self._emulator_channel(
                transport=grpc_transport,
                options=_GRPC_CHANNEL_OPTIONS,
            )
        else:
            channel = grpc_transport.create_channel(
                host=api_endpoint,
                credentials=self._credentials,
                options=_GRPC_CHANNEL_OPTIONS,
            )
        return grpc_transport(channel=channel, host=api_endpoint)

    @property
    def project_path(self):
        """Project name to be used with Instance Admin API.

        .. note::

            This property will not change if ``project`` does not, but the
            return value is not cached.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_project_path]
            :end-before: [END bigtable_api_project_path]
            :dedent: 4

        The project name is of the form

            ``"projects/{project}"``

        :rtype: str
        :returns: Return a fully-qualified project string.
        """
        return self.instance_admin_client.common_project_path(self.project)

    @property
    def table_data_client(self):
        """Getter for the gRPC stub used for the Table Admin API.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_table_data_client]
            :end-before: [END bigtable_api_table_data_client]
            :dedent: 4

        :rtype: :class:`.bigtable_v2.BigtableClient`
        :returns: A BigtableClient object.
        """
        if self._table_data_client is None:
            transport = self._create_gapic_client_channel(
                bigtable_v2.BigtableClient,
                BigtableGrpcTransport,
            )
            klass = _create_gapic_client(
                bigtable_v2.BigtableClient,
                client_options=self._client_options,
                transport=transport,
            )
            self._table_data_client = klass(self)
        return self._table_data_client

    @property
    def table_admin_client(self):
        """Getter for the gRPC stub used for the Table Admin API.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_table_admin_client]
            :end-before: [END bigtable_api_table_admin_client]
            :dedent: 4

        :rtype: :class:`.bigtable_admin_pb2.BigtableTableAdmin`
        :returns: A BigtableTableAdmin instance.
        :raises: :class:`ValueError <exceptions.ValueError>` if the current
                 client is not an admin client or if it has not been
                 :meth:`start`-ed.
        """
        if self._table_admin_client is None:
            if not self._admin:
                raise ValueError("Client is not an admin client.")

            transport = self._create_gapic_client_channel(
                bigtable_admin_v2.BaseBigtableTableAdminClient,
                BigtableTableAdminGrpcTransport,
            )
            klass = _create_gapic_client(
                bigtable_admin_v2.BaseBigtableTableAdminClient,
                client_options=self._admin_client_options,
                transport=transport,
            )
            self._table_admin_client = klass(self)
        return self._table_admin_client

    @property
    def instance_admin_client(self):
        """Getter for the gRPC stub used for the Table Admin API.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_instance_admin_client]
            :end-before: [END bigtable_api_instance_admin_client]
            :dedent: 4

        :rtype: :class:`.bigtable_admin_pb2.BigtableInstanceAdmin`
        :returns: A BigtableInstanceAdmin instance.
        :raises: :class:`ValueError <exceptions.ValueError>` if the current
                 client is not an admin client or if it has not been
                 :meth:`start`-ed.
        """
        if self._instance_admin_client is None:
            if not self._admin:
                raise ValueError("Client is not an admin client.")

            transport = self._create_gapic_client_channel(
                bigtable_admin_v2.BigtableInstanceAdminClient,
                BigtableInstanceAdminGrpcTransport,
            )
            klass = _create_gapic_client(
                bigtable_admin_v2.BigtableInstanceAdminClient,
                client_options=self._admin_client_options,
                transport=transport,
            )
            self._instance_admin_client = klass(self)
        return self._instance_admin_client

    def instance(self, instance_id, display_name=None, instance_type=None, labels=None):
        """Factory to create a instance associated with this client.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_create_prod_instance]
            :end-before: [END bigtable_api_create_prod_instance]
            :dedent: 4

        :type instance_id: str
        :param instance_id: The ID of the instance.

        :type display_name: str
        :param display_name: (Optional) The display name for the instance in
                             the Cloud Console UI. (Must be between 4 and 30
                             characters.) If this value is not set in the
                             constructor, will fall back to the instance ID.

        :type instance_type: int
        :param instance_type: (Optional) The type of the instance.
                               Possible values are represented
                               by the following constants:
                               :data:`google.cloud.bigtable.instance.InstanceType.PRODUCTION`.
                               :data:`google.cloud.bigtable.instance.InstanceType.DEVELOPMENT`,
                               Defaults to
                               :data:`google.cloud.bigtable.instance.InstanceType.UNSPECIFIED`.

        :type labels: dict
        :param labels: (Optional) Labels are a flexible and lightweight
                       mechanism for organizing cloud resources into groups
                       that reflect a customer's organizational needs and
                       deployment strategies. They can be used to filter
                       resources and aggregate metrics. Label keys must be
                       between 1 and 63 characters long. Maximum 64 labels can
                       be associated with a given resource. Label values must
                       be between 0 and 63 characters long. Keys and values
                       must both be under 128 bytes.

        :rtype: :class:`~google.cloud.bigtable.instance.Instance`
        :returns: an instance owned by this client.
        """
        return Instance(
            instance_id,
            self,
            display_name=display_name,
            instance_type=instance_type,
            labels=labels,
        )

    def list_instances(self):
        """List instances owned by the project.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_list_instances]
            :end-before: [END bigtable_api_list_instances]
            :dedent: 4

        :rtype: tuple
        :returns:
            (instances, failed_locations), where 'instances' is list of
            :class:`google.cloud.bigtable.instance.Instance`, and
            'failed_locations' is a list of locations which could not
            be resolved.
        """
        resp = self.instance_admin_client.list_instances(
            request={"parent": self.project_path}
        )
        instances = [Instance.from_pb(instance, self) for instance in resp.instances]
        return instances, resp.failed_locations

    def list_clusters(self):
        """List the clusters in the project.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_list_clusters_in_project]
            :end-before: [END bigtable_api_list_clusters_in_project]
            :dedent: 4

        :rtype: tuple
        :returns:
            (clusters, failed_locations), where 'clusters' is list of
            :class:`google.cloud.bigtable.instance.Cluster`, and
            'failed_locations' is a list of strings representing
            locations which could not be resolved.
        """
        resp = self.instance_admin_client.list_clusters(
            request={
                "parent": self.instance_admin_client.instance_path(self.project, "-")
            }
        )
        clusters = []
        instances = {}
        for cluster in resp.clusters:
            match_cluster_name = _CLUSTER_NAME_RE.match(cluster.name)
            instance_id = match_cluster_name.group("instance")
            if instance_id not in instances:
                instances[instance_id] = self.instance(instance_id)
            clusters.append(Cluster.from_pb(cluster, instances[instance_id]))
        return clusters, resp.failed_locations

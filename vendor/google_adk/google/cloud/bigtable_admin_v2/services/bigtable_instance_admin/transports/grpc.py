# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
#
import json
import logging as std_logging
import pickle
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from google.api_core import grpc_helpers
from google.api_core import operations_v1
from google.api_core import gapic_v1
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message

import grpc  # type: ignore
import proto  # type: ignore

from google.cloud.bigtable_admin_v2.types import bigtable_instance_admin
from google.cloud.bigtable_admin_v2.types import instance
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from .base import BigtableInstanceAdminTransport, DEFAULT_CLIENT_INFO

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class _LoggingClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # pragma: NO COVER
    def intercept_unary_unary(self, continuation, client_call_details, request):
        logging_enabled = CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        )
        if logging_enabled:  # pragma: NO COVER
            request_metadata = client_call_details.metadata
            if isinstance(request, proto.Message):
                request_payload = type(request).to_json(request)
            elif isinstance(request, google.protobuf.message.Message):
                request_payload = MessageToJson(request)
            else:
                request_payload = f"{type(request).__name__}: {pickle.dumps(request)}"

            request_metadata = {
                key: value.decode("utf-8") if isinstance(value, bytes) else value
                for key, value in request_metadata
            }
            grpc_request = {
                "payload": request_payload,
                "requestMethod": "grpc",
                "metadata": dict(request_metadata),
            }
            _LOGGER.debug(
                f"Sending request for {client_call_details.method}",
                extra={
                    "serviceName": "google.bigtable.admin.v2.BigtableInstanceAdmin",
                    "rpcName": str(client_call_details.method),
                    "request": grpc_request,
                    "metadata": grpc_request["metadata"],
                },
            )
        response = continuation(client_call_details, request)
        if logging_enabled:  # pragma: NO COVER
            response_metadata = response.trailing_metadata()
            # Convert gRPC metadata `<class 'grpc.aio._metadata.Metadata'>` to list of tuples
            metadata = (
                dict([(k, str(v)) for k, v in response_metadata])
                if response_metadata
                else None
            )
            result = response.result()
            if isinstance(result, proto.Message):
                response_payload = type(result).to_json(result)
            elif isinstance(result, google.protobuf.message.Message):
                response_payload = MessageToJson(result)
            else:
                response_payload = f"{type(result).__name__}: {pickle.dumps(result)}"
            grpc_response = {
                "payload": response_payload,
                "metadata": metadata,
                "status": "OK",
            }
            _LOGGER.debug(
                f"Received response for {client_call_details.method}.",
                extra={
                    "serviceName": "google.bigtable.admin.v2.BigtableInstanceAdmin",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class BigtableInstanceAdminGrpcTransport(BigtableInstanceAdminTransport):
    """gRPC backend transport for BigtableInstanceAdmin.

    Service for creating, configuring, and deleting Cloud
    Bigtable Instances and Clusters. Provides access to the Instance
    and Cluster schemas only, not the tables' metadata or data
    stored in those tables.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _stubs: Dict[str, Callable]

    def __init__(
        self,
        *,
        host: str = "bigtableadmin.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[Union[grpc.Channel, Callable[..., grpc.Channel]]] = None,
        api_mtls_endpoint: Optional[str] = None,
        client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        ssl_channel_credentials: Optional[grpc.ChannelCredentials] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'bigtableadmin.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
                This argument is ignored if a ``channel`` instance is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if a ``channel`` instance is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if a ``channel`` instance is provided.
            channel (Optional[Union[grpc.Channel, Callable[..., grpc.Channel]]]):
                A ``Channel`` instance through which to make calls, or a Callable
                that constructs and returns one. If set to None, ``self.create_channel``
                is used to create the channel. If a Callable is given, it will be called
                with the same arguments as used in ``self.create_channel``.
            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.
                If provided, it overrides the ``host`` argument and tries to create
                a mutual TLS channel with client SSL credentials from
                ``client_cert_source`` or application default SSL credentials.
            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):
                Deprecated. A callback to provide client SSL certificate bytes and
                private key bytes, both in PEM format. It is ignored if
                ``api_mtls_endpoint`` is None.
            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials
                for the grpc channel. It is ignored if a ``channel`` instance is provided.
            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):
                A callback to provide client certificate bytes and private key bytes,
                both in PEM format. It is used to configure a mutual TLS channel. It is
                ignored if a ``channel`` instance or ``ssl_channel_credentials`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.

        Raises:
          google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}
        self._operations_client: Optional[operations_v1.OperationsClient] = None

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if isinstance(channel, grpc.Channel):
            # Ignore credentials if a channel was passed.
            credentials = None
            self._ignore_credentials = True
            # If a channel was explicitly provided, set it.
            self._grpc_channel = channel
            self._ssl_channel_credentials = None

        else:
            if api_mtls_endpoint:
                host = api_mtls_endpoint

                # Create SSL credentials with client_cert_source or application
                # default SSL credentials.
                if client_cert_source:
                    cert, key = client_cert_source()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )
                else:
                    self._ssl_channel_credentials = SslCredentials().ssl_credentials

            else:
                if client_cert_source_for_mtls and not ssl_channel_credentials:
                    cert, key = client_cert_source_for_mtls()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )

        # The base transport sets the host, credentials and scopes
        super().__init__(
            host=host,
            credentials=credentials,
            credentials_file=credentials_file,
            scopes=scopes,
            quota_project_id=quota_project_id,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience,
        )

        if not self._grpc_channel:
            # initialize with the provided callable or the default channel
            channel_init = channel or type(self).create_channel
            self._grpc_channel = channel_init(
                self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                # Set ``credentials_file`` to ``None`` here as
                # the credentials that we saved earlier should be used.
                credentials_file=None,
                scopes=self._scopes,
                ssl_credentials=self._ssl_channel_credentials,
                quota_project_id=quota_project_id,
                options=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                ],
            )

        self._interceptor = _LoggingClientInterceptor()
        self._logged_channel = grpc.intercept_channel(
            self._grpc_channel, self._interceptor
        )

        # Wrap messages. This must be done after self._logged_channel exists
        self._prep_wrapped_messages(client_info)

    @classmethod
    def create_channel(
        cls,
        host: str = "bigtableadmin.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> grpc.Channel:
        """Create and return a gRPC channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            grpc.Channel: A gRPC channel object.

        Raises:
            google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """

        return grpc_helpers.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    @property
    def grpc_channel(self) -> grpc.Channel:
        """Return the channel designed to connect to this service."""
        return self._grpc_channel

    @property
    def operations_client(self) -> operations_v1.OperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Quick check: Only create a new client if we do not already have one.
        if self._operations_client is None:
            self._operations_client = operations_v1.OperationsClient(
                self._logged_channel
            )

        # Return the client from cache.
        return self._operations_client

    @property
    def create_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateInstanceRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create instance method over gRPC.

        Create an instance within a project.

        Note that exactly one of Cluster.serve_nodes and
        Cluster.cluster_config.cluster_autoscaling_config can be set. If
        serve_nodes is set to non-zero, then the cluster is manually
        scaled. If cluster_config.cluster_autoscaling_config is
        non-empty, then autoscaling is enabled.

        Returns:
            Callable[[~.CreateInstanceRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_instance" not in self._stubs:
            self._stubs["create_instance"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateInstance",
                request_serializer=bigtable_instance_admin.CreateInstanceRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_instance"]

    @property
    def get_instance(
        self,
    ) -> Callable[[bigtable_instance_admin.GetInstanceRequest], instance.Instance]:
        r"""Return a callable for the get instance method over gRPC.

        Gets information about an instance.

        Returns:
            Callable[[~.GetInstanceRequest],
                    ~.Instance]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_instance" not in self._stubs:
            self._stubs["get_instance"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetInstance",
                request_serializer=bigtable_instance_admin.GetInstanceRequest.serialize,
                response_deserializer=instance.Instance.deserialize,
            )
        return self._stubs["get_instance"]

    @property
    def list_instances(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListInstancesRequest],
        bigtable_instance_admin.ListInstancesResponse,
    ]:
        r"""Return a callable for the list instances method over gRPC.

        Lists information about instances in a project.

        Returns:
            Callable[[~.ListInstancesRequest],
                    ~.ListInstancesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instances" not in self._stubs:
            self._stubs["list_instances"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListInstances",
                request_serializer=bigtable_instance_admin.ListInstancesRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListInstancesResponse.deserialize,
            )
        return self._stubs["list_instances"]

    @property
    def update_instance(self) -> Callable[[instance.Instance], instance.Instance]:
        r"""Return a callable for the update instance method over gRPC.

        Updates an instance within a project. This method
        updates only the display name and type for an Instance.
        To update other Instance properties, such as labels, use
        PartialUpdateInstance.

        Returns:
            Callable[[~.Instance],
                    ~.Instance]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_instance" not in self._stubs:
            self._stubs["update_instance"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateInstance",
                request_serializer=instance.Instance.serialize,
                response_deserializer=instance.Instance.deserialize,
            )
        return self._stubs["update_instance"]

    @property
    def partial_update_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.PartialUpdateInstanceRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the partial update instance method over gRPC.

        Partially updates an instance within a project. This
        method can modify all fields of an Instance and is the
        preferred way to update an Instance.

        Returns:
            Callable[[~.PartialUpdateInstanceRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "partial_update_instance" not in self._stubs:
            self._stubs["partial_update_instance"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/PartialUpdateInstance",
                request_serializer=bigtable_instance_admin.PartialUpdateInstanceRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["partial_update_instance"]

    @property
    def delete_instance(
        self,
    ) -> Callable[[bigtable_instance_admin.DeleteInstanceRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete instance method over gRPC.

        Delete an instance from a project.

        Returns:
            Callable[[~.DeleteInstanceRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_instance" not in self._stubs:
            self._stubs["delete_instance"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteInstance",
                request_serializer=bigtable_instance_admin.DeleteInstanceRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_instance"]

    @property
    def create_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateClusterRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create cluster method over gRPC.

        Creates a cluster within an instance.

        Note that exactly one of Cluster.serve_nodes and
        Cluster.cluster_config.cluster_autoscaling_config can be set. If
        serve_nodes is set to non-zero, then the cluster is manually
        scaled. If cluster_config.cluster_autoscaling_config is
        non-empty, then autoscaling is enabled.

        Returns:
            Callable[[~.CreateClusterRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_cluster" not in self._stubs:
            self._stubs["create_cluster"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateCluster",
                request_serializer=bigtable_instance_admin.CreateClusterRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_cluster"]

    @property
    def get_cluster(
        self,
    ) -> Callable[[bigtable_instance_admin.GetClusterRequest], instance.Cluster]:
        r"""Return a callable for the get cluster method over gRPC.

        Gets information about a cluster.

        Returns:
            Callable[[~.GetClusterRequest],
                    ~.Cluster]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_cluster" not in self._stubs:
            self._stubs["get_cluster"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetCluster",
                request_serializer=bigtable_instance_admin.GetClusterRequest.serialize,
                response_deserializer=instance.Cluster.deserialize,
            )
        return self._stubs["get_cluster"]

    @property
    def list_clusters(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListClustersRequest],
        bigtable_instance_admin.ListClustersResponse,
    ]:
        r"""Return a callable for the list clusters method over gRPC.

        Lists information about clusters in an instance.

        Returns:
            Callable[[~.ListClustersRequest],
                    ~.ListClustersResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_clusters" not in self._stubs:
            self._stubs["list_clusters"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListClusters",
                request_serializer=bigtable_instance_admin.ListClustersRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListClustersResponse.deserialize,
            )
        return self._stubs["list_clusters"]

    @property
    def update_cluster(self) -> Callable[[instance.Cluster], operations_pb2.Operation]:
        r"""Return a callable for the update cluster method over gRPC.

        Updates a cluster within an instance.

        Note that UpdateCluster does not support updating
        cluster_config.cluster_autoscaling_config. In order to update
        it, you must use PartialUpdateCluster.

        Returns:
            Callable[[~.Cluster],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_cluster" not in self._stubs:
            self._stubs["update_cluster"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateCluster",
                request_serializer=instance.Cluster.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_cluster"]

    @property
    def partial_update_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.PartialUpdateClusterRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the partial update cluster method over gRPC.

        Partially updates a cluster within a project. This method is the
        preferred way to update a Cluster.

        To enable and update autoscaling, set
        cluster_config.cluster_autoscaling_config. When autoscaling is
        enabled, serve_nodes is treated as an OUTPUT_ONLY field, meaning
        that updates to it are ignored. Note that an update cannot
        simultaneously set serve_nodes to non-zero and
        cluster_config.cluster_autoscaling_config to non-empty, and also
        specify both in the update_mask.

        To disable autoscaling, clear
        cluster_config.cluster_autoscaling_config, and explicitly set a
        serve_node count via the update_mask.

        Returns:
            Callable[[~.PartialUpdateClusterRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "partial_update_cluster" not in self._stubs:
            self._stubs["partial_update_cluster"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/PartialUpdateCluster",
                request_serializer=bigtable_instance_admin.PartialUpdateClusterRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["partial_update_cluster"]

    @property
    def delete_cluster(
        self,
    ) -> Callable[[bigtable_instance_admin.DeleteClusterRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete cluster method over gRPC.

        Deletes a cluster from an instance.

        Returns:
            Callable[[~.DeleteClusterRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_cluster" not in self._stubs:
            self._stubs["delete_cluster"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteCluster",
                request_serializer=bigtable_instance_admin.DeleteClusterRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_cluster"]

    @property
    def create_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateAppProfileRequest], instance.AppProfile
    ]:
        r"""Return a callable for the create app profile method over gRPC.

        Creates an app profile within an instance.

        Returns:
            Callable[[~.CreateAppProfileRequest],
                    ~.AppProfile]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_app_profile" not in self._stubs:
            self._stubs["create_app_profile"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateAppProfile",
                request_serializer=bigtable_instance_admin.CreateAppProfileRequest.serialize,
                response_deserializer=instance.AppProfile.deserialize,
            )
        return self._stubs["create_app_profile"]

    @property
    def get_app_profile(
        self,
    ) -> Callable[[bigtable_instance_admin.GetAppProfileRequest], instance.AppProfile]:
        r"""Return a callable for the get app profile method over gRPC.

        Gets information about an app profile.

        Returns:
            Callable[[~.GetAppProfileRequest],
                    ~.AppProfile]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_app_profile" not in self._stubs:
            self._stubs["get_app_profile"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetAppProfile",
                request_serializer=bigtable_instance_admin.GetAppProfileRequest.serialize,
                response_deserializer=instance.AppProfile.deserialize,
            )
        return self._stubs["get_app_profile"]

    @property
    def list_app_profiles(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListAppProfilesRequest],
        bigtable_instance_admin.ListAppProfilesResponse,
    ]:
        r"""Return a callable for the list app profiles method over gRPC.

        Lists information about app profiles in an instance.

        Returns:
            Callable[[~.ListAppProfilesRequest],
                    ~.ListAppProfilesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_app_profiles" not in self._stubs:
            self._stubs["list_app_profiles"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListAppProfiles",
                request_serializer=bigtable_instance_admin.ListAppProfilesRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListAppProfilesResponse.deserialize,
            )
        return self._stubs["list_app_profiles"]

    @property
    def update_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateAppProfileRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update app profile method over gRPC.

        Updates an app profile within an instance.

        Returns:
            Callable[[~.UpdateAppProfileRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_app_profile" not in self._stubs:
            self._stubs["update_app_profile"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateAppProfile",
                request_serializer=bigtable_instance_admin.UpdateAppProfileRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_app_profile"]

    @property
    def delete_app_profile(
        self,
    ) -> Callable[[bigtable_instance_admin.DeleteAppProfileRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete app profile method over gRPC.

        Deletes an app profile from an instance.

        Returns:
            Callable[[~.DeleteAppProfileRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_app_profile" not in self._stubs:
            self._stubs["delete_app_profile"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteAppProfile",
                request_serializer=bigtable_instance_admin.DeleteAppProfileRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_app_profile"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the get iam policy method over gRPC.

        Gets the access control policy for an instance
        resource. Returns an empty policy if an instance exists
        but does not have a policy set.

        Returns:
            Callable[[~.GetIamPolicyRequest],
                    ~.Policy]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_iam_policy" not in self._stubs:
            self._stubs["get_iam_policy"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the set iam policy method over gRPC.

        Sets the access control policy on an instance
        resource. Replaces any existing policy.

        Returns:
            Callable[[~.SetIamPolicyRequest],
                    ~.Policy]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "set_iam_policy" not in self._stubs:
            self._stubs["set_iam_policy"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/SetIamPolicy",
                request_serializer=iam_policy_pb2.SetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["set_iam_policy"]

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        r"""Return a callable for the test iam permissions method over gRPC.

        Returns permissions that the caller has on the
        specified instance resource.

        Returns:
            Callable[[~.TestIamPermissionsRequest],
                    ~.TestIamPermissionsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "test_iam_permissions" not in self._stubs:
            self._stubs["test_iam_permissions"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def list_hot_tablets(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListHotTabletsRequest],
        bigtable_instance_admin.ListHotTabletsResponse,
    ]:
        r"""Return a callable for the list hot tablets method over gRPC.

        Lists hot tablets in a cluster, within the time range
        provided. Hot tablets are ordered based on CPU usage.

        Returns:
            Callable[[~.ListHotTabletsRequest],
                    ~.ListHotTabletsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_hot_tablets" not in self._stubs:
            self._stubs["list_hot_tablets"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListHotTablets",
                request_serializer=bigtable_instance_admin.ListHotTabletsRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListHotTabletsResponse.deserialize,
            )
        return self._stubs["list_hot_tablets"]

    @property
    def create_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateLogicalViewRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create logical view method over gRPC.

        Creates a logical view within an instance.

        Returns:
            Callable[[~.CreateLogicalViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_logical_view" not in self._stubs:
            self._stubs["create_logical_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateLogicalView",
                request_serializer=bigtable_instance_admin.CreateLogicalViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_logical_view"]

    @property
    def get_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetLogicalViewRequest], instance.LogicalView
    ]:
        r"""Return a callable for the get logical view method over gRPC.

        Gets information about a logical view.

        Returns:
            Callable[[~.GetLogicalViewRequest],
                    ~.LogicalView]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_logical_view" not in self._stubs:
            self._stubs["get_logical_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetLogicalView",
                request_serializer=bigtable_instance_admin.GetLogicalViewRequest.serialize,
                response_deserializer=instance.LogicalView.deserialize,
            )
        return self._stubs["get_logical_view"]

    @property
    def list_logical_views(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListLogicalViewsRequest],
        bigtable_instance_admin.ListLogicalViewsResponse,
    ]:
        r"""Return a callable for the list logical views method over gRPC.

        Lists information about logical views in an instance.

        Returns:
            Callable[[~.ListLogicalViewsRequest],
                    ~.ListLogicalViewsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_logical_views" not in self._stubs:
            self._stubs["list_logical_views"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListLogicalViews",
                request_serializer=bigtable_instance_admin.ListLogicalViewsRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListLogicalViewsResponse.deserialize,
            )
        return self._stubs["list_logical_views"]

    @property
    def update_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateLogicalViewRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update logical view method over gRPC.

        Updates a logical view within an instance.

        Returns:
            Callable[[~.UpdateLogicalViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_logical_view" not in self._stubs:
            self._stubs["update_logical_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateLogicalView",
                request_serializer=bigtable_instance_admin.UpdateLogicalViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_logical_view"]

    @property
    def delete_logical_view(
        self,
    ) -> Callable[[bigtable_instance_admin.DeleteLogicalViewRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete logical view method over gRPC.

        Deletes a logical view from an instance.

        Returns:
            Callable[[~.DeleteLogicalViewRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_logical_view" not in self._stubs:
            self._stubs["delete_logical_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteLogicalView",
                request_serializer=bigtable_instance_admin.DeleteLogicalViewRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_logical_view"]

    @property
    def create_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateMaterializedViewRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the create materialized view method over gRPC.

        Creates a materialized view within an instance.

        Returns:
            Callable[[~.CreateMaterializedViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_materialized_view" not in self._stubs:
            self._stubs["create_materialized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateMaterializedView",
                request_serializer=bigtable_instance_admin.CreateMaterializedViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_materialized_view"]

    @property
    def get_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetMaterializedViewRequest], instance.MaterializedView
    ]:
        r"""Return a callable for the get materialized view method over gRPC.

        Gets information about a materialized view.

        Returns:
            Callable[[~.GetMaterializedViewRequest],
                    ~.MaterializedView]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_materialized_view" not in self._stubs:
            self._stubs["get_materialized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/GetMaterializedView",
                request_serializer=bigtable_instance_admin.GetMaterializedViewRequest.serialize,
                response_deserializer=instance.MaterializedView.deserialize,
            )
        return self._stubs["get_materialized_view"]

    @property
    def list_materialized_views(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListMaterializedViewsRequest],
        bigtable_instance_admin.ListMaterializedViewsResponse,
    ]:
        r"""Return a callable for the list materialized views method over gRPC.

        Lists information about materialized views in an
        instance.

        Returns:
            Callable[[~.ListMaterializedViewsRequest],
                    ~.ListMaterializedViewsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_materialized_views" not in self._stubs:
            self._stubs["list_materialized_views"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/ListMaterializedViews",
                request_serializer=bigtable_instance_admin.ListMaterializedViewsRequest.serialize,
                response_deserializer=bigtable_instance_admin.ListMaterializedViewsResponse.deserialize,
            )
        return self._stubs["list_materialized_views"]

    @property
    def update_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateMaterializedViewRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the update materialized view method over gRPC.

        Updates a materialized view within an instance.

        Returns:
            Callable[[~.UpdateMaterializedViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_materialized_view" not in self._stubs:
            self._stubs["update_materialized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateMaterializedView",
                request_serializer=bigtable_instance_admin.UpdateMaterializedViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_materialized_view"]

    @property
    def delete_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteMaterializedViewRequest], empty_pb2.Empty
    ]:
        r"""Return a callable for the delete materialized view method over gRPC.

        Deletes a materialized view from an instance.

        Returns:
            Callable[[~.DeleteMaterializedViewRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_materialized_view" not in self._stubs:
            self._stubs["delete_materialized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteMaterializedView",
                request_serializer=bigtable_instance_admin.DeleteMaterializedViewRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_materialized_view"]

    def close(self):
        self._logged_channel.close()

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("BigtableInstanceAdminGrpcTransport",)

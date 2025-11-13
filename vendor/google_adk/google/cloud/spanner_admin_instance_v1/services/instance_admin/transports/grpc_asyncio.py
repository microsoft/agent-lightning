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
import inspect
import json
import pickle
import logging as std_logging
import warnings
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union

from google.api_core import gapic_v1
from google.api_core import grpc_helpers_async
from google.api_core import exceptions as core_exceptions
from google.api_core import retry_async as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message

import grpc  # type: ignore
import proto  # type: ignore
from grpc.experimental import aio  # type: ignore

from google.cloud.spanner_admin_instance_v1.types import spanner_instance_admin
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from .base import InstanceAdminTransport, DEFAULT_CLIENT_INFO
from .grpc import InstanceAdminGrpcTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class _LoggingClientAIOInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor
):  # pragma: NO COVER
    async def intercept_unary_unary(self, continuation, client_call_details, request):
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
                    "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                    "rpcName": str(client_call_details.method),
                    "request": grpc_request,
                    "metadata": grpc_request["metadata"],
                },
            )
        response = await continuation(client_call_details, request)
        if logging_enabled:  # pragma: NO COVER
            response_metadata = await response.trailing_metadata()
            # Convert gRPC metadata `<class 'grpc.aio._metadata.Metadata'>` to list of tuples
            metadata = (
                dict([(k, str(v)) for k, v in response_metadata])
                if response_metadata
                else None
            )
            result = await response
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
                f"Received response to rpc {client_call_details.method}.",
                extra={
                    "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                    "rpcName": str(client_call_details.method),
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class InstanceAdminGrpcAsyncIOTransport(InstanceAdminTransport):
    """gRPC AsyncIO backend transport for InstanceAdmin.

    Cloud Spanner Instance Admin API

    The Cloud Spanner Instance Admin API can be used to create,
    delete, modify and list instances. Instances are dedicated Cloud
    Spanner serving and storage resources to be used by Cloud
    Spanner databases.

    Each instance has a "configuration", which dictates where the
    serving resources for the Cloud Spanner instance are located
    (e.g., US-central, Europe). Configurations are created by Google
    based on resource availability.

    Cloud Spanner billing is based on the instances that exist and
    their sizes. After an instance exists, there are no additional
    per-database or per-operation charges for use of the instance
    (though there may be additional network bandwidth charges).
    Instances offer isolation: problems with databases in one
    instance will not affect other instances. However, within an
    instance databases can affect each other. For example, if one
    database in an instance receives a lot of requests and consumes
    most of the instance resources, fewer resources are available
    for other databases in that instance, and their performance may
    suffer.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _grpc_channel: aio.Channel
    _stubs: Dict[str, Callable] = {}

    @classmethod
    def create_channel(
        cls,
        host: str = "spanner.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> aio.Channel:
        """Create and return a gRPC AsyncIO channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            aio.Channel: A gRPC AsyncIO channel object.
        """

        return grpc_helpers_async.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    def __init__(
        self,
        *,
        host: str = "spanner.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[Union[aio.Channel, Callable[..., aio.Channel]]] = None,
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
                 The hostname to connect to (default: 'spanner.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
                This argument is ignored if a ``channel`` instance is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if a ``channel`` instance is provided.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            channel (Optional[Union[aio.Channel, Callable[..., aio.Channel]]]):
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
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}
        self._operations_client: Optional[operations_v1.OperationsAsyncClient] = None

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if isinstance(channel, aio.Channel):
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

        self._interceptor = _LoggingClientAIOInterceptor()
        self._grpc_channel._unary_unary_interceptors.append(self._interceptor)
        self._logged_channel = self._grpc_channel
        self._wrap_with_kind = (
            "kind" in inspect.signature(gapic_v1.method_async.wrap_method).parameters
        )
        # Wrap messages. This must be done after self._logged_channel exists
        self._prep_wrapped_messages(client_info)

    @property
    def grpc_channel(self) -> aio.Channel:
        """Create the channel designed to connect to this service.

        This property caches on the instance; repeated calls return
        the same channel.
        """
        # Return the channel from cache.
        return self._grpc_channel

    @property
    def operations_client(self) -> operations_v1.OperationsAsyncClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Quick check: Only create a new client if we do not already have one.
        if self._operations_client is None:
            self._operations_client = operations_v1.OperationsAsyncClient(
                self._logged_channel
            )

        # Return the client from cache.
        return self._operations_client

    @property
    def list_instance_configs(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstanceConfigsRequest],
        Awaitable[spanner_instance_admin.ListInstanceConfigsResponse],
    ]:
        r"""Return a callable for the list instance configs method over gRPC.

        Lists the supported instance configurations for a
        given project.
        Returns both Google-managed configurations and
        user-managed configurations.

        Returns:
            Callable[[~.ListInstanceConfigsRequest],
                    Awaitable[~.ListInstanceConfigsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instance_configs" not in self._stubs:
            self._stubs["list_instance_configs"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/ListInstanceConfigs",
                request_serializer=spanner_instance_admin.ListInstanceConfigsRequest.serialize,
                response_deserializer=spanner_instance_admin.ListInstanceConfigsResponse.deserialize,
            )
        return self._stubs["list_instance_configs"]

    @property
    def get_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstanceConfigRequest],
        Awaitable[spanner_instance_admin.InstanceConfig],
    ]:
        r"""Return a callable for the get instance config method over gRPC.

        Gets information about a particular instance
        configuration.

        Returns:
            Callable[[~.GetInstanceConfigRequest],
                    Awaitable[~.InstanceConfig]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_instance_config" not in self._stubs:
            self._stubs["get_instance_config"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/GetInstanceConfig",
                request_serializer=spanner_instance_admin.GetInstanceConfigRequest.serialize,
                response_deserializer=spanner_instance_admin.InstanceConfig.deserialize,
            )
        return self._stubs["get_instance_config"]

    @property
    def create_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstanceConfigRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the create instance config method over gRPC.

        Creates an instance configuration and begins preparing it to be
        used. The returned long-running operation can be used to track
        the progress of preparing the new instance configuration. The
        instance configuration name is assigned by the caller. If the
        named instance configuration already exists,
        ``CreateInstanceConfig`` returns ``ALREADY_EXISTS``.

        Immediately after the request returns:

        - The instance configuration is readable via the API, with all
          requested attributes. The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field is set to true. Its state is ``CREATING``.

        While the operation is pending:

        - Cancelling the operation renders the instance configuration
          immediately unreadable via the API.
        - Except for deleting the creating resource, all other attempts
          to modify the instance configuration are rejected.

        Upon completion of the returned operation:

        - Instances can be created using the instance configuration.
        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field becomes false. Its state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_config_name>/operations/<operation_id>`` and
        can be used to track creation of the instance configuration. The
        metadata field type is
        [CreateInstanceConfigMetadata][google.spanner.admin.instance.v1.CreateInstanceConfigMetadata].
        The response field type is
        [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig],
        if successful.

        Authorization requires ``spanner.instanceConfigs.create``
        permission on the resource
        [parent][google.spanner.admin.instance.v1.CreateInstanceConfigRequest.parent].

        Returns:
            Callable[[~.CreateInstanceConfigRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_instance_config" not in self._stubs:
            self._stubs["create_instance_config"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/CreateInstanceConfig",
                request_serializer=spanner_instance_admin.CreateInstanceConfigRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_instance_config"]

    @property
    def update_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstanceConfigRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the update instance config method over gRPC.

        Updates an instance configuration. The returned long-running
        operation can be used to track the progress of updating the
        instance. If the named instance configuration does not exist,
        returns ``NOT_FOUND``.

        Only user-managed configurations can be updated.

        Immediately after the request returns:

        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field is set to true.

        While the operation is pending:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstanceConfigMetadata.cancel_time].
          The operation is guaranteed to succeed at undoing all changes,
          after which point it terminates with a ``CANCELLED`` status.
        - All other attempts to modify the instance configuration are
          rejected.
        - Reading the instance configuration via the API continues to
          give the pre-request values.

        Upon completion of the returned operation:

        - Creating instances using the instance configuration uses the
          new values.
        - The new values of the instance configuration are readable via
          the API.
        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field becomes false.

        The returned long-running operation will have a name of the
        format ``<instance_config_name>/operations/<operation_id>`` and
        can be used to track the instance configuration modification.
        The metadata field type is
        [UpdateInstanceConfigMetadata][google.spanner.admin.instance.v1.UpdateInstanceConfigMetadata].
        The response field type is
        [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig],
        if successful.

        Authorization requires ``spanner.instanceConfigs.update``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstanceConfig.name].

        Returns:
            Callable[[~.UpdateInstanceConfigRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_instance_config" not in self._stubs:
            self._stubs["update_instance_config"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/UpdateInstanceConfig",
                request_serializer=spanner_instance_admin.UpdateInstanceConfigRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_instance_config"]

    @property
    def delete_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.DeleteInstanceConfigRequest], Awaitable[empty_pb2.Empty]
    ]:
        r"""Return a callable for the delete instance config method over gRPC.

        Deletes the instance configuration. Deletion is only allowed
        when no instances are using the configuration. If any instances
        are using the configuration, returns ``FAILED_PRECONDITION``.

        Only user-managed configurations can be deleted.

        Authorization requires ``spanner.instanceConfigs.delete``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstanceConfig.name].

        Returns:
            Callable[[~.DeleteInstanceConfigRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_instance_config" not in self._stubs:
            self._stubs["delete_instance_config"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/DeleteInstanceConfig",
                request_serializer=spanner_instance_admin.DeleteInstanceConfigRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_instance_config"]

    @property
    def list_instance_config_operations(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstanceConfigOperationsRequest],
        Awaitable[spanner_instance_admin.ListInstanceConfigOperationsResponse],
    ]:
        r"""Return a callable for the list instance config
        operations method over gRPC.

        Lists the user-managed instance configuration long-running
        operations in the given project. An instance configuration
        operation has a name of the form
        ``projects/<project>/instanceConfigs/<instance_config>/operations/<operation>``.
        The long-running operation metadata field type
        ``metadata.type_url`` describes the type of the metadata.
        Operations returned include those that have
        completed/failed/canceled within the last 7 days, and pending
        operations. Operations returned are ordered by
        ``operation.metadata.value.start_time`` in descending order
        starting from the most recently started operation.

        Returns:
            Callable[[~.ListInstanceConfigOperationsRequest],
                    Awaitable[~.ListInstanceConfigOperationsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instance_config_operations" not in self._stubs:
            self._stubs[
                "list_instance_config_operations"
            ] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/ListInstanceConfigOperations",
                request_serializer=spanner_instance_admin.ListInstanceConfigOperationsRequest.serialize,
                response_deserializer=spanner_instance_admin.ListInstanceConfigOperationsResponse.deserialize,
            )
        return self._stubs["list_instance_config_operations"]

    @property
    def list_instances(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancesRequest],
        Awaitable[spanner_instance_admin.ListInstancesResponse],
    ]:
        r"""Return a callable for the list instances method over gRPC.

        Lists all instances in the given project.

        Returns:
            Callable[[~.ListInstancesRequest],
                    Awaitable[~.ListInstancesResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instances" not in self._stubs:
            self._stubs["list_instances"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/ListInstances",
                request_serializer=spanner_instance_admin.ListInstancesRequest.serialize,
                response_deserializer=spanner_instance_admin.ListInstancesResponse.deserialize,
            )
        return self._stubs["list_instances"]

    @property
    def list_instance_partitions(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancePartitionsRequest],
        Awaitable[spanner_instance_admin.ListInstancePartitionsResponse],
    ]:
        r"""Return a callable for the list instance partitions method over gRPC.

        Lists all instance partitions for the given instance.

        Returns:
            Callable[[~.ListInstancePartitionsRequest],
                    Awaitable[~.ListInstancePartitionsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instance_partitions" not in self._stubs:
            self._stubs["list_instance_partitions"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/ListInstancePartitions",
                request_serializer=spanner_instance_admin.ListInstancePartitionsRequest.serialize,
                response_deserializer=spanner_instance_admin.ListInstancePartitionsResponse.deserialize,
            )
        return self._stubs["list_instance_partitions"]

    @property
    def get_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstanceRequest],
        Awaitable[spanner_instance_admin.Instance],
    ]:
        r"""Return a callable for the get instance method over gRPC.

        Gets information about a particular instance.

        Returns:
            Callable[[~.GetInstanceRequest],
                    Awaitable[~.Instance]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_instance" not in self._stubs:
            self._stubs["get_instance"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/GetInstance",
                request_serializer=spanner_instance_admin.GetInstanceRequest.serialize,
                response_deserializer=spanner_instance_admin.Instance.deserialize,
            )
        return self._stubs["get_instance"]

    @property
    def create_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstanceRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the create instance method over gRPC.

        Creates an instance and begins preparing it to begin serving.
        The returned long-running operation can be used to track the
        progress of preparing the new instance. The instance name is
        assigned by the caller. If the named instance already exists,
        ``CreateInstance`` returns ``ALREADY_EXISTS``.

        Immediately upon completion of this request:

        - The instance is readable via the API, with all requested
          attributes but no allocated resources. Its state is
          ``CREATING``.

        Until completion of the returned operation:

        - Cancelling the operation renders the instance immediately
          unreadable via the API.
        - The instance can be deleted.
        - All other attempts to modify the instance are rejected.

        Upon completion of the returned operation:

        - Billing for all successfully-allocated resources begins (some
          types may have lower than the requested levels).
        - Databases can be created in the instance.
        - The instance's allocated resource levels are readable via the
          API.
        - The instance's state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_name>/operations/<operation_id>`` and can be
        used to track creation of the instance. The metadata field type
        is
        [CreateInstanceMetadata][google.spanner.admin.instance.v1.CreateInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful.

        Returns:
            Callable[[~.CreateInstanceRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_instance" not in self._stubs:
            self._stubs["create_instance"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/CreateInstance",
                request_serializer=spanner_instance_admin.CreateInstanceRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_instance"]

    @property
    def update_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstanceRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the update instance method over gRPC.

        Updates an instance, and begins allocating or releasing
        resources as requested. The returned long-running operation can
        be used to track the progress of updating the instance. If the
        named instance does not exist, returns ``NOT_FOUND``.

        Immediately upon completion of this request:

        - For resource types for which a decrease in the instance's
          allocation has been requested, billing is based on the
          newly-requested level.

        Until completion of the returned operation:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstanceMetadata.cancel_time],
          and begins restoring resources to their pre-request values.
          The operation is guaranteed to succeed at undoing all resource
          changes, after which point it terminates with a ``CANCELLED``
          status.
        - All other attempts to modify the instance are rejected.
        - Reading the instance via the API continues to give the
          pre-request resource levels.

        Upon completion of the returned operation:

        - Billing begins for all successfully-allocated resources (some
          types may have lower than the requested levels).
        - All newly-reserved resources are available for serving the
          instance's tables.
        - The instance's new resource levels are readable via the API.

        The returned long-running operation will have a name of the
        format ``<instance_name>/operations/<operation_id>`` and can be
        used to track the instance modification. The metadata field type
        is
        [UpdateInstanceMetadata][google.spanner.admin.instance.v1.UpdateInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful.

        Authorization requires ``spanner.instances.update`` permission
        on the resource
        [name][google.spanner.admin.instance.v1.Instance.name].

        Returns:
            Callable[[~.UpdateInstanceRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_instance" not in self._stubs:
            self._stubs["update_instance"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/UpdateInstance",
                request_serializer=spanner_instance_admin.UpdateInstanceRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_instance"]

    @property
    def delete_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.DeleteInstanceRequest], Awaitable[empty_pb2.Empty]
    ]:
        r"""Return a callable for the delete instance method over gRPC.

        Deletes an instance.

        Immediately upon completion of the request:

        - Billing ceases for all of the instance's reserved resources.

        Soon afterward:

        - The instance and *all of its databases* immediately and
          irrevocably disappear from the API. All data in the databases
          is permanently deleted.

        Returns:
            Callable[[~.DeleteInstanceRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_instance" not in self._stubs:
            self._stubs["delete_instance"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/DeleteInstance",
                request_serializer=spanner_instance_admin.DeleteInstanceRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_instance"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], Awaitable[policy_pb2.Policy]]:
        r"""Return a callable for the set iam policy method over gRPC.

        Sets the access control policy on an instance resource. Replaces
        any existing policy.

        Authorization requires ``spanner.instances.setIamPolicy`` on
        [resource][google.iam.v1.SetIamPolicyRequest.resource].

        Returns:
            Callable[[~.SetIamPolicyRequest],
                    Awaitable[~.Policy]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "set_iam_policy" not in self._stubs:
            self._stubs["set_iam_policy"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/SetIamPolicy",
                request_serializer=iam_policy_pb2.SetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["set_iam_policy"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], Awaitable[policy_pb2.Policy]]:
        r"""Return a callable for the get iam policy method over gRPC.

        Gets the access control policy for an instance resource. Returns
        an empty policy if an instance exists but does not have a policy
        set.

        Authorization requires ``spanner.instances.getIamPolicy`` on
        [resource][google.iam.v1.GetIamPolicyRequest.resource].

        Returns:
            Callable[[~.GetIamPolicyRequest],
                    Awaitable[~.Policy]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_iam_policy" not in self._stubs:
            self._stubs["get_iam_policy"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        Awaitable[iam_policy_pb2.TestIamPermissionsResponse],
    ]:
        r"""Return a callable for the test iam permissions method over gRPC.

        Returns permissions that the caller has on the specified
        instance resource.

        Attempting this RPC on a non-existent Cloud Spanner instance
        resource will result in a NOT_FOUND error if the user has
        ``spanner.instances.list`` permission on the containing Google
        Cloud Project. Otherwise returns an empty set of permissions.

        Returns:
            Callable[[~.TestIamPermissionsRequest],
                    Awaitable[~.TestIamPermissionsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "test_iam_permissions" not in self._stubs:
            self._stubs["test_iam_permissions"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def get_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstancePartitionRequest],
        Awaitable[spanner_instance_admin.InstancePartition],
    ]:
        r"""Return a callable for the get instance partition method over gRPC.

        Gets information about a particular instance
        partition.

        Returns:
            Callable[[~.GetInstancePartitionRequest],
                    Awaitable[~.InstancePartition]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_instance_partition" not in self._stubs:
            self._stubs["get_instance_partition"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/GetInstancePartition",
                request_serializer=spanner_instance_admin.GetInstancePartitionRequest.serialize,
                response_deserializer=spanner_instance_admin.InstancePartition.deserialize,
            )
        return self._stubs["get_instance_partition"]

    @property
    def create_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstancePartitionRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the create instance partition method over gRPC.

        Creates an instance partition and begins preparing it to be
        used. The returned long-running operation can be used to track
        the progress of preparing the new instance partition. The
        instance partition name is assigned by the caller. If the named
        instance partition already exists, ``CreateInstancePartition``
        returns ``ALREADY_EXISTS``.

        Immediately upon completion of this request:

        - The instance partition is readable via the API, with all
          requested attributes but no allocated resources. Its state is
          ``CREATING``.

        Until completion of the returned operation:

        - Cancelling the operation renders the instance partition
          immediately unreadable via the API.
        - The instance partition can be deleted.
        - All other attempts to modify the instance partition are
          rejected.

        Upon completion of the returned operation:

        - Billing for all successfully-allocated resources begins (some
          types may have lower than the requested levels).
        - Databases can start using this instance partition.
        - The instance partition's allocated resource levels are
          readable via the API.
        - The instance partition's state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_partition_name>/operations/<operation_id>``
        and can be used to track creation of the instance partition. The
        metadata field type is
        [CreateInstancePartitionMetadata][google.spanner.admin.instance.v1.CreateInstancePartitionMetadata].
        The response field type is
        [InstancePartition][google.spanner.admin.instance.v1.InstancePartition],
        if successful.

        Returns:
            Callable[[~.CreateInstancePartitionRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_instance_partition" not in self._stubs:
            self._stubs["create_instance_partition"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/CreateInstancePartition",
                request_serializer=spanner_instance_admin.CreateInstancePartitionRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_instance_partition"]

    @property
    def delete_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.DeleteInstancePartitionRequest],
        Awaitable[empty_pb2.Empty],
    ]:
        r"""Return a callable for the delete instance partition method over gRPC.

        Deletes an existing instance partition. Requires that the
        instance partition is not used by any database or backup and is
        not the default instance partition of an instance.

        Authorization requires ``spanner.instancePartitions.delete``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstancePartition.name].

        Returns:
            Callable[[~.DeleteInstancePartitionRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_instance_partition" not in self._stubs:
            self._stubs["delete_instance_partition"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/DeleteInstancePartition",
                request_serializer=spanner_instance_admin.DeleteInstancePartitionRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_instance_partition"]

    @property
    def update_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstancePartitionRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the update instance partition method over gRPC.

        Updates an instance partition, and begins allocating or
        releasing resources as requested. The returned long-running
        operation can be used to track the progress of updating the
        instance partition. If the named instance partition does not
        exist, returns ``NOT_FOUND``.

        Immediately upon completion of this request:

        - For resource types for which a decrease in the instance
          partition's allocation has been requested, billing is based on
          the newly-requested level.

        Until completion of the returned operation:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstancePartitionMetadata.cancel_time],
          and begins restoring resources to their pre-request values.
          The operation is guaranteed to succeed at undoing all resource
          changes, after which point it terminates with a ``CANCELLED``
          status.
        - All other attempts to modify the instance partition are
          rejected.
        - Reading the instance partition via the API continues to give
          the pre-request resource levels.

        Upon completion of the returned operation:

        - Billing begins for all successfully-allocated resources (some
          types may have lower than the requested levels).
        - All newly-reserved resources are available for serving the
          instance partition's tables.
        - The instance partition's new resource levels are readable via
          the API.

        The returned long-running operation will have a name of the
        format ``<instance_partition_name>/operations/<operation_id>``
        and can be used to track the instance partition modification.
        The metadata field type is
        [UpdateInstancePartitionMetadata][google.spanner.admin.instance.v1.UpdateInstancePartitionMetadata].
        The response field type is
        [InstancePartition][google.spanner.admin.instance.v1.InstancePartition],
        if successful.

        Authorization requires ``spanner.instancePartitions.update``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstancePartition.name].

        Returns:
            Callable[[~.UpdateInstancePartitionRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_instance_partition" not in self._stubs:
            self._stubs["update_instance_partition"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/UpdateInstancePartition",
                request_serializer=spanner_instance_admin.UpdateInstancePartitionRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_instance_partition"]

    @property
    def list_instance_partition_operations(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancePartitionOperationsRequest],
        Awaitable[spanner_instance_admin.ListInstancePartitionOperationsResponse],
    ]:
        r"""Return a callable for the list instance partition
        operations method over gRPC.

        Lists instance partition long-running operations in the given
        instance. An instance partition operation has a name of the form
        ``projects/<project>/instances/<instance>/instancePartitions/<instance_partition>/operations/<operation>``.
        The long-running operation metadata field type
        ``metadata.type_url`` describes the type of the metadata.
        Operations returned include those that have
        completed/failed/canceled within the last 7 days, and pending
        operations. Operations returned are ordered by
        ``operation.metadata.value.start_time`` in descending order
        starting from the most recently started operation.

        Authorization requires
        ``spanner.instancePartitionOperations.list`` permission on the
        resource
        [parent][google.spanner.admin.instance.v1.ListInstancePartitionOperationsRequest.parent].

        Returns:
            Callable[[~.ListInstancePartitionOperationsRequest],
                    Awaitable[~.ListInstancePartitionOperationsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_instance_partition_operations" not in self._stubs:
            self._stubs[
                "list_instance_partition_operations"
            ] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/ListInstancePartitionOperations",
                request_serializer=spanner_instance_admin.ListInstancePartitionOperationsRequest.serialize,
                response_deserializer=spanner_instance_admin.ListInstancePartitionOperationsResponse.deserialize,
            )
        return self._stubs["list_instance_partition_operations"]

    @property
    def move_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.MoveInstanceRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the move instance method over gRPC.

        Moves an instance to the target instance configuration. You can
        use the returned long-running operation to track the progress of
        moving the instance.

        ``MoveInstance`` returns ``FAILED_PRECONDITION`` if the instance
        meets any of the following criteria:

        - Is undergoing a move to a different instance configuration
        - Has backups
        - Has an ongoing update
        - Contains any CMEK-enabled databases
        - Is a free trial instance

        While the operation is pending:

        - All other attempts to modify the instance, including changes
          to its compute capacity, are rejected.

        - The following database and backup admin operations are
          rejected:

          - ``DatabaseAdmin.CreateDatabase``
          - ``DatabaseAdmin.UpdateDatabaseDdl`` (disabled if
            default_leader is specified in the request.)
          - ``DatabaseAdmin.RestoreDatabase``
          - ``DatabaseAdmin.CreateBackup``
          - ``DatabaseAdmin.CopyBackup``

        - Both the source and target instance configurations are subject
          to hourly compute and storage charges.

        - The instance might experience higher read-write latencies and
          a higher transaction abort rate. However, moving an instance
          doesn't cause any downtime.

        The returned long-running operation has a name of the format
        ``<instance_name>/operations/<operation_id>`` and can be used to
        track the move instance operation. The metadata field type is
        [MoveInstanceMetadata][google.spanner.admin.instance.v1.MoveInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful. Cancelling the operation sets its metadata's
        [cancel_time][google.spanner.admin.instance.v1.MoveInstanceMetadata.cancel_time].
        Cancellation is not immediate because it involves moving any
        data previously moved to the target instance configuration back
        to the original instance configuration. You can use this
        operation to track the progress of the cancellation. Upon
        successful completion of the cancellation, the operation
        terminates with ``CANCELLED`` status.

        If not cancelled, upon completion of the returned operation:

        - The instance successfully moves to the target instance
          configuration.
        - You are billed for compute and storage in target instance
          configuration.

        Authorization requires the ``spanner.instances.update``
        permission on the resource
        [instance][google.spanner.admin.instance.v1.Instance].

        For more details, see `Move an
        instance <https://cloud.google.com/spanner/docs/move-instance>`__.

        Returns:
            Callable[[~.MoveInstanceRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "move_instance" not in self._stubs:
            self._stubs["move_instance"] = self._logged_channel.unary_unary(
                "/google.spanner.admin.instance.v1.InstanceAdmin/MoveInstance",
                request_serializer=spanner_instance_admin.MoveInstanceRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["move_instance"]

    def _prep_wrapped_messages(self, client_info):
        """Precompute the wrapped methods, overriding the base class method to use async wrappers."""
        self._wrapped_methods = {
            self.list_instance_configs: self._wrap_method(
                self.list_instance_configs,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.get_instance_config: self._wrap_method(
                self.get_instance_config,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.create_instance_config: self._wrap_method(
                self.create_instance_config,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_instance_config: self._wrap_method(
                self.update_instance_config,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_instance_config: self._wrap_method(
                self.delete_instance_config,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_instance_config_operations: self._wrap_method(
                self.list_instance_config_operations,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_instances: self._wrap_method(
                self.list_instances,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_instance_partitions: self._wrap_method(
                self.list_instance_partitions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_instance: self._wrap_method(
                self.get_instance,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.create_instance: self._wrap_method(
                self.create_instance,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.update_instance: self._wrap_method(
                self.update_instance,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.delete_instance: self._wrap_method(
                self.delete_instance,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.set_iam_policy: self._wrap_method(
                self.set_iam_policy,
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.get_iam_policy: self._wrap_method(
                self.get_iam_policy,
                default_retry=retries.AsyncRetry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.test_iam_permissions: self._wrap_method(
                self.test_iam_permissions,
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.get_instance_partition: self._wrap_method(
                self.get_instance_partition,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_instance_partition: self._wrap_method(
                self.create_instance_partition,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_instance_partition: self._wrap_method(
                self.delete_instance_partition,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_instance_partition: self._wrap_method(
                self.update_instance_partition,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_instance_partition_operations: self._wrap_method(
                self.list_instance_partition_operations,
                default_timeout=None,
                client_info=client_info,
            ),
            self.move_instance: self._wrap_method(
                self.move_instance,
                default_timeout=None,
                client_info=client_info,
            ),
            self.cancel_operation: self._wrap_method(
                self.cancel_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_operation: self._wrap_method(
                self.delete_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_operation: self._wrap_method(
                self.get_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_operations: self._wrap_method(
                self.list_operations,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def _wrap_method(self, func, *args, **kwargs):
        if self._wrap_with_kind:  # pragma: NO COVER
            kwargs["kind"] = self.kind
        return gapic_v1.method_async.wrap_method(func, *args, **kwargs)

    def close(self):
        return self._logged_channel.close()

    @property
    def kind(self) -> str:
        return "grpc_asyncio"

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], None]:
        r"""Return a callable for the delete_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_operation" not in self._stubs:
            self._stubs["delete_operation"] = self._logged_channel.unary_unary(
                "/google.longrunning.Operations/DeleteOperation",
                request_serializer=operations_pb2.DeleteOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["delete_operation"]

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], None]:
        r"""Return a callable for the cancel_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "cancel_operation" not in self._stubs:
            self._stubs["cancel_operation"] = self._logged_channel.unary_unary(
                "/google.longrunning.Operations/CancelOperation",
                request_serializer=operations_pb2.CancelOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["cancel_operation"]

    @property
    def get_operation(
        self,
    ) -> Callable[[operations_pb2.GetOperationRequest], operations_pb2.Operation]:
        r"""Return a callable for the get_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_operation" not in self._stubs:
            self._stubs["get_operation"] = self._logged_channel.unary_unary(
                "/google.longrunning.Operations/GetOperation",
                request_serializer=operations_pb2.GetOperationRequest.SerializeToString,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["get_operation"]

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest], operations_pb2.ListOperationsResponse
    ]:
        r"""Return a callable for the list_operations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_operations" not in self._stubs:
            self._stubs["list_operations"] = self._logged_channel.unary_unary(
                "/google.longrunning.Operations/ListOperations",
                request_serializer=operations_pb2.ListOperationsRequest.SerializeToString,
                response_deserializer=operations_pb2.ListOperationsResponse.FromString,
            )
        return self._stubs["list_operations"]


__all__ = ("InstanceAdminGrpcAsyncIOTransport",)

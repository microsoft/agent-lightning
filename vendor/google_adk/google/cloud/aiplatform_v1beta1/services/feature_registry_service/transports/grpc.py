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

from google.cloud.aiplatform_v1beta1.types import feature
from google.cloud.aiplatform_v1beta1.types import feature_group
from google.cloud.aiplatform_v1beta1.types import feature_monitor
from google.cloud.aiplatform_v1beta1.types import feature_monitor_job
from google.cloud.aiplatform_v1beta1.types import (
    feature_monitor_job as gca_feature_monitor_job,
)
from google.cloud.aiplatform_v1beta1.types import feature_registry_service
from google.cloud.aiplatform_v1beta1.types import featurestore_service
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from .base import FeatureRegistryServiceTransport, DEFAULT_CLIENT_INFO

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
                    "serviceName": "google.cloud.aiplatform.v1beta1.FeatureRegistryService",
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
                    "serviceName": "google.cloud.aiplatform.v1beta1.FeatureRegistryService",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class FeatureRegistryServiceGrpcTransport(FeatureRegistryServiceTransport):
    """gRPC backend transport for FeatureRegistryService.

    The service that handles CRUD and List for resources for
    FeatureRegistry.

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
        host: str = "aiplatform.googleapis.com",
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
                 The hostname to connect to (default: 'aiplatform.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
                This argument is ignored if a ``channel`` instance is provided.
            credentials_file (Optional[str]): Deprecated. A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if a ``channel`` instance is provided.
                This argument will be removed in the next major version of this library.
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
        host: str = "aiplatform.googleapis.com",
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
            credentials_file (Optional[str]): Deprecated. A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.  This argument will be
                removed in the next major version of this library.
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
    def create_feature_group(
        self,
    ) -> Callable[
        [feature_registry_service.CreateFeatureGroupRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create feature group method over gRPC.

        Creates a new FeatureGroup in a given project and
        location.

        Returns:
            Callable[[~.CreateFeatureGroupRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_feature_group" not in self._stubs:
            self._stubs["create_feature_group"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/CreateFeatureGroup",
                request_serializer=feature_registry_service.CreateFeatureGroupRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_feature_group"]

    @property
    def get_feature_group(
        self,
    ) -> Callable[
        [feature_registry_service.GetFeatureGroupRequest], feature_group.FeatureGroup
    ]:
        r"""Return a callable for the get feature group method over gRPC.

        Gets details of a single FeatureGroup.

        Returns:
            Callable[[~.GetFeatureGroupRequest],
                    ~.FeatureGroup]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_feature_group" not in self._stubs:
            self._stubs["get_feature_group"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/GetFeatureGroup",
                request_serializer=feature_registry_service.GetFeatureGroupRequest.serialize,
                response_deserializer=feature_group.FeatureGroup.deserialize,
            )
        return self._stubs["get_feature_group"]

    @property
    def list_feature_groups(
        self,
    ) -> Callable[
        [feature_registry_service.ListFeatureGroupsRequest],
        feature_registry_service.ListFeatureGroupsResponse,
    ]:
        r"""Return a callable for the list feature groups method over gRPC.

        Lists FeatureGroups in a given project and location.

        Returns:
            Callable[[~.ListFeatureGroupsRequest],
                    ~.ListFeatureGroupsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_feature_groups" not in self._stubs:
            self._stubs["list_feature_groups"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/ListFeatureGroups",
                request_serializer=feature_registry_service.ListFeatureGroupsRequest.serialize,
                response_deserializer=feature_registry_service.ListFeatureGroupsResponse.deserialize,
            )
        return self._stubs["list_feature_groups"]

    @property
    def update_feature_group(
        self,
    ) -> Callable[
        [feature_registry_service.UpdateFeatureGroupRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update feature group method over gRPC.

        Updates the parameters of a single FeatureGroup.

        Returns:
            Callable[[~.UpdateFeatureGroupRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_feature_group" not in self._stubs:
            self._stubs["update_feature_group"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/UpdateFeatureGroup",
                request_serializer=feature_registry_service.UpdateFeatureGroupRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_feature_group"]

    @property
    def delete_feature_group(
        self,
    ) -> Callable[
        [feature_registry_service.DeleteFeatureGroupRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete feature group method over gRPC.

        Deletes a single FeatureGroup.

        Returns:
            Callable[[~.DeleteFeatureGroupRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_feature_group" not in self._stubs:
            self._stubs["delete_feature_group"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/DeleteFeatureGroup",
                request_serializer=feature_registry_service.DeleteFeatureGroupRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_feature_group"]

    @property
    def create_feature(
        self,
    ) -> Callable[
        [featurestore_service.CreateFeatureRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create feature method over gRPC.

        Creates a new Feature in a given FeatureGroup.

        Returns:
            Callable[[~.CreateFeatureRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_feature" not in self._stubs:
            self._stubs["create_feature"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/CreateFeature",
                request_serializer=featurestore_service.CreateFeatureRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_feature"]

    @property
    def batch_create_features(
        self,
    ) -> Callable[
        [featurestore_service.BatchCreateFeaturesRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the batch create features method over gRPC.

        Creates a batch of Features in a given FeatureGroup.

        Returns:
            Callable[[~.BatchCreateFeaturesRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_features" not in self._stubs:
            self._stubs["batch_create_features"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/BatchCreateFeatures",
                request_serializer=featurestore_service.BatchCreateFeaturesRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["batch_create_features"]

    @property
    def get_feature(
        self,
    ) -> Callable[[featurestore_service.GetFeatureRequest], feature.Feature]:
        r"""Return a callable for the get feature method over gRPC.

        Gets details of a single Feature.

        Returns:
            Callable[[~.GetFeatureRequest],
                    ~.Feature]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_feature" not in self._stubs:
            self._stubs["get_feature"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/GetFeature",
                request_serializer=featurestore_service.GetFeatureRequest.serialize,
                response_deserializer=feature.Feature.deserialize,
            )
        return self._stubs["get_feature"]

    @property
    def list_features(
        self,
    ) -> Callable[
        [featurestore_service.ListFeaturesRequest],
        featurestore_service.ListFeaturesResponse,
    ]:
        r"""Return a callable for the list features method over gRPC.

        Lists Features in a given FeatureGroup.

        Returns:
            Callable[[~.ListFeaturesRequest],
                    ~.ListFeaturesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_features" not in self._stubs:
            self._stubs["list_features"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/ListFeatures",
                request_serializer=featurestore_service.ListFeaturesRequest.serialize,
                response_deserializer=featurestore_service.ListFeaturesResponse.deserialize,
            )
        return self._stubs["list_features"]

    @property
    def update_feature(
        self,
    ) -> Callable[
        [featurestore_service.UpdateFeatureRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update feature method over gRPC.

        Updates the parameters of a single Feature.

        Returns:
            Callable[[~.UpdateFeatureRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_feature" not in self._stubs:
            self._stubs["update_feature"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/UpdateFeature",
                request_serializer=featurestore_service.UpdateFeatureRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_feature"]

    @property
    def delete_feature(
        self,
    ) -> Callable[
        [featurestore_service.DeleteFeatureRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete feature method over gRPC.

        Deletes a single Feature.

        Returns:
            Callable[[~.DeleteFeatureRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_feature" not in self._stubs:
            self._stubs["delete_feature"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/DeleteFeature",
                request_serializer=featurestore_service.DeleteFeatureRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_feature"]

    @property
    def create_feature_monitor(
        self,
    ) -> Callable[
        [feature_registry_service.CreateFeatureMonitorRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create feature monitor method over gRPC.

        Creates a new FeatureMonitor in a given project,
        location and FeatureGroup.

        Returns:
            Callable[[~.CreateFeatureMonitorRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_feature_monitor" not in self._stubs:
            self._stubs["create_feature_monitor"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/CreateFeatureMonitor",
                request_serializer=feature_registry_service.CreateFeatureMonitorRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_feature_monitor"]

    @property
    def get_feature_monitor(
        self,
    ) -> Callable[
        [feature_registry_service.GetFeatureMonitorRequest],
        feature_monitor.FeatureMonitor,
    ]:
        r"""Return a callable for the get feature monitor method over gRPC.

        Gets details of a single FeatureMonitor.

        Returns:
            Callable[[~.GetFeatureMonitorRequest],
                    ~.FeatureMonitor]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_feature_monitor" not in self._stubs:
            self._stubs["get_feature_monitor"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/GetFeatureMonitor",
                request_serializer=feature_registry_service.GetFeatureMonitorRequest.serialize,
                response_deserializer=feature_monitor.FeatureMonitor.deserialize,
            )
        return self._stubs["get_feature_monitor"]

    @property
    def list_feature_monitors(
        self,
    ) -> Callable[
        [feature_registry_service.ListFeatureMonitorsRequest],
        feature_registry_service.ListFeatureMonitorsResponse,
    ]:
        r"""Return a callable for the list feature monitors method over gRPC.

        Lists FeatureGroups in a given project and location.

        Returns:
            Callable[[~.ListFeatureMonitorsRequest],
                    ~.ListFeatureMonitorsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_feature_monitors" not in self._stubs:
            self._stubs["list_feature_monitors"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/ListFeatureMonitors",
                request_serializer=feature_registry_service.ListFeatureMonitorsRequest.serialize,
                response_deserializer=feature_registry_service.ListFeatureMonitorsResponse.deserialize,
            )
        return self._stubs["list_feature_monitors"]

    @property
    def update_feature_monitor(
        self,
    ) -> Callable[
        [feature_registry_service.UpdateFeatureMonitorRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update feature monitor method over gRPC.

        Updates the parameters of a single FeatureMonitor.

        Returns:
            Callable[[~.UpdateFeatureMonitorRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_feature_monitor" not in self._stubs:
            self._stubs["update_feature_monitor"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/UpdateFeatureMonitor",
                request_serializer=feature_registry_service.UpdateFeatureMonitorRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_feature_monitor"]

    @property
    def delete_feature_monitor(
        self,
    ) -> Callable[
        [feature_registry_service.DeleteFeatureMonitorRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete feature monitor method over gRPC.

        Deletes a single FeatureMonitor.

        Returns:
            Callable[[~.DeleteFeatureMonitorRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_feature_monitor" not in self._stubs:
            self._stubs["delete_feature_monitor"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/DeleteFeatureMonitor",
                request_serializer=feature_registry_service.DeleteFeatureMonitorRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_feature_monitor"]

    @property
    def create_feature_monitor_job(
        self,
    ) -> Callable[
        [feature_registry_service.CreateFeatureMonitorJobRequest],
        gca_feature_monitor_job.FeatureMonitorJob,
    ]:
        r"""Return a callable for the create feature monitor job method over gRPC.

        Creates a new feature monitor job.

        Returns:
            Callable[[~.CreateFeatureMonitorJobRequest],
                    ~.FeatureMonitorJob]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_feature_monitor_job" not in self._stubs:
            self._stubs["create_feature_monitor_job"] = (
                self._logged_channel.unary_unary(
                    "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/CreateFeatureMonitorJob",
                    request_serializer=feature_registry_service.CreateFeatureMonitorJobRequest.serialize,
                    response_deserializer=gca_feature_monitor_job.FeatureMonitorJob.deserialize,
                )
            )
        return self._stubs["create_feature_monitor_job"]

    @property
    def get_feature_monitor_job(
        self,
    ) -> Callable[
        [feature_registry_service.GetFeatureMonitorJobRequest],
        feature_monitor_job.FeatureMonitorJob,
    ]:
        r"""Return a callable for the get feature monitor job method over gRPC.

        Get a feature monitor job.

        Returns:
            Callable[[~.GetFeatureMonitorJobRequest],
                    ~.FeatureMonitorJob]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_feature_monitor_job" not in self._stubs:
            self._stubs["get_feature_monitor_job"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/GetFeatureMonitorJob",
                request_serializer=feature_registry_service.GetFeatureMonitorJobRequest.serialize,
                response_deserializer=feature_monitor_job.FeatureMonitorJob.deserialize,
            )
        return self._stubs["get_feature_monitor_job"]

    @property
    def list_feature_monitor_jobs(
        self,
    ) -> Callable[
        [feature_registry_service.ListFeatureMonitorJobsRequest],
        feature_registry_service.ListFeatureMonitorJobsResponse,
    ]:
        r"""Return a callable for the list feature monitor jobs method over gRPC.

        List feature monitor jobs.

        Returns:
            Callable[[~.ListFeatureMonitorJobsRequest],
                    ~.ListFeatureMonitorJobsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_feature_monitor_jobs" not in self._stubs:
            self._stubs["list_feature_monitor_jobs"] = self._logged_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.FeatureRegistryService/ListFeatureMonitorJobs",
                request_serializer=feature_registry_service.ListFeatureMonitorJobsRequest.serialize,
                response_deserializer=feature_registry_service.ListFeatureMonitorJobsResponse.deserialize,
            )
        return self._stubs["list_feature_monitor_jobs"]

    def close(self):
        self._logged_channel.close()

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
    def wait_operation(
        self,
    ) -> Callable[[operations_pb2.WaitOperationRequest], None]:
        r"""Return a callable for the wait_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "wait_operation" not in self._stubs:
            self._stubs["wait_operation"] = self._logged_channel.unary_unary(
                "/google.longrunning.Operations/WaitOperation",
                request_serializer=operations_pb2.WaitOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["wait_operation"]

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

    @property
    def list_locations(
        self,
    ) -> Callable[
        [locations_pb2.ListLocationsRequest], locations_pb2.ListLocationsResponse
    ]:
        r"""Return a callable for the list locations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_locations" not in self._stubs:
            self._stubs["list_locations"] = self._logged_channel.unary_unary(
                "/google.cloud.location.Locations/ListLocations",
                request_serializer=locations_pb2.ListLocationsRequest.SerializeToString,
                response_deserializer=locations_pb2.ListLocationsResponse.FromString,
            )
        return self._stubs["list_locations"]

    @property
    def get_location(
        self,
    ) -> Callable[[locations_pb2.GetLocationRequest], locations_pb2.Location]:
        r"""Return a callable for the list locations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_location" not in self._stubs:
            self._stubs["get_location"] = self._logged_channel.unary_unary(
                "/google.cloud.location.Locations/GetLocation",
                request_serializer=locations_pb2.GetLocationRequest.SerializeToString,
                response_deserializer=locations_pb2.Location.FromString,
            )
        return self._stubs["get_location"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the set iam policy method over gRPC.
        Sets the IAM access control policy on the specified
        function. Replaces any existing policy.
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
                "/google.iam.v1.IAMPolicy/SetIamPolicy",
                request_serializer=iam_policy_pb2.SetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["set_iam_policy"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the get iam policy method over gRPC.
        Gets the IAM access control policy for a function.
        Returns an empty policy if the function exists and does
        not have a policy set.
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
                "/google.iam.v1.IAMPolicy/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        r"""Return a callable for the test iam permissions method over gRPC.
        Tests the specified permissions against the IAM access control
        policy for a function. If the function does not exist, this will
        return an empty set of permissions, not a NOT_FOUND error.
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
                "/google.iam.v1.IAMPolicy/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("FeatureRegistryServiceGrpcTransport",)

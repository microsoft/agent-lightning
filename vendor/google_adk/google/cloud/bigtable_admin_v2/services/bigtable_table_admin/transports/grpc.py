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

from google.cloud.bigtable_admin_v2.types import bigtable_table_admin
from google.cloud.bigtable_admin_v2.types import table
from google.cloud.bigtable_admin_v2.types import table as gba_table
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from .base import BigtableTableAdminTransport, DEFAULT_CLIENT_INFO

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
                    "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
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
                    "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class BigtableTableAdminGrpcTransport(BigtableTableAdminTransport):
    """gRPC backend transport for BigtableTableAdmin.

    Service for creating, configuring, and deleting Cloud
    Bigtable tables.

    Provides access to the table schemas only, not the data stored
    within the tables.

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
    def create_table(
        self,
    ) -> Callable[[bigtable_table_admin.CreateTableRequest], gba_table.Table]:
        r"""Return a callable for the create table method over gRPC.

        Creates a new table in the specified instance.
        The table can be created with a full set of initial
        column families, specified in the request.

        Returns:
            Callable[[~.CreateTableRequest],
                    ~.Table]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_table" not in self._stubs:
            self._stubs["create_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CreateTable",
                request_serializer=bigtable_table_admin.CreateTableRequest.serialize,
                response_deserializer=gba_table.Table.deserialize,
            )
        return self._stubs["create_table"]

    @property
    def create_table_from_snapshot(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateTableFromSnapshotRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create table from snapshot method over gRPC.

        Creates a new table from the specified snapshot. The
        target table must not exist. The snapshot and the table
        must be in the same instance.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        Returns:
            Callable[[~.CreateTableFromSnapshotRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_table_from_snapshot" not in self._stubs:
            self._stubs[
                "create_table_from_snapshot"
            ] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CreateTableFromSnapshot",
                request_serializer=bigtable_table_admin.CreateTableFromSnapshotRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_table_from_snapshot"]

    @property
    def list_tables(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListTablesRequest],
        bigtable_table_admin.ListTablesResponse,
    ]:
        r"""Return a callable for the list tables method over gRPC.

        Lists all tables served from a specified instance.

        Returns:
            Callable[[~.ListTablesRequest],
                    ~.ListTablesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_tables" not in self._stubs:
            self._stubs["list_tables"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ListTables",
                request_serializer=bigtable_table_admin.ListTablesRequest.serialize,
                response_deserializer=bigtable_table_admin.ListTablesResponse.deserialize,
            )
        return self._stubs["list_tables"]

    @property
    def get_table(
        self,
    ) -> Callable[[bigtable_table_admin.GetTableRequest], table.Table]:
        r"""Return a callable for the get table method over gRPC.

        Gets metadata information about the specified table.

        Returns:
            Callable[[~.GetTableRequest],
                    ~.Table]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_table" not in self._stubs:
            self._stubs["get_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetTable",
                request_serializer=bigtable_table_admin.GetTableRequest.serialize,
                response_deserializer=table.Table.deserialize,
            )
        return self._stubs["get_table"]

    @property
    def update_table(
        self,
    ) -> Callable[[bigtable_table_admin.UpdateTableRequest], operations_pb2.Operation]:
        r"""Return a callable for the update table method over gRPC.

        Updates a specified table.

        Returns:
            Callable[[~.UpdateTableRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_table" not in self._stubs:
            self._stubs["update_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/UpdateTable",
                request_serializer=bigtable_table_admin.UpdateTableRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_table"]

    @property
    def delete_table(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteTableRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete table method over gRPC.

        Permanently deletes a specified table and all of its
        data.

        Returns:
            Callable[[~.DeleteTableRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_table" not in self._stubs:
            self._stubs["delete_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DeleteTable",
                request_serializer=bigtable_table_admin.DeleteTableRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_table"]

    @property
    def undelete_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.UndeleteTableRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the undelete table method over gRPC.

        Restores a specified table which was accidentally
        deleted.

        Returns:
            Callable[[~.UndeleteTableRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "undelete_table" not in self._stubs:
            self._stubs["undelete_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/UndeleteTable",
                request_serializer=bigtable_table_admin.UndeleteTableRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["undelete_table"]

    @property
    def create_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateAuthorizedViewRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create authorized view method over gRPC.

        Creates a new AuthorizedView in a table.

        Returns:
            Callable[[~.CreateAuthorizedViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_authorized_view" not in self._stubs:
            self._stubs["create_authorized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CreateAuthorizedView",
                request_serializer=bigtable_table_admin.CreateAuthorizedViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_authorized_view"]

    @property
    def list_authorized_views(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListAuthorizedViewsRequest],
        bigtable_table_admin.ListAuthorizedViewsResponse,
    ]:
        r"""Return a callable for the list authorized views method over gRPC.

        Lists all AuthorizedViews from a specific table.

        Returns:
            Callable[[~.ListAuthorizedViewsRequest],
                    ~.ListAuthorizedViewsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_authorized_views" not in self._stubs:
            self._stubs["list_authorized_views"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ListAuthorizedViews",
                request_serializer=bigtable_table_admin.ListAuthorizedViewsRequest.serialize,
                response_deserializer=bigtable_table_admin.ListAuthorizedViewsResponse.deserialize,
            )
        return self._stubs["list_authorized_views"]

    @property
    def get_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetAuthorizedViewRequest], table.AuthorizedView
    ]:
        r"""Return a callable for the get authorized view method over gRPC.

        Gets information from a specified AuthorizedView.

        Returns:
            Callable[[~.GetAuthorizedViewRequest],
                    ~.AuthorizedView]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_authorized_view" not in self._stubs:
            self._stubs["get_authorized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetAuthorizedView",
                request_serializer=bigtable_table_admin.GetAuthorizedViewRequest.serialize,
                response_deserializer=table.AuthorizedView.deserialize,
            )
        return self._stubs["get_authorized_view"]

    @property
    def update_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateAuthorizedViewRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update authorized view method over gRPC.

        Updates an AuthorizedView in a table.

        Returns:
            Callable[[~.UpdateAuthorizedViewRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_authorized_view" not in self._stubs:
            self._stubs["update_authorized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/UpdateAuthorizedView",
                request_serializer=bigtable_table_admin.UpdateAuthorizedViewRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_authorized_view"]

    @property
    def delete_authorized_view(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteAuthorizedViewRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete authorized view method over gRPC.

        Permanently deletes a specified AuthorizedView.

        Returns:
            Callable[[~.DeleteAuthorizedViewRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_authorized_view" not in self._stubs:
            self._stubs["delete_authorized_view"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DeleteAuthorizedView",
                request_serializer=bigtable_table_admin.DeleteAuthorizedViewRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_authorized_view"]

    @property
    def modify_column_families(
        self,
    ) -> Callable[[bigtable_table_admin.ModifyColumnFamiliesRequest], table.Table]:
        r"""Return a callable for the modify column families method over gRPC.

        Performs a series of column family modifications on
        the specified table. Either all or none of the
        modifications will occur before this method returns, but
        data requests received prior to that point may see a
        table where only some modifications have taken effect.

        Returns:
            Callable[[~.ModifyColumnFamiliesRequest],
                    ~.Table]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "modify_column_families" not in self._stubs:
            self._stubs["modify_column_families"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ModifyColumnFamilies",
                request_serializer=bigtable_table_admin.ModifyColumnFamiliesRequest.serialize,
                response_deserializer=table.Table.deserialize,
            )
        return self._stubs["modify_column_families"]

    @property
    def drop_row_range(
        self,
    ) -> Callable[[bigtable_table_admin.DropRowRangeRequest], empty_pb2.Empty]:
        r"""Return a callable for the drop row range method over gRPC.

        Permanently drop/delete a row range from a specified
        table. The request can specify whether to delete all
        rows in a table, or only those that match a particular
        prefix.

        Returns:
            Callable[[~.DropRowRangeRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "drop_row_range" not in self._stubs:
            self._stubs["drop_row_range"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DropRowRange",
                request_serializer=bigtable_table_admin.DropRowRangeRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["drop_row_range"]

    @property
    def generate_consistency_token(
        self,
    ) -> Callable[
        [bigtable_table_admin.GenerateConsistencyTokenRequest],
        bigtable_table_admin.GenerateConsistencyTokenResponse,
    ]:
        r"""Return a callable for the generate consistency token method over gRPC.

        Generates a consistency token for a Table, which can
        be used in CheckConsistency to check whether mutations
        to the table that finished before this call started have
        been replicated. The tokens will be available for 90
        days.

        Returns:
            Callable[[~.GenerateConsistencyTokenRequest],
                    ~.GenerateConsistencyTokenResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "generate_consistency_token" not in self._stubs:
            self._stubs[
                "generate_consistency_token"
            ] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GenerateConsistencyToken",
                request_serializer=bigtable_table_admin.GenerateConsistencyTokenRequest.serialize,
                response_deserializer=bigtable_table_admin.GenerateConsistencyTokenResponse.deserialize,
            )
        return self._stubs["generate_consistency_token"]

    @property
    def check_consistency(
        self,
    ) -> Callable[
        [bigtable_table_admin.CheckConsistencyRequest],
        bigtable_table_admin.CheckConsistencyResponse,
    ]:
        r"""Return a callable for the check consistency method over gRPC.

        Checks replication consistency based on a consistency
        token, that is, if replication has caught up based on
        the conditions specified in the token and the check
        request.

        Returns:
            Callable[[~.CheckConsistencyRequest],
                    ~.CheckConsistencyResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "check_consistency" not in self._stubs:
            self._stubs["check_consistency"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CheckConsistency",
                request_serializer=bigtable_table_admin.CheckConsistencyRequest.serialize,
                response_deserializer=bigtable_table_admin.CheckConsistencyResponse.deserialize,
            )
        return self._stubs["check_consistency"]

    @property
    def snapshot_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.SnapshotTableRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the snapshot table method over gRPC.

        Creates a new snapshot in the specified cluster from
        the specified source table. The cluster and the table
        must be in the same instance.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        Returns:
            Callable[[~.SnapshotTableRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "snapshot_table" not in self._stubs:
            self._stubs["snapshot_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/SnapshotTable",
                request_serializer=bigtable_table_admin.SnapshotTableRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["snapshot_table"]

    @property
    def get_snapshot(
        self,
    ) -> Callable[[bigtable_table_admin.GetSnapshotRequest], table.Snapshot]:
        r"""Return a callable for the get snapshot method over gRPC.

        Gets metadata information about the specified
        snapshot.
        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        Returns:
            Callable[[~.GetSnapshotRequest],
                    ~.Snapshot]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_snapshot" not in self._stubs:
            self._stubs["get_snapshot"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetSnapshot",
                request_serializer=bigtable_table_admin.GetSnapshotRequest.serialize,
                response_deserializer=table.Snapshot.deserialize,
            )
        return self._stubs["get_snapshot"]

    @property
    def list_snapshots(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSnapshotsRequest],
        bigtable_table_admin.ListSnapshotsResponse,
    ]:
        r"""Return a callable for the list snapshots method over gRPC.

        Lists all snapshots associated with the specified
        cluster.
        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        Returns:
            Callable[[~.ListSnapshotsRequest],
                    ~.ListSnapshotsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_snapshots" not in self._stubs:
            self._stubs["list_snapshots"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ListSnapshots",
                request_serializer=bigtable_table_admin.ListSnapshotsRequest.serialize,
                response_deserializer=bigtable_table_admin.ListSnapshotsResponse.deserialize,
            )
        return self._stubs["list_snapshots"]

    @property
    def delete_snapshot(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteSnapshotRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete snapshot method over gRPC.

        Permanently deletes the specified snapshot.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        Returns:
            Callable[[~.DeleteSnapshotRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_snapshot" not in self._stubs:
            self._stubs["delete_snapshot"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DeleteSnapshot",
                request_serializer=bigtable_table_admin.DeleteSnapshotRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_snapshot"]

    @property
    def create_backup(
        self,
    ) -> Callable[[bigtable_table_admin.CreateBackupRequest], operations_pb2.Operation]:
        r"""Return a callable for the create backup method over gRPC.

        Starts creating a new Cloud Bigtable Backup. The returned backup
        [long-running operation][google.longrunning.Operation] can be
        used to track creation of the backup. The
        [metadata][google.longrunning.Operation.metadata] field type is
        [CreateBackupMetadata][google.bigtable.admin.v2.CreateBackupMetadata].
        The [response][google.longrunning.Operation.response] field type
        is [Backup][google.bigtable.admin.v2.Backup], if successful.
        Cancelling the returned operation will stop the creation and
        delete the backup.

        Returns:
            Callable[[~.CreateBackupRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_backup" not in self._stubs:
            self._stubs["create_backup"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CreateBackup",
                request_serializer=bigtable_table_admin.CreateBackupRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_backup"]

    @property
    def get_backup(
        self,
    ) -> Callable[[bigtable_table_admin.GetBackupRequest], table.Backup]:
        r"""Return a callable for the get backup method over gRPC.

        Gets metadata on a pending or completed Cloud
        Bigtable Backup.

        Returns:
            Callable[[~.GetBackupRequest],
                    ~.Backup]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_backup" not in self._stubs:
            self._stubs["get_backup"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetBackup",
                request_serializer=bigtable_table_admin.GetBackupRequest.serialize,
                response_deserializer=table.Backup.deserialize,
            )
        return self._stubs["get_backup"]

    @property
    def update_backup(
        self,
    ) -> Callable[[bigtable_table_admin.UpdateBackupRequest], table.Backup]:
        r"""Return a callable for the update backup method over gRPC.

        Updates a pending or completed Cloud Bigtable Backup.

        Returns:
            Callable[[~.UpdateBackupRequest],
                    ~.Backup]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_backup" not in self._stubs:
            self._stubs["update_backup"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/UpdateBackup",
                request_serializer=bigtable_table_admin.UpdateBackupRequest.serialize,
                response_deserializer=table.Backup.deserialize,
            )
        return self._stubs["update_backup"]

    @property
    def delete_backup(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteBackupRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete backup method over gRPC.

        Deletes a pending or completed Cloud Bigtable backup.

        Returns:
            Callable[[~.DeleteBackupRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_backup" not in self._stubs:
            self._stubs["delete_backup"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DeleteBackup",
                request_serializer=bigtable_table_admin.DeleteBackupRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_backup"]

    @property
    def list_backups(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListBackupsRequest],
        bigtable_table_admin.ListBackupsResponse,
    ]:
        r"""Return a callable for the list backups method over gRPC.

        Lists Cloud Bigtable backups. Returns both completed
        and pending backups.

        Returns:
            Callable[[~.ListBackupsRequest],
                    ~.ListBackupsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_backups" not in self._stubs:
            self._stubs["list_backups"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ListBackups",
                request_serializer=bigtable_table_admin.ListBackupsRequest.serialize,
                response_deserializer=bigtable_table_admin.ListBackupsResponse.deserialize,
            )
        return self._stubs["list_backups"]

    @property
    def restore_table(
        self,
    ) -> Callable[[bigtable_table_admin.RestoreTableRequest], operations_pb2.Operation]:
        r"""Return a callable for the restore table method over gRPC.

        Create a new table by restoring from a completed backup. The
        returned table [long-running
        operation][google.longrunning.Operation] can be used to track
        the progress of the operation, and to cancel it. The
        [metadata][google.longrunning.Operation.metadata] field type is
        [RestoreTableMetadata][google.bigtable.admin.v2.RestoreTableMetadata].
        The [response][google.longrunning.Operation.response] type is
        [Table][google.bigtable.admin.v2.Table], if successful.

        Returns:
            Callable[[~.RestoreTableRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "restore_table" not in self._stubs:
            self._stubs["restore_table"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/RestoreTable",
                request_serializer=bigtable_table_admin.RestoreTableRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["restore_table"]

    @property
    def copy_backup(
        self,
    ) -> Callable[[bigtable_table_admin.CopyBackupRequest], operations_pb2.Operation]:
        r"""Return a callable for the copy backup method over gRPC.

        Copy a Cloud Bigtable backup to a new backup in the
        destination cluster located in the destination instance
        and project.

        Returns:
            Callable[[~.CopyBackupRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "copy_backup" not in self._stubs:
            self._stubs["copy_backup"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CopyBackup",
                request_serializer=bigtable_table_admin.CopyBackupRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["copy_backup"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the get iam policy method over gRPC.

        Gets the access control policy for a Bigtable
        resource. Returns an empty policy if the resource exists
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
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the set iam policy method over gRPC.

        Sets the access control policy on a Bigtable
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
                "/google.bigtable.admin.v2.BigtableTableAdmin/SetIamPolicy",
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
        specified Bigtable resource.

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
                "/google.bigtable.admin.v2.BigtableTableAdmin/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def create_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateSchemaBundleRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create schema bundle method over gRPC.

        Creates a new schema bundle in the specified table.

        Returns:
            Callable[[~.CreateSchemaBundleRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_schema_bundle" not in self._stubs:
            self._stubs["create_schema_bundle"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/CreateSchemaBundle",
                request_serializer=bigtable_table_admin.CreateSchemaBundleRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_schema_bundle"]

    @property
    def update_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateSchemaBundleRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update schema bundle method over gRPC.

        Updates a schema bundle in the specified table.

        Returns:
            Callable[[~.UpdateSchemaBundleRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_schema_bundle" not in self._stubs:
            self._stubs["update_schema_bundle"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/UpdateSchemaBundle",
                request_serializer=bigtable_table_admin.UpdateSchemaBundleRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_schema_bundle"]

    @property
    def get_schema_bundle(
        self,
    ) -> Callable[[bigtable_table_admin.GetSchemaBundleRequest], table.SchemaBundle]:
        r"""Return a callable for the get schema bundle method over gRPC.

        Gets metadata information about the specified schema
        bundle.

        Returns:
            Callable[[~.GetSchemaBundleRequest],
                    ~.SchemaBundle]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_schema_bundle" not in self._stubs:
            self._stubs["get_schema_bundle"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/GetSchemaBundle",
                request_serializer=bigtable_table_admin.GetSchemaBundleRequest.serialize,
                response_deserializer=table.SchemaBundle.deserialize,
            )
        return self._stubs["get_schema_bundle"]

    @property
    def list_schema_bundles(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSchemaBundlesRequest],
        bigtable_table_admin.ListSchemaBundlesResponse,
    ]:
        r"""Return a callable for the list schema bundles method over gRPC.

        Lists all schema bundles associated with the
        specified table.

        Returns:
            Callable[[~.ListSchemaBundlesRequest],
                    ~.ListSchemaBundlesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_schema_bundles" not in self._stubs:
            self._stubs["list_schema_bundles"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/ListSchemaBundles",
                request_serializer=bigtable_table_admin.ListSchemaBundlesRequest.serialize,
                response_deserializer=bigtable_table_admin.ListSchemaBundlesResponse.deserialize,
            )
        return self._stubs["list_schema_bundles"]

    @property
    def delete_schema_bundle(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteSchemaBundleRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete schema bundle method over gRPC.

        Deletes a schema bundle in the specified table.

        Returns:
            Callable[[~.DeleteSchemaBundleRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_schema_bundle" not in self._stubs:
            self._stubs["delete_schema_bundle"] = self._logged_channel.unary_unary(
                "/google.bigtable.admin.v2.BigtableTableAdmin/DeleteSchemaBundle",
                request_serializer=bigtable_table_admin.DeleteSchemaBundleRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_schema_bundle"]

    def close(self):
        self._logged_channel.close()

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("BigtableTableAdminGrpcTransport",)

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
from google.api_core import gapic_v1
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message

import grpc  # type: ignore
import proto  # type: ignore

from google.cloud.bigtable_v2.types import bigtable
from .base import BigtableTransport, DEFAULT_CLIENT_INFO

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
                    "serviceName": "google.bigtable.v2.Bigtable",
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
                    "serviceName": "google.bigtable.v2.Bigtable",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class BigtableGrpcTransport(BigtableTransport):
    """gRPC backend transport for Bigtable.

    Service for reading from and writing to existing Bigtable
    tables.

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
        host: str = "bigtable.googleapis.com",
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
                 The hostname to connect to (default: 'bigtable.googleapis.com').
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
        host: str = "bigtable.googleapis.com",
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
    def read_rows(
        self,
    ) -> Callable[[bigtable.ReadRowsRequest], bigtable.ReadRowsResponse]:
        r"""Return a callable for the read rows method over gRPC.

        Streams back the contents of all requested rows in
        key order, optionally applying the same Reader filter to
        each. Depending on their size, rows and cells may be
        broken up across multiple responses, but atomicity of
        each row will still be preserved. See the
        ReadRowsResponse documentation for details.

        Returns:
            Callable[[~.ReadRowsRequest],
                    ~.ReadRowsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_rows" not in self._stubs:
            self._stubs["read_rows"] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/ReadRows",
                request_serializer=bigtable.ReadRowsRequest.serialize,
                response_deserializer=bigtable.ReadRowsResponse.deserialize,
            )
        return self._stubs["read_rows"]

    @property
    def sample_row_keys(
        self,
    ) -> Callable[[bigtable.SampleRowKeysRequest], bigtable.SampleRowKeysResponse]:
        r"""Return a callable for the sample row keys method over gRPC.

        Returns a sample of row keys in the table. The
        returned row keys will delimit contiguous sections of
        the table of approximately equal size, which can be used
        to break up the data for distributed tasks like
        mapreduces.

        Returns:
            Callable[[~.SampleRowKeysRequest],
                    ~.SampleRowKeysResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "sample_row_keys" not in self._stubs:
            self._stubs["sample_row_keys"] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/SampleRowKeys",
                request_serializer=bigtable.SampleRowKeysRequest.serialize,
                response_deserializer=bigtable.SampleRowKeysResponse.deserialize,
            )
        return self._stubs["sample_row_keys"]

    @property
    def mutate_row(
        self,
    ) -> Callable[[bigtable.MutateRowRequest], bigtable.MutateRowResponse]:
        r"""Return a callable for the mutate row method over gRPC.

        Mutates a row atomically. Cells already present in the row are
        left unchanged unless explicitly changed by ``mutation``.

        Returns:
            Callable[[~.MutateRowRequest],
                    ~.MutateRowResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "mutate_row" not in self._stubs:
            self._stubs["mutate_row"] = self._logged_channel.unary_unary(
                "/google.bigtable.v2.Bigtable/MutateRow",
                request_serializer=bigtable.MutateRowRequest.serialize,
                response_deserializer=bigtable.MutateRowResponse.deserialize,
            )
        return self._stubs["mutate_row"]

    @property
    def mutate_rows(
        self,
    ) -> Callable[[bigtable.MutateRowsRequest], bigtable.MutateRowsResponse]:
        r"""Return a callable for the mutate rows method over gRPC.

        Mutates multiple rows in a batch. Each individual row
        is mutated atomically as in MutateRow, but the entire
        batch is not executed atomically.

        Returns:
            Callable[[~.MutateRowsRequest],
                    ~.MutateRowsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "mutate_rows" not in self._stubs:
            self._stubs["mutate_rows"] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/MutateRows",
                request_serializer=bigtable.MutateRowsRequest.serialize,
                response_deserializer=bigtable.MutateRowsResponse.deserialize,
            )
        return self._stubs["mutate_rows"]

    @property
    def check_and_mutate_row(
        self,
    ) -> Callable[
        [bigtable.CheckAndMutateRowRequest], bigtable.CheckAndMutateRowResponse
    ]:
        r"""Return a callable for the check and mutate row method over gRPC.

        Mutates a row atomically based on the output of a
        predicate Reader filter.

        Returns:
            Callable[[~.CheckAndMutateRowRequest],
                    ~.CheckAndMutateRowResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "check_and_mutate_row" not in self._stubs:
            self._stubs["check_and_mutate_row"] = self._logged_channel.unary_unary(
                "/google.bigtable.v2.Bigtable/CheckAndMutateRow",
                request_serializer=bigtable.CheckAndMutateRowRequest.serialize,
                response_deserializer=bigtable.CheckAndMutateRowResponse.deserialize,
            )
        return self._stubs["check_and_mutate_row"]

    @property
    def ping_and_warm(
        self,
    ) -> Callable[[bigtable.PingAndWarmRequest], bigtable.PingAndWarmResponse]:
        r"""Return a callable for the ping and warm method over gRPC.

        Warm up associated instance metadata for this
        connection. This call is not required but may be useful
        for connection keep-alive.

        Returns:
            Callable[[~.PingAndWarmRequest],
                    ~.PingAndWarmResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "ping_and_warm" not in self._stubs:
            self._stubs["ping_and_warm"] = self._logged_channel.unary_unary(
                "/google.bigtable.v2.Bigtable/PingAndWarm",
                request_serializer=bigtable.PingAndWarmRequest.serialize,
                response_deserializer=bigtable.PingAndWarmResponse.deserialize,
            )
        return self._stubs["ping_and_warm"]

    @property
    def read_modify_write_row(
        self,
    ) -> Callable[
        [bigtable.ReadModifyWriteRowRequest], bigtable.ReadModifyWriteRowResponse
    ]:
        r"""Return a callable for the read modify write row method over gRPC.

        Modifies a row atomically on the server. The method
        reads the latest existing timestamp and value from the
        specified columns and writes a new entry based on
        pre-defined read/modify/write rules. The new value for
        the timestamp is the greater of the existing timestamp
        or the current server time. The method returns the new
        contents of all modified cells.

        Returns:
            Callable[[~.ReadModifyWriteRowRequest],
                    ~.ReadModifyWriteRowResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_modify_write_row" not in self._stubs:
            self._stubs["read_modify_write_row"] = self._logged_channel.unary_unary(
                "/google.bigtable.v2.Bigtable/ReadModifyWriteRow",
                request_serializer=bigtable.ReadModifyWriteRowRequest.serialize,
                response_deserializer=bigtable.ReadModifyWriteRowResponse.deserialize,
            )
        return self._stubs["read_modify_write_row"]

    @property
    def generate_initial_change_stream_partitions(
        self,
    ) -> Callable[
        [bigtable.GenerateInitialChangeStreamPartitionsRequest],
        bigtable.GenerateInitialChangeStreamPartitionsResponse,
    ]:
        r"""Return a callable for the generate initial change stream
        partitions method over gRPC.

        Returns the current list of partitions that make up the table's
        change stream. The union of partitions will cover the entire
        keyspace. Partitions can be read with ``ReadChangeStream``.
        NOTE: This API is only intended to be used by Apache Beam
        BigtableIO.

        Returns:
            Callable[[~.GenerateInitialChangeStreamPartitionsRequest],
                    ~.GenerateInitialChangeStreamPartitionsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "generate_initial_change_stream_partitions" not in self._stubs:
            self._stubs[
                "generate_initial_change_stream_partitions"
            ] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/GenerateInitialChangeStreamPartitions",
                request_serializer=bigtable.GenerateInitialChangeStreamPartitionsRequest.serialize,
                response_deserializer=bigtable.GenerateInitialChangeStreamPartitionsResponse.deserialize,
            )
        return self._stubs["generate_initial_change_stream_partitions"]

    @property
    def read_change_stream(
        self,
    ) -> Callable[
        [bigtable.ReadChangeStreamRequest], bigtable.ReadChangeStreamResponse
    ]:
        r"""Return a callable for the read change stream method over gRPC.

        Reads changes from a table's change stream. Changes
        will reflect both user-initiated mutations and mutations
        that are caused by garbage collection.
        NOTE: This API is only intended to be used by Apache
        Beam BigtableIO.

        Returns:
            Callable[[~.ReadChangeStreamRequest],
                    ~.ReadChangeStreamResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_change_stream" not in self._stubs:
            self._stubs["read_change_stream"] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/ReadChangeStream",
                request_serializer=bigtable.ReadChangeStreamRequest.serialize,
                response_deserializer=bigtable.ReadChangeStreamResponse.deserialize,
            )
        return self._stubs["read_change_stream"]

    @property
    def prepare_query(
        self,
    ) -> Callable[[bigtable.PrepareQueryRequest], bigtable.PrepareQueryResponse]:
        r"""Return a callable for the prepare query method over gRPC.

        Prepares a GoogleSQL query for execution on a
        particular Bigtable instance.

        Returns:
            Callable[[~.PrepareQueryRequest],
                    ~.PrepareQueryResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "prepare_query" not in self._stubs:
            self._stubs["prepare_query"] = self._logged_channel.unary_unary(
                "/google.bigtable.v2.Bigtable/PrepareQuery",
                request_serializer=bigtable.PrepareQueryRequest.serialize,
                response_deserializer=bigtable.PrepareQueryResponse.deserialize,
            )
        return self._stubs["prepare_query"]

    @property
    def execute_query(
        self,
    ) -> Callable[[bigtable.ExecuteQueryRequest], bigtable.ExecuteQueryResponse]:
        r"""Return a callable for the execute query method over gRPC.

        Executes a SQL query against a particular Bigtable
        instance.

        Returns:
            Callable[[~.ExecuteQueryRequest],
                    ~.ExecuteQueryResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "execute_query" not in self._stubs:
            self._stubs["execute_query"] = self._logged_channel.unary_stream(
                "/google.bigtable.v2.Bigtable/ExecuteQuery",
                request_serializer=bigtable.ExecuteQueryRequest.serialize,
                response_deserializer=bigtable.ExecuteQueryResponse.deserialize,
            )
        return self._stubs["execute_query"]

    def close(self):
        self._logged_channel.close()

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("BigtableGrpcTransport",)

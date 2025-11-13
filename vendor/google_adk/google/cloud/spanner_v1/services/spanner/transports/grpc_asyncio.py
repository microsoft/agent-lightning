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
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message

import grpc  # type: ignore
import proto  # type: ignore
from grpc.experimental import aio  # type: ignore

from google.cloud.spanner_v1.types import commit_response
from google.cloud.spanner_v1.types import result_set
from google.cloud.spanner_v1.types import spanner
from google.cloud.spanner_v1.types import transaction
from google.cloud.spanner_v1.metrics.metrics_interceptor import MetricsInterceptor
from google.protobuf import empty_pb2  # type: ignore
from .base import SpannerTransport, DEFAULT_CLIENT_INFO
from .grpc import SpannerGrpcTransport

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
                    "serviceName": "google.spanner.v1.Spanner",
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
                    "serviceName": "google.spanner.v1.Spanner",
                    "rpcName": str(client_call_details.method),
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class SpannerGrpcAsyncIOTransport(SpannerTransport):
    """gRPC AsyncIO backend transport for Spanner.

    Cloud Spanner API

    The Cloud Spanner API can be used to manage sessions and execute
    transactions on data stored in Cloud Spanner databases.

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
        metrics_interceptor: Optional[MetricsInterceptor] = None,
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
    def create_session(
        self,
    ) -> Callable[[spanner.CreateSessionRequest], Awaitable[spanner.Session]]:
        r"""Return a callable for the create session method over gRPC.

        Creates a new session. A session can be used to perform
        transactions that read and/or modify data in a Cloud Spanner
        database. Sessions are meant to be reused for many consecutive
        transactions.

        Sessions can only execute one transaction at a time. To execute
        multiple concurrent read-write/write-only transactions, create
        multiple sessions. Note that standalone reads and queries use a
        transaction internally, and count toward the one transaction
        limit.

        Active sessions use additional server resources, so it's a good
        idea to delete idle and unneeded sessions. Aside from explicit
        deletes, Cloud Spanner can delete sessions when no operations
        are sent for more than an hour. If a session is deleted,
        requests to it return ``NOT_FOUND``.

        Idle sessions can be kept alive by sending a trivial SQL query
        periodically, for example, ``"SELECT 1"``.

        Returns:
            Callable[[~.CreateSessionRequest],
                    Awaitable[~.Session]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_session" not in self._stubs:
            self._stubs["create_session"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/CreateSession",
                request_serializer=spanner.CreateSessionRequest.serialize,
                response_deserializer=spanner.Session.deserialize,
            )
        return self._stubs["create_session"]

    @property
    def batch_create_sessions(
        self,
    ) -> Callable[
        [spanner.BatchCreateSessionsRequest],
        Awaitable[spanner.BatchCreateSessionsResponse],
    ]:
        r"""Return a callable for the batch create sessions method over gRPC.

        Creates multiple new sessions.

        This API can be used to initialize a session cache on
        the clients. See https://goo.gl/TgSFN2 for best
        practices on session cache management.

        Returns:
            Callable[[~.BatchCreateSessionsRequest],
                    Awaitable[~.BatchCreateSessionsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_sessions" not in self._stubs:
            self._stubs["batch_create_sessions"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/BatchCreateSessions",
                request_serializer=spanner.BatchCreateSessionsRequest.serialize,
                response_deserializer=spanner.BatchCreateSessionsResponse.deserialize,
            )
        return self._stubs["batch_create_sessions"]

    @property
    def get_session(
        self,
    ) -> Callable[[spanner.GetSessionRequest], Awaitable[spanner.Session]]:
        r"""Return a callable for the get session method over gRPC.

        Gets a session. Returns ``NOT_FOUND`` if the session doesn't
        exist. This is mainly useful for determining whether a session
        is still alive.

        Returns:
            Callable[[~.GetSessionRequest],
                    Awaitable[~.Session]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_session" not in self._stubs:
            self._stubs["get_session"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/GetSession",
                request_serializer=spanner.GetSessionRequest.serialize,
                response_deserializer=spanner.Session.deserialize,
            )
        return self._stubs["get_session"]

    @property
    def list_sessions(
        self,
    ) -> Callable[
        [spanner.ListSessionsRequest], Awaitable[spanner.ListSessionsResponse]
    ]:
        r"""Return a callable for the list sessions method over gRPC.

        Lists all sessions in a given database.

        Returns:
            Callable[[~.ListSessionsRequest],
                    Awaitable[~.ListSessionsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_sessions" not in self._stubs:
            self._stubs["list_sessions"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/ListSessions",
                request_serializer=spanner.ListSessionsRequest.serialize,
                response_deserializer=spanner.ListSessionsResponse.deserialize,
            )
        return self._stubs["list_sessions"]

    @property
    def delete_session(
        self,
    ) -> Callable[[spanner.DeleteSessionRequest], Awaitable[empty_pb2.Empty]]:
        r"""Return a callable for the delete session method over gRPC.

        Ends a session, releasing server resources associated
        with it. This asynchronously triggers the cancellation
        of any operations that are running with this session.

        Returns:
            Callable[[~.DeleteSessionRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_session" not in self._stubs:
            self._stubs["delete_session"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/DeleteSession",
                request_serializer=spanner.DeleteSessionRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_session"]

    @property
    def execute_sql(
        self,
    ) -> Callable[[spanner.ExecuteSqlRequest], Awaitable[result_set.ResultSet]]:
        r"""Return a callable for the execute sql method over gRPC.

        Executes an SQL statement, returning all results in a single
        reply. This method can't be used to return a result set larger
        than 10 MiB; if the query yields more data than that, the query
        fails with a ``FAILED_PRECONDITION`` error.

        Operations inside read-write transactions might return
        ``ABORTED``. If this occurs, the application should restart the
        transaction from the beginning. See
        [Transaction][google.spanner.v1.Transaction] for more details.

        Larger result sets can be fetched in streaming fashion by
        calling
        [ExecuteStreamingSql][google.spanner.v1.Spanner.ExecuteStreamingSql]
        instead.

        The query string can be SQL or `Graph Query Language
        (GQL) <https://cloud.google.com/spanner/docs/reference/standard-sql/graph-intro>`__.

        Returns:
            Callable[[~.ExecuteSqlRequest],
                    Awaitable[~.ResultSet]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "execute_sql" not in self._stubs:
            self._stubs["execute_sql"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/ExecuteSql",
                request_serializer=spanner.ExecuteSqlRequest.serialize,
                response_deserializer=result_set.ResultSet.deserialize,
            )
        return self._stubs["execute_sql"]

    @property
    def execute_streaming_sql(
        self,
    ) -> Callable[[spanner.ExecuteSqlRequest], Awaitable[result_set.PartialResultSet]]:
        r"""Return a callable for the execute streaming sql method over gRPC.

        Like [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql], except
        returns the result set as a stream. Unlike
        [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql], there is no
        limit on the size of the returned result set. However, no
        individual row in the result set can exceed 100 MiB, and no
        column value can exceed 10 MiB.

        The query string can be SQL or `Graph Query Language
        (GQL) <https://cloud.google.com/spanner/docs/reference/standard-sql/graph-intro>`__.

        Returns:
            Callable[[~.ExecuteSqlRequest],
                    Awaitable[~.PartialResultSet]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "execute_streaming_sql" not in self._stubs:
            self._stubs["execute_streaming_sql"] = self._logged_channel.unary_stream(
                "/google.spanner.v1.Spanner/ExecuteStreamingSql",
                request_serializer=spanner.ExecuteSqlRequest.serialize,
                response_deserializer=result_set.PartialResultSet.deserialize,
            )
        return self._stubs["execute_streaming_sql"]

    @property
    def execute_batch_dml(
        self,
    ) -> Callable[
        [spanner.ExecuteBatchDmlRequest], Awaitable[spanner.ExecuteBatchDmlResponse]
    ]:
        r"""Return a callable for the execute batch dml method over gRPC.

        Executes a batch of SQL DML statements. This method allows many
        statements to be run with lower latency than submitting them
        sequentially with
        [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql].

        Statements are executed in sequential order. A request can
        succeed even if a statement fails. The
        [ExecuteBatchDmlResponse.status][google.spanner.v1.ExecuteBatchDmlResponse.status]
        field in the response provides information about the statement
        that failed. Clients must inspect this field to determine
        whether an error occurred.

        Execution stops after the first failed statement; the remaining
        statements are not executed.

        Returns:
            Callable[[~.ExecuteBatchDmlRequest],
                    Awaitable[~.ExecuteBatchDmlResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "execute_batch_dml" not in self._stubs:
            self._stubs["execute_batch_dml"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/ExecuteBatchDml",
                request_serializer=spanner.ExecuteBatchDmlRequest.serialize,
                response_deserializer=spanner.ExecuteBatchDmlResponse.deserialize,
            )
        return self._stubs["execute_batch_dml"]

    @property
    def read(self) -> Callable[[spanner.ReadRequest], Awaitable[result_set.ResultSet]]:
        r"""Return a callable for the read method over gRPC.

        Reads rows from the database using key lookups and scans, as a
        simple key/value style alternative to
        [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql]. This method
        can't be used to return a result set larger than 10 MiB; if the
        read matches more data than that, the read fails with a
        ``FAILED_PRECONDITION`` error.

        Reads inside read-write transactions might return ``ABORTED``.
        If this occurs, the application should restart the transaction
        from the beginning. See
        [Transaction][google.spanner.v1.Transaction] for more details.

        Larger result sets can be yielded in streaming fashion by
        calling [StreamingRead][google.spanner.v1.Spanner.StreamingRead]
        instead.

        Returns:
            Callable[[~.ReadRequest],
                    Awaitable[~.ResultSet]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read" not in self._stubs:
            self._stubs["read"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/Read",
                request_serializer=spanner.ReadRequest.serialize,
                response_deserializer=result_set.ResultSet.deserialize,
            )
        return self._stubs["read"]

    @property
    def streaming_read(
        self,
    ) -> Callable[[spanner.ReadRequest], Awaitable[result_set.PartialResultSet]]:
        r"""Return a callable for the streaming read method over gRPC.

        Like [Read][google.spanner.v1.Spanner.Read], except returns the
        result set as a stream. Unlike
        [Read][google.spanner.v1.Spanner.Read], there is no limit on the
        size of the returned result set. However, no individual row in
        the result set can exceed 100 MiB, and no column value can
        exceed 10 MiB.

        Returns:
            Callable[[~.ReadRequest],
                    Awaitable[~.PartialResultSet]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "streaming_read" not in self._stubs:
            self._stubs["streaming_read"] = self._logged_channel.unary_stream(
                "/google.spanner.v1.Spanner/StreamingRead",
                request_serializer=spanner.ReadRequest.serialize,
                response_deserializer=result_set.PartialResultSet.deserialize,
            )
        return self._stubs["streaming_read"]

    @property
    def begin_transaction(
        self,
    ) -> Callable[
        [spanner.BeginTransactionRequest], Awaitable[transaction.Transaction]
    ]:
        r"""Return a callable for the begin transaction method over gRPC.

        Begins a new transaction. This step can often be skipped:
        [Read][google.spanner.v1.Spanner.Read],
        [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql] and
        [Commit][google.spanner.v1.Spanner.Commit] can begin a new
        transaction as a side-effect.

        Returns:
            Callable[[~.BeginTransactionRequest],
                    Awaitable[~.Transaction]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "begin_transaction" not in self._stubs:
            self._stubs["begin_transaction"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/BeginTransaction",
                request_serializer=spanner.BeginTransactionRequest.serialize,
                response_deserializer=transaction.Transaction.deserialize,
            )
        return self._stubs["begin_transaction"]

    @property
    def commit(
        self,
    ) -> Callable[[spanner.CommitRequest], Awaitable[commit_response.CommitResponse]]:
        r"""Return a callable for the commit method over gRPC.

        Commits a transaction. The request includes the mutations to be
        applied to rows in the database.

        ``Commit`` might return an ``ABORTED`` error. This can occur at
        any time; commonly, the cause is conflicts with concurrent
        transactions. However, it can also happen for a variety of other
        reasons. If ``Commit`` returns ``ABORTED``, the caller should
        retry the transaction from the beginning, reusing the same
        session.

        On very rare occasions, ``Commit`` might return ``UNKNOWN``.
        This can happen, for example, if the client job experiences a 1+
        hour networking failure. At that point, Cloud Spanner has lost
        track of the transaction outcome and we recommend that you
        perform another read from the database to see the state of
        things as they are now.

        Returns:
            Callable[[~.CommitRequest],
                    Awaitable[~.CommitResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "commit" not in self._stubs:
            self._stubs["commit"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/Commit",
                request_serializer=spanner.CommitRequest.serialize,
                response_deserializer=commit_response.CommitResponse.deserialize,
            )
        return self._stubs["commit"]

    @property
    def rollback(
        self,
    ) -> Callable[[spanner.RollbackRequest], Awaitable[empty_pb2.Empty]]:
        r"""Return a callable for the rollback method over gRPC.

        Rolls back a transaction, releasing any locks it holds. It's a
        good idea to call this for any transaction that includes one or
        more [Read][google.spanner.v1.Spanner.Read] or
        [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql] requests and
        ultimately decides not to commit.

        ``Rollback`` returns ``OK`` if it successfully aborts the
        transaction, the transaction was already aborted, or the
        transaction isn't found. ``Rollback`` never returns ``ABORTED``.

        Returns:
            Callable[[~.RollbackRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "rollback" not in self._stubs:
            self._stubs["rollback"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/Rollback",
                request_serializer=spanner.RollbackRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["rollback"]

    @property
    def partition_query(
        self,
    ) -> Callable[
        [spanner.PartitionQueryRequest], Awaitable[spanner.PartitionResponse]
    ]:
        r"""Return a callable for the partition query method over gRPC.

        Creates a set of partition tokens that can be used to execute a
        query operation in parallel. Each of the returned partition
        tokens can be used by
        [ExecuteStreamingSql][google.spanner.v1.Spanner.ExecuteStreamingSql]
        to specify a subset of the query result to read. The same
        session and read-only transaction must be used by the
        ``PartitionQueryRequest`` used to create the partition tokens
        and the ``ExecuteSqlRequests`` that use the partition tokens.

        Partition tokens become invalid when the session used to create
        them is deleted, is idle for too long, begins a new transaction,
        or becomes too old. When any of these happen, it isn't possible
        to resume the query, and the whole operation must be restarted
        from the beginning.

        Returns:
            Callable[[~.PartitionQueryRequest],
                    Awaitable[~.PartitionResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "partition_query" not in self._stubs:
            self._stubs["partition_query"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/PartitionQuery",
                request_serializer=spanner.PartitionQueryRequest.serialize,
                response_deserializer=spanner.PartitionResponse.deserialize,
            )
        return self._stubs["partition_query"]

    @property
    def partition_read(
        self,
    ) -> Callable[[spanner.PartitionReadRequest], Awaitable[spanner.PartitionResponse]]:
        r"""Return a callable for the partition read method over gRPC.

        Creates a set of partition tokens that can be used to execute a
        read operation in parallel. Each of the returned partition
        tokens can be used by
        [StreamingRead][google.spanner.v1.Spanner.StreamingRead] to
        specify a subset of the read result to read. The same session
        and read-only transaction must be used by the
        ``PartitionReadRequest`` used to create the partition tokens and
        the ``ReadRequests`` that use the partition tokens. There are no
        ordering guarantees on rows returned among the returned
        partition tokens, or even within each individual
        ``StreamingRead`` call issued with a ``partition_token``.

        Partition tokens become invalid when the session used to create
        them is deleted, is idle for too long, begins a new transaction,
        or becomes too old. When any of these happen, it isn't possible
        to resume the read, and the whole operation must be restarted
        from the beginning.

        Returns:
            Callable[[~.PartitionReadRequest],
                    Awaitable[~.PartitionResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "partition_read" not in self._stubs:
            self._stubs["partition_read"] = self._logged_channel.unary_unary(
                "/google.spanner.v1.Spanner/PartitionRead",
                request_serializer=spanner.PartitionReadRequest.serialize,
                response_deserializer=spanner.PartitionResponse.deserialize,
            )
        return self._stubs["partition_read"]

    @property
    def batch_write(
        self,
    ) -> Callable[[spanner.BatchWriteRequest], Awaitable[spanner.BatchWriteResponse]]:
        r"""Return a callable for the batch write method over gRPC.

        Batches the supplied mutation groups in a collection of
        efficient transactions. All mutations in a group are committed
        atomically. However, mutations across groups can be committed
        non-atomically in an unspecified order and thus, they must be
        independent of each other. Partial failure is possible, that is,
        some groups might have been committed successfully, while some
        might have failed. The results of individual batches are
        streamed into the response as the batches are applied.

        ``BatchWrite`` requests are not replay protected, meaning that
        each mutation group can be applied more than once. Replays of
        non-idempotent mutations can have undesirable effects. For
        example, replays of an insert mutation can produce an already
        exists error or if you use generated or commit timestamp-based
        keys, it can result in additional rows being added to the
        mutation's table. We recommend structuring your mutation groups
        to be idempotent to avoid this issue.

        Returns:
            Callable[[~.BatchWriteRequest],
                    Awaitable[~.BatchWriteResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_write" not in self._stubs:
            self._stubs["batch_write"] = self._logged_channel.unary_stream(
                "/google.spanner.v1.Spanner/BatchWrite",
                request_serializer=spanner.BatchWriteRequest.serialize,
                response_deserializer=spanner.BatchWriteResponse.deserialize,
            )
        return self._stubs["batch_write"]

    def _prep_wrapped_messages(self, client_info):
        """Precompute the wrapped methods, overriding the base class method to use async wrappers."""
        self._wrapped_methods = {
            self.create_session: self._wrap_method(
                self.create_session,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.batch_create_sessions: self._wrap_method(
                self.batch_create_sessions,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_session: self._wrap_method(
                self.get_session,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.list_sessions: self._wrap_method(
                self.list_sessions,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.delete_session: self._wrap_method(
                self.delete_session,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.execute_sql: self._wrap_method(
                self.execute_sql,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.execute_streaming_sql: self._wrap_method(
                self.execute_streaming_sql,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.execute_batch_dml: self._wrap_method(
                self.execute_batch_dml,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.read: self._wrap_method(
                self.read,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.streaming_read: self._wrap_method(
                self.streaming_read,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.begin_transaction: self._wrap_method(
                self.begin_transaction,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.commit: self._wrap_method(
                self.commit,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.rollback: self._wrap_method(
                self.rollback,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.partition_query: self._wrap_method(
                self.partition_query,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.partition_read: self._wrap_method(
                self.partition_read,
                default_retry=retries.AsyncRetry(
                    initial=0.25,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ResourceExhausted,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.batch_write: self._wrap_method(
                self.batch_write,
                default_timeout=3600.0,
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


__all__ = ("SpannerGrpcAsyncIOTransport",)

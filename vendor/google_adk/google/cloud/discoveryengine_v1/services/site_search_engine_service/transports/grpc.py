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
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, grpc_helpers, operations_v1
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message
import grpc  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import (
    site_search_engine,
    site_search_engine_service,
)

from .base import DEFAULT_CLIENT_INFO, SiteSearchEngineServiceTransport

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
                    "serviceName": "google.cloud.discoveryengine.v1.SiteSearchEngineService",
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
                    "serviceName": "google.cloud.discoveryengine.v1.SiteSearchEngineService",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class SiteSearchEngineServiceGrpcTransport(SiteSearchEngineServiceTransport):
    """gRPC backend transport for SiteSearchEngineService.

    Service for managing site search related resources.

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
        host: str = "discoveryengine.googleapis.com",
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
                 The hostname to connect to (default: 'discoveryengine.googleapis.com').
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
        host: str = "discoveryengine.googleapis.com",
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
    def get_site_search_engine(
        self,
    ) -> Callable[
        [site_search_engine_service.GetSiteSearchEngineRequest],
        site_search_engine.SiteSearchEngine,
    ]:
        r"""Return a callable for the get site search engine method over gRPC.

        Gets the
        [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine].

        Returns:
            Callable[[~.GetSiteSearchEngineRequest],
                    ~.SiteSearchEngine]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_site_search_engine" not in self._stubs:
            self._stubs["get_site_search_engine"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/GetSiteSearchEngine",
                request_serializer=site_search_engine_service.GetSiteSearchEngineRequest.serialize,
                response_deserializer=site_search_engine.SiteSearchEngine.deserialize,
            )
        return self._stubs["get_site_search_engine"]

    @property
    def create_target_site(
        self,
    ) -> Callable[
        [site_search_engine_service.CreateTargetSiteRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create target site method over gRPC.

        Creates a
        [TargetSite][google.cloud.discoveryengine.v1.TargetSite].

        Returns:
            Callable[[~.CreateTargetSiteRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_target_site" not in self._stubs:
            self._stubs["create_target_site"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/CreateTargetSite",
                request_serializer=site_search_engine_service.CreateTargetSiteRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_target_site"]

    @property
    def batch_create_target_sites(
        self,
    ) -> Callable[
        [site_search_engine_service.BatchCreateTargetSitesRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the batch create target sites method over gRPC.

        Creates [TargetSite][google.cloud.discoveryengine.v1.TargetSite]
        in a batch.

        Returns:
            Callable[[~.BatchCreateTargetSitesRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_target_sites" not in self._stubs:
            self._stubs["batch_create_target_sites"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/BatchCreateTargetSites",
                request_serializer=site_search_engine_service.BatchCreateTargetSitesRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["batch_create_target_sites"]

    @property
    def get_target_site(
        self,
    ) -> Callable[
        [site_search_engine_service.GetTargetSiteRequest], site_search_engine.TargetSite
    ]:
        r"""Return a callable for the get target site method over gRPC.

        Gets a [TargetSite][google.cloud.discoveryengine.v1.TargetSite].

        Returns:
            Callable[[~.GetTargetSiteRequest],
                    ~.TargetSite]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_target_site" not in self._stubs:
            self._stubs["get_target_site"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/GetTargetSite",
                request_serializer=site_search_engine_service.GetTargetSiteRequest.serialize,
                response_deserializer=site_search_engine.TargetSite.deserialize,
            )
        return self._stubs["get_target_site"]

    @property
    def update_target_site(
        self,
    ) -> Callable[
        [site_search_engine_service.UpdateTargetSiteRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update target site method over gRPC.

        Updates a
        [TargetSite][google.cloud.discoveryengine.v1.TargetSite].

        Returns:
            Callable[[~.UpdateTargetSiteRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_target_site" not in self._stubs:
            self._stubs["update_target_site"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/UpdateTargetSite",
                request_serializer=site_search_engine_service.UpdateTargetSiteRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_target_site"]

    @property
    def delete_target_site(
        self,
    ) -> Callable[
        [site_search_engine_service.DeleteTargetSiteRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete target site method over gRPC.

        Deletes a
        [TargetSite][google.cloud.discoveryengine.v1.TargetSite].

        Returns:
            Callable[[~.DeleteTargetSiteRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_target_site" not in self._stubs:
            self._stubs["delete_target_site"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/DeleteTargetSite",
                request_serializer=site_search_engine_service.DeleteTargetSiteRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_target_site"]

    @property
    def list_target_sites(
        self,
    ) -> Callable[
        [site_search_engine_service.ListTargetSitesRequest],
        site_search_engine_service.ListTargetSitesResponse,
    ]:
        r"""Return a callable for the list target sites method over gRPC.

        Gets a list of
        [TargetSite][google.cloud.discoveryengine.v1.TargetSite]s.

        Returns:
            Callable[[~.ListTargetSitesRequest],
                    ~.ListTargetSitesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_target_sites" not in self._stubs:
            self._stubs["list_target_sites"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/ListTargetSites",
                request_serializer=site_search_engine_service.ListTargetSitesRequest.serialize,
                response_deserializer=site_search_engine_service.ListTargetSitesResponse.deserialize,
            )
        return self._stubs["list_target_sites"]

    @property
    def create_sitemap(
        self,
    ) -> Callable[
        [site_search_engine_service.CreateSitemapRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create sitemap method over gRPC.

        Creates a [Sitemap][google.cloud.discoveryengine.v1.Sitemap].

        Returns:
            Callable[[~.CreateSitemapRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_sitemap" not in self._stubs:
            self._stubs["create_sitemap"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/CreateSitemap",
                request_serializer=site_search_engine_service.CreateSitemapRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_sitemap"]

    @property
    def delete_sitemap(
        self,
    ) -> Callable[
        [site_search_engine_service.DeleteSitemapRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete sitemap method over gRPC.

        Deletes a [Sitemap][google.cloud.discoveryengine.v1.Sitemap].

        Returns:
            Callable[[~.DeleteSitemapRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_sitemap" not in self._stubs:
            self._stubs["delete_sitemap"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/DeleteSitemap",
                request_serializer=site_search_engine_service.DeleteSitemapRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_sitemap"]

    @property
    def fetch_sitemaps(
        self,
    ) -> Callable[
        [site_search_engine_service.FetchSitemapsRequest],
        site_search_engine_service.FetchSitemapsResponse,
    ]:
        r"""Return a callable for the fetch sitemaps method over gRPC.

        Fetch [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s in a
        [DataStore][google.cloud.discoveryengine.v1.DataStore].

        Returns:
            Callable[[~.FetchSitemapsRequest],
                    ~.FetchSitemapsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "fetch_sitemaps" not in self._stubs:
            self._stubs["fetch_sitemaps"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/FetchSitemaps",
                request_serializer=site_search_engine_service.FetchSitemapsRequest.serialize,
                response_deserializer=site_search_engine_service.FetchSitemapsResponse.deserialize,
            )
        return self._stubs["fetch_sitemaps"]

    @property
    def enable_advanced_site_search(
        self,
    ) -> Callable[
        [site_search_engine_service.EnableAdvancedSiteSearchRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the enable advanced site search method over gRPC.

        Upgrade from basic site search to advanced site
        search.

        Returns:
            Callable[[~.EnableAdvancedSiteSearchRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "enable_advanced_site_search" not in self._stubs:
            self._stubs[
                "enable_advanced_site_search"
            ] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/EnableAdvancedSiteSearch",
                request_serializer=site_search_engine_service.EnableAdvancedSiteSearchRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["enable_advanced_site_search"]

    @property
    def disable_advanced_site_search(
        self,
    ) -> Callable[
        [site_search_engine_service.DisableAdvancedSiteSearchRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the disable advanced site search method over gRPC.

        Downgrade from advanced site search to basic site
        search.

        Returns:
            Callable[[~.DisableAdvancedSiteSearchRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "disable_advanced_site_search" not in self._stubs:
            self._stubs[
                "disable_advanced_site_search"
            ] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/DisableAdvancedSiteSearch",
                request_serializer=site_search_engine_service.DisableAdvancedSiteSearchRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["disable_advanced_site_search"]

    @property
    def recrawl_uris(
        self,
    ) -> Callable[
        [site_search_engine_service.RecrawlUrisRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the recrawl uris method over gRPC.

        Request on-demand recrawl for a list of URIs.

        Returns:
            Callable[[~.RecrawlUrisRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "recrawl_uris" not in self._stubs:
            self._stubs["recrawl_uris"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/RecrawlUris",
                request_serializer=site_search_engine_service.RecrawlUrisRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["recrawl_uris"]

    @property
    def batch_verify_target_sites(
        self,
    ) -> Callable[
        [site_search_engine_service.BatchVerifyTargetSitesRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the batch verify target sites method over gRPC.

        Verify target sites' ownership and validity.
        This API sends all the target sites under site search
        engine for verification.

        Returns:
            Callable[[~.BatchVerifyTargetSitesRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_verify_target_sites" not in self._stubs:
            self._stubs["batch_verify_target_sites"] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/BatchVerifyTargetSites",
                request_serializer=site_search_engine_service.BatchVerifyTargetSitesRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["batch_verify_target_sites"]

    @property
    def fetch_domain_verification_status(
        self,
    ) -> Callable[
        [site_search_engine_service.FetchDomainVerificationStatusRequest],
        site_search_engine_service.FetchDomainVerificationStatusResponse,
    ]:
        r"""Return a callable for the fetch domain verification
        status method over gRPC.

        Returns list of target sites with its domain verification
        status. This method can only be called under data store with
        BASIC_SITE_SEARCH state at the moment.

        Returns:
            Callable[[~.FetchDomainVerificationStatusRequest],
                    ~.FetchDomainVerificationStatusResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "fetch_domain_verification_status" not in self._stubs:
            self._stubs[
                "fetch_domain_verification_status"
            ] = self._logged_channel.unary_unary(
                "/google.cloud.discoveryengine.v1.SiteSearchEngineService/FetchDomainVerificationStatus",
                request_serializer=site_search_engine_service.FetchDomainVerificationStatusRequest.serialize,
                response_deserializer=site_search_engine_service.FetchDomainVerificationStatusResponse.deserialize,
            )
        return self._stubs["fetch_domain_verification_status"]

    def close(self):
        self._logged_channel.close()

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

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("SiteSearchEngineServiceGrpcTransport",)

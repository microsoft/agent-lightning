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
from collections import OrderedDict
import logging as std_logging
import re
from typing import (
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry_async as retries
from google.api_core.client_options import ClientOptions
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.discoveryengine_v1beta import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore

from google.cloud.discoveryengine_v1beta.services.site_search_engine_service import (
    pagers,
)
from google.cloud.discoveryengine_v1beta.types import (
    site_search_engine,
    site_search_engine_service,
)

from .client import SiteSearchEngineServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, SiteSearchEngineServiceTransport
from .transports.grpc_asyncio import SiteSearchEngineServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class SiteSearchEngineServiceAsyncClient:
    """Service for managing site search related resources."""

    _client: SiteSearchEngineServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = SiteSearchEngineServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = SiteSearchEngineServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = (
        SiteSearchEngineServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    )
    _DEFAULT_UNIVERSE = SiteSearchEngineServiceClient._DEFAULT_UNIVERSE

    sitemap_path = staticmethod(SiteSearchEngineServiceClient.sitemap_path)
    parse_sitemap_path = staticmethod(SiteSearchEngineServiceClient.parse_sitemap_path)
    site_search_engine_path = staticmethod(
        SiteSearchEngineServiceClient.site_search_engine_path
    )
    parse_site_search_engine_path = staticmethod(
        SiteSearchEngineServiceClient.parse_site_search_engine_path
    )
    target_site_path = staticmethod(SiteSearchEngineServiceClient.target_site_path)
    parse_target_site_path = staticmethod(
        SiteSearchEngineServiceClient.parse_target_site_path
    )
    common_billing_account_path = staticmethod(
        SiteSearchEngineServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        SiteSearchEngineServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(SiteSearchEngineServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        SiteSearchEngineServiceClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        SiteSearchEngineServiceClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        SiteSearchEngineServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(
        SiteSearchEngineServiceClient.common_project_path
    )
    parse_common_project_path = staticmethod(
        SiteSearchEngineServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(
        SiteSearchEngineServiceClient.common_location_path
    )
    parse_common_location_path = staticmethod(
        SiteSearchEngineServiceClient.parse_common_location_path
    )

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            SiteSearchEngineServiceAsyncClient: The constructed client.
        """
        return SiteSearchEngineServiceClient.from_service_account_info.__func__(SiteSearchEngineServiceAsyncClient, info, *args, **kwargs)  # type: ignore

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            SiteSearchEngineServiceAsyncClient: The constructed client.
        """
        return SiteSearchEngineServiceClient.from_service_account_file.__func__(SiteSearchEngineServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

    from_service_account_json = from_service_account_file

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[ClientOptions] = None
    ):
        """Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """
        return SiteSearchEngineServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> SiteSearchEngineServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            SiteSearchEngineServiceTransport: The transport used by the client instance.
        """
        return self._client.transport

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._client._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used
                by the client instance.
        """
        return self._client._universe_domain

    get_transport_class = SiteSearchEngineServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                SiteSearchEngineServiceTransport,
                Callable[..., SiteSearchEngineServiceTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the site search engine service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,SiteSearchEngineServiceTransport,Callable[..., SiteSearchEngineServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the SiteSearchEngineServiceTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]):
                Custom options for the client.

                1. The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client when ``transport`` is
                not explicitly provided. Only if this property is not set and
                ``transport`` was not explicitly provided, the endpoint is
                determined by the GOOGLE_API_USE_MTLS_ENDPOINT environment
                variable, which have one of the following values:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto-switch to the
                default mTLS endpoint if client certificate is present; this is
                the default value).

                2. If the GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide a client certificate for mTLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

                3. The ``universe_domain`` property can be used to override the
                default "googleapis.com" universe. Note that ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client = SiteSearchEngineServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.cloud.discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient`.",
                extra={
                    "serviceName": "google.cloud.discoveryengine.v1beta.SiteSearchEngineService",
                    "universeDomain": getattr(
                        self._client._transport._credentials, "universe_domain", ""
                    ),
                    "credentialsType": f"{type(self._client._transport._credentials).__module__}.{type(self._client._transport._credentials).__qualname__}",
                    "credentialsInfo": getattr(
                        self.transport._credentials, "get_cred_info", lambda: None
                    )(),
                }
                if hasattr(self._client._transport, "_credentials")
                else {
                    "serviceName": "google.cloud.discoveryengine.v1beta.SiteSearchEngineService",
                    "credentialsType": None,
                },
            )

    async def get_site_search_engine(
        self,
        request: Optional[
            Union[site_search_engine_service.GetSiteSearchEngineRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> site_search_engine.SiteSearchEngine:
        r"""Gets the
        [SiteSearchEngine][google.cloud.discoveryengine.v1beta.SiteSearchEngine].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_get_site_search_engine():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.GetSiteSearchEngineRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_site_search_engine(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.GetSiteSearchEngineRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.GetSiteSearchEngine][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.GetSiteSearchEngine]
                method.
            name (:class:`str`):
                Required. Resource name of
                [SiteSearchEngine][google.cloud.discoveryengine.v1beta.SiteSearchEngine],
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.

                If the caller does not have permission to access the
                [SiteSearchEngine], regardless of whether or not it
                exists, a PERMISSION_DENIED error is returned.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.discoveryengine_v1beta.types.SiteSearchEngine:
                SiteSearchEngine captures DataStore
                level site search persisting
                configurations. It is a singleton value
                per data store.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.GetSiteSearchEngineRequest
        ):
            request = site_search_engine_service.GetSiteSearchEngineRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_site_search_engine
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def create_target_site(
        self,
        request: Optional[
            Union[site_search_engine_service.CreateTargetSiteRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        target_site: Optional[site_search_engine.TargetSite] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_create_target_site():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                target_site = discoveryengine_v1beta.TargetSite()
                target_site.provided_uri_pattern = "provided_uri_pattern_value"

                request = discoveryengine_v1beta.CreateTargetSiteRequest(
                    parent="parent_value",
                    target_site=target_site,
                )

                # Make the request
                operation = client.create_target_site(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.CreateTargetSiteRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.CreateTargetSite][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.CreateTargetSite]
                method.
            parent (:class:`str`):
                Required. Parent resource name of
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            target_site (:class:`google.cloud.discoveryengine_v1beta.types.TargetSite`):
                Required. The
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]
                to create.

                This corresponds to the ``target_site`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.discoveryengine_v1beta.types.TargetSite`
                A target site for the SiteSearchEngine.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, target_site]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.CreateTargetSiteRequest):
            request = site_search_engine_service.CreateTargetSiteRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if target_site is not None:
            request.target_site = target_site

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_target_site
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine.TargetSite,
            metadata_type=site_search_engine_service.CreateTargetSiteMetadata,
        )

        # Done; return the response.
        return response

    async def batch_create_target_sites(
        self,
        request: Optional[
            Union[site_search_engine_service.BatchCreateTargetSitesRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite] in
        a batch.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_batch_create_target_sites():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                requests = discoveryengine_v1beta.CreateTargetSiteRequest()
                requests.parent = "parent_value"
                requests.target_site.provided_uri_pattern = "provided_uri_pattern_value"

                request = discoveryengine_v1beta.BatchCreateTargetSitesRequest(
                    parent="parent_value",
                    requests=requests,
                )

                # Make the request
                operation = client.batch_create_target_sites(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.BatchCreateTargetSitesRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.BatchCreateTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.BatchCreateTargetSites]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.discoveryengine_v1beta.types.BatchCreateTargetSitesResponse` Response message for
                   [SiteSearchEngineService.BatchCreateTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.BatchCreateTargetSites]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.BatchCreateTargetSitesRequest
        ):
            request = site_search_engine_service.BatchCreateTargetSitesRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.batch_create_target_sites
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine_service.BatchCreateTargetSitesResponse,
            metadata_type=site_search_engine_service.BatchCreateTargetSiteMetadata,
        )

        # Done; return the response.
        return response

    async def get_target_site(
        self,
        request: Optional[
            Union[site_search_engine_service.GetTargetSiteRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> site_search_engine.TargetSite:
        r"""Gets a
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_get_target_site():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.GetTargetSiteRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_target_site(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.GetTargetSiteRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.GetTargetSite][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.GetTargetSite]
                method.
            name (:class:`str`):
                Required. Full resource name of
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/targetSites/{target_site}``.

                If the caller does not have permission to access the
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                regardless of whether or not it exists, a
                PERMISSION_DENIED error is returned.

                If the requested
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]
                does not exist, a NOT_FOUND error is returned.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.discoveryengine_v1beta.types.TargetSite:
                A target site for the
                SiteSearchEngine.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.GetTargetSiteRequest):
            request = site_search_engine_service.GetTargetSiteRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_target_site
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_target_site(
        self,
        request: Optional[
            Union[site_search_engine_service.UpdateTargetSiteRequest, dict]
        ] = None,
        *,
        target_site: Optional[site_search_engine.TargetSite] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_update_target_site():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                target_site = discoveryengine_v1beta.TargetSite()
                target_site.provided_uri_pattern = "provided_uri_pattern_value"

                request = discoveryengine_v1beta.UpdateTargetSiteRequest(
                    target_site=target_site,
                )

                # Make the request
                operation = client.update_target_site(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.UpdateTargetSiteRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.UpdateTargetSite][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.UpdateTargetSite]
                method.
            target_site (:class:`google.cloud.discoveryengine_v1beta.types.TargetSite`):
                Required. The target site to update. If the caller does
                not have permission to update the
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                regardless of whether or not it exists, a
                PERMISSION_DENIED error is returned.

                If the
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]
                to update does not exist, a NOT_FOUND error is returned.

                This corresponds to the ``target_site`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.discoveryengine_v1beta.types.TargetSite`
                A target site for the SiteSearchEngine.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [target_site]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.UpdateTargetSiteRequest):
            request = site_search_engine_service.UpdateTargetSiteRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if target_site is not None:
            request.target_site = target_site

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_target_site
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("target_site.name", request.target_site.name),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine.TargetSite,
            metadata_type=site_search_engine_service.UpdateTargetSiteMetadata,
        )

        # Done; return the response.
        return response

    async def delete_target_site(
        self,
        request: Optional[
            Union[site_search_engine_service.DeleteTargetSiteRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Deletes a
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_delete_target_site():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.DeleteTargetSiteRequest(
                    name="name_value",
                )

                # Make the request
                operation = client.delete_target_site(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.DeleteTargetSiteRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.DeleteTargetSite][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.DeleteTargetSite]
                method.
            name (:class:`str`):
                Required. Full resource name of
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/targetSites/{target_site}``.

                If the caller does not have permission to access the
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite],
                regardless of whether or not it exists, a
                PERMISSION_DENIED error is returned.

                If the requested
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]
                does not exist, a NOT_FOUND error is returned.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.protobuf.empty_pb2.Empty` A generic empty message that you can re-use to avoid defining duplicated
                   empty messages in your APIs. A typical example is to
                   use it as the request or the response type of an API
                   method. For instance:

                      service Foo {
                         rpc Bar(google.protobuf.Empty) returns
                         (google.protobuf.Empty);

                      }

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.DeleteTargetSiteRequest):
            request = site_search_engine_service.DeleteTargetSiteRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_target_site
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            empty_pb2.Empty,
            metadata_type=site_search_engine_service.DeleteTargetSiteMetadata,
        )

        # Done; return the response.
        return response

    async def list_target_sites(
        self,
        request: Optional[
            Union[site_search_engine_service.ListTargetSitesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListTargetSitesAsyncPager:
        r"""Gets a list of
        [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]s.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_list_target_sites():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.ListTargetSitesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_target_sites(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.ListTargetSitesRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.ListTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.ListTargetSites]
                method.
            parent (:class:`str`):
                Required. The parent site search engine resource name,
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.

                If the caller does not have permission to list
                [TargetSite][google.cloud.discoveryengine.v1beta.TargetSite]s
                under this site search engine, regardless of whether or
                not this branch exists, a PERMISSION_DENIED error is
                returned.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.discoveryengine_v1beta.services.site_search_engine_service.pagers.ListTargetSitesAsyncPager:
                Response message for
                   [SiteSearchEngineService.ListTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.ListTargetSites]
                   method.

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.ListTargetSitesRequest):
            request = site_search_engine_service.ListTargetSitesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_target_sites
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListTargetSitesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def create_sitemap(
        self,
        request: Optional[
            Union[site_search_engine_service.CreateSitemapRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        sitemap: Optional[site_search_engine.Sitemap] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a
        [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_create_sitemap():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                sitemap = discoveryengine_v1beta.Sitemap()
                sitemap.uri = "uri_value"

                request = discoveryengine_v1beta.CreateSitemapRequest(
                    parent="parent_value",
                    sitemap=sitemap,
                )

                # Make the request
                operation = client.create_sitemap(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.CreateSitemapRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.CreateSitemap][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.CreateSitemap]
                method.
            parent (:class:`str`):
                Required. Parent resource name of the
                [SiteSearchEngine][google.cloud.discoveryengine.v1beta.SiteSearchEngine],
                such as
                ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            sitemap (:class:`google.cloud.discoveryengine_v1beta.types.Sitemap`):
                Required. The
                [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap]
                to create.

                This corresponds to the ``sitemap`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.discoveryengine_v1beta.types.Sitemap`
                A sitemap for the SiteSearchEngine.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, sitemap]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.CreateSitemapRequest):
            request = site_search_engine_service.CreateSitemapRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if sitemap is not None:
            request.sitemap = sitemap

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_sitemap
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine.Sitemap,
            metadata_type=site_search_engine_service.CreateSitemapMetadata,
        )

        # Done; return the response.
        return response

    async def delete_sitemap(
        self,
        request: Optional[
            Union[site_search_engine_service.DeleteSitemapRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Deletes a
        [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_delete_sitemap():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.DeleteSitemapRequest(
                    name="name_value",
                )

                # Make the request
                operation = client.delete_sitemap(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.DeleteSitemapRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.DeleteSitemap][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.DeleteSitemap]
                method.
            name (:class:`str`):
                Required. Full resource name of
                [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap],
                such as
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/sitemaps/{sitemap}``.

                If the caller does not have permission to access the
                [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap],
                regardless of whether or not it exists, a
                PERMISSION_DENIED error is returned.

                If the requested
                [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap]
                does not exist, a NOT_FOUND error is returned.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.protobuf.empty_pb2.Empty` A generic empty message that you can re-use to avoid defining duplicated
                   empty messages in your APIs. A typical example is to
                   use it as the request or the response type of an API
                   method. For instance:

                      service Foo {
                         rpc Bar(google.protobuf.Empty) returns
                         (google.protobuf.Empty);

                      }

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.DeleteSitemapRequest):
            request = site_search_engine_service.DeleteSitemapRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_sitemap
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            empty_pb2.Empty,
            metadata_type=site_search_engine_service.DeleteSitemapMetadata,
        )

        # Done; return the response.
        return response

    async def fetch_sitemaps(
        self,
        request: Optional[
            Union[site_search_engine_service.FetchSitemapsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> site_search_engine_service.FetchSitemapsResponse:
        r"""Fetch [Sitemap][google.cloud.discoveryengine.v1beta.Sitemap]s in
        a [DataStore][google.cloud.discoveryengine.v1beta.DataStore].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_fetch_sitemaps():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.FetchSitemapsRequest(
                    parent="parent_value",
                )

                # Make the request
                response = await client.fetch_sitemaps(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.FetchSitemapsRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.FetchSitemaps][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.FetchSitemaps]
                method.
            parent (:class:`str`):
                Required. Parent resource name of the
                [SiteSearchEngine][google.cloud.discoveryengine.v1beta.SiteSearchEngine],
                such as
                ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.discoveryengine_v1beta.types.FetchSitemapsResponse:
                Response message for
                   [SiteSearchEngineService.FetchSitemaps][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.FetchSitemaps]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.FetchSitemapsRequest):
            request = site_search_engine_service.FetchSitemapsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.fetch_sitemaps
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def enable_advanced_site_search(
        self,
        request: Optional[
            Union[site_search_engine_service.EnableAdvancedSiteSearchRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Upgrade from basic site search to advanced site
        search.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_enable_advanced_site_search():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.EnableAdvancedSiteSearchRequest(
                    site_search_engine="site_search_engine_value",
                )

                # Make the request
                operation = client.enable_advanced_site_search(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.EnableAdvancedSiteSearchRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.EnableAdvancedSiteSearch][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.EnableAdvancedSiteSearch]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.discoveryengine_v1beta.types.EnableAdvancedSiteSearchResponse` Response message for
                   [SiteSearchEngineService.EnableAdvancedSiteSearch][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.EnableAdvancedSiteSearch]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.EnableAdvancedSiteSearchRequest
        ):
            request = site_search_engine_service.EnableAdvancedSiteSearchRequest(
                request
            )

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.enable_advanced_site_search
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("site_search_engine", request.site_search_engine),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine_service.EnableAdvancedSiteSearchResponse,
            metadata_type=site_search_engine_service.EnableAdvancedSiteSearchMetadata,
        )

        # Done; return the response.
        return response

    async def disable_advanced_site_search(
        self,
        request: Optional[
            Union[site_search_engine_service.DisableAdvancedSiteSearchRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Downgrade from advanced site search to basic site
        search.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_disable_advanced_site_search():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.DisableAdvancedSiteSearchRequest(
                    site_search_engine="site_search_engine_value",
                )

                # Make the request
                operation = client.disable_advanced_site_search(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.DisableAdvancedSiteSearchRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.DisableAdvancedSiteSearch][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.DisableAdvancedSiteSearch]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.discoveryengine_v1beta.types.DisableAdvancedSiteSearchResponse` Response message for
                   [SiteSearchEngineService.DisableAdvancedSiteSearch][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.DisableAdvancedSiteSearch]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.DisableAdvancedSiteSearchRequest
        ):
            request = site_search_engine_service.DisableAdvancedSiteSearchRequest(
                request
            )

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.disable_advanced_site_search
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("site_search_engine", request.site_search_engine),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine_service.DisableAdvancedSiteSearchResponse,
            metadata_type=site_search_engine_service.DisableAdvancedSiteSearchMetadata,
        )

        # Done; return the response.
        return response

    async def recrawl_uris(
        self,
        request: Optional[
            Union[site_search_engine_service.RecrawlUrisRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Request on-demand recrawl for a list of URIs.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_recrawl_uris():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.RecrawlUrisRequest(
                    site_search_engine="site_search_engine_value",
                    uris=['uris_value1', 'uris_value2'],
                )

                # Make the request
                operation = client.recrawl_uris(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.RecrawlUrisRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.RecrawlUris][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.RecrawlUris]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.discoveryengine_v1beta.types.RecrawlUrisResponse` Response message for
                   [SiteSearchEngineService.RecrawlUris][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.RecrawlUris]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, site_search_engine_service.RecrawlUrisRequest):
            request = site_search_engine_service.RecrawlUrisRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.recrawl_uris
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("site_search_engine", request.site_search_engine),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine_service.RecrawlUrisResponse,
            metadata_type=site_search_engine_service.RecrawlUrisMetadata,
        )

        # Done; return the response.
        return response

    async def batch_verify_target_sites(
        self,
        request: Optional[
            Union[site_search_engine_service.BatchVerifyTargetSitesRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Verify target sites' ownership and validity.
        This API sends all the target sites under site search
        engine for verification.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_batch_verify_target_sites():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.BatchVerifyTargetSitesRequest(
                    parent="parent_value",
                )

                # Make the request
                operation = client.batch_verify_target_sites(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.BatchVerifyTargetSitesRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.BatchVerifyTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.BatchVerifyTargetSites]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.discoveryengine_v1beta.types.BatchVerifyTargetSitesResponse` Response message for
                   [SiteSearchEngineService.BatchVerifyTargetSites][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.BatchVerifyTargetSites]
                   method.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.BatchVerifyTargetSitesRequest
        ):
            request = site_search_engine_service.BatchVerifyTargetSitesRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.batch_verify_target_sites
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            site_search_engine_service.BatchVerifyTargetSitesResponse,
            metadata_type=site_search_engine_service.BatchVerifyTargetSitesMetadata,
        )

        # Done; return the response.
        return response

    async def fetch_domain_verification_status(
        self,
        request: Optional[
            Union[site_search_engine_service.FetchDomainVerificationStatusRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.FetchDomainVerificationStatusAsyncPager:
        r"""Returns list of target sites with its domain verification
        status. This method can only be called under data store with
        BASIC_SITE_SEARCH state at the moment.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import discoveryengine_v1beta

            async def sample_fetch_domain_verification_status():
                # Create a client
                client = discoveryengine_v1beta.SiteSearchEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = discoveryengine_v1beta.FetchDomainVerificationStatusRequest(
                    site_search_engine="site_search_engine_value",
                )

                # Make the request
                page_result = client.fetch_domain_verification_status(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.discoveryengine_v1beta.types.FetchDomainVerificationStatusRequest, dict]]):
                The request object. Request message for
                [SiteSearchEngineService.FetchDomainVerificationStatus][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.FetchDomainVerificationStatus]
                method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.discoveryengine_v1beta.services.site_search_engine_service.pagers.FetchDomainVerificationStatusAsyncPager:
                Response message for
                   [SiteSearchEngineService.FetchDomainVerificationStatus][google.cloud.discoveryengine.v1beta.SiteSearchEngineService.FetchDomainVerificationStatus]
                   method.

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, site_search_engine_service.FetchDomainVerificationStatusRequest
        ):
            request = site_search_engine_service.FetchDomainVerificationStatusRequest(
                request
            )

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.fetch_domain_verification_status
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("site_search_engine", request.site_search_engine),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.FetchDomainVerificationStatusAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def list_operations(
        self,
        request: Optional[operations_pb2.ListOperationsRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operations_pb2.ListOperationsResponse:
        r"""Lists operations that match the specified filter in the request.

        Args:
            request (:class:`~.operations_pb2.ListOperationsRequest`):
                The request object. Request message for
                `ListOperations` method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            ~.operations_pb2.ListOperationsResponse:
                Response message for ``ListOperations`` method.
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.ListOperationsRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self.transport._wrapped_methods[self._client._transport.list_operations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_operation(
        self,
        request: Optional[operations_pb2.GetOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operations_pb2.Operation:
        r"""Gets the latest state of a long-running operation.

        Args:
            request (:class:`~.operations_pb2.GetOperationRequest`):
                The request object. Request message for
                `GetOperation` method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            ~.operations_pb2.Operation:
                An ``Operation`` object.
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.GetOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self.transport._wrapped_methods[self._client._transport.get_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def cancel_operation(
        self,
        request: Optional[operations_pb2.CancelOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Starts asynchronous cancellation on a long-running operation.

        The server makes a best effort to cancel the operation, but success
        is not guaranteed.  If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.

        Args:
            request (:class:`~.operations_pb2.CancelOperationRequest`):
                The request object. Request message for
                `CancelOperation` method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            None
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.CancelOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self.transport._wrapped_methods[self._client._transport.cancel_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    async def __aenter__(self) -> "SiteSearchEngineServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("SiteSearchEngineServiceAsyncClient",)

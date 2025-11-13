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

from google.cloud.monitoring_v3 import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api import monitored_resource_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore

from google.cloud.monitoring_v3.services.uptime_check_service import pagers
from google.cloud.monitoring_v3.types import uptime, uptime_service

from .client import UptimeCheckServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, UptimeCheckServiceTransport
from .transports.grpc_asyncio import UptimeCheckServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class UptimeCheckServiceAsyncClient:
    """The UptimeCheckService API is used to manage (list, create, delete,
    edit) Uptime check configurations in the Cloud Monitoring product.
    An Uptime check is a piece of configuration that determines which
    resources and services to monitor for availability. These
    configurations can also be configured interactively by navigating to
    the [Cloud console] (https://console.cloud.google.com), selecting
    the appropriate project, clicking on "Monitoring" on the left-hand
    side to navigate to Cloud Monitoring, and then clicking on "Uptime".
    """

    _client: UptimeCheckServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = UptimeCheckServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = UptimeCheckServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = UptimeCheckServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = UptimeCheckServiceClient._DEFAULT_UNIVERSE

    function_path = staticmethod(UptimeCheckServiceClient.function_path)
    parse_function_path = staticmethod(UptimeCheckServiceClient.parse_function_path)
    uptime_check_config_path = staticmethod(
        UptimeCheckServiceClient.uptime_check_config_path
    )
    parse_uptime_check_config_path = staticmethod(
        UptimeCheckServiceClient.parse_uptime_check_config_path
    )
    common_billing_account_path = staticmethod(
        UptimeCheckServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        UptimeCheckServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(UptimeCheckServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        UptimeCheckServiceClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        UptimeCheckServiceClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        UptimeCheckServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(UptimeCheckServiceClient.common_project_path)
    parse_common_project_path = staticmethod(
        UptimeCheckServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(UptimeCheckServiceClient.common_location_path)
    parse_common_location_path = staticmethod(
        UptimeCheckServiceClient.parse_common_location_path
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
            UptimeCheckServiceAsyncClient: The constructed client.
        """
        return UptimeCheckServiceClient.from_service_account_info.__func__(UptimeCheckServiceAsyncClient, info, *args, **kwargs)  # type: ignore

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
            UptimeCheckServiceAsyncClient: The constructed client.
        """
        return UptimeCheckServiceClient.from_service_account_file.__func__(UptimeCheckServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return UptimeCheckServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> UptimeCheckServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            UptimeCheckServiceTransport: The transport used by the client instance.
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

    get_transport_class = UptimeCheckServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                UptimeCheckServiceTransport,
                Callable[..., UptimeCheckServiceTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the uptime check service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,UptimeCheckServiceTransport,Callable[..., UptimeCheckServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the UptimeCheckServiceTransport constructor.
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
        self._client = UptimeCheckServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.monitoring_v3.UptimeCheckServiceAsyncClient`.",
                extra={
                    "serviceName": "google.monitoring.v3.UptimeCheckService",
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
                    "serviceName": "google.monitoring.v3.UptimeCheckService",
                    "credentialsType": None,
                },
            )

    async def list_uptime_check_configs(
        self,
        request: Optional[
            Union[uptime_service.ListUptimeCheckConfigsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListUptimeCheckConfigsAsyncPager:
        r"""Lists the existing valid Uptime check configurations
        for the project (leaving out any invalid
        configurations).

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_uptime_check_configs():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListUptimeCheckConfigsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_uptime_check_configs(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListUptimeCheckConfigsRequest, dict]]):
                The request object. The protocol for the ``ListUptimeCheckConfigs`` request.
            parent (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                whose Uptime check configurations are listed. The format
                is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

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
            google.cloud.monitoring_v3.services.uptime_check_service.pagers.ListUptimeCheckConfigsAsyncPager:
                The protocol for the ListUptimeCheckConfigs response.

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
        if not isinstance(request, uptime_service.ListUptimeCheckConfigsRequest):
            request = uptime_service.ListUptimeCheckConfigsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_uptime_check_configs
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
        response = pagers.ListUptimeCheckConfigsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_uptime_check_config(
        self,
        request: Optional[
            Union[uptime_service.GetUptimeCheckConfigRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> uptime.UptimeCheckConfig:
        r"""Gets a single Uptime check configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_uptime_check_config():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetUptimeCheckConfigRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_uptime_check_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetUptimeCheckConfigRequest, dict]]):
                The request object. The protocol for the ``GetUptimeCheckConfig`` request.
            name (:class:`str`):
                Required. The Uptime check configuration to retrieve.
                The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]

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
            google.cloud.monitoring_v3.types.UptimeCheckConfig:
                This message configures which
                resources and services to monitor for
                availability.

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
        if not isinstance(request, uptime_service.GetUptimeCheckConfigRequest):
            request = uptime_service.GetUptimeCheckConfigRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_uptime_check_config
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

    async def create_uptime_check_config(
        self,
        request: Optional[
            Union[uptime_service.CreateUptimeCheckConfigRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        uptime_check_config: Optional[uptime.UptimeCheckConfig] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> uptime.UptimeCheckConfig:
        r"""Creates a new Uptime check configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_uptime_check_config():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateUptimeCheckConfigRequest(
                    parent="parent_value",
                )

                # Make the request
                response = await client.create_uptime_check_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateUptimeCheckConfigRequest, dict]]):
                The request object. The protocol for the ``CreateUptimeCheckConfig``
                request.
            parent (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                in which to create the Uptime check. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            uptime_check_config (:class:`google.cloud.monitoring_v3.types.UptimeCheckConfig`):
                Required. The new Uptime check
                configuration.

                This corresponds to the ``uptime_check_config`` field
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
            google.cloud.monitoring_v3.types.UptimeCheckConfig:
                This message configures which
                resources and services to monitor for
                availability.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, uptime_check_config]
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
        if not isinstance(request, uptime_service.CreateUptimeCheckConfigRequest):
            request = uptime_service.CreateUptimeCheckConfigRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if uptime_check_config is not None:
            request.uptime_check_config = uptime_check_config

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_uptime_check_config
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

    async def update_uptime_check_config(
        self,
        request: Optional[
            Union[uptime_service.UpdateUptimeCheckConfigRequest, dict]
        ] = None,
        *,
        uptime_check_config: Optional[uptime.UptimeCheckConfig] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> uptime.UptimeCheckConfig:
        r"""Updates an Uptime check configuration. You can either replace
        the entire configuration with a new one or replace only certain
        fields in the current configuration by specifying the fields to
        be updated via ``updateMask``. Returns the updated
        configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_update_uptime_check_config():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.UpdateUptimeCheckConfigRequest(
                )

                # Make the request
                response = await client.update_uptime_check_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.UpdateUptimeCheckConfigRequest, dict]]):
                The request object. The protocol for the ``UpdateUptimeCheckConfig``
                request.
            uptime_check_config (:class:`google.cloud.monitoring_v3.types.UptimeCheckConfig`):
                Required. If an ``updateMask`` has been specified, this
                field gives the values for the set of fields mentioned
                in the ``updateMask``. If an ``updateMask`` has not been
                given, this Uptime check configuration replaces the
                current configuration. If a field is mentioned in
                ``updateMask`` but the corresponding field is omitted in
                this partial Uptime check configuration, it has the
                effect of deleting/clearing the field from the
                configuration on the server.

                The following fields can be updated: ``display_name``,
                ``http_check``, ``tcp_check``, ``timeout``,
                ``content_matchers``, and ``selected_regions``.

                This corresponds to the ``uptime_check_config`` field
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
            google.cloud.monitoring_v3.types.UptimeCheckConfig:
                This message configures which
                resources and services to monitor for
                availability.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [uptime_check_config]
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
        if not isinstance(request, uptime_service.UpdateUptimeCheckConfigRequest):
            request = uptime_service.UpdateUptimeCheckConfigRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if uptime_check_config is not None:
            request.uptime_check_config = uptime_check_config

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_uptime_check_config
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("uptime_check_config.name", request.uptime_check_config.name),)
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

        # Done; return the response.
        return response

    async def delete_uptime_check_config(
        self,
        request: Optional[
            Union[uptime_service.DeleteUptimeCheckConfigRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes an Uptime check configuration. Note that this
        method will fail if the Uptime check configuration is
        referenced by an alert policy or other dependent configs
        that would be rendered invalid by the deletion.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_delete_uptime_check_config():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.DeleteUptimeCheckConfigRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_uptime_check_config(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.DeleteUptimeCheckConfigRequest, dict]]):
                The request object. The protocol for the ``DeleteUptimeCheckConfig``
                request.
            name (:class:`str`):
                Required. The Uptime check configuration to delete. The
                format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]

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
        if not isinstance(request, uptime_service.DeleteUptimeCheckConfigRequest):
            request = uptime_service.DeleteUptimeCheckConfigRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_uptime_check_config
        ]

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

    async def list_uptime_check_ips(
        self,
        request: Optional[Union[uptime_service.ListUptimeCheckIpsRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListUptimeCheckIpsAsyncPager:
        r"""Returns the list of IP addresses that checkers run
        from.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_uptime_check_ips():
                # Create a client
                client = monitoring_v3.UptimeCheckServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListUptimeCheckIpsRequest(
                )

                # Make the request
                page_result = client.list_uptime_check_ips(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListUptimeCheckIpsRequest, dict]]):
                The request object. The protocol for the ``ListUptimeCheckIps`` request.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.services.uptime_check_service.pagers.ListUptimeCheckIpsAsyncPager:
                The protocol for the ListUptimeCheckIps response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, uptime_service.ListUptimeCheckIpsRequest):
            request = uptime_service.ListUptimeCheckIpsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_uptime_check_ips
        ]

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
        response = pagers.ListUptimeCheckIpsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def __aenter__(self) -> "UptimeCheckServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("UptimeCheckServiceAsyncClient",)

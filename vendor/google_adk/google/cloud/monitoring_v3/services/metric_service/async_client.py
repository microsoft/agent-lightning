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

from google.api import label_pb2  # type: ignore
from google.api import launch_stage_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore

from google.cloud.monitoring_v3.services.metric_service import pagers
from google.cloud.monitoring_v3.types import common
from google.cloud.monitoring_v3.types import metric as gm_metric
from google.cloud.monitoring_v3.types import metric_service

from .client import MetricServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, MetricServiceTransport
from .transports.grpc_asyncio import MetricServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class MetricServiceAsyncClient:
    """Manages metric descriptors, monitored resource descriptors,
    and time series data.
    """

    _client: MetricServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = MetricServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = MetricServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = MetricServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = MetricServiceClient._DEFAULT_UNIVERSE

    metric_descriptor_path = staticmethod(MetricServiceClient.metric_descriptor_path)
    parse_metric_descriptor_path = staticmethod(
        MetricServiceClient.parse_metric_descriptor_path
    )
    monitored_resource_descriptor_path = staticmethod(
        MetricServiceClient.monitored_resource_descriptor_path
    )
    parse_monitored_resource_descriptor_path = staticmethod(
        MetricServiceClient.parse_monitored_resource_descriptor_path
    )
    time_series_path = staticmethod(MetricServiceClient.time_series_path)
    parse_time_series_path = staticmethod(MetricServiceClient.parse_time_series_path)
    common_billing_account_path = staticmethod(
        MetricServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        MetricServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(MetricServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        MetricServiceClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        MetricServiceClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        MetricServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(MetricServiceClient.common_project_path)
    parse_common_project_path = staticmethod(
        MetricServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(MetricServiceClient.common_location_path)
    parse_common_location_path = staticmethod(
        MetricServiceClient.parse_common_location_path
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
            MetricServiceAsyncClient: The constructed client.
        """
        return MetricServiceClient.from_service_account_info.__func__(MetricServiceAsyncClient, info, *args, **kwargs)  # type: ignore

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
            MetricServiceAsyncClient: The constructed client.
        """
        return MetricServiceClient.from_service_account_file.__func__(MetricServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return MetricServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> MetricServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            MetricServiceTransport: The transport used by the client instance.
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

    get_transport_class = MetricServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, MetricServiceTransport, Callable[..., MetricServiceTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the metric service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,MetricServiceTransport,Callable[..., MetricServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the MetricServiceTransport constructor.
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
        self._client = MetricServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.monitoring_v3.MetricServiceAsyncClient`.",
                extra={
                    "serviceName": "google.monitoring.v3.MetricService",
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
                    "serviceName": "google.monitoring.v3.MetricService",
                    "credentialsType": None,
                },
            )

    async def list_monitored_resource_descriptors(
        self,
        request: Optional[
            Union[metric_service.ListMonitoredResourceDescriptorsRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListMonitoredResourceDescriptorsAsyncPager:
        r"""Lists monitored resource descriptors that match a
        filter.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_monitored_resource_descriptors():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListMonitoredResourceDescriptorsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_monitored_resource_descriptors(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListMonitoredResourceDescriptorsRequest, dict]]):
                The request object. The ``ListMonitoredResourceDescriptors`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

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
            google.cloud.monitoring_v3.services.metric_service.pagers.ListMonitoredResourceDescriptorsAsyncPager:
                The ListMonitoredResourceDescriptors response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

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
            request, metric_service.ListMonitoredResourceDescriptorsRequest
        ):
            request = metric_service.ListMonitoredResourceDescriptorsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_monitored_resource_descriptors
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

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListMonitoredResourceDescriptorsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_monitored_resource_descriptor(
        self,
        request: Optional[
            Union[metric_service.GetMonitoredResourceDescriptorRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> monitored_resource_pb2.MonitoredResourceDescriptor:
        r"""Gets a single monitored resource descriptor.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_monitored_resource_descriptor():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetMonitoredResourceDescriptorRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_monitored_resource_descriptor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetMonitoredResourceDescriptorRequest, dict]]):
                The request object. The ``GetMonitoredResourceDescriptor`` request.
            name (:class:`str`):
                Required. The monitored resource descriptor to get. The
                format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/monitoredResourceDescriptors/[RESOURCE_TYPE]

                The ``[RESOURCE_TYPE]`` is a predefined type, such as
                ``cloudsql_database``.

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
            google.api.monitored_resource_pb2.MonitoredResourceDescriptor:
                An object that describes the schema of a
                   [MonitoredResource][google.api.MonitoredResource]
                   object using a type name and a set of labels. For
                   example, the monitored resource descriptor for Google
                   Compute Engine VM instances has a type of
                   "gce_instance" and specifies the use of the labels
                   "instance_id" and "zone" to identify particular VM
                   instances.

                   Different APIs can support different monitored
                   resource types. APIs generally provide a list method
                   that returns the monitored resource descriptors used
                   by the API.

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
            request, metric_service.GetMonitoredResourceDescriptorRequest
        ):
            request = metric_service.GetMonitoredResourceDescriptorRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_monitored_resource_descriptor
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

    async def list_metric_descriptors(
        self,
        request: Optional[
            Union[metric_service.ListMetricDescriptorsRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListMetricDescriptorsAsyncPager:
        r"""Lists metric descriptors that match a filter.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_metric_descriptors():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListMetricDescriptorsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_metric_descriptors(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListMetricDescriptorsRequest, dict]]):
                The request object. The ``ListMetricDescriptors`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

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
            google.cloud.monitoring_v3.services.metric_service.pagers.ListMetricDescriptorsAsyncPager:
                The ListMetricDescriptors response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

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
        if not isinstance(request, metric_service.ListMetricDescriptorsRequest):
            request = metric_service.ListMetricDescriptorsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_metric_descriptors
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

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListMetricDescriptorsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_metric_descriptor(
        self,
        request: Optional[
            Union[metric_service.GetMetricDescriptorRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> metric_pb2.MetricDescriptor:
        r"""Gets a single metric descriptor.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_metric_descriptor():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetMetricDescriptorRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_metric_descriptor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetMetricDescriptorRequest, dict]]):
                The request object. The ``GetMetricDescriptor`` request.
            name (:class:`str`):
                Required. The metric descriptor on which to execute the
                request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]

                An example value of ``[METRIC_ID]`` is
                ``"compute.googleapis.com/instance/disk/read_bytes_count"``.

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
            google.api.metric_pb2.MetricDescriptor:
                Defines a metric type and its schema.
                Once a metric descriptor is created,
                deleting or altering it stops data
                collection and makes the metric type's
                existing data unusable.

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
        if not isinstance(request, metric_service.GetMetricDescriptorRequest):
            request = metric_service.GetMetricDescriptorRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_metric_descriptor
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

    async def create_metric_descriptor(
        self,
        request: Optional[
            Union[metric_service.CreateMetricDescriptorRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        metric_descriptor: Optional[metric_pb2.MetricDescriptor] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> metric_pb2.MetricDescriptor:
        r"""Creates a new metric descriptor. The creation is executed
        asynchronously. User-created metric descriptors define `custom
        metrics <https://cloud.google.com/monitoring/custom-metrics>`__.
        The metric descriptor is updated if it already exists, except
        that metric labels are never removed.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_metric_descriptor():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateMetricDescriptorRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.create_metric_descriptor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateMetricDescriptorRequest, dict]]):
                The request object. The ``CreateMetricDescriptor`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is: 4
                projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            metric_descriptor (:class:`google.api.metric_pb2.MetricDescriptor`):
                Required. The new `custom
                metric <https://cloud.google.com/monitoring/custom-metrics>`__
                descriptor.

                This corresponds to the ``metric_descriptor`` field
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
            google.api.metric_pb2.MetricDescriptor:
                Defines a metric type and its schema.
                Once a metric descriptor is created,
                deleting or altering it stops data
                collection and makes the metric type's
                existing data unusable.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, metric_descriptor]
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
        if not isinstance(request, metric_service.CreateMetricDescriptorRequest):
            request = metric_service.CreateMetricDescriptorRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if metric_descriptor is not None:
            request.metric_descriptor = metric_descriptor

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_metric_descriptor
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

    async def delete_metric_descriptor(
        self,
        request: Optional[
            Union[metric_service.DeleteMetricDescriptorRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a metric descriptor. Only user-created `custom
        metrics <https://cloud.google.com/monitoring/custom-metrics>`__
        can be deleted.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_delete_metric_descriptor():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.DeleteMetricDescriptorRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_metric_descriptor(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.DeleteMetricDescriptorRequest, dict]]):
                The request object. The ``DeleteMetricDescriptor`` request.
            name (:class:`str`):
                Required. The metric descriptor on which to execute the
                request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]

                An example of ``[METRIC_ID]`` is:
                ``"custom.googleapis.com/my_test_metric"``.

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
        if not isinstance(request, metric_service.DeleteMetricDescriptorRequest):
            request = metric_service.DeleteMetricDescriptorRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_metric_descriptor
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

    async def list_time_series(
        self,
        request: Optional[Union[metric_service.ListTimeSeriesRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        filter: Optional[str] = None,
        interval: Optional[common.TimeInterval] = None,
        view: Optional[metric_service.ListTimeSeriesRequest.TimeSeriesView] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListTimeSeriesAsyncPager:
        r"""Lists time series that match a filter.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_time_series():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListTimeSeriesRequest(
                    name="name_value",
                    filter="filter_value",
                    view="HEADERS",
                )

                # Make the request
                page_result = client.list_time_series(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListTimeSeriesRequest, dict]]):
                The request object. The ``ListTimeSeries`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__,
                organization or folder on which to execute the request.
                The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]
                    organizations/[ORGANIZATION_ID]
                    folders/[FOLDER_ID]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            filter (:class:`str`):
                Required. A `monitoring
                filter <https://cloud.google.com/monitoring/api/v3/filters>`__
                that specifies which time series should be returned. The
                filter must specify a single metric type, and can
                additionally specify metric labels and other
                information. For example:

                ::

                    metric.type = "compute.googleapis.com/instance/cpu/usage_time" AND
                        metric.labels.instance_name = "my-instance-name"

                This corresponds to the ``filter`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            interval (:class:`google.cloud.monitoring_v3.types.TimeInterval`):
                Required. The time interval for which
                results should be returned. Only time
                series that contain data points in the
                specified interval are included in the
                response.

                This corresponds to the ``interval`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            view (:class:`google.cloud.monitoring_v3.types.ListTimeSeriesRequest.TimeSeriesView`):
                Required. Specifies which information
                is returned about the time series.

                This corresponds to the ``view`` field
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
            google.cloud.monitoring_v3.services.metric_service.pagers.ListTimeSeriesAsyncPager:
                The ListTimeSeries response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, filter, interval, view]
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
        if not isinstance(request, metric_service.ListTimeSeriesRequest):
            request = metric_service.ListTimeSeriesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if filter is not None:
            request.filter = filter
        if interval is not None:
            request.interval = interval
        if view is not None:
            request.view = view

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_time_series
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

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListTimeSeriesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def create_time_series(
        self,
        request: Optional[Union[metric_service.CreateTimeSeriesRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        time_series: Optional[MutableSequence[gm_metric.TimeSeries]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Creates or adds data to one or more time series. The response is
        empty if all time series in the request were written. If any
        time series could not be written, a corresponding failure
        message is included in the error response. This method does not
        support `resource locations constraint of an organization
        policy <https://cloud.google.com/resource-manager/docs/organization-policy/defining-locations#setting_the_organization_policy>`__.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_time_series():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateTimeSeriesRequest(
                    name="name_value",
                )

                # Make the request
                await client.create_time_series(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateTimeSeriesRequest, dict]]):
                The request object. The ``CreateTimeSeries`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            time_series (:class:`MutableSequence[google.cloud.monitoring_v3.types.TimeSeries]`):
                Required. The new data to be added to a list of time
                series. Adds at most one data point to each of several
                time series. The new data point must be more recent than
                any other point in its time series. Each ``TimeSeries``
                value must fully specify a unique time series by
                supplying all label values for the metric and the
                monitored resource.

                The maximum number of ``TimeSeries`` objects per
                ``Create`` request is 200.

                This corresponds to the ``time_series`` field
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
        flattened_params = [name, time_series]
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
        if not isinstance(request, metric_service.CreateTimeSeriesRequest):
            request = metric_service.CreateTimeSeriesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if time_series:
            request.time_series.extend(time_series)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_time_series
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

    async def create_service_time_series(
        self,
        request: Optional[Union[metric_service.CreateTimeSeriesRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        time_series: Optional[MutableSequence[gm_metric.TimeSeries]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Creates or adds data to one or more service time series. A
        service time series is a time series for a metric from a Google
        Cloud service. The response is empty if all time series in the
        request were written. If any time series could not be written, a
        corresponding failure message is included in the error response.
        This endpoint rejects writes to user-defined metrics. This
        method is only for use by Google Cloud services. Use
        [projects.timeSeries.create][google.monitoring.v3.MetricService.CreateTimeSeries]
        instead.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_service_time_series():
                # Create a client
                client = monitoring_v3.MetricServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateTimeSeriesRequest(
                    name="name_value",
                )

                # Make the request
                await client.create_service_time_series(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateTimeSeriesRequest, dict]]):
                The request object. The ``CreateTimeSeries`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            time_series (:class:`MutableSequence[google.cloud.monitoring_v3.types.TimeSeries]`):
                Required. The new data to be added to a list of time
                series. Adds at most one data point to each of several
                time series. The new data point must be more recent than
                any other point in its time series. Each ``TimeSeries``
                value must fully specify a unique time series by
                supplying all label values for the metric and the
                monitored resource.

                The maximum number of ``TimeSeries`` objects per
                ``Create`` request is 200.

                This corresponds to the ``time_series`` field
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
        flattened_params = [name, time_series]
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
        if not isinstance(request, metric_service.CreateTimeSeriesRequest):
            request = metric_service.CreateTimeSeriesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if time_series:
            request.time_series.extend(time_series)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_service_time_series
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

    async def __aenter__(self) -> "MetricServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("MetricServiceAsyncClient",)

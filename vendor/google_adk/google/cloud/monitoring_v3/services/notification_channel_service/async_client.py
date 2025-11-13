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
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore

from google.cloud.monitoring_v3.services.notification_channel_service import pagers
from google.cloud.monitoring_v3.types import (
    common,
    mutation_record,
    notification,
    notification_service,
)

from .client import NotificationChannelServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, NotificationChannelServiceTransport
from .transports.grpc_asyncio import NotificationChannelServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class NotificationChannelServiceAsyncClient:
    """The Notification Channel API provides access to configuration
    that controls how messages related to incidents are sent.
    """

    _client: NotificationChannelServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = NotificationChannelServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = NotificationChannelServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = (
        NotificationChannelServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    )
    _DEFAULT_UNIVERSE = NotificationChannelServiceClient._DEFAULT_UNIVERSE

    notification_channel_path = staticmethod(
        NotificationChannelServiceClient.notification_channel_path
    )
    parse_notification_channel_path = staticmethod(
        NotificationChannelServiceClient.parse_notification_channel_path
    )
    notification_channel_descriptor_path = staticmethod(
        NotificationChannelServiceClient.notification_channel_descriptor_path
    )
    parse_notification_channel_descriptor_path = staticmethod(
        NotificationChannelServiceClient.parse_notification_channel_descriptor_path
    )
    common_billing_account_path = staticmethod(
        NotificationChannelServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        NotificationChannelServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(
        NotificationChannelServiceClient.common_folder_path
    )
    parse_common_folder_path = staticmethod(
        NotificationChannelServiceClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        NotificationChannelServiceClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        NotificationChannelServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(
        NotificationChannelServiceClient.common_project_path
    )
    parse_common_project_path = staticmethod(
        NotificationChannelServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(
        NotificationChannelServiceClient.common_location_path
    )
    parse_common_location_path = staticmethod(
        NotificationChannelServiceClient.parse_common_location_path
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
            NotificationChannelServiceAsyncClient: The constructed client.
        """
        return NotificationChannelServiceClient.from_service_account_info.__func__(NotificationChannelServiceAsyncClient, info, *args, **kwargs)  # type: ignore

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
            NotificationChannelServiceAsyncClient: The constructed client.
        """
        return NotificationChannelServiceClient.from_service_account_file.__func__(NotificationChannelServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return NotificationChannelServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> NotificationChannelServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            NotificationChannelServiceTransport: The transport used by the client instance.
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

    get_transport_class = NotificationChannelServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                NotificationChannelServiceTransport,
                Callable[..., NotificationChannelServiceTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the notification channel service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,NotificationChannelServiceTransport,Callable[..., NotificationChannelServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the NotificationChannelServiceTransport constructor.
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
        self._client = NotificationChannelServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.monitoring_v3.NotificationChannelServiceAsyncClient`.",
                extra={
                    "serviceName": "google.monitoring.v3.NotificationChannelService",
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
                    "serviceName": "google.monitoring.v3.NotificationChannelService",
                    "credentialsType": None,
                },
            )

    async def list_notification_channel_descriptors(
        self,
        request: Optional[
            Union[notification_service.ListNotificationChannelDescriptorsRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListNotificationChannelDescriptorsAsyncPager:
        r"""Lists the descriptors for supported channel types.
        The use of descriptors makes it possible for new channel
        types to be dynamically added.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_notification_channel_descriptors():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListNotificationChannelDescriptorsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_notification_channel_descriptors(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListNotificationChannelDescriptorsRequest, dict]]):
                The request object. The ``ListNotificationChannelDescriptors`` request.
            name (:class:`str`):
                Required. The REST resource name of the parent from
                which to retrieve the notification channel descriptors.
                The expected syntax is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                Note that this
                `names <https://cloud.google.com/monitoring/api/v3#project_name>`__
                the parent container in which to look for the
                descriptors; to retrieve a single descriptor by name,
                use the
                [GetNotificationChannelDescriptor][google.monitoring.v3.NotificationChannelService.GetNotificationChannelDescriptor]
                operation, instead.

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
            google.cloud.monitoring_v3.services.notification_channel_service.pagers.ListNotificationChannelDescriptorsAsyncPager:
                The ListNotificationChannelDescriptors response.

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
            request, notification_service.ListNotificationChannelDescriptorsRequest
        ):
            request = notification_service.ListNotificationChannelDescriptorsRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_notification_channel_descriptors
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
        response = pagers.ListNotificationChannelDescriptorsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_notification_channel_descriptor(
        self,
        request: Optional[
            Union[notification_service.GetNotificationChannelDescriptorRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification.NotificationChannelDescriptor:
        r"""Gets a single channel descriptor. The descriptor
        indicates which fields are expected / permitted for a
        notification channel of the given type.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_notification_channel_descriptor():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetNotificationChannelDescriptorRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_notification_channel_descriptor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetNotificationChannelDescriptorRequest, dict]]):
                The request object. The ``GetNotificationChannelDescriptor`` response.
            name (:class:`str`):
                Required. The channel type for which to execute the
                request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/notificationChannelDescriptors/[CHANNEL_TYPE]

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
            google.cloud.monitoring_v3.types.NotificationChannelDescriptor:
                A description of a notification
                channel. The descriptor includes the
                properties of the channel and the set of
                labels or fields that must be specified
                to configure channels of a given type.

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
            request, notification_service.GetNotificationChannelDescriptorRequest
        ):
            request = notification_service.GetNotificationChannelDescriptorRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_notification_channel_descriptor
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

    async def list_notification_channels(
        self,
        request: Optional[
            Union[notification_service.ListNotificationChannelsRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListNotificationChannelsAsyncPager:
        r"""Lists the notification channels that have been created for the
        project. To list the types of notification channels that are
        supported, use the ``ListNotificationChannelDescriptors``
        method.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_notification_channels():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListNotificationChannelsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_notification_channels(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListNotificationChannelsRequest, dict]]):
                The request object. The ``ListNotificationChannels`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This names the container in which to look for the
                notification channels; it does not name a specific
                channel. To query a specific channel by REST resource
                name, use the
                [``GetNotificationChannel``][google.monitoring.v3.NotificationChannelService.GetNotificationChannel]
                operation.

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
            google.cloud.monitoring_v3.services.notification_channel_service.pagers.ListNotificationChannelsAsyncPager:
                The ListNotificationChannels response.

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
            request, notification_service.ListNotificationChannelsRequest
        ):
            request = notification_service.ListNotificationChannelsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_notification_channels
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
        response = pagers.ListNotificationChannelsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_notification_channel(
        self,
        request: Optional[
            Union[notification_service.GetNotificationChannelRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification.NotificationChannel:
        r"""Gets a single notification channel. The channel
        includes the relevant configuration details with which
        the channel was created. However, the response may
        truncate or omit passwords, API keys, or other private
        key matter and thus the response may not be 100%
        identical to the information that was supplied in the
        call to the create method.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_notification_channel():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetNotificationChannelRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_notification_channel(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetNotificationChannelRequest, dict]]):
                The request object. The ``GetNotificationChannel`` request.
            name (:class:`str`):
                Required. The channel for which to execute the request.
                The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]

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
            google.cloud.monitoring_v3.types.NotificationChannel:
                A NotificationChannel is a medium through which an alert is
                   delivered when a policy violation is detected.
                   Examples of channels include email, SMS, and
                   third-party messaging applications. Fields containing
                   sensitive information like authentication tokens or
                   contact info are only partially populated on
                   retrieval.

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
        if not isinstance(request, notification_service.GetNotificationChannelRequest):
            request = notification_service.GetNotificationChannelRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_notification_channel
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

    async def create_notification_channel(
        self,
        request: Optional[
            Union[notification_service.CreateNotificationChannelRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        notification_channel: Optional[notification.NotificationChannel] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification.NotificationChannel:
        r"""Creates a new notification channel, representing a
        single notification endpoint such as an email address,
        SMS number, or PagerDuty service.

        Design your application to single-thread API calls that
        modify the state of notification channels in a single
        project. This includes calls to
        CreateNotificationChannel, DeleteNotificationChannel and
        UpdateNotificationChannel.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_notification_channel():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateNotificationChannelRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.create_notification_channel(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateNotificationChannelRequest, dict]]):
                The request object. The ``CreateNotificationChannel`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                on which to execute the request. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This names the container into which the channel will be
                written, this does not name the newly created channel.
                The resulting channel's name will have a normalized
                version of this field as a prefix, but will add
                ``/notificationChannels/[CHANNEL_ID]`` to identify the
                channel.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            notification_channel (:class:`google.cloud.monitoring_v3.types.NotificationChannel`):
                Required. The definition of the ``NotificationChannel``
                to create.

                This corresponds to the ``notification_channel`` field
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
            google.cloud.monitoring_v3.types.NotificationChannel:
                A NotificationChannel is a medium through which an alert is
                   delivered when a policy violation is detected.
                   Examples of channels include email, SMS, and
                   third-party messaging applications. Fields containing
                   sensitive information like authentication tokens or
                   contact info are only partially populated on
                   retrieval.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, notification_channel]
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
            request, notification_service.CreateNotificationChannelRequest
        ):
            request = notification_service.CreateNotificationChannelRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if notification_channel is not None:
            request.notification_channel = notification_channel

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_notification_channel
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

    async def update_notification_channel(
        self,
        request: Optional[
            Union[notification_service.UpdateNotificationChannelRequest, dict]
        ] = None,
        *,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        notification_channel: Optional[notification.NotificationChannel] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification.NotificationChannel:
        r"""Updates a notification channel. Fields not specified
        in the field mask remain unchanged.

        Design your application to single-thread API calls that
        modify the state of notification channels in a single
        project. This includes calls to
        CreateNotificationChannel, DeleteNotificationChannel and
        UpdateNotificationChannel.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_update_notification_channel():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.UpdateNotificationChannelRequest(
                )

                # Make the request
                response = await client.update_notification_channel(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.UpdateNotificationChannelRequest, dict]]):
                The request object. The ``UpdateNotificationChannel`` request.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Optional. The fields to update.
                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            notification_channel (:class:`google.cloud.monitoring_v3.types.NotificationChannel`):
                Required. A description of the changes to be applied to
                the specified notification channel. The description must
                provide a definition for fields to be updated; the names
                of these fields should also be included in the
                ``update_mask``.

                This corresponds to the ``notification_channel`` field
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
            google.cloud.monitoring_v3.types.NotificationChannel:
                A NotificationChannel is a medium through which an alert is
                   delivered when a policy violation is detected.
                   Examples of channels include email, SMS, and
                   third-party messaging applications. Fields containing
                   sensitive information like authentication tokens or
                   contact info are only partially populated on
                   retrieval.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [update_mask, notification_channel]
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
            request, notification_service.UpdateNotificationChannelRequest
        ):
            request = notification_service.UpdateNotificationChannelRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if update_mask is not None:
            request.update_mask = update_mask
        if notification_channel is not None:
            request.notification_channel = notification_channel

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_notification_channel
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("notification_channel.name", request.notification_channel.name),)
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

    async def delete_notification_channel(
        self,
        request: Optional[
            Union[notification_service.DeleteNotificationChannelRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        force: Optional[bool] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a notification channel.

        Design your application to single-thread API calls that
        modify the state of notification channels in a single
        project. This includes calls to
        CreateNotificationChannel, DeleteNotificationChannel and
        UpdateNotificationChannel.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_delete_notification_channel():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.DeleteNotificationChannelRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_notification_channel(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.DeleteNotificationChannelRequest, dict]]):
                The request object. The ``DeleteNotificationChannel`` request.
            name (:class:`str`):
                Required. The channel for which to execute the request.
                The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            force (:class:`bool`):
                If true, the notification channel
                will be deleted regardless of its use in
                alert policies (the policies will be
                updated to remove the channel). If
                false, this operation will fail if the
                notification channel is referenced by
                existing alerting policies.

                This corresponds to the ``force`` field
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
        flattened_params = [name, force]
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
            request, notification_service.DeleteNotificationChannelRequest
        ):
            request = notification_service.DeleteNotificationChannelRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if force is not None:
            request.force = force

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_notification_channel
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

    async def send_notification_channel_verification_code(
        self,
        request: Optional[
            Union[
                notification_service.SendNotificationChannelVerificationCodeRequest,
                dict,
            ]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Causes a verification code to be delivered to the channel. The
        code can then be supplied in ``VerifyNotificationChannel`` to
        verify the channel.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_send_notification_channel_verification_code():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.SendNotificationChannelVerificationCodeRequest(
                    name="name_value",
                )

                # Make the request
                await client.send_notification_channel_verification_code(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.SendNotificationChannelVerificationCodeRequest, dict]]):
                The request object. The ``SendNotificationChannelVerificationCode`` request.
            name (:class:`str`):
                Required. The notification channel to
                which to send a verification code.

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
        if not isinstance(
            request, notification_service.SendNotificationChannelVerificationCodeRequest
        ):
            request = (
                notification_service.SendNotificationChannelVerificationCodeRequest(
                    request
                )
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.send_notification_channel_verification_code
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

    async def get_notification_channel_verification_code(
        self,
        request: Optional[
            Union[
                notification_service.GetNotificationChannelVerificationCodeRequest, dict
            ]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification_service.GetNotificationChannelVerificationCodeResponse:
        r"""Requests a verification code for an already verified
        channel that can then be used in a call to
        VerifyNotificationChannel() on a different channel with
        an equivalent identity in the same or in a different
        project. This makes it possible to copy a channel
        between projects without requiring manual reverification
        of the channel. If the channel is not in the verified
        state, this method will fail (in other words, this may
        only be used if the
        SendNotificationChannelVerificationCode and
        VerifyNotificationChannel paths have already been used
        to put the given channel into the verified state).

        There is no guarantee that the verification codes
        returned by this method will be of a similar structure
        or form as the ones that are delivered to the channel
        via SendNotificationChannelVerificationCode; while
        VerifyNotificationChannel() will recognize both the
        codes delivered via
        SendNotificationChannelVerificationCode() and returned
        from GetNotificationChannelVerificationCode(), it is
        typically the case that the verification codes delivered
        via
        SendNotificationChannelVerificationCode() will be
        shorter and also have a shorter expiration (e.g. codes
        such as "G-123456") whereas GetVerificationCode() will
        typically return a much longer, websafe base 64 encoded
        string that has a longer expiration time.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_notification_channel_verification_code():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetNotificationChannelVerificationCodeRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_notification_channel_verification_code(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetNotificationChannelVerificationCodeRequest, dict]]):
                The request object. The ``GetNotificationChannelVerificationCode`` request.
            name (:class:`str`):
                Required. The notification channel
                for which a verification code is to be
                generated and retrieved. This must name
                a channel that is already verified; if
                the specified channel is not verified,
                the request will fail.

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
            google.cloud.monitoring_v3.types.GetNotificationChannelVerificationCodeResponse:
                The GetNotificationChannelVerificationCode request.
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
            request, notification_service.GetNotificationChannelVerificationCodeRequest
        ):
            request = (
                notification_service.GetNotificationChannelVerificationCodeRequest(
                    request
                )
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_notification_channel_verification_code
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

    async def verify_notification_channel(
        self,
        request: Optional[
            Union[notification_service.VerifyNotificationChannelRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        code: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> notification.NotificationChannel:
        r"""Verifies a ``NotificationChannel`` by proving receipt of the
        code delivered to the channel as a result of calling
        ``SendNotificationChannelVerificationCode``.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_verify_notification_channel():
                # Create a client
                client = monitoring_v3.NotificationChannelServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.VerifyNotificationChannelRequest(
                    name="name_value",
                    code="code_value",
                )

                # Make the request
                response = await client.verify_notification_channel(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.VerifyNotificationChannelRequest, dict]]):
                The request object. The ``VerifyNotificationChannel`` request.
            name (:class:`str`):
                Required. The notification channel to
                verify.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            code (:class:`str`):
                Required. The verification code that was delivered to
                the channel as a result of invoking the
                ``SendNotificationChannelVerificationCode`` API method
                or that was retrieved from a verified channel via
                ``GetNotificationChannelVerificationCode``. For example,
                one might have "G-123456" or "TKNZGhhd2EyN3I1MnRnMjRv"
                (in general, one is only guaranteed that the code is
                valid UTF-8; one should not make any assumptions
                regarding the structure or format of the code).

                This corresponds to the ``code`` field
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
            google.cloud.monitoring_v3.types.NotificationChannel:
                A NotificationChannel is a medium through which an alert is
                   delivered when a policy violation is detected.
                   Examples of channels include email, SMS, and
                   third-party messaging applications. Fields containing
                   sensitive information like authentication tokens or
                   contact info are only partially populated on
                   retrieval.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, code]
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
            request, notification_service.VerifyNotificationChannelRequest
        ):
            request = notification_service.VerifyNotificationChannelRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if code is not None:
            request.code = code

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.verify_notification_channel
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

    async def __aenter__(self) -> "NotificationChannelServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("NotificationChannelServiceAsyncClient",)

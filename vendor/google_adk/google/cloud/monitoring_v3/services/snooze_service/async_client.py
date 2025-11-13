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

from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore

from google.cloud.monitoring_v3.services.snooze_service import pagers
from google.cloud.monitoring_v3.types import common
from google.cloud.monitoring_v3.types import snooze
from google.cloud.monitoring_v3.types import snooze as gm_snooze
from google.cloud.monitoring_v3.types import snooze_service

from .client import SnoozeServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, SnoozeServiceTransport
from .transports.grpc_asyncio import SnoozeServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class SnoozeServiceAsyncClient:
    """The SnoozeService API is used to temporarily prevent an alert
    policy from generating alerts. A Snooze is a description of the
    criteria under which one or more alert policies should not fire
    alerts for the specified duration.
    """

    _client: SnoozeServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = SnoozeServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = SnoozeServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = SnoozeServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = SnoozeServiceClient._DEFAULT_UNIVERSE

    alert_policy_path = staticmethod(SnoozeServiceClient.alert_policy_path)
    parse_alert_policy_path = staticmethod(SnoozeServiceClient.parse_alert_policy_path)
    snooze_path = staticmethod(SnoozeServiceClient.snooze_path)
    parse_snooze_path = staticmethod(SnoozeServiceClient.parse_snooze_path)
    common_billing_account_path = staticmethod(
        SnoozeServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        SnoozeServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(SnoozeServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        SnoozeServiceClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        SnoozeServiceClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        SnoozeServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(SnoozeServiceClient.common_project_path)
    parse_common_project_path = staticmethod(
        SnoozeServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(SnoozeServiceClient.common_location_path)
    parse_common_location_path = staticmethod(
        SnoozeServiceClient.parse_common_location_path
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
            SnoozeServiceAsyncClient: The constructed client.
        """
        return SnoozeServiceClient.from_service_account_info.__func__(SnoozeServiceAsyncClient, info, *args, **kwargs)  # type: ignore

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
            SnoozeServiceAsyncClient: The constructed client.
        """
        return SnoozeServiceClient.from_service_account_file.__func__(SnoozeServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return SnoozeServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> SnoozeServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            SnoozeServiceTransport: The transport used by the client instance.
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

    get_transport_class = SnoozeServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, SnoozeServiceTransport, Callable[..., SnoozeServiceTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the snooze service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,SnoozeServiceTransport,Callable[..., SnoozeServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the SnoozeServiceTransport constructor.
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
        self._client = SnoozeServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.monitoring_v3.SnoozeServiceAsyncClient`.",
                extra={
                    "serviceName": "google.monitoring.v3.SnoozeService",
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
                    "serviceName": "google.monitoring.v3.SnoozeService",
                    "credentialsType": None,
                },
            )

    async def create_snooze(
        self,
        request: Optional[Union[snooze_service.CreateSnoozeRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        snooze: Optional[gm_snooze.Snooze] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> gm_snooze.Snooze:
        r"""Creates a ``Snooze`` that will prevent alerts, which match the
        provided criteria, from being opened. The ``Snooze`` applies for
        a specific time interval.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_snooze():
                # Create a client
                client = monitoring_v3.SnoozeServiceAsyncClient()

                # Initialize request argument(s)
                snooze = monitoring_v3.Snooze()
                snooze.display_name = "display_name_value"

                request = monitoring_v3.CreateSnoozeRequest(
                    parent="parent_value",
                    snooze=snooze,
                )

                # Make the request
                response = await client.create_snooze(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateSnoozeRequest, dict]]):
                The request object. The message definition for creating a ``Snooze``. Users
                must provide the body of the ``Snooze`` to be created
                but must omit the ``Snooze`` field, ``name``.
            parent (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                in which a ``Snooze`` should be created. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            snooze (:class:`google.cloud.monitoring_v3.types.Snooze`):
                Required. The ``Snooze`` to create. Omit the ``name``
                field, as it will be filled in by the API.

                This corresponds to the ``snooze`` field
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
            google.cloud.monitoring_v3.types.Snooze:
                A Snooze will prevent any alerts from being opened, and close any that
                   are already open. The Snooze will work on alerts that
                   match the criteria defined in the Snooze. The Snooze
                   will be active from interval.start_time through
                   interval.end_time.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, snooze]
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
        if not isinstance(request, snooze_service.CreateSnoozeRequest):
            request = snooze_service.CreateSnoozeRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if snooze is not None:
            request.snooze = snooze

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_snooze
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

    async def list_snoozes(
        self,
        request: Optional[Union[snooze_service.ListSnoozesRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListSnoozesAsyncPager:
        r"""Lists the ``Snooze``\ s associated with a project. Can
        optionally pass in ``filter``, which specifies predicates to
        match ``Snooze``\ s.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_snoozes():
                # Create a client
                client = monitoring_v3.SnoozeServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListSnoozesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_snoozes(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListSnoozesRequest, dict]]):
                The request object. The message definition for listing ``Snooze``\ s
                associated with the given ``parent``, satisfying the
                optional ``filter``.
            parent (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                whose ``Snooze``\ s should be listed. The format is:

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
            google.cloud.monitoring_v3.services.snooze_service.pagers.ListSnoozesAsyncPager:
                The results of a successful ListSnoozes call, containing the matching
                   \`Snooze`s.

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
        if not isinstance(request, snooze_service.ListSnoozesRequest):
            request = snooze_service.ListSnoozesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_snoozes
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
        response = pagers.ListSnoozesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_snooze(
        self,
        request: Optional[Union[snooze_service.GetSnoozeRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> snooze.Snooze:
        r"""Retrieves a ``Snooze`` by ``name``.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_snooze():
                # Create a client
                client = monitoring_v3.SnoozeServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetSnoozeRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_snooze(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetSnoozeRequest, dict]]):
                The request object. The message definition for retrieving a ``Snooze``.
                Users must specify the field, ``name``, which identifies
                the ``Snooze``.
            name (:class:`str`):
                Required. The ID of the ``Snooze`` to retrieve. The
                format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/snoozes/[SNOOZE_ID]

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
            google.cloud.monitoring_v3.types.Snooze:
                A Snooze will prevent any alerts from being opened, and close any that
                   are already open. The Snooze will work on alerts that
                   match the criteria defined in the Snooze. The Snooze
                   will be active from interval.start_time through
                   interval.end_time.

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
        if not isinstance(request, snooze_service.GetSnoozeRequest):
            request = snooze_service.GetSnoozeRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_snooze
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

    async def update_snooze(
        self,
        request: Optional[Union[snooze_service.UpdateSnoozeRequest, dict]] = None,
        *,
        snooze: Optional[gm_snooze.Snooze] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> gm_snooze.Snooze:
        r"""Updates a ``Snooze``, identified by its ``name``, with the
        parameters in the given ``Snooze`` object.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_update_snooze():
                # Create a client
                client = monitoring_v3.SnoozeServiceAsyncClient()

                # Initialize request argument(s)
                snooze = monitoring_v3.Snooze()
                snooze.display_name = "display_name_value"

                request = monitoring_v3.UpdateSnoozeRequest(
                    snooze=snooze,
                )

                # Make the request
                response = await client.update_snooze(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.UpdateSnoozeRequest, dict]]):
                The request object. The message definition for updating a ``Snooze``. The
                field, ``snooze.name`` identifies the ``Snooze`` to be
                updated. The remainder of ``snooze`` gives the content
                the ``Snooze`` in question will be assigned.

                What fields can be updated depends on the start time and
                end time of the ``Snooze``.

                - end time is in the past: These ``Snooze``\ s are
                  considered read-only and cannot be updated.
                - start time is in the past and end time is in the
                  future: ``display_name`` and ``interval.end_time`` can
                  be updated.
                - start time is in the future: ``display_name``,
                  ``interval.start_time`` and ``interval.end_time`` can
                  be updated.
            snooze (:class:`google.cloud.monitoring_v3.types.Snooze`):
                Required. The ``Snooze`` to update. Must have the name
                field present.

                This corresponds to the ``snooze`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. The fields to update.

                For each field listed in ``update_mask``:

                - If the ``Snooze`` object supplied in the
                  ``UpdateSnoozeRequest`` has a value for that field,
                  the value of the field in the existing ``Snooze`` will
                  be set to the value of the field in the supplied
                  ``Snooze``.
                - If the field does not have a value in the supplied
                  ``Snooze``, the field in the existing ``Snooze`` is
                  set to its default value.

                Fields not listed retain their existing value.

                The following are the field names that are accepted in
                ``update_mask``:

                - ``display_name``
                - ``interval.start_time``
                - ``interval.end_time``

                That said, the start time and end time of the ``Snooze``
                determines which fields can legally be updated. Before
                attempting an update, users should consult the
                documentation for ``UpdateSnoozeRequest``, which talks
                about which fields can be updated.

                This corresponds to the ``update_mask`` field
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
            google.cloud.monitoring_v3.types.Snooze:
                A Snooze will prevent any alerts from being opened, and close any that
                   are already open. The Snooze will work on alerts that
                   match the criteria defined in the Snooze. The Snooze
                   will be active from interval.start_time through
                   interval.end_time.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [snooze, update_mask]
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
        if not isinstance(request, snooze_service.UpdateSnoozeRequest):
            request = snooze_service.UpdateSnoozeRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if snooze is not None:
            request.snooze = snooze
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_snooze
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("snooze.name", request.snooze.name),)
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

    async def __aenter__(self) -> "SnoozeServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("SnoozeServiceAsyncClient",)

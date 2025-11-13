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
import logging as std_logging
from collections import OrderedDict
import re
from typing import (
    Dict,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from google.cloud.bigtable_admin_v2 import gapic_version as package_version

from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry_async as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf


try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.bigtable_admin_v2.services.bigtable_instance_admin import pagers
from google.cloud.bigtable_admin_v2.types import bigtable_instance_admin
from google.cloud.bigtable_admin_v2.types import common
from google.cloud.bigtable_admin_v2.types import instance
from google.cloud.bigtable_admin_v2.types import instance as gba_instance
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from .transports.base import BigtableInstanceAdminTransport, DEFAULT_CLIENT_INFO
from .transports.grpc_asyncio import BigtableInstanceAdminGrpcAsyncIOTransport
from .client import BigtableInstanceAdminClient

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class BigtableInstanceAdminAsyncClient:
    """Service for creating, configuring, and deleting Cloud
    Bigtable Instances and Clusters. Provides access to the Instance
    and Cluster schemas only, not the tables' metadata or data
    stored in those tables.
    """

    _client: BigtableInstanceAdminClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = BigtableInstanceAdminClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = BigtableInstanceAdminClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = BigtableInstanceAdminClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = BigtableInstanceAdminClient._DEFAULT_UNIVERSE

    app_profile_path = staticmethod(BigtableInstanceAdminClient.app_profile_path)
    parse_app_profile_path = staticmethod(
        BigtableInstanceAdminClient.parse_app_profile_path
    )
    cluster_path = staticmethod(BigtableInstanceAdminClient.cluster_path)
    parse_cluster_path = staticmethod(BigtableInstanceAdminClient.parse_cluster_path)
    crypto_key_path = staticmethod(BigtableInstanceAdminClient.crypto_key_path)
    parse_crypto_key_path = staticmethod(
        BigtableInstanceAdminClient.parse_crypto_key_path
    )
    hot_tablet_path = staticmethod(BigtableInstanceAdminClient.hot_tablet_path)
    parse_hot_tablet_path = staticmethod(
        BigtableInstanceAdminClient.parse_hot_tablet_path
    )
    instance_path = staticmethod(BigtableInstanceAdminClient.instance_path)
    parse_instance_path = staticmethod(BigtableInstanceAdminClient.parse_instance_path)
    logical_view_path = staticmethod(BigtableInstanceAdminClient.logical_view_path)
    parse_logical_view_path = staticmethod(
        BigtableInstanceAdminClient.parse_logical_view_path
    )
    materialized_view_path = staticmethod(
        BigtableInstanceAdminClient.materialized_view_path
    )
    parse_materialized_view_path = staticmethod(
        BigtableInstanceAdminClient.parse_materialized_view_path
    )
    table_path = staticmethod(BigtableInstanceAdminClient.table_path)
    parse_table_path = staticmethod(BigtableInstanceAdminClient.parse_table_path)
    common_billing_account_path = staticmethod(
        BigtableInstanceAdminClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        BigtableInstanceAdminClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(BigtableInstanceAdminClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        BigtableInstanceAdminClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        BigtableInstanceAdminClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        BigtableInstanceAdminClient.parse_common_organization_path
    )
    common_project_path = staticmethod(BigtableInstanceAdminClient.common_project_path)
    parse_common_project_path = staticmethod(
        BigtableInstanceAdminClient.parse_common_project_path
    )
    common_location_path = staticmethod(
        BigtableInstanceAdminClient.common_location_path
    )
    parse_common_location_path = staticmethod(
        BigtableInstanceAdminClient.parse_common_location_path
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
            BigtableInstanceAdminAsyncClient: The constructed client.
        """
        return BigtableInstanceAdminClient.from_service_account_info.__func__(BigtableInstanceAdminAsyncClient, info, *args, **kwargs)  # type: ignore

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
            BigtableInstanceAdminAsyncClient: The constructed client.
        """
        return BigtableInstanceAdminClient.from_service_account_file.__func__(BigtableInstanceAdminAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return BigtableInstanceAdminClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> BigtableInstanceAdminTransport:
        """Returns the transport used by the client instance.

        Returns:
            BigtableInstanceAdminTransport: The transport used by the client instance.
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

    get_transport_class = BigtableInstanceAdminClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                BigtableInstanceAdminTransport,
                Callable[..., BigtableInstanceAdminTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the bigtable instance admin async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,BigtableInstanceAdminTransport,Callable[..., BigtableInstanceAdminTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the BigtableInstanceAdminTransport constructor.
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
        self._client = BigtableInstanceAdminClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.bigtable.admin_v2.BigtableInstanceAdminAsyncClient`.",
                extra={
                    "serviceName": "google.bigtable.admin.v2.BigtableInstanceAdmin",
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
                    "serviceName": "google.bigtable.admin.v2.BigtableInstanceAdmin",
                    "credentialsType": None,
                },
            )

    async def create_instance(
        self,
        request: Optional[
            Union[bigtable_instance_admin.CreateInstanceRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        instance_id: Optional[str] = None,
        instance: Optional[gba_instance.Instance] = None,
        clusters: Optional[MutableMapping[str, gba_instance.Cluster]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Create an instance within a project.

        Note that exactly one of Cluster.serve_nodes and
        Cluster.cluster_config.cluster_autoscaling_config can be set. If
        serve_nodes is set to non-zero, then the cluster is manually
        scaled. If cluster_config.cluster_autoscaling_config is
        non-empty, then autoscaling is enabled.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_instance():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                instance = bigtable_admin_v2.Instance()
                instance.display_name = "display_name_value"

                request = bigtable_admin_v2.CreateInstanceRequest(
                    parent="parent_value",
                    instance_id="instance_id_value",
                    instance=instance,
                )

                # Make the request
                operation = client.create_instance(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateInstanceRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.CreateInstance.
            parent (:class:`str`):
                Required. The unique name of the project in which to
                create the new instance. Values are of the form
                ``projects/{project}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_id (:class:`str`):
                Required. The ID to be used when referring to the new
                instance within its project, e.g., just ``myinstance``
                rather than ``projects/myproject/instances/myinstance``.

                This corresponds to the ``instance_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance (:class:`google.cloud.bigtable_admin_v2.types.Instance`):
                Required. The instance to create. Fields marked
                ``OutputOnly`` must be left blank.

                This corresponds to the ``instance`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            clusters (:class:`MutableMapping[str, google.cloud.bigtable_admin_v2.types.Cluster]`):
                Required. The clusters to be created within the
                instance, mapped by desired cluster ID, e.g., just
                ``mycluster`` rather than
                ``projects/myproject/instances/myinstance/clusters/mycluster``.
                Fields marked ``OutputOnly`` must be left blank.

                This corresponds to the ``clusters`` field
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Instance` A collection of Bigtable [Tables][google.bigtable.admin.v2.Table] and
                   the resources that serve them. All tables in an
                   instance are served from all
                   [Clusters][google.bigtable.admin.v2.Cluster] in the
                   instance.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, instance_id, instance, clusters]
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
        if not isinstance(request, bigtable_instance_admin.CreateInstanceRequest):
            request = bigtable_instance_admin.CreateInstanceRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if instance_id is not None:
            request.instance_id = instance_id
        if instance is not None:
            request.instance = instance

        if clusters:
            request.clusters.update(clusters)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_instance
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
            gba_instance.Instance,
            metadata_type=bigtable_instance_admin.CreateInstanceMetadata,
        )

        # Done; return the response.
        return response

    async def get_instance(
        self,
        request: Optional[
            Union[bigtable_instance_admin.GetInstanceRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.Instance:
        r"""Gets information about an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_instance():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetInstanceRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_instance(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetInstanceRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.GetInstance.
            name (:class:`str`):
                Required. The unique name of the requested instance.
                Values are of the form
                ``projects/{project}/instances/{instance}``.

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
            google.cloud.bigtable_admin_v2.types.Instance:
                A collection of Bigtable [Tables][google.bigtable.admin.v2.Table] and
                   the resources that serve them. All tables in an
                   instance are served from all
                   [Clusters][google.bigtable.admin.v2.Cluster] in the
                   instance.

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
        if not isinstance(request, bigtable_instance_admin.GetInstanceRequest):
            request = bigtable_instance_admin.GetInstanceRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_instance
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

    async def list_instances(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListInstancesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable_instance_admin.ListInstancesResponse:
        r"""Lists information about instances in a project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_instances():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListInstancesRequest(
                    parent="parent_value",
                )

                # Make the request
                response = await client.list_instances(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListInstancesRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListInstances.
            parent (:class:`str`):
                Required. The unique name of the project for which a
                list of instances is requested. Values are of the form
                ``projects/{project}``.

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
            google.cloud.bigtable_admin_v2.types.ListInstancesResponse:
                Response message for
                BigtableInstanceAdmin.ListInstances.

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
        if not isinstance(request, bigtable_instance_admin.ListInstancesRequest):
            request = bigtable_instance_admin.ListInstancesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_instances
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

    async def update_instance(
        self,
        request: Optional[Union[instance.Instance, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.Instance:
        r"""Updates an instance within a project. This method
        updates only the display name and type for an Instance.
        To update other Instance properties, such as labels, use
        PartialUpdateInstance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_instance():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.Instance(
                    display_name="display_name_value",
                )

                # Make the request
                response = await client.update_instance(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.Instance, dict]]):
                The request object. A collection of Bigtable
                [Tables][google.bigtable.admin.v2.Table] and the
                resources that serve them. All tables in an instance are
                served from all
                [Clusters][google.bigtable.admin.v2.Cluster] in the
                instance.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.bigtable_admin_v2.types.Instance:
                A collection of Bigtable [Tables][google.bigtable.admin.v2.Table] and
                   the resources that serve them. All tables in an
                   instance are served from all
                   [Clusters][google.bigtable.admin.v2.Cluster] in the
                   instance.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, instance.Instance):
            request = instance.Instance(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_instance
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

    async def partial_update_instance(
        self,
        request: Optional[
            Union[bigtable_instance_admin.PartialUpdateInstanceRequest, dict]
        ] = None,
        *,
        instance: Optional[gba_instance.Instance] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Partially updates an instance within a project. This
        method can modify all fields of an Instance and is the
        preferred way to update an Instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_partial_update_instance():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                instance = bigtable_admin_v2.Instance()
                instance.display_name = "display_name_value"

                request = bigtable_admin_v2.PartialUpdateInstanceRequest(
                    instance=instance,
                )

                # Make the request
                operation = client.partial_update_instance(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.PartialUpdateInstanceRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.PartialUpdateInstance.
            instance (:class:`google.cloud.bigtable_admin_v2.types.Instance`):
                Required. The Instance which will
                (partially) replace the current value.

                This corresponds to the ``instance`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. The subset of Instance
                fields which should be replaced. Must be
                explicitly set.

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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Instance` A collection of Bigtable [Tables][google.bigtable.admin.v2.Table] and
                   the resources that serve them. All tables in an
                   instance are served from all
                   [Clusters][google.bigtable.admin.v2.Cluster] in the
                   instance.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance, update_mask]
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
            request, bigtable_instance_admin.PartialUpdateInstanceRequest
        ):
            request = bigtable_instance_admin.PartialUpdateInstanceRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if instance is not None:
            request.instance = instance
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.partial_update_instance
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("instance.name", request.instance.name),)
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
            gba_instance.Instance,
            metadata_type=bigtable_instance_admin.UpdateInstanceMetadata,
        )

        # Done; return the response.
        return response

    async def delete_instance(
        self,
        request: Optional[
            Union[bigtable_instance_admin.DeleteInstanceRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Delete an instance from a project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_instance():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteInstanceRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_instance(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteInstanceRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.DeleteInstance.
            name (:class:`str`):
                Required. The unique name of the instance to be deleted.
                Values are of the form
                ``projects/{project}/instances/{instance}``.

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
        if not isinstance(request, bigtable_instance_admin.DeleteInstanceRequest):
            request = bigtable_instance_admin.DeleteInstanceRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_instance
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

    async def create_cluster(
        self,
        request: Optional[
            Union[bigtable_instance_admin.CreateClusterRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        cluster_id: Optional[str] = None,
        cluster: Optional[instance.Cluster] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a cluster within an instance.

        Note that exactly one of Cluster.serve_nodes and
        Cluster.cluster_config.cluster_autoscaling_config can be set. If
        serve_nodes is set to non-zero, then the cluster is manually
        scaled. If cluster_config.cluster_autoscaling_config is
        non-empty, then autoscaling is enabled.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_cluster():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CreateClusterRequest(
                    parent="parent_value",
                    cluster_id="cluster_id_value",
                )

                # Make the request
                operation = client.create_cluster(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateClusterRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.CreateCluster.
            parent (:class:`str`):
                Required. The unique name of the instance in which to
                create the new cluster. Values are of the form
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            cluster_id (:class:`str`):
                Required. The ID to be used when referring to the new
                cluster within its instance, e.g., just ``mycluster``
                rather than
                ``projects/myproject/instances/myinstance/clusters/mycluster``.

                This corresponds to the ``cluster_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            cluster (:class:`google.cloud.bigtable_admin_v2.types.Cluster`):
                Required. The cluster to be created. Fields marked
                ``OutputOnly`` must be left blank.

                This corresponds to the ``cluster`` field
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Cluster` A resizable group of nodes in a particular cloud location, capable
                   of serving all
                   [Tables][google.bigtable.admin.v2.Table] in the
                   parent [Instance][google.bigtable.admin.v2.Instance].

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, cluster_id, cluster]
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
        if not isinstance(request, bigtable_instance_admin.CreateClusterRequest):
            request = bigtable_instance_admin.CreateClusterRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if cluster_id is not None:
            request.cluster_id = cluster_id
        if cluster is not None:
            request.cluster = cluster

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_cluster
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
            instance.Cluster,
            metadata_type=bigtable_instance_admin.CreateClusterMetadata,
        )

        # Done; return the response.
        return response

    async def get_cluster(
        self,
        request: Optional[
            Union[bigtable_instance_admin.GetClusterRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.Cluster:
        r"""Gets information about a cluster.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_cluster():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetClusterRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_cluster(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetClusterRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.GetCluster.
            name (:class:`str`):
                Required. The unique name of the requested cluster.
                Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

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
            google.cloud.bigtable_admin_v2.types.Cluster:
                A resizable group of nodes in a particular cloud location, capable
                   of serving all
                   [Tables][google.bigtable.admin.v2.Table] in the
                   parent [Instance][google.bigtable.admin.v2.Instance].

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
        if not isinstance(request, bigtable_instance_admin.GetClusterRequest):
            request = bigtable_instance_admin.GetClusterRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_cluster
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

    async def list_clusters(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListClustersRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable_instance_admin.ListClustersResponse:
        r"""Lists information about clusters in an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_clusters():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListClustersRequest(
                    parent="parent_value",
                )

                # Make the request
                response = await client.list_clusters(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListClustersRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListClusters.
            parent (:class:`str`):
                Required. The unique name of the instance for which a
                list of clusters is requested. Values are of the form
                ``projects/{project}/instances/{instance}``. Use
                ``{instance} = '-'`` to list Clusters for all Instances
                in a project, e.g., ``projects/myproject/instances/-``.

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
            google.cloud.bigtable_admin_v2.types.ListClustersResponse:
                Response message for
                BigtableInstanceAdmin.ListClusters.

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
        if not isinstance(request, bigtable_instance_admin.ListClustersRequest):
            request = bigtable_instance_admin.ListClustersRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_clusters
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

    async def update_cluster(
        self,
        request: Optional[Union[instance.Cluster, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a cluster within an instance.

        Note that UpdateCluster does not support updating
        cluster_config.cluster_autoscaling_config. In order to update
        it, you must use PartialUpdateCluster.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_cluster():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.Cluster(
                )

                # Make the request
                operation = client.update_cluster(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.Cluster, dict]]):
                The request object. A resizable group of nodes in a particular cloud
                location, capable of serving all
                [Tables][google.bigtable.admin.v2.Table] in the parent
                [Instance][google.bigtable.admin.v2.Instance].
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Cluster` A resizable group of nodes in a particular cloud location, capable
                   of serving all
                   [Tables][google.bigtable.admin.v2.Table] in the
                   parent [Instance][google.bigtable.admin.v2.Instance].

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, instance.Cluster):
            request = instance.Cluster(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_cluster
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
            instance.Cluster,
            metadata_type=bigtable_instance_admin.UpdateClusterMetadata,
        )

        # Done; return the response.
        return response

    async def partial_update_cluster(
        self,
        request: Optional[
            Union[bigtable_instance_admin.PartialUpdateClusterRequest, dict]
        ] = None,
        *,
        cluster: Optional[instance.Cluster] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Partially updates a cluster within a project. This method is the
        preferred way to update a Cluster.

        To enable and update autoscaling, set
        cluster_config.cluster_autoscaling_config. When autoscaling is
        enabled, serve_nodes is treated as an OUTPUT_ONLY field, meaning
        that updates to it are ignored. Note that an update cannot
        simultaneously set serve_nodes to non-zero and
        cluster_config.cluster_autoscaling_config to non-empty, and also
        specify both in the update_mask.

        To disable autoscaling, clear
        cluster_config.cluster_autoscaling_config, and explicitly set a
        serve_node count via the update_mask.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_partial_update_cluster():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.PartialUpdateClusterRequest(
                )

                # Make the request
                operation = client.partial_update_cluster(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.PartialUpdateClusterRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.PartialUpdateCluster.
            cluster (:class:`google.cloud.bigtable_admin_v2.types.Cluster`):
                Required. The Cluster which contains the partial updates
                to be applied, subject to the update_mask.

                This corresponds to the ``cluster`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. The subset of Cluster
                fields which should be replaced.

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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Cluster` A resizable group of nodes in a particular cloud location, capable
                   of serving all
                   [Tables][google.bigtable.admin.v2.Table] in the
                   parent [Instance][google.bigtable.admin.v2.Instance].

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [cluster, update_mask]
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
        if not isinstance(request, bigtable_instance_admin.PartialUpdateClusterRequest):
            request = bigtable_instance_admin.PartialUpdateClusterRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if cluster is not None:
            request.cluster = cluster
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.partial_update_cluster
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("cluster.name", request.cluster.name),)
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
            instance.Cluster,
            metadata_type=bigtable_instance_admin.PartialUpdateClusterMetadata,
        )

        # Done; return the response.
        return response

    async def delete_cluster(
        self,
        request: Optional[
            Union[bigtable_instance_admin.DeleteClusterRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a cluster from an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_cluster():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteClusterRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_cluster(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteClusterRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.DeleteCluster.
            name (:class:`str`):
                Required. The unique name of the cluster to be deleted.
                Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

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
        if not isinstance(request, bigtable_instance_admin.DeleteClusterRequest):
            request = bigtable_instance_admin.DeleteClusterRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_cluster
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

    async def create_app_profile(
        self,
        request: Optional[
            Union[bigtable_instance_admin.CreateAppProfileRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        app_profile: Optional[instance.AppProfile] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.AppProfile:
        r"""Creates an app profile within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_app_profile():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                app_profile = bigtable_admin_v2.AppProfile()
                app_profile.priority = "PRIORITY_HIGH"

                request = bigtable_admin_v2.CreateAppProfileRequest(
                    parent="parent_value",
                    app_profile_id="app_profile_id_value",
                    app_profile=app_profile,
                )

                # Make the request
                response = await client.create_app_profile(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateAppProfileRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.CreateAppProfile.
            parent (:class:`str`):
                Required. The unique name of the instance in which to
                create the new app profile. Values are of the form
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                Required. The ID to be used when referring to the new
                app profile within its instance, e.g., just
                ``myprofile`` rather than
                ``projects/myproject/instances/myinstance/appProfiles/myprofile``.

                This corresponds to the ``app_profile_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile (:class:`google.cloud.bigtable_admin_v2.types.AppProfile`):
                Required. The app profile to be created. Fields marked
                ``OutputOnly`` will be ignored.

                This corresponds to the ``app_profile`` field
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
            google.cloud.bigtable_admin_v2.types.AppProfile:
                A configuration object describing how
                Cloud Bigtable should treat traffic from
                a particular end user application.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, app_profile_id, app_profile]
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
        if not isinstance(request, bigtable_instance_admin.CreateAppProfileRequest):
            request = bigtable_instance_admin.CreateAppProfileRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id
        if app_profile is not None:
            request.app_profile = app_profile

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_app_profile
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

    async def get_app_profile(
        self,
        request: Optional[
            Union[bigtable_instance_admin.GetAppProfileRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.AppProfile:
        r"""Gets information about an app profile.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_app_profile():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetAppProfileRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_app_profile(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetAppProfileRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.GetAppProfile.
            name (:class:`str`):
                Required. The unique name of the requested app profile.
                Values are of the form
                ``projects/{project}/instances/{instance}/appProfiles/{app_profile}``.

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
            google.cloud.bigtable_admin_v2.types.AppProfile:
                A configuration object describing how
                Cloud Bigtable should treat traffic from
                a particular end user application.

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
        if not isinstance(request, bigtable_instance_admin.GetAppProfileRequest):
            request = bigtable_instance_admin.GetAppProfileRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_app_profile
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

    async def list_app_profiles(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListAppProfilesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListAppProfilesAsyncPager:
        r"""Lists information about app profiles in an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_app_profiles():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListAppProfilesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_app_profiles(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListAppProfilesRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListAppProfiles.
            parent (:class:`str`):
                Required. The unique name of the instance for which a
                list of app profiles is requested. Values are of the
                form ``projects/{project}/instances/{instance}``. Use
                ``{instance} = '-'`` to list AppProfiles for all
                Instances in a project, e.g.,
                ``projects/myproject/instances/-``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_instance_admin.pagers.ListAppProfilesAsyncPager:
                Response message for
                BigtableInstanceAdmin.ListAppProfiles.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

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
        if not isinstance(request, bigtable_instance_admin.ListAppProfilesRequest):
            request = bigtable_instance_admin.ListAppProfilesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_app_profiles
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
        response = pagers.ListAppProfilesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_app_profile(
        self,
        request: Optional[
            Union[bigtable_instance_admin.UpdateAppProfileRequest, dict]
        ] = None,
        *,
        app_profile: Optional[instance.AppProfile] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates an app profile within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_app_profile():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                app_profile = bigtable_admin_v2.AppProfile()
                app_profile.priority = "PRIORITY_HIGH"

                request = bigtable_admin_v2.UpdateAppProfileRequest(
                    app_profile=app_profile,
                )

                # Make the request
                operation = client.update_app_profile(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateAppProfileRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.UpdateAppProfile.
            app_profile (:class:`google.cloud.bigtable_admin_v2.types.AppProfile`):
                Required. The app profile which will
                (partially) replace the current value.

                This corresponds to the ``app_profile`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. The subset of app profile
                fields which should be replaced. If
                unset, all fields will be replaced.

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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.AppProfile` A configuration object describing how Cloud Bigtable should treat traffic
                   from a particular end user application.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [app_profile, update_mask]
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
        if not isinstance(request, bigtable_instance_admin.UpdateAppProfileRequest):
            request = bigtable_instance_admin.UpdateAppProfileRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if app_profile is not None:
            request.app_profile = app_profile
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_app_profile
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("app_profile.name", request.app_profile.name),)
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
            instance.AppProfile,
            metadata_type=bigtable_instance_admin.UpdateAppProfileMetadata,
        )

        # Done; return the response.
        return response

    async def delete_app_profile(
        self,
        request: Optional[
            Union[bigtable_instance_admin.DeleteAppProfileRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        ignore_warnings: Optional[bool] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes an app profile from an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_app_profile():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteAppProfileRequest(
                    name="name_value",
                    ignore_warnings=True,
                )

                # Make the request
                await client.delete_app_profile(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteAppProfileRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.DeleteAppProfile.
            name (:class:`str`):
                Required. The unique name of the app profile to be
                deleted. Values are of the form
                ``projects/{project}/instances/{instance}/appProfiles/{app_profile}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            ignore_warnings (:class:`bool`):
                Required. If true, ignore safety
                checks when deleting the app profile.

                This corresponds to the ``ignore_warnings`` field
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
        flattened_params = [name, ignore_warnings]
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
        if not isinstance(request, bigtable_instance_admin.DeleteAppProfileRequest):
            request = bigtable_instance_admin.DeleteAppProfileRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if ignore_warnings is not None:
            request.ignore_warnings = ignore_warnings

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_app_profile
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

    async def get_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.GetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Gets the access control policy for an instance
        resource. Returns an empty policy if an instance exists
        but does not have a policy set.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            async def sample_get_iam_policy():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.GetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = await client.get_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.iam.v1.iam_policy_pb2.GetIamPolicyRequest, dict]]):
                The request object. Request message for ``GetIamPolicy`` method.
            resource (:class:`str`):
                REQUIRED: The resource for which the
                policy is being requested. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
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
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](\ https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:`\`     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:`\`     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](\ https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - The request isn't a proto-plus wrapped type,
        #   so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = iam_policy_pb2.GetIamPolicyRequest(**request)
        elif not request:
            request = iam_policy_pb2.GetIamPolicyRequest(resource=resource)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_iam_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
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

    async def set_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.SetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Sets the access control policy on an instance
        resource. Replaces any existing policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            async def sample_set_iam_policy():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.SetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = await client.set_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.iam.v1.iam_policy_pb2.SetIamPolicyRequest, dict]]):
                The request object. Request message for ``SetIamPolicy`` method.
            resource (:class:`str`):
                REQUIRED: The resource for which the
                policy is being specified. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
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
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](\ https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:`\`     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:`\`     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](\ https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - The request isn't a proto-plus wrapped type,
        #   so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = iam_policy_pb2.SetIamPolicyRequest(**request)
        elif not request:
            request = iam_policy_pb2.SetIamPolicyRequest(resource=resource)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.set_iam_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
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

    async def test_iam_permissions(
        self,
        request: Optional[Union[iam_policy_pb2.TestIamPermissionsRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        permissions: Optional[MutableSequence[str]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        r"""Returns permissions that the caller has on the
        specified instance resource.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            async def sample_test_iam_permissions():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.TestIamPermissionsRequest(
                    resource="resource_value",
                    permissions=['permissions_value1', 'permissions_value2'],
                )

                # Make the request
                response = await client.test_iam_permissions(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.iam.v1.iam_policy_pb2.TestIamPermissionsRequest, dict]]):
                The request object. Request message for ``TestIamPermissions`` method.
            resource (:class:`str`):
                REQUIRED: The resource for which the
                policy detail is being requested. See
                the operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            permissions (:class:`MutableSequence[str]`):
                The set of permissions to check for the ``resource``.
                Permissions with wildcards (such as '*' or 'storage.*')
                are not allowed. For more information see `IAM
                Overview <https://cloud.google.com/iam/docs/overview#permissions>`__.

                This corresponds to the ``permissions`` field
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
            google.iam.v1.iam_policy_pb2.TestIamPermissionsResponse:
                Response message for TestIamPermissions method.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource, permissions]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - The request isn't a proto-plus wrapped type,
        #   so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = iam_policy_pb2.TestIamPermissionsRequest(**request)
        elif not request:
            request = iam_policy_pb2.TestIamPermissionsRequest(
                resource=resource, permissions=permissions
            )

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.test_iam_permissions
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
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

    async def list_hot_tablets(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListHotTabletsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListHotTabletsAsyncPager:
        r"""Lists hot tablets in a cluster, within the time range
        provided. Hot tablets are ordered based on CPU usage.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_hot_tablets():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListHotTabletsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_hot_tablets(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListHotTabletsRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListHotTablets.
            parent (:class:`str`):
                Required. The cluster name to list hot tablets. Value is
                in the following form:
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_instance_admin.pagers.ListHotTabletsAsyncPager:
                Response message for
                BigtableInstanceAdmin.ListHotTablets.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

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
        if not isinstance(request, bigtable_instance_admin.ListHotTabletsRequest):
            request = bigtable_instance_admin.ListHotTabletsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_hot_tablets
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
        response = pagers.ListHotTabletsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def create_logical_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.CreateLogicalViewRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        logical_view: Optional[instance.LogicalView] = None,
        logical_view_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a logical view within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_logical_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                logical_view = bigtable_admin_v2.LogicalView()
                logical_view.query = "query_value"

                request = bigtable_admin_v2.CreateLogicalViewRequest(
                    parent="parent_value",
                    logical_view_id="logical_view_id_value",
                    logical_view=logical_view,
                )

                # Make the request
                operation = client.create_logical_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateLogicalViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.CreateLogicalView.
            parent (:class:`str`):
                Required. The parent instance where this logical view
                will be created. Format:
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            logical_view (:class:`google.cloud.bigtable_admin_v2.types.LogicalView`):
                Required. The logical view to create.
                This corresponds to the ``logical_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            logical_view_id (:class:`str`):
                Required. The ID to use for the
                logical view, which will become the
                final component of the logical view's
                resource name.

                This corresponds to the ``logical_view_id`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.LogicalView`
                A SQL logical view object that can be referenced in SQL
                queries.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, logical_view, logical_view_id]
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
        if not isinstance(request, bigtable_instance_admin.CreateLogicalViewRequest):
            request = bigtable_instance_admin.CreateLogicalViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if logical_view is not None:
            request.logical_view = logical_view
        if logical_view_id is not None:
            request.logical_view_id = logical_view_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_logical_view
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
            instance.LogicalView,
            metadata_type=bigtable_instance_admin.CreateLogicalViewMetadata,
        )

        # Done; return the response.
        return response

    async def get_logical_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.GetLogicalViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.LogicalView:
        r"""Gets information about a logical view.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_logical_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetLogicalViewRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_logical_view(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetLogicalViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.GetLogicalView.
            name (:class:`str`):
                Required. The unique name of the requested logical view.
                Values are of the form
                ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.

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
            google.cloud.bigtable_admin_v2.types.LogicalView:
                A SQL logical view object that can be
                referenced in SQL queries.

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
        if not isinstance(request, bigtable_instance_admin.GetLogicalViewRequest):
            request = bigtable_instance_admin.GetLogicalViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_logical_view
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

    async def list_logical_views(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListLogicalViewsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListLogicalViewsAsyncPager:
        r"""Lists information about logical views in an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_logical_views():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListLogicalViewsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_logical_views(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListLogicalViewsRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListLogicalViews.
            parent (:class:`str`):
                Required. The unique name of the instance for which the
                list of logical views is requested. Values are of the
                form ``projects/{project}/instances/{instance}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_instance_admin.pagers.ListLogicalViewsAsyncPager:
                Response message for
                BigtableInstanceAdmin.ListLogicalViews.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

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
        if not isinstance(request, bigtable_instance_admin.ListLogicalViewsRequest):
            request = bigtable_instance_admin.ListLogicalViewsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_logical_views
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
        response = pagers.ListLogicalViewsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_logical_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.UpdateLogicalViewRequest, dict]
        ] = None,
        *,
        logical_view: Optional[instance.LogicalView] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a logical view within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_logical_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                logical_view = bigtable_admin_v2.LogicalView()
                logical_view.query = "query_value"

                request = bigtable_admin_v2.UpdateLogicalViewRequest(
                    logical_view=logical_view,
                )

                # Make the request
                operation = client.update_logical_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateLogicalViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.UpdateLogicalView.
            logical_view (:class:`google.cloud.bigtable_admin_v2.types.LogicalView`):
                Required. The logical view to update.

                The logical view's ``name`` field is used to identify
                the view to update. Format:
                ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.

                This corresponds to the ``logical_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Optional. The list of fields to
                update.

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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.bigtable_admin_v2.types.LogicalView`
                A SQL logical view object that can be referenced in SQL
                queries.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [logical_view, update_mask]
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
        if not isinstance(request, bigtable_instance_admin.UpdateLogicalViewRequest):
            request = bigtable_instance_admin.UpdateLogicalViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if logical_view is not None:
            request.logical_view = logical_view
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_logical_view
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("logical_view.name", request.logical_view.name),)
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
            instance.LogicalView,
            metadata_type=bigtable_instance_admin.UpdateLogicalViewMetadata,
        )

        # Done; return the response.
        return response

    async def delete_logical_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.DeleteLogicalViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a logical view from an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_logical_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteLogicalViewRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_logical_view(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteLogicalViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.DeleteLogicalView.
            name (:class:`str`):
                Required. The unique name of the logical view to be
                deleted. Format:
                ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.

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
        if not isinstance(request, bigtable_instance_admin.DeleteLogicalViewRequest):
            request = bigtable_instance_admin.DeleteLogicalViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_logical_view
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

    async def create_materialized_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.CreateMaterializedViewRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        materialized_view: Optional[instance.MaterializedView] = None,
        materialized_view_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a materialized view within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_materialized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                materialized_view = bigtable_admin_v2.MaterializedView()
                materialized_view.query = "query_value"

                request = bigtable_admin_v2.CreateMaterializedViewRequest(
                    parent="parent_value",
                    materialized_view_id="materialized_view_id_value",
                    materialized_view=materialized_view,
                )

                # Make the request
                operation = client.create_materialized_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateMaterializedViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.CreateMaterializedView.
            parent (:class:`str`):
                Required. The parent instance where this materialized
                view will be created. Format:
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            materialized_view (:class:`google.cloud.bigtable_admin_v2.types.MaterializedView`):
                Required. The materialized view to
                create.

                This corresponds to the ``materialized_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            materialized_view_id (:class:`str`):
                Required. The ID to use for the
                materialized view, which will become the
                final component of the materialized
                view's resource name.

                This corresponds to the ``materialized_view_id`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.MaterializedView`
                A materialized view object that can be referenced in SQL
                queries.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, materialized_view, materialized_view_id]
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
            request, bigtable_instance_admin.CreateMaterializedViewRequest
        ):
            request = bigtable_instance_admin.CreateMaterializedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if materialized_view is not None:
            request.materialized_view = materialized_view
        if materialized_view_id is not None:
            request.materialized_view_id = materialized_view_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_materialized_view
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
            instance.MaterializedView,
            metadata_type=bigtable_instance_admin.CreateMaterializedViewMetadata,
        )

        # Done; return the response.
        return response

    async def get_materialized_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.GetMaterializedViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> instance.MaterializedView:
        r"""Gets information about a materialized view.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_materialized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetMaterializedViewRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_materialized_view(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetMaterializedViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.GetMaterializedView.
            name (:class:`str`):
                Required. The unique name of the requested materialized
                view. Values are of the form
                ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.

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
            google.cloud.bigtable_admin_v2.types.MaterializedView:
                A materialized view object that can
                be referenced in SQL queries.

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
        if not isinstance(request, bigtable_instance_admin.GetMaterializedViewRequest):
            request = bigtable_instance_admin.GetMaterializedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_materialized_view
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

    async def list_materialized_views(
        self,
        request: Optional[
            Union[bigtable_instance_admin.ListMaterializedViewsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListMaterializedViewsAsyncPager:
        r"""Lists information about materialized views in an
        instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_materialized_views():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListMaterializedViewsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_materialized_views(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListMaterializedViewsRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.ListMaterializedViews.
            parent (:class:`str`):
                Required. The unique name of the instance for which the
                list of materialized views is requested. Values are of
                the form ``projects/{project}/instances/{instance}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_instance_admin.pagers.ListMaterializedViewsAsyncPager:
                Response message for
                BigtableInstanceAdmin.ListMaterializedViews.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

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
        if not isinstance(
            request, bigtable_instance_admin.ListMaterializedViewsRequest
        ):
            request = bigtable_instance_admin.ListMaterializedViewsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_materialized_views
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
        response = pagers.ListMaterializedViewsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_materialized_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.UpdateMaterializedViewRequest, dict]
        ] = None,
        *,
        materialized_view: Optional[instance.MaterializedView] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a materialized view within an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_materialized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                materialized_view = bigtable_admin_v2.MaterializedView()
                materialized_view.query = "query_value"

                request = bigtable_admin_v2.UpdateMaterializedViewRequest(
                    materialized_view=materialized_view,
                )

                # Make the request
                operation = client.update_materialized_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateMaterializedViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.UpdateMaterializedView.
            materialized_view (:class:`google.cloud.bigtable_admin_v2.types.MaterializedView`):
                Required. The materialized view to update.

                The materialized view's ``name`` field is used to
                identify the view to update. Format:
                ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.

                This corresponds to the ``materialized_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Optional. The list of fields to
                update.

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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.bigtable_admin_v2.types.MaterializedView`
                A materialized view object that can be referenced in SQL
                queries.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [materialized_view, update_mask]
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
            request, bigtable_instance_admin.UpdateMaterializedViewRequest
        ):
            request = bigtable_instance_admin.UpdateMaterializedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if materialized_view is not None:
            request.materialized_view = materialized_view
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_materialized_view
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("materialized_view.name", request.materialized_view.name),)
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
            instance.MaterializedView,
            metadata_type=bigtable_instance_admin.UpdateMaterializedViewMetadata,
        )

        # Done; return the response.
        return response

    async def delete_materialized_view(
        self,
        request: Optional[
            Union[bigtable_instance_admin.DeleteMaterializedViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a materialized view from an instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_materialized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableInstanceAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteMaterializedViewRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_materialized_view(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteMaterializedViewRequest, dict]]):
                The request object. Request message for
                BigtableInstanceAdmin.DeleteMaterializedView.
            name (:class:`str`):
                Required. The unique name of the materialized view to be
                deleted. Format:
                ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.

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
            request, bigtable_instance_admin.DeleteMaterializedViewRequest
        ):
            request = bigtable_instance_admin.DeleteMaterializedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_materialized_view
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

    async def __aenter__(self) -> "BigtableInstanceAdminAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("BigtableInstanceAdminAsyncClient",)

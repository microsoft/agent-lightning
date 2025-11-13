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
from google.cloud.bigtable_admin_v2.services.bigtable_table_admin import pagers
from google.cloud.bigtable_admin_v2.types import bigtable_table_admin
from google.cloud.bigtable_admin_v2.types import table
from google.cloud.bigtable_admin_v2.types import table as gba_table
from google.cloud.bigtable_admin_v2.types import types
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from .transports.base import BigtableTableAdminTransport, DEFAULT_CLIENT_INFO
from .transports.grpc_asyncio import BigtableTableAdminGrpcAsyncIOTransport
from .client import BaseBigtableTableAdminClient

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class BaseBigtableTableAdminAsyncClient:
    """Service for creating, configuring, and deleting Cloud
    Bigtable tables.

    Provides access to the table schemas only, not the data stored
    within the tables.
    """

    _client: BaseBigtableTableAdminClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = BaseBigtableTableAdminClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = BaseBigtableTableAdminClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = BaseBigtableTableAdminClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = BaseBigtableTableAdminClient._DEFAULT_UNIVERSE

    authorized_view_path = staticmethod(
        BaseBigtableTableAdminClient.authorized_view_path
    )
    parse_authorized_view_path = staticmethod(
        BaseBigtableTableAdminClient.parse_authorized_view_path
    )
    backup_path = staticmethod(BaseBigtableTableAdminClient.backup_path)
    parse_backup_path = staticmethod(BaseBigtableTableAdminClient.parse_backup_path)
    cluster_path = staticmethod(BaseBigtableTableAdminClient.cluster_path)
    parse_cluster_path = staticmethod(BaseBigtableTableAdminClient.parse_cluster_path)
    crypto_key_version_path = staticmethod(
        BaseBigtableTableAdminClient.crypto_key_version_path
    )
    parse_crypto_key_version_path = staticmethod(
        BaseBigtableTableAdminClient.parse_crypto_key_version_path
    )
    instance_path = staticmethod(BaseBigtableTableAdminClient.instance_path)
    parse_instance_path = staticmethod(BaseBigtableTableAdminClient.parse_instance_path)
    schema_bundle_path = staticmethod(BaseBigtableTableAdminClient.schema_bundle_path)
    parse_schema_bundle_path = staticmethod(
        BaseBigtableTableAdminClient.parse_schema_bundle_path
    )
    snapshot_path = staticmethod(BaseBigtableTableAdminClient.snapshot_path)
    parse_snapshot_path = staticmethod(BaseBigtableTableAdminClient.parse_snapshot_path)
    table_path = staticmethod(BaseBigtableTableAdminClient.table_path)
    parse_table_path = staticmethod(BaseBigtableTableAdminClient.parse_table_path)
    common_billing_account_path = staticmethod(
        BaseBigtableTableAdminClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        BaseBigtableTableAdminClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(BaseBigtableTableAdminClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        BaseBigtableTableAdminClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        BaseBigtableTableAdminClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        BaseBigtableTableAdminClient.parse_common_organization_path
    )
    common_project_path = staticmethod(BaseBigtableTableAdminClient.common_project_path)
    parse_common_project_path = staticmethod(
        BaseBigtableTableAdminClient.parse_common_project_path
    )
    common_location_path = staticmethod(
        BaseBigtableTableAdminClient.common_location_path
    )
    parse_common_location_path = staticmethod(
        BaseBigtableTableAdminClient.parse_common_location_path
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
            BaseBigtableTableAdminAsyncClient: The constructed client.
        """
        return BaseBigtableTableAdminClient.from_service_account_info.__func__(BaseBigtableTableAdminAsyncClient, info, *args, **kwargs)  # type: ignore

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
            BaseBigtableTableAdminAsyncClient: The constructed client.
        """
        return BaseBigtableTableAdminClient.from_service_account_file.__func__(BaseBigtableTableAdminAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return BaseBigtableTableAdminClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> BigtableTableAdminTransport:
        """Returns the transport used by the client instance.

        Returns:
            BigtableTableAdminTransport: The transport used by the client instance.
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

    get_transport_class = BaseBigtableTableAdminClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                BigtableTableAdminTransport,
                Callable[..., BigtableTableAdminTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the base bigtable table admin async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,BigtableTableAdminTransport,Callable[..., BigtableTableAdminTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the BigtableTableAdminTransport constructor.
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
        self._client = BaseBigtableTableAdminClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.bigtable.admin_v2.BaseBigtableTableAdminAsyncClient`.",
                extra={
                    "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
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
                    "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                    "credentialsType": None,
                },
            )

    async def create_table(
        self,
        request: Optional[Union[bigtable_table_admin.CreateTableRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        table_id: Optional[str] = None,
        table: Optional[gba_table.Table] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> gba_table.Table:
        r"""Creates a new table in the specified instance.
        The table can be created with a full set of initial
        column families, specified in the request.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CreateTableRequest(
                    parent="parent_value",
                    table_id="table_id_value",
                )

                # Make the request
                response = await client.create_table(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateTableRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CreateTable][google.bigtable.admin.v2.BigtableTableAdmin.CreateTable]
            parent (:class:`str`):
                Required. The unique name of the instance in which to
                create the table. Values are of the form
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            table_id (:class:`str`):
                Required. The name by which the new table should be
                referred to within the parent instance, e.g., ``foobar``
                rather than ``{parent}/tables/foobar``. Maximum 50
                characters.

                This corresponds to the ``table_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            table (:class:`google.cloud.bigtable_admin_v2.types.Table`):
                Required. The Table to create.
                This corresponds to the ``table`` field
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
            google.cloud.bigtable_admin_v2.types.Table:
                A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, table_id, table]
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
        if not isinstance(request, bigtable_table_admin.CreateTableRequest):
            request = bigtable_table_admin.CreateTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if table_id is not None:
            request.table_id = table_id
        if table is not None:
            request.table = table

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_table
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

    async def create_table_from_snapshot(
        self,
        request: Optional[
            Union[bigtable_table_admin.CreateTableFromSnapshotRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        table_id: Optional[str] = None,
        source_snapshot: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a new table from the specified snapshot. The
        target table must not exist. The snapshot and the table
        must be in the same instance.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_table_from_snapshot():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CreateTableFromSnapshotRequest(
                    parent="parent_value",
                    table_id="table_id_value",
                    source_snapshot="source_snapshot_value",
                )

                # Make the request
                operation = client.create_table_from_snapshot(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateTableFromSnapshotRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
            parent (:class:`str`):
                Required. The unique name of the instance in which to
                create the table. Values are of the form
                ``projects/{project}/instances/{instance}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            table_id (:class:`str`):
                Required. The name by which the new table should be
                referred to within the parent instance, e.g., ``foobar``
                rather than ``{parent}/tables/foobar``.

                This corresponds to the ``table_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            source_snapshot (:class:`str`):
                Required. The unique name of the snapshot from which to
                restore the table. The snapshot and the table must be in
                the same instance. Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.

                This corresponds to the ``source_snapshot`` field
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Table` A collection of user data indexed by row, column, and timestamp.
                   Each table is served using the resources of its
                   parent cluster.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, table_id, source_snapshot]
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
        if not isinstance(request, bigtable_table_admin.CreateTableFromSnapshotRequest):
            request = bigtable_table_admin.CreateTableFromSnapshotRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if table_id is not None:
            request.table_id = table_id
        if source_snapshot is not None:
            request.source_snapshot = source_snapshot

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_table_from_snapshot
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
            table.Table,
            metadata_type=bigtable_table_admin.CreateTableFromSnapshotMetadata,
        )

        # Done; return the response.
        return response

    async def list_tables(
        self,
        request: Optional[Union[bigtable_table_admin.ListTablesRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListTablesAsyncPager:
        r"""Lists all tables served from a specified instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_tables():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListTablesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_tables(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListTablesRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]
            parent (:class:`str`):
                Required. The unique name of the instance for which
                tables should be listed. Values are of the form
                ``projects/{project}/instances/{instance}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_table_admin.pagers.ListTablesAsyncPager:
                Response message for
                   [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]

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
        if not isinstance(request, bigtable_table_admin.ListTablesRequest):
            request = bigtable_table_admin.ListTablesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_tables
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
        response = pagers.ListTablesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_table(
        self,
        request: Optional[Union[bigtable_table_admin.GetTableRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.Table:
        r"""Gets metadata information about the specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetTableRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_table(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetTableRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetTable][google.bigtable.admin.v2.BigtableTableAdmin.GetTable]
            name (:class:`str`):
                Required. The unique name of the requested table. Values
                are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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
            google.cloud.bigtable_admin_v2.types.Table:
                A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

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
        if not isinstance(request, bigtable_table_admin.GetTableRequest):
            request = bigtable_table_admin.GetTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_table
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

    async def update_table(
        self,
        request: Optional[Union[bigtable_table_admin.UpdateTableRequest, dict]] = None,
        *,
        table: Optional[gba_table.Table] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.UpdateTableRequest(
                )

                # Make the request
                operation = client.update_table(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateTableRequest, dict]]):
                The request object. The request for
                [UpdateTable][google.bigtable.admin.v2.BigtableTableAdmin.UpdateTable].
            table (:class:`google.cloud.bigtable_admin_v2.types.Table`):
                Required. The table to update. The table's ``name``
                field is used to identify the table to update.

                This corresponds to the ``table`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. The list of fields to update. A mask
                specifying which fields (e.g. ``change_stream_config``)
                in the ``table`` field should be updated. This mask is
                relative to the ``table`` field, not to the request
                message. The wildcard (*) path is currently not
                supported. Currently UpdateTable is only supported for
                the following fields:

                -  ``change_stream_config``
                -  ``change_stream_config.retention_period``
                -  ``deletion_protection``
                -  ``row_key_schema``

                If ``column_families`` is set in ``update_mask``, it
                will return an UNIMPLEMENTED error.

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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Table` A collection of user data indexed by row, column, and timestamp.
                   Each table is served using the resources of its
                   parent cluster.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table, update_mask]
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
        if not isinstance(request, bigtable_table_admin.UpdateTableRequest):
            request = bigtable_table_admin.UpdateTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table is not None:
            request.table = table
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_table
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("table.name", request.table.name),)
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
            gba_table.Table,
            metadata_type=bigtable_table_admin.UpdateTableMetadata,
        )

        # Done; return the response.
        return response

    async def delete_table(
        self,
        request: Optional[Union[bigtable_table_admin.DeleteTableRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Permanently deletes a specified table and all of its
        data.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteTableRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_table(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteTableRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable]
            name (:class:`str`):
                Required. The unique name of the table to be deleted.
                Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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
        if not isinstance(request, bigtable_table_admin.DeleteTableRequest):
            request = bigtable_table_admin.DeleteTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_table
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

    async def undelete_table(
        self,
        request: Optional[
            Union[bigtable_table_admin.UndeleteTableRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Restores a specified table which was accidentally
        deleted.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_undelete_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.UndeleteTableRequest(
                    name="name_value",
                )

                # Make the request
                operation = client.undelete_table(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UndeleteTableRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable]
            name (:class:`str`):
                Required. The unique name of the table to be restored.
                Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Table` A collection of user data indexed by row, column, and timestamp.
                   Each table is served using the resources of its
                   parent cluster.

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
        if not isinstance(request, bigtable_table_admin.UndeleteTableRequest):
            request = bigtable_table_admin.UndeleteTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.undelete_table
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
            table.Table,
            metadata_type=bigtable_table_admin.UndeleteTableMetadata,
        )

        # Done; return the response.
        return response

    async def create_authorized_view(
        self,
        request: Optional[
            Union[bigtable_table_admin.CreateAuthorizedViewRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        authorized_view: Optional[table.AuthorizedView] = None,
        authorized_view_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a new AuthorizedView in a table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_authorized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CreateAuthorizedViewRequest(
                    parent="parent_value",
                    authorized_view_id="authorized_view_id_value",
                )

                # Make the request
                operation = client.create_authorized_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateAuthorizedViewRequest, dict]]):
                The request object. The request for
                [CreateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.CreateAuthorizedView]
            parent (:class:`str`):
                Required. This is the name of the table the
                AuthorizedView belongs to. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            authorized_view (:class:`google.cloud.bigtable_admin_v2.types.AuthorizedView`):
                Required. The AuthorizedView to
                create.

                This corresponds to the ``authorized_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            authorized_view_id (:class:`str`):
                Required. The id of the AuthorizedView to create. This
                AuthorizedView must not already exist. The
                ``authorized_view_id`` appended to ``parent`` forms the
                full AuthorizedView name of the form
                ``projects/{project}/instances/{instance}/tables/{table}/authorizedView/{authorized_view}``.

                This corresponds to the ``authorized_view_id`` field
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.AuthorizedView` AuthorizedViews represent subsets of a particular Cloud Bigtable table. Users
                   can configure access to each Authorized View
                   independently from the table and use the existing
                   Data APIs to access the subset of data.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, authorized_view, authorized_view_id]
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
        if not isinstance(request, bigtable_table_admin.CreateAuthorizedViewRequest):
            request = bigtable_table_admin.CreateAuthorizedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if authorized_view is not None:
            request.authorized_view = authorized_view
        if authorized_view_id is not None:
            request.authorized_view_id = authorized_view_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_authorized_view
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
            table.AuthorizedView,
            metadata_type=bigtable_table_admin.CreateAuthorizedViewMetadata,
        )

        # Done; return the response.
        return response

    async def list_authorized_views(
        self,
        request: Optional[
            Union[bigtable_table_admin.ListAuthorizedViewsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListAuthorizedViewsAsyncPager:
        r"""Lists all AuthorizedViews from a specific table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_authorized_views():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListAuthorizedViewsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_authorized_views(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListAuthorizedViewsRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]
            parent (:class:`str`):
                Required. The unique name of the table for which
                AuthorizedViews should be listed. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_table_admin.pagers.ListAuthorizedViewsAsyncPager:
                Response message for
                   [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]

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
        if not isinstance(request, bigtable_table_admin.ListAuthorizedViewsRequest):
            request = bigtable_table_admin.ListAuthorizedViewsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_authorized_views
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
        response = pagers.ListAuthorizedViewsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_authorized_view(
        self,
        request: Optional[
            Union[bigtable_table_admin.GetAuthorizedViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.AuthorizedView:
        r"""Gets information from a specified AuthorizedView.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_authorized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetAuthorizedViewRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_authorized_view(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetAuthorizedViewRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView]
            name (:class:`str`):
                Required. The unique name of the requested
                AuthorizedView. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.

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
            google.cloud.bigtable_admin_v2.types.AuthorizedView:
                AuthorizedViews represent subsets of
                a particular Cloud Bigtable table. Users
                can configure access to each Authorized
                View independently from the table and
                use the existing Data APIs to access the
                subset of data.

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
        if not isinstance(request, bigtable_table_admin.GetAuthorizedViewRequest):
            request = bigtable_table_admin.GetAuthorizedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_authorized_view
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

    async def update_authorized_view(
        self,
        request: Optional[
            Union[bigtable_table_admin.UpdateAuthorizedViewRequest, dict]
        ] = None,
        *,
        authorized_view: Optional[table.AuthorizedView] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates an AuthorizedView in a table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_authorized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.UpdateAuthorizedViewRequest(
                )

                # Make the request
                operation = client.update_authorized_view(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateAuthorizedViewRequest, dict]]):
                The request object. The request for
                [UpdateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.UpdateAuthorizedView].
            authorized_view (:class:`google.cloud.bigtable_admin_v2.types.AuthorizedView`):
                Required. The AuthorizedView to update. The ``name`` in
                ``authorized_view`` is used to identify the
                AuthorizedView. AuthorizedView name must in this format:
                ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.

                This corresponds to the ``authorized_view`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Optional. The list of fields to update. A mask
                specifying which fields in the AuthorizedView resource
                should be updated. This mask is relative to the
                AuthorizedView resource, not to the request message. A
                field will be overwritten if it is in the mask. If
                empty, all fields set in the request will be
                overwritten. A special value ``*`` means to overwrite
                all fields (including fields not set in the request).

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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.AuthorizedView` AuthorizedViews represent subsets of a particular Cloud Bigtable table. Users
                   can configure access to each Authorized View
                   independently from the table and use the existing
                   Data APIs to access the subset of data.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [authorized_view, update_mask]
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
        if not isinstance(request, bigtable_table_admin.UpdateAuthorizedViewRequest):
            request = bigtable_table_admin.UpdateAuthorizedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if authorized_view is not None:
            request.authorized_view = authorized_view
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_authorized_view
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("authorized_view.name", request.authorized_view.name),)
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
            table.AuthorizedView,
            metadata_type=bigtable_table_admin.UpdateAuthorizedViewMetadata,
        )

        # Done; return the response.
        return response

    async def delete_authorized_view(
        self,
        request: Optional[
            Union[bigtable_table_admin.DeleteAuthorizedViewRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Permanently deletes a specified AuthorizedView.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_authorized_view():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteAuthorizedViewRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_authorized_view(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteAuthorizedViewRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView]
            name (:class:`str`):
                Required. The unique name of the AuthorizedView to be
                deleted. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.

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
        if not isinstance(request, bigtable_table_admin.DeleteAuthorizedViewRequest):
            request = bigtable_table_admin.DeleteAuthorizedViewRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_authorized_view
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

    async def modify_column_families(
        self,
        request: Optional[
            Union[bigtable_table_admin.ModifyColumnFamiliesRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        modifications: Optional[
            MutableSequence[
                bigtable_table_admin.ModifyColumnFamiliesRequest.Modification
            ]
        ] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.Table:
        r"""Performs a series of column family modifications on
        the specified table. Either all or none of the
        modifications will occur before this method returns, but
        data requests received prior to that point may see a
        table where only some modifications have taken effect.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_modify_column_families():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ModifyColumnFamiliesRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.modify_column_families(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ModifyColumnFamiliesRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies][google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies]
            name (:class:`str`):
                Required. The unique name of the table whose families
                should be modified. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            modifications (:class:`MutableSequence[google.cloud.bigtable_admin_v2.types.ModifyColumnFamiliesRequest.Modification]`):
                Required. Modifications to be
                atomically applied to the specified
                table's families. Entries are applied in
                order, meaning that earlier
                modifications can be masked by later
                ones (in the case of repeated updates to
                the same family, for example).

                This corresponds to the ``modifications`` field
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
            google.cloud.bigtable_admin_v2.types.Table:
                A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, modifications]
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
        if not isinstance(request, bigtable_table_admin.ModifyColumnFamiliesRequest):
            request = bigtable_table_admin.ModifyColumnFamiliesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if modifications:
            request.modifications.extend(modifications)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.modify_column_families
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

    async def drop_row_range(
        self,
        request: Optional[Union[bigtable_table_admin.DropRowRangeRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Permanently drop/delete a row range from a specified
        table. The request can specify whether to delete all
        rows in a table, or only those that match a particular
        prefix.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_drop_row_range():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DropRowRangeRequest(
                    row_key_prefix=b'row_key_prefix_blob',
                    name="name_value",
                )

                # Make the request
                await client.drop_row_range(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DropRowRangeRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange][google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange]
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, bigtable_table_admin.DropRowRangeRequest):
            request = bigtable_table_admin.DropRowRangeRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.drop_row_range
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

    async def generate_consistency_token(
        self,
        request: Optional[
            Union[bigtable_table_admin.GenerateConsistencyTokenRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable_table_admin.GenerateConsistencyTokenResponse:
        r"""Generates a consistency token for a Table, which can
        be used in CheckConsistency to check whether mutations
        to the table that finished before this call started have
        been replicated. The tokens will be available for 90
        days.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_generate_consistency_token():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GenerateConsistencyTokenRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.generate_consistency_token(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GenerateConsistencyTokenRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]
            name (:class:`str`):
                Required. The unique name of the Table for which to
                create a consistency token. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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
            google.cloud.bigtable_admin_v2.types.GenerateConsistencyTokenResponse:
                Response message for
                   [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]

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
            request, bigtable_table_admin.GenerateConsistencyTokenRequest
        ):
            request = bigtable_table_admin.GenerateConsistencyTokenRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.generate_consistency_token
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

    async def check_consistency(
        self,
        request: Optional[
            Union[bigtable_table_admin.CheckConsistencyRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        consistency_token: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable_table_admin.CheckConsistencyResponse:
        r"""Checks replication consistency based on a consistency
        token, that is, if replication has caught up based on
        the conditions specified in the token and the check
        request.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_check_consistency():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CheckConsistencyRequest(
                    name="name_value",
                    consistency_token="consistency_token_value",
                )

                # Make the request
                response = await client.check_consistency(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CheckConsistencyRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]
            name (:class:`str`):
                Required. The unique name of the Table for which to
                check replication consistency. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            consistency_token (:class:`str`):
                Required. The token created using
                GenerateConsistencyToken for the Table.

                This corresponds to the ``consistency_token`` field
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
            google.cloud.bigtable_admin_v2.types.CheckConsistencyResponse:
                Response message for
                   [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, consistency_token]
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
        if not isinstance(request, bigtable_table_admin.CheckConsistencyRequest):
            request = bigtable_table_admin.CheckConsistencyRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if consistency_token is not None:
            request.consistency_token = consistency_token

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.check_consistency
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

    async def snapshot_table(
        self,
        request: Optional[
            Union[bigtable_table_admin.SnapshotTableRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        cluster: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        description: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a new snapshot in the specified cluster from
        the specified source table. The cluster and the table
        must be in the same instance.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_snapshot_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.SnapshotTableRequest(
                    name="name_value",
                    cluster="cluster_value",
                    snapshot_id="snapshot_id_value",
                )

                # Make the request
                operation = client.snapshot_table(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.SnapshotTableRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable][google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
            name (:class:`str`):
                Required. The unique name of the table to have the
                snapshot taken. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            cluster (:class:`str`):
                Required. The name of the cluster where the snapshot
                will be created in. Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

                This corresponds to the ``cluster`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            snapshot_id (:class:`str`):
                Required. The ID by which the new snapshot should be
                referred to within the parent cluster, e.g.,
                ``mysnapshot`` of the form:
                ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*`` rather than
                ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/mysnapshot``.

                This corresponds to the ``snapshot_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            description (:class:`str`):
                Description of the snapshot.
                This corresponds to the ``description`` field
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Snapshot` A snapshot of a table at a particular time. A snapshot can be used as a
                   checkpoint for data restoration or a data source for
                   a new table.

                   Note: This is a private alpha release of Cloud
                   Bigtable snapshots. This feature is not currently
                   available to most Cloud Bigtable customers. This
                   feature might be changed in backward-incompatible
                   ways and is not recommended for production use. It is
                   not subject to any SLA or deprecation policy.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, cluster, snapshot_id, description]
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
        if not isinstance(request, bigtable_table_admin.SnapshotTableRequest):
            request = bigtable_table_admin.SnapshotTableRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if cluster is not None:
            request.cluster = cluster
        if snapshot_id is not None:
            request.snapshot_id = snapshot_id
        if description is not None:
            request.description = description

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.snapshot_table
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
            table.Snapshot,
            metadata_type=bigtable_table_admin.SnapshotTableMetadata,
        )

        # Done; return the response.
        return response

    async def get_snapshot(
        self,
        request: Optional[Union[bigtable_table_admin.GetSnapshotRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.Snapshot:
        r"""Gets metadata information about the specified
        snapshot.
        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_snapshot():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetSnapshotRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_snapshot(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetSnapshotRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
            name (:class:`str`):
                Required. The unique name of the requested snapshot.
                Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.

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
            google.cloud.bigtable_admin_v2.types.Snapshot:
                A snapshot of a table at a particular
                time. A snapshot can be used as a
                checkpoint for data restoration or a
                data source for a new table.

                Note: This is a private alpha release of
                Cloud Bigtable snapshots. This feature
                is not currently available to most Cloud
                Bigtable customers. This feature might
                be changed in backward-incompatible ways
                and is not recommended for production
                use. It is not subject to any SLA or
                deprecation policy.

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
        if not isinstance(request, bigtable_table_admin.GetSnapshotRequest):
            request = bigtable_table_admin.GetSnapshotRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_snapshot
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

    async def list_snapshots(
        self,
        request: Optional[
            Union[bigtable_table_admin.ListSnapshotsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListSnapshotsAsyncPager:
        r"""Lists all snapshots associated with the specified
        cluster.
        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_snapshots():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListSnapshotsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_snapshots(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListSnapshotsRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
            parent (:class:`str`):
                Required. The unique name of the cluster for which
                snapshots should be listed. Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.
                Use ``{cluster} = '-'`` to list snapshots for all
                clusters in an instance, e.g.,
                ``projects/{project}/instances/{instance}/clusters/-``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_table_admin.pagers.ListSnapshotsAsyncPager:
                Response message for
                   [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

                   Note: This is a private alpha release of Cloud
                   Bigtable snapshots. This feature is not currently
                   available to most Cloud Bigtable customers. This
                   feature might be changed in backward-incompatible
                   ways and is not recommended for production use. It is
                   not subject to any SLA or deprecation policy.

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
        if not isinstance(request, bigtable_table_admin.ListSnapshotsRequest):
            request = bigtable_table_admin.ListSnapshotsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_snapshots
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
        response = pagers.ListSnapshotsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_snapshot(
        self,
        request: Optional[
            Union[bigtable_table_admin.DeleteSnapshotRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Permanently deletes the specified snapshot.

        Note: This is a private alpha release of Cloud Bigtable
        snapshots. This feature is not currently available to
        most Cloud Bigtable customers. This feature might be
        changed in backward-incompatible ways and is not
        recommended for production use. It is not subject to any
        SLA or deprecation policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_snapshot():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteSnapshotRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_snapshot(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteSnapshotRequest, dict]]):
                The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
            name (:class:`str`):
                Required. The unique name of the snapshot to be deleted.
                Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.

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
        if not isinstance(request, bigtable_table_admin.DeleteSnapshotRequest):
            request = bigtable_table_admin.DeleteSnapshotRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_snapshot
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

    async def create_backup(
        self,
        request: Optional[Union[bigtable_table_admin.CreateBackupRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        backup_id: Optional[str] = None,
        backup: Optional[table.Backup] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Starts creating a new Cloud Bigtable Backup. The returned backup
        [long-running operation][google.longrunning.Operation] can be
        used to track creation of the backup. The
        [metadata][google.longrunning.Operation.metadata] field type is
        [CreateBackupMetadata][google.bigtable.admin.v2.CreateBackupMetadata].
        The [response][google.longrunning.Operation.response] field type
        is [Backup][google.bigtable.admin.v2.Backup], if successful.
        Cancelling the returned operation will stop the creation and
        delete the backup.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_backup():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                backup = bigtable_admin_v2.Backup()
                backup.source_table = "source_table_value"

                request = bigtable_admin_v2.CreateBackupRequest(
                    parent="parent_value",
                    backup_id="backup_id_value",
                    backup=backup,
                )

                # Make the request
                operation = client.create_backup(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateBackupRequest, dict]]):
                The request object. The request for
                [CreateBackup][google.bigtable.admin.v2.BigtableTableAdmin.CreateBackup].
            parent (:class:`str`):
                Required. This must be one of the clusters in the
                instance in which this table is located. The backup will
                be stored in this cluster. Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            backup_id (:class:`str`):
                Required. The id of the backup to be created. The
                ``backup_id`` along with the parent ``parent`` are
                combined as {parent}/backups/{backup_id} to create the
                full backup name, of the form:
                ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup_id}``.
                This string must be between 1 and 50 characters in
                length and match the regex [*a-zA-Z0-9][-*.a-zA-Z0-9]*.

                This corresponds to the ``backup_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            backup (:class:`google.cloud.bigtable_admin_v2.types.Backup`):
                Required. The backup to create.
                This corresponds to the ``backup`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.Backup` A
                backup of a Cloud Bigtable table.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, backup_id, backup]
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
        if not isinstance(request, bigtable_table_admin.CreateBackupRequest):
            request = bigtable_table_admin.CreateBackupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if backup_id is not None:
            request.backup_id = backup_id
        if backup is not None:
            request.backup = backup

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_backup
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
            table.Backup,
            metadata_type=bigtable_table_admin.CreateBackupMetadata,
        )

        # Done; return the response.
        return response

    async def get_backup(
        self,
        request: Optional[Union[bigtable_table_admin.GetBackupRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.Backup:
        r"""Gets metadata on a pending or completed Cloud
        Bigtable Backup.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_backup():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetBackupRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_backup(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetBackupRequest, dict]]):
                The request object. The request for
                [GetBackup][google.bigtable.admin.v2.BigtableTableAdmin.GetBackup].
            name (:class:`str`):
                Required. Name of the backup. Values are of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup}``.

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
            google.cloud.bigtable_admin_v2.types.Backup:
                A backup of a Cloud Bigtable table.
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
        if not isinstance(request, bigtable_table_admin.GetBackupRequest):
            request = bigtable_table_admin.GetBackupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_backup
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

    async def update_backup(
        self,
        request: Optional[Union[bigtable_table_admin.UpdateBackupRequest, dict]] = None,
        *,
        backup: Optional[table.Backup] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.Backup:
        r"""Updates a pending or completed Cloud Bigtable Backup.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_backup():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                backup = bigtable_admin_v2.Backup()
                backup.source_table = "source_table_value"

                request = bigtable_admin_v2.UpdateBackupRequest(
                    backup=backup,
                )

                # Make the request
                response = await client.update_backup(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateBackupRequest, dict]]):
                The request object. The request for
                [UpdateBackup][google.bigtable.admin.v2.BigtableTableAdmin.UpdateBackup].
            backup (:class:`google.cloud.bigtable_admin_v2.types.Backup`):
                Required. The backup to update. ``backup.name``, and the
                fields to be updated as specified by ``update_mask`` are
                required. Other fields are ignored. Update is only
                supported for the following fields:

                -  ``backup.expire_time``.

                This corresponds to the ``backup`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Required. A mask specifying which fields (e.g.
                ``expire_time``) in the Backup resource should be
                updated. This mask is relative to the Backup resource,
                not to the request message. The field mask must always
                be specified; this prevents any future fields from being
                erased accidentally by clients that do not know about
                them.

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
            google.cloud.bigtable_admin_v2.types.Backup:
                A backup of a Cloud Bigtable table.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [backup, update_mask]
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
        if not isinstance(request, bigtable_table_admin.UpdateBackupRequest):
            request = bigtable_table_admin.UpdateBackupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if backup is not None:
            request.backup = backup
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_backup
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("backup.name", request.backup.name),)
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

    async def delete_backup(
        self,
        request: Optional[Union[bigtable_table_admin.DeleteBackupRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a pending or completed Cloud Bigtable backup.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_backup():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteBackupRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_backup(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteBackupRequest, dict]]):
                The request object. The request for
                [DeleteBackup][google.bigtable.admin.v2.BigtableTableAdmin.DeleteBackup].
            name (:class:`str`):
                Required. Name of the backup to delete. Values are of
                the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup}``.

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
        if not isinstance(request, bigtable_table_admin.DeleteBackupRequest):
            request = bigtable_table_admin.DeleteBackupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_backup
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

    async def list_backups(
        self,
        request: Optional[Union[bigtable_table_admin.ListBackupsRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListBackupsAsyncPager:
        r"""Lists Cloud Bigtable backups. Returns both completed
        and pending backups.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_backups():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListBackupsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_backups(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListBackupsRequest, dict]]):
                The request object. The request for
                [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].
            parent (:class:`str`):
                Required. The cluster to list backups from. Values are
                of the form
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.
                Use ``{cluster} = '-'`` to list backups for all clusters
                in an instance, e.g.,
                ``projects/{project}/instances/{instance}/clusters/-``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_table_admin.pagers.ListBackupsAsyncPager:
                The response for
                   [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].

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
        if not isinstance(request, bigtable_table_admin.ListBackupsRequest):
            request = bigtable_table_admin.ListBackupsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_backups
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
        response = pagers.ListBackupsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def _restore_table(
        self,
        request: Optional[Union[bigtable_table_admin.RestoreTableRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Create a new table by restoring from a completed backup. The
        returned table [long-running
        operation][google.longrunning.Operation] can be used to track
        the progress of the operation, and to cancel it. The
        [metadata][google.longrunning.Operation.metadata] field type is
        [RestoreTableMetadata][google.bigtable.admin.v2.RestoreTableMetadata].
        The [response][google.longrunning.Operation.response] type is
        [Table][google.bigtable.admin.v2.Table], if successful.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_restore_table():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.RestoreTableRequest(
                    backup="backup_value",
                    parent="parent_value",
                    table_id="table_id_value",
                )

                # Make the request
                operation = client._restore_table(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.RestoreTableRequest, dict]]):
                The request object. The request for
                [RestoreTable][google.bigtable.admin.v2.BigtableTableAdmin.RestoreTable].
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

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Table` A collection of user data indexed by row, column, and timestamp.
                   Each table is served using the resources of its
                   parent cluster.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, bigtable_table_admin.RestoreTableRequest):
            request = bigtable_table_admin.RestoreTableRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.restore_table
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
            table.Table,
            metadata_type=bigtable_table_admin.RestoreTableMetadata,
        )

        # Done; return the response.
        return response

    async def copy_backup(
        self,
        request: Optional[Union[bigtable_table_admin.CopyBackupRequest, dict]] = None,
        *,
        parent: Optional[str] = None,
        backup_id: Optional[str] = None,
        source_backup: Optional[str] = None,
        expire_time: Optional[timestamp_pb2.Timestamp] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Copy a Cloud Bigtable backup to a new backup in the
        destination cluster located in the destination instance
        and project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_copy_backup():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.CopyBackupRequest(
                    parent="parent_value",
                    backup_id="backup_id_value",
                    source_backup="source_backup_value",
                )

                # Make the request
                operation = client.copy_backup(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CopyBackupRequest, dict]]):
                The request object. The request for
                [CopyBackup][google.bigtable.admin.v2.BigtableTableAdmin.CopyBackup].
            parent (:class:`str`):
                Required. The name of the destination cluster that will
                contain the backup copy. The cluster must already exist.
                Values are of the form:
                ``projects/{project}/instances/{instance}/clusters/{cluster}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            backup_id (:class:`str`):
                Required. The id of the new backup. The ``backup_id``
                along with ``parent`` are combined as
                {parent}/backups/{backup_id} to create the full backup
                name, of the form:
                ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup_id}``.
                This string must be between 1 and 50 characters in
                length and match the regex [*a-zA-Z0-9][-*.a-zA-Z0-9]*.

                This corresponds to the ``backup_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            source_backup (:class:`str`):
                Required. The source backup to be copied from. The
                source backup needs to be in READY state for it to be
                copied. Copying a copied backup is not allowed. Once
                CopyBackup is in progress, the source backup cannot be
                deleted or cleaned up on expiration until CopyBackup is
                finished. Values are of the form:
                ``projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>``.

                This corresponds to the ``source_backup`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            expire_time (:class:`google.protobuf.timestamp_pb2.Timestamp`):
                Required. Required. The expiration time of the copied
                backup with microsecond granularity that must be at
                least 6 hours and at most 30 days from the time the
                request is received. Once the ``expire_time`` has
                passed, Cloud Bigtable will delete the backup and free
                the resources used by the backup.

                This corresponds to the ``expire_time`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.Backup` A
                backup of a Cloud Bigtable table.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, backup_id, source_backup, expire_time]
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
        if not isinstance(request, bigtable_table_admin.CopyBackupRequest):
            request = bigtable_table_admin.CopyBackupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if backup_id is not None:
            request.backup_id = backup_id
        if source_backup is not None:
            request.source_backup = source_backup
        if expire_time is not None:
            request.expire_time = expire_time

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.copy_backup
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
            table.Backup,
            metadata_type=bigtable_table_admin.CopyBackupMetadata,
        )

        # Done; return the response.
        return response

    async def get_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.GetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Gets the access control policy for a Bigtable
        resource. Returns an empty policy if the resource exists
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
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

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
        r"""Sets the access control policy on a Bigtable
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
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

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
        specified Bigtable resource.

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
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

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

    async def create_schema_bundle(
        self,
        request: Optional[
            Union[bigtable_table_admin.CreateSchemaBundleRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        schema_bundle_id: Optional[str] = None,
        schema_bundle: Optional[table.SchemaBundle] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a new schema bundle in the specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_create_schema_bundle():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                schema_bundle = bigtable_admin_v2.SchemaBundle()
                schema_bundle.proto_schema.proto_descriptors = b'proto_descriptors_blob'

                request = bigtable_admin_v2.CreateSchemaBundleRequest(
                    parent="parent_value",
                    schema_bundle_id="schema_bundle_id_value",
                    schema_bundle=schema_bundle,
                )

                # Make the request
                operation = client.create_schema_bundle(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.CreateSchemaBundleRequest, dict]]):
                The request object. The request for
                [CreateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.CreateSchemaBundle].
            parent (:class:`str`):
                Required. The parent resource where this schema bundle
                will be created. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            schema_bundle_id (:class:`str`):
                Required. The unique ID to use for
                the schema bundle, which will become the
                final component of the schema bundle's
                resource name.

                This corresponds to the ``schema_bundle_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            schema_bundle (:class:`google.cloud.bigtable_admin_v2.types.SchemaBundle`):
                Required. The schema bundle to
                create.

                This corresponds to the ``schema_bundle`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.SchemaBundle`
                A named collection of related schemas.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, schema_bundle_id, schema_bundle]
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
        if not isinstance(request, bigtable_table_admin.CreateSchemaBundleRequest):
            request = bigtable_table_admin.CreateSchemaBundleRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if schema_bundle_id is not None:
            request.schema_bundle_id = schema_bundle_id
        if schema_bundle is not None:
            request.schema_bundle = schema_bundle

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_schema_bundle
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
            table.SchemaBundle,
            metadata_type=bigtable_table_admin.CreateSchemaBundleMetadata,
        )

        # Done; return the response.
        return response

    async def update_schema_bundle(
        self,
        request: Optional[
            Union[bigtable_table_admin.UpdateSchemaBundleRequest, dict]
        ] = None,
        *,
        schema_bundle: Optional[table.SchemaBundle] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a schema bundle in the specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_update_schema_bundle():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                schema_bundle = bigtable_admin_v2.SchemaBundle()
                schema_bundle.proto_schema.proto_descriptors = b'proto_descriptors_blob'

                request = bigtable_admin_v2.UpdateSchemaBundleRequest(
                    schema_bundle=schema_bundle,
                )

                # Make the request
                operation = client.update_schema_bundle(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.UpdateSchemaBundleRequest, dict]]):
                The request object. The request for
                [UpdateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.UpdateSchemaBundle].
            schema_bundle (:class:`google.cloud.bigtable_admin_v2.types.SchemaBundle`):
                Required. The schema bundle to update.

                The schema bundle's ``name`` field is used to identify
                the schema bundle to update. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``

                This corresponds to the ``schema_bundle`` field
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
                :class:`google.cloud.bigtable_admin_v2.types.SchemaBundle`
                A named collection of related schemas.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [schema_bundle, update_mask]
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
        if not isinstance(request, bigtable_table_admin.UpdateSchemaBundleRequest):
            request = bigtable_table_admin.UpdateSchemaBundleRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if schema_bundle is not None:
            request.schema_bundle = schema_bundle
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_schema_bundle
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("schema_bundle.name", request.schema_bundle.name),)
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
            table.SchemaBundle,
            metadata_type=bigtable_table_admin.UpdateSchemaBundleMetadata,
        )

        # Done; return the response.
        return response

    async def get_schema_bundle(
        self,
        request: Optional[
            Union[bigtable_table_admin.GetSchemaBundleRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> table.SchemaBundle:
        r"""Gets metadata information about the specified schema
        bundle.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_get_schema_bundle():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.GetSchemaBundleRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_schema_bundle(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.GetSchemaBundleRequest, dict]]):
                The request object. The request for
                [GetSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.GetSchemaBundle].
            name (:class:`str`):
                Required. The unique name of the schema bundle to
                retrieve. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``

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
            google.cloud.bigtable_admin_v2.types.SchemaBundle:
                A named collection of related
                schemas.

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
        if not isinstance(request, bigtable_table_admin.GetSchemaBundleRequest):
            request = bigtable_table_admin.GetSchemaBundleRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_schema_bundle
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

    async def list_schema_bundles(
        self,
        request: Optional[
            Union[bigtable_table_admin.ListSchemaBundlesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListSchemaBundlesAsyncPager:
        r"""Lists all schema bundles associated with the
        specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_list_schema_bundles():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.ListSchemaBundlesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_schema_bundles(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.ListSchemaBundlesRequest, dict]]):
                The request object. The request for
                [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].
            parent (:class:`str`):
                Required. The parent, which owns this collection of
                schema bundles. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

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
            google.cloud.bigtable_admin_v2.services.bigtable_table_admin.pagers.ListSchemaBundlesAsyncPager:
                The response for
                   [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].

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
        if not isinstance(request, bigtable_table_admin.ListSchemaBundlesRequest):
            request = bigtable_table_admin.ListSchemaBundlesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_schema_bundles
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
        response = pagers.ListSchemaBundlesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_schema_bundle(
        self,
        request: Optional[
            Union[bigtable_table_admin.DeleteSchemaBundleRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a schema bundle in the specified table.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import bigtable_admin_v2

            async def sample_delete_schema_bundle():
                # Create a client
                client = bigtable_admin_v2.BigtableTableAdminAsyncClient()

                # Initialize request argument(s)
                request = bigtable_admin_v2.DeleteSchemaBundleRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_schema_bundle(request=request)

        Args:
            request (Optional[Union[google.cloud.bigtable_admin_v2.types.DeleteSchemaBundleRequest, dict]]):
                The request object. The request for
                [DeleteSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSchemaBundle].
            name (:class:`str`):
                Required. The unique name of the schema bundle to
                delete. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``

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
        if not isinstance(request, bigtable_table_admin.DeleteSchemaBundleRequest):
            request = bigtable_table_admin.DeleteSchemaBundleRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_schema_bundle
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

    async def __aenter__(self) -> "BaseBigtableTableAdminAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("BaseBigtableTableAdminAsyncClient",)

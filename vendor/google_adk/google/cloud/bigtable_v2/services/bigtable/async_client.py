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
    AsyncIterable,
    Awaitable,
    Sequence,
    Tuple,
    Type,
    Union,
)

from google.cloud.bigtable_v2 import gapic_version as package_version

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

from google.cloud.bigtable_v2.types import bigtable
from google.cloud.bigtable_v2.types import data
from google.cloud.bigtable_v2.types import request_stats
from google.protobuf import timestamp_pb2  # type: ignore
from .transports.base import BigtableTransport, DEFAULT_CLIENT_INFO
from .transports.grpc_asyncio import BigtableGrpcAsyncIOTransport
from .client import BigtableClient

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class BigtableAsyncClient:
    """Service for reading from and writing to existing Bigtable
    tables.
    """

    _client: BigtableClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = BigtableClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = BigtableClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = BigtableClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = BigtableClient._DEFAULT_UNIVERSE

    authorized_view_path = staticmethod(BigtableClient.authorized_view_path)
    parse_authorized_view_path = staticmethod(BigtableClient.parse_authorized_view_path)
    instance_path = staticmethod(BigtableClient.instance_path)
    parse_instance_path = staticmethod(BigtableClient.parse_instance_path)
    materialized_view_path = staticmethod(BigtableClient.materialized_view_path)
    parse_materialized_view_path = staticmethod(
        BigtableClient.parse_materialized_view_path
    )
    table_path = staticmethod(BigtableClient.table_path)
    parse_table_path = staticmethod(BigtableClient.parse_table_path)
    common_billing_account_path = staticmethod(
        BigtableClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        BigtableClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(BigtableClient.common_folder_path)
    parse_common_folder_path = staticmethod(BigtableClient.parse_common_folder_path)
    common_organization_path = staticmethod(BigtableClient.common_organization_path)
    parse_common_organization_path = staticmethod(
        BigtableClient.parse_common_organization_path
    )
    common_project_path = staticmethod(BigtableClient.common_project_path)
    parse_common_project_path = staticmethod(BigtableClient.parse_common_project_path)
    common_location_path = staticmethod(BigtableClient.common_location_path)
    parse_common_location_path = staticmethod(BigtableClient.parse_common_location_path)

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            BigtableAsyncClient: The constructed client.
        """
        return BigtableClient.from_service_account_info.__func__(BigtableAsyncClient, info, *args, **kwargs)  # type: ignore

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
            BigtableAsyncClient: The constructed client.
        """
        return BigtableClient.from_service_account_file.__func__(BigtableAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return BigtableClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> BigtableTransport:
        """Returns the transport used by the client instance.

        Returns:
            BigtableTransport: The transport used by the client instance.
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

    get_transport_class = BigtableClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, BigtableTransport, Callable[..., BigtableTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the bigtable async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,BigtableTransport,Callable[..., BigtableTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the BigtableTransport constructor.
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
        self._client = BigtableClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.bigtable_v2.BigtableAsyncClient`.",
                extra={
                    "serviceName": "google.bigtable.v2.Bigtable",
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
                    "serviceName": "google.bigtable.v2.Bigtable",
                    "credentialsType": None,
                },
            )

    def read_rows(
        self,
        request: Optional[Union[bigtable.ReadRowsRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[bigtable.ReadRowsResponse]]:
        r"""Streams back the contents of all requested rows in
        key order, optionally applying the same Reader filter to
        each. Depending on their size, rows and cells may be
        broken up across multiple responses, but atomicity of
        each row will still be preserved. See the
        ReadRowsResponse documentation for details.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.ReadRowsRequest, dict]]):
                The request object. Request message for
                Bigtable.ReadRows.
            table_name (:class:`str`):
                Optional. The unique name of the table from which to
                read.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.ReadRowsResponse]:
                Response message for
                Bigtable.ReadRows.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, app_profile_id]
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
        if not isinstance(request, bigtable.ReadRowsRequest):
            request = bigtable.ReadRowsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.read_rows
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
            )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def sample_row_keys(
        self,
        request: Optional[Union[bigtable.SampleRowKeysRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[bigtable.SampleRowKeysResponse]]:
        r"""Returns a sample of row keys in the table. The
        returned row keys will delimit contiguous sections of
        the table of approximately equal size, which can be used
        to break up the data for distributed tasks like
        mapreduces.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.SampleRowKeysRequest, dict]]):
                The request object. Request message for
                Bigtable.SampleRowKeys.
            table_name (:class:`str`):
                Optional. The unique name of the table from which to
                sample row keys.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.SampleRowKeysResponse]:
                Response message for
                Bigtable.SampleRowKeys.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, app_profile_id]
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
        if not isinstance(request, bigtable.SampleRowKeysRequest):
            request = bigtable.SampleRowKeysRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.sample_row_keys
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
            )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def mutate_row(
        self,
        request: Optional[Union[bigtable.MutateRowRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        row_key: Optional[bytes] = None,
        mutations: Optional[MutableSequence[data.Mutation]] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable.MutateRowResponse:
        r"""Mutates a row atomically. Cells already present in the row are
        left unchanged unless explicitly changed by ``mutation``.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.MutateRowRequest, dict]]):
                The request object. Request message for
                Bigtable.MutateRow.
            table_name (:class:`str`):
                Optional. The unique name of the table to which the
                mutation should be applied.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            row_key (:class:`bytes`):
                Required. The key of the row to which
                the mutation should be applied.

                This corresponds to the ``row_key`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            mutations (:class:`MutableSequence[google.cloud.bigtable_v2.types.Mutation]`):
                Required. Changes to be atomically
                applied to the specified row. Entries
                are applied in order, meaning that
                earlier mutations can be masked by later
                ones. Must contain at least one entry
                and at most 100000.

                This corresponds to the ``mutations`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            google.cloud.bigtable_v2.types.MutateRowResponse:
                Response message for
                Bigtable.MutateRow.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, row_key, mutations, app_profile_id]
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
        if not isinstance(request, bigtable.MutateRowRequest):
            request = bigtable.MutateRowRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if row_key is not None:
            request.row_key = row_key
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id
        if mutations:
            request.mutations.extend(mutations)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.mutate_row
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
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

    def mutate_rows(
        self,
        request: Optional[Union[bigtable.MutateRowsRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        entries: Optional[MutableSequence[bigtable.MutateRowsRequest.Entry]] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[bigtable.MutateRowsResponse]]:
        r"""Mutates multiple rows in a batch. Each individual row
        is mutated atomically as in MutateRow, but the entire
        batch is not executed atomically.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.MutateRowsRequest, dict]]):
                The request object. Request message for
                BigtableService.MutateRows.
            table_name (:class:`str`):
                Optional. The unique name of the table to which the
                mutations should be applied.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            entries (:class:`MutableSequence[google.cloud.bigtable_v2.types.MutateRowsRequest.Entry]`):
                Required. The row keys and
                corresponding mutations to be applied in
                bulk. Each entry is applied as an atomic
                mutation, but the entries may be applied
                in arbitrary order (even between entries
                for the same row). At least one entry
                must be specified, and in total the
                entries can contain at most 100000
                mutations.

                This corresponds to the ``entries`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.MutateRowsResponse]:
                Response message for
                BigtableService.MutateRows.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, entries, app_profile_id]
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
        if not isinstance(request, bigtable.MutateRowsRequest):
            request = bigtable.MutateRowsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id
        if entries:
            request.entries.extend(entries)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.mutate_rows
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
            )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def check_and_mutate_row(
        self,
        request: Optional[Union[bigtable.CheckAndMutateRowRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        row_key: Optional[bytes] = None,
        predicate_filter: Optional[data.RowFilter] = None,
        true_mutations: Optional[MutableSequence[data.Mutation]] = None,
        false_mutations: Optional[MutableSequence[data.Mutation]] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable.CheckAndMutateRowResponse:
        r"""Mutates a row atomically based on the output of a
        predicate Reader filter.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.CheckAndMutateRowRequest, dict]]):
                The request object. Request message for
                Bigtable.CheckAndMutateRow.
            table_name (:class:`str`):
                Optional. The unique name of the table to which the
                conditional mutation should be applied.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            row_key (:class:`bytes`):
                Required. The key of the row to which
                the conditional mutation should be
                applied.

                This corresponds to the ``row_key`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            predicate_filter (:class:`google.cloud.bigtable_v2.types.RowFilter`):
                The filter to be applied to the contents of the
                specified row. Depending on whether or not any results
                are yielded, either ``true_mutations`` or
                ``false_mutations`` will be executed. If unset, checks
                that the row contains any values at all.

                This corresponds to the ``predicate_filter`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            true_mutations (:class:`MutableSequence[google.cloud.bigtable_v2.types.Mutation]`):
                Changes to be atomically applied to the specified row if
                ``predicate_filter`` yields at least one cell when
                applied to ``row_key``. Entries are applied in order,
                meaning that earlier mutations can be masked by later
                ones. Must contain at least one entry if
                ``false_mutations`` is empty, and at most 100000.

                This corresponds to the ``true_mutations`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            false_mutations (:class:`MutableSequence[google.cloud.bigtable_v2.types.Mutation]`):
                Changes to be atomically applied to the specified row if
                ``predicate_filter`` does not yield any cells when
                applied to ``row_key``. Entries are applied in order,
                meaning that earlier mutations can be masked by later
                ones. Must contain at least one entry if
                ``true_mutations`` is empty, and at most 100000.

                This corresponds to the ``false_mutations`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            google.cloud.bigtable_v2.types.CheckAndMutateRowResponse:
                Response message for
                Bigtable.CheckAndMutateRow.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [
            table_name,
            row_key,
            predicate_filter,
            true_mutations,
            false_mutations,
            app_profile_id,
        ]
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
        if not isinstance(request, bigtable.CheckAndMutateRowRequest):
            request = bigtable.CheckAndMutateRowRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if row_key is not None:
            request.row_key = row_key
        if predicate_filter is not None:
            request.predicate_filter = predicate_filter
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id
        if true_mutations:
            request.true_mutations.extend(true_mutations)
        if false_mutations:
            request.false_mutations.extend(false_mutations)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.check_and_mutate_row
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
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

    async def ping_and_warm(
        self,
        request: Optional[Union[bigtable.PingAndWarmRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable.PingAndWarmResponse:
        r"""Warm up associated instance metadata for this
        connection. This call is not required but may be useful
        for connection keep-alive.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.PingAndWarmRequest, dict]]):
                The request object. Request message for client connection
                keep-alive and warming.
            name (:class:`str`):
                Required. The unique name of the instance to check
                permissions for as well as respond. Values are of the
                form ``projects/<project>/instances/<instance>``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            google.cloud.bigtable_v2.types.PingAndWarmResponse:
                Response message for
                Bigtable.PingAndWarm connection
                keepalive and warming.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, app_profile_id]
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
        if not isinstance(request, bigtable.PingAndWarmRequest):
            request = bigtable.PingAndWarmRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.ping_and_warm
        ]

        header_params = {}

        routing_param_regex = re.compile("^(?P<name>projects/[^/]+/instances/[^/]+)$")
        regex_match = routing_param_regex.match(request.name)
        if regex_match and regex_match.group("name"):
            header_params["name"] = regex_match.group("name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
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

    async def read_modify_write_row(
        self,
        request: Optional[Union[bigtable.ReadModifyWriteRowRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        row_key: Optional[bytes] = None,
        rules: Optional[MutableSequence[data.ReadModifyWriteRule]] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable.ReadModifyWriteRowResponse:
        r"""Modifies a row atomically on the server. The method
        reads the latest existing timestamp and value from the
        specified columns and writes a new entry based on
        pre-defined read/modify/write rules. The new value for
        the timestamp is the greater of the existing timestamp
        or the current server time. The method returns the new
        contents of all modified cells.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.ReadModifyWriteRowRequest, dict]]):
                The request object. Request message for
                Bigtable.ReadModifyWriteRow.
            table_name (:class:`str`):
                Optional. The unique name of the table to which the
                read/modify/write rules should be applied.

                Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            row_key (:class:`bytes`):
                Required. The key of the row to which
                the read/modify/write rules should be
                applied.

                This corresponds to the ``row_key`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            rules (:class:`MutableSequence[google.cloud.bigtable_v2.types.ReadModifyWriteRule]`):
                Required. Rules specifying how the
                specified row's contents are to be
                transformed into writes. Entries are
                applied in order, meaning that earlier
                rules will affect the results of later
                ones. At least one entry must be
                specified, and there can be at most
                100000 rules.

                This corresponds to the ``rules`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used.

                This corresponds to the ``app_profile_id`` field
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
            google.cloud.bigtable_v2.types.ReadModifyWriteRowResponse:
                Response message for
                Bigtable.ReadModifyWriteRow.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, row_key, rules, app_profile_id]
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
        if not isinstance(request, bigtable.ReadModifyWriteRowRequest):
            request = bigtable.ReadModifyWriteRowRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if row_key is not None:
            request.row_key = row_key
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id
        if rules:
            request.rules.extend(rules)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.read_modify_write_row
        ]

        header_params = {}

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)$"
        )
        regex_match = routing_param_regex.match(request.table_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        routing_param_regex = re.compile(
            "^(?P<table_name>projects/[^/]+/instances/[^/]+/tables/[^/]+)(?:/.*)?$"
        )
        regex_match = routing_param_regex.match(request.authorized_view_name)
        if regex_match and regex_match.group("table_name"):
            header_params["table_name"] = regex_match.group("table_name")

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
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

    def generate_initial_change_stream_partitions(
        self,
        request: Optional[
            Union[bigtable.GenerateInitialChangeStreamPartitionsRequest, dict]
        ] = None,
        *,
        table_name: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[
        AsyncIterable[bigtable.GenerateInitialChangeStreamPartitionsResponse]
    ]:
        r"""Returns the current list of partitions that make up the table's
        change stream. The union of partitions will cover the entire
        keyspace. Partitions can be read with ``ReadChangeStream``.
        NOTE: This API is only intended to be used by Apache Beam
        BigtableIO.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.GenerateInitialChangeStreamPartitionsRequest, dict]]):
                The request object. NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Request
                message for
                Bigtable.GenerateInitialChangeStreamPartitions.
            table_name (:class:`str`):
                Required. The unique name of the table from which to get
                change stream partitions. Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.
                Change streaming must be enabled on the table.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used. Single cluster routing must be
                configured on the profile.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.GenerateInitialChangeStreamPartitionsResponse]:
                NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Response
                message for
                Bigtable.GenerateInitialChangeStreamPartitions.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, app_profile_id]
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
            request, bigtable.GenerateInitialChangeStreamPartitionsRequest
        ):
            request = bigtable.GenerateInitialChangeStreamPartitionsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.generate_initial_change_stream_partitions
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("table_name", request.table_name),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def read_change_stream(
        self,
        request: Optional[Union[bigtable.ReadChangeStreamRequest, dict]] = None,
        *,
        table_name: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[bigtable.ReadChangeStreamResponse]]:
        r"""Reads changes from a table's change stream. Changes
        will reflect both user-initiated mutations and mutations
        that are caused by garbage collection.
        NOTE: This API is only intended to be used by Apache
        Beam BigtableIO.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.ReadChangeStreamRequest, dict]]):
                The request object. NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Request
                message for Bigtable.ReadChangeStream.
            table_name (:class:`str`):
                Required. The unique name of the table from which to
                read a change stream. Values are of the form
                ``projects/<project>/instances/<instance>/tables/<table>``.
                Change streaming must be enabled on the table.

                This corresponds to the ``table_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                This value specifies routing for
                replication. If not specified, the
                "default" application profile will be
                used. Single cluster routing must be
                configured on the profile.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.ReadChangeStreamResponse]:
                NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Response
                message for Bigtable.ReadChangeStream.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [table_name, app_profile_id]
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
        if not isinstance(request, bigtable.ReadChangeStreamRequest):
            request = bigtable.ReadChangeStreamRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if table_name is not None:
            request.table_name = table_name
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.read_change_stream
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("table_name", request.table_name),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def prepare_query(
        self,
        request: Optional[Union[bigtable.PrepareQueryRequest, dict]] = None,
        *,
        instance_name: Optional[str] = None,
        query: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bigtable.PrepareQueryResponse:
        r"""Prepares a GoogleSQL query for execution on a
        particular Bigtable instance.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.PrepareQueryRequest, dict]]):
                The request object. Request message for
                Bigtable.PrepareQuery
            instance_name (:class:`str`):
                Required. The unique name of the instance against which
                the query should be executed. Values are of the form
                ``projects/<project>/instances/<instance>``

                This corresponds to the ``instance_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            query (:class:`str`):
                Required. The query string.
                This corresponds to the ``query`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                Optional. This value specifies routing for preparing the
                query. Note that this ``app_profile_id`` is only used
                for preparing the query. The actual query execution will
                use the app profile specified in the
                ``ExecuteQueryRequest``. If not specified, the
                ``default`` application profile will be used.

                This corresponds to the ``app_profile_id`` field
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
            google.cloud.bigtable_v2.types.PrepareQueryResponse:
                Response message for
                Bigtable.PrepareQueryResponse

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance_name, query, app_profile_id]
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
        if not isinstance(request, bigtable.PrepareQueryRequest):
            request = bigtable.PrepareQueryRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if instance_name is not None:
            request.instance_name = instance_name
        if query is not None:
            request.query = query
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.prepare_query
        ]

        header_params = {}

        routing_param_regex = re.compile("^(?P<name>projects/[^/]+/instances/[^/]+)$")
        regex_match = routing_param_regex.match(request.instance_name)
        if regex_match and regex_match.group("name"):
            header_params["name"] = regex_match.group("name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
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

    def execute_query(
        self,
        request: Optional[Union[bigtable.ExecuteQueryRequest, dict]] = None,
        *,
        instance_name: Optional[str] = None,
        query: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[bigtable.ExecuteQueryResponse]]:
        r"""Executes a SQL query against a particular Bigtable
        instance.

        Args:
            request (Optional[Union[google.cloud.bigtable_v2.types.ExecuteQueryRequest, dict]]):
                The request object. Request message for
                Bigtable.ExecuteQuery
            instance_name (:class:`str`):
                Required. The unique name of the instance against which
                the query should be executed. Values are of the form
                ``projects/<project>/instances/<instance>``

                This corresponds to the ``instance_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            query (:class:`str`):
                Required. The query string.

                Exactly one of ``query`` and ``prepared_query`` is
                required. Setting both or neither is an
                ``INVALID_ARGUMENT``.

                This corresponds to the ``query`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            app_profile_id (:class:`str`):
                Optional. This value specifies routing for replication.
                If not specified, the ``default`` application profile
                will be used.

                This corresponds to the ``app_profile_id`` field
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
            AsyncIterable[google.cloud.bigtable_v2.types.ExecuteQueryResponse]:
                Response message for
                Bigtable.ExecuteQuery

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance_name, query, app_profile_id]
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
        if not isinstance(request, bigtable.ExecuteQueryRequest):
            request = bigtable.ExecuteQueryRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if instance_name is not None:
            request.instance_name = instance_name
        if query is not None:
            request.query = query
        if app_profile_id is not None:
            request.app_profile_id = app_profile_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.execute_query
        ]

        header_params = {}

        routing_param_regex = re.compile("^(?P<name>projects/[^/]+/instances/[^/]+)$")
        regex_match = routing_param_regex.match(request.instance_name)
        if regex_match and regex_match.group("name"):
            header_params["name"] = regex_match.group("name")

        if True:  # always attach app_profile_id, even if empty string
            header_params["app_profile_id"] = request.app_profile_id

        if header_params:
            metadata = tuple(metadata) + (
                gapic_v1.routing_header.to_grpc_metadata(header_params),
            )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def __aenter__(self) -> "BigtableAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("BigtableAsyncClient",)

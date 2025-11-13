# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import copy
import functools

from typing import Callable, Optional, Sequence, Tuple, Union
from google.api_core import gapic_v1
from google.api_core import retry as retries

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

from google.api_core import client_options as client_options_lib
from google.auth import credentials as ga_credentials  # type: ignore

from google.cloud.bigtable_admin_v2.types import bigtable_table_admin

from google.cloud.bigtable_admin_v2.services.bigtable_table_admin import (
    client as base_client,
)
from google.cloud.bigtable_admin_v2.services.bigtable_table_admin.transports.base import (
    BigtableTableAdminTransport,
)
from google.cloud.bigtable_admin_v2.overlay.types import (
    consistency,
    restore_table,
    wait_for_consistency_request,
)

from google.cloud.bigtable.gapic_version import __version__ as bigtable_version


DEFAULT_CLIENT_INFO = copy.copy(base_client.DEFAULT_CLIENT_INFO)
DEFAULT_CLIENT_INFO.client_library_version = f"{bigtable_version}-admin-overlay"


class BigtableTableAdminClient(base_client.BaseBigtableTableAdminClient):
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
        ] = None,
        client_options: Optional[Union[client_options_lib.ClientOptions, dict]] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the Bigtable table admin client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,BigtableTableAdminTransport,Callable[..., BigtableTableAdminTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport.
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
                default "googleapis.com" universe. Note that the ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        super(BigtableTableAdminClient, self).__init__(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

    def restore_table(
        self,
        request: Optional[Union[bigtable_table_admin.RestoreTableRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> restore_table.RestoreTableOperation:
        r"""Create a new table by restoring from a completed backup. The
        returned table :class:`long-running operation
        <google.cloud.bigtable_admin_v2.overlay.types.restore_table.RestoreTableOperation>`
        can be used to track the progress of the operation, and to cancel it. The
        :attr:`metadata <google.api_core.operation.Operation.metadata>` field type is
        :class:`RestoreTableMetadata <google.cloud.bigtable_admin_v2.types.RestoreTableMetadata>`.
        The :meth:`response <google.api_core.operation.Operation.result>` type is
        :class:`google.cloud.bigtable_admin_v2.types.Table`, if successful.

        Additionally, the returned :class:`long-running-operation <google.cloud.bigtable_admin_v2.overlay.types.restore_table.RestoreTableOperation>`
        provides a method, :meth:`google.cloud.bigtable_admin_v2.overlay.types.restore_table.RestoreTableOperation.optimize_restore_table_operation` that
        provides access to a :class:`google.api_core.operation.Operation` object representing the OptimizeRestoreTable long-running-operation
        after the current one has completed.

        .. code-block:: python

            # This snippet should be regarded as a code template only.
            #
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud.bigtable import admin_v2

            def sample_restore_table():
                # Create a client
                client = admin_v2.BigtableTableAdminClient()

                # Initialize request argument(s)
                request = admin_v2.RestoreTableRequest(
                    backup="backup_value",
                    parent="parent_value",
                    table_id="table_id_value",
                )

                # Make the request
                operation = client.restore_table(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

                # Handle LRO2
                optimize_operation = operation.optimize_restore_table_operation()

                if optimize_operation:
                    print("Waiting for table optimization to complete...")

                    response = optimize_operation.result()

        Args:
            request (Union[google.cloud.bigtable_admin_v2.types.RestoreTableRequest, dict]):
                The request object. The request for
                [RestoreTable][google.bigtable.admin.v2.BigtableTableAdmin.RestoreTable].
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.bigtable_admin_v2.overlay.types.restore_table.RestoreTableOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.bigtable_admin_v2.types.Table` A collection of user data indexed by row, column, and timestamp.
                   Each table is served using the resources of its
                   parent cluster.
        """
        operation = self._restore_table(
            request=request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        restore_table_operation = restore_table.RestoreTableOperation(
            self._transport.operations_client, operation
        )
        return restore_table_operation

    def wait_for_consistency(
        self,
        request: Optional[
            Union[wait_for_consistency_request.WaitForConsistencyRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> bool:
        r"""Blocks until the mutations for the specified Table that have been
        made before the call have been replicated or reads using an app profile with `DataBoostIsolationReadOnly`
        can see all writes committed before the token was created. This is done by generating
        a consistency token for the Table, then polling :meth:`check_consistency`
        for the specified table until the call returns True.

        .. code-block:: python

            # This snippet should be regarded as a code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud.bigtable import admin_v2

            def sample_wait_for_consistency():
                # Create a client
                client = admin_v2.BigtableTableAdminClient()

                # Initialize request argument(s)
                request = admin_v2.WaitForConsistencyRequest(
                    name="name_value",
                )

                # Make the request
                print("Waiting for operation to complete...")

                response = client.wait_for_replication(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.bigtable_admin_v2.overlay.types.WaitForConsistencyRequest, dict]):
                The request object.
            name (str):
                Required. The unique name of the Table for which to
                create a consistency token. Values are of the form
                ``projects/{project}/instances/{instance}/tables/{table}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            bool:
                If the `standard_read_remote_writes` mode is specified in the request object, returns
                `True` after the mutations of the specified table have been fully replicated. If the
                `data_boost_read_local_writes` mode is specified in the request object, returns `True`
                after reads using an app profile with `DataBoostIsolationReadOnly` can see all writes
                committed before the token was created.

        Raises:
            google.api_core.GoogleAPICallError: If the operation errors or if
                the timeout is reached before the operation completes.
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
            request, wait_for_consistency_request.WaitForConsistencyRequest
        ):
            request = wait_for_consistency_request.WaitForConsistencyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Generate the consistency token.
        generate_consistency_token_request = (
            bigtable_table_admin.GenerateConsistencyTokenRequest(
                name=request.name,
            )
        )

        generate_consistency_response = self.generate_consistency_token(
            generate_consistency_token_request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Create the CheckConsistencyRequest object.
        check_consistency_request = bigtable_table_admin.CheckConsistencyRequest(
            name=request.name,
            consistency_token=generate_consistency_response.consistency_token,
        )

        # Since the default values of StandardReadRemoteWrites and DataBoostReadLocalWrites evaluate to
        # False in proto plus, we cannot do a simple "if request.standard_read_remote_writes" to check
        # whether or not that field is defined in the original request object.
        mode_oneof_field = request._pb.WhichOneof("mode")
        if mode_oneof_field:
            setattr(
                check_consistency_request,
                mode_oneof_field,
                getattr(request, mode_oneof_field),
            )

        check_consistency_call = functools.partial(
            self.check_consistency,
            check_consistency_request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Block and wait until the polling harness returns True.
        check_consistency_future = consistency._CheckConsistencyPollingFuture(
            check_consistency_call
        )
        return check_consistency_future.result()

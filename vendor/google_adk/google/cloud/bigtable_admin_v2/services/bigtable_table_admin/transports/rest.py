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
import logging
import json  # type: ignore

from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import gapic_v1
import google.protobuf

from google.protobuf import json_format
from google.api_core import operations_v1

from requests import __version__ as requests_version
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings


from google.cloud.bigtable_admin_v2.types import bigtable_table_admin
from google.cloud.bigtable_admin_v2.types import table
from google.cloud.bigtable_admin_v2.types import table as gba_table
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BaseBigtableTableAdminRestTransport
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class BigtableTableAdminRestInterceptor:
    """Interceptor for BigtableTableAdmin.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the BigtableTableAdminRestTransport.

    .. code-block:: python
        class MyCustomBigtableTableAdminInterceptor(BigtableTableAdminRestInterceptor):
            def pre_check_consistency(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_check_consistency(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_copy_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_copy_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_authorized_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_authorized_view(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_schema_bundle(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_schema_bundle(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_table(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_table_from_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_table_from_snapshot(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_authorized_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_schema_bundle(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_drop_row_range(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_generate_consistency_token(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_generate_consistency_token(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_authorized_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_authorized_view(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_schema_bundle(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_schema_bundle(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_snapshot(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_table(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_authorized_views(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_authorized_views(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_backups(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_backups(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_schema_bundles(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_schema_bundles(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_snapshots(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_snapshots(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_tables(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_tables(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_modify_column_families(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_modify_column_families(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_restore_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_restore_table(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_set_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_set_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_snapshot_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_snapshot_table(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_test_iam_permissions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_test_iam_permissions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_undelete_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_undelete_table(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_authorized_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_authorized_view(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_schema_bundle(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_schema_bundle(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_table(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_table(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = BigtableTableAdminRestTransport(interceptor=MyCustomBigtableTableAdminInterceptor())
        client = BaseBigtableTableAdminClient(transport=transport)


    """

    def pre_check_consistency(
        self,
        request: bigtable_table_admin.CheckConsistencyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CheckConsistencyRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for check_consistency

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_check_consistency(
        self, response: bigtable_table_admin.CheckConsistencyResponse
    ) -> bigtable_table_admin.CheckConsistencyResponse:
        """Post-rpc interceptor for check_consistency

        DEPRECATED. Please use the `post_check_consistency_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_check_consistency` interceptor runs
        before the `post_check_consistency_with_metadata` interceptor.
        """
        return response

    def post_check_consistency_with_metadata(
        self,
        response: bigtable_table_admin.CheckConsistencyResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CheckConsistencyResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for check_consistency

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_check_consistency_with_metadata`
        interceptor in new development instead of the `post_check_consistency` interceptor.
        When both interceptors are used, this `post_check_consistency_with_metadata` interceptor runs after the
        `post_check_consistency` interceptor. The (possibly modified) response returned by
        `post_check_consistency` will be passed to
        `post_check_consistency_with_metadata`.
        """
        return response, metadata

    def pre_copy_backup(
        self,
        request: bigtable_table_admin.CopyBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CopyBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for copy_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_copy_backup(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for copy_backup

        DEPRECATED. Please use the `post_copy_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_copy_backup` interceptor runs
        before the `post_copy_backup_with_metadata` interceptor.
        """
        return response

    def post_copy_backup_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for copy_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_copy_backup_with_metadata`
        interceptor in new development instead of the `post_copy_backup` interceptor.
        When both interceptors are used, this `post_copy_backup_with_metadata` interceptor runs after the
        `post_copy_backup` interceptor. The (possibly modified) response returned by
        `post_copy_backup` will be passed to
        `post_copy_backup_with_metadata`.
        """
        return response, metadata

    def pre_create_authorized_view(
        self,
        request: bigtable_table_admin.CreateAuthorizedViewRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CreateAuthorizedViewRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_authorized_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_create_authorized_view(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_authorized_view

        DEPRECATED. Please use the `post_create_authorized_view_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_create_authorized_view` interceptor runs
        before the `post_create_authorized_view_with_metadata` interceptor.
        """
        return response

    def post_create_authorized_view_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_authorized_view

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_create_authorized_view_with_metadata`
        interceptor in new development instead of the `post_create_authorized_view` interceptor.
        When both interceptors are used, this `post_create_authorized_view_with_metadata` interceptor runs after the
        `post_create_authorized_view` interceptor. The (possibly modified) response returned by
        `post_create_authorized_view` will be passed to
        `post_create_authorized_view_with_metadata`.
        """
        return response, metadata

    def pre_create_backup(
        self,
        request: bigtable_table_admin.CreateBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CreateBackupRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_create_backup(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_backup

        DEPRECATED. Please use the `post_create_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_create_backup` interceptor runs
        before the `post_create_backup_with_metadata` interceptor.
        """
        return response

    def post_create_backup_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_create_backup_with_metadata`
        interceptor in new development instead of the `post_create_backup` interceptor.
        When both interceptors are used, this `post_create_backup_with_metadata` interceptor runs after the
        `post_create_backup` interceptor. The (possibly modified) response returned by
        `post_create_backup` will be passed to
        `post_create_backup_with_metadata`.
        """
        return response, metadata

    def pre_create_schema_bundle(
        self,
        request: bigtable_table_admin.CreateSchemaBundleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CreateSchemaBundleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_schema_bundle

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_create_schema_bundle(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_schema_bundle

        DEPRECATED. Please use the `post_create_schema_bundle_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_create_schema_bundle` interceptor runs
        before the `post_create_schema_bundle_with_metadata` interceptor.
        """
        return response

    def post_create_schema_bundle_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_schema_bundle

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_create_schema_bundle_with_metadata`
        interceptor in new development instead of the `post_create_schema_bundle` interceptor.
        When both interceptors are used, this `post_create_schema_bundle_with_metadata` interceptor runs after the
        `post_create_schema_bundle` interceptor. The (possibly modified) response returned by
        `post_create_schema_bundle` will be passed to
        `post_create_schema_bundle_with_metadata`.
        """
        return response, metadata

    def pre_create_table(
        self,
        request: bigtable_table_admin.CreateTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CreateTableRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_create_table(self, response: gba_table.Table) -> gba_table.Table:
        """Post-rpc interceptor for create_table

        DEPRECATED. Please use the `post_create_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_create_table` interceptor runs
        before the `post_create_table_with_metadata` interceptor.
        """
        return response

    def post_create_table_with_metadata(
        self,
        response: gba_table.Table,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gba_table.Table, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_create_table_with_metadata`
        interceptor in new development instead of the `post_create_table` interceptor.
        When both interceptors are used, this `post_create_table_with_metadata` interceptor runs after the
        `post_create_table` interceptor. The (possibly modified) response returned by
        `post_create_table` will be passed to
        `post_create_table_with_metadata`.
        """
        return response, metadata

    def pre_create_table_from_snapshot(
        self,
        request: bigtable_table_admin.CreateTableFromSnapshotRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.CreateTableFromSnapshotRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_table_from_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_create_table_from_snapshot(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_table_from_snapshot

        DEPRECATED. Please use the `post_create_table_from_snapshot_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_create_table_from_snapshot` interceptor runs
        before the `post_create_table_from_snapshot_with_metadata` interceptor.
        """
        return response

    def post_create_table_from_snapshot_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_table_from_snapshot

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_create_table_from_snapshot_with_metadata`
        interceptor in new development instead of the `post_create_table_from_snapshot` interceptor.
        When both interceptors are used, this `post_create_table_from_snapshot_with_metadata` interceptor runs after the
        `post_create_table_from_snapshot` interceptor. The (possibly modified) response returned by
        `post_create_table_from_snapshot` will be passed to
        `post_create_table_from_snapshot_with_metadata`.
        """
        return response, metadata

    def pre_delete_authorized_view(
        self,
        request: bigtable_table_admin.DeleteAuthorizedViewRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DeleteAuthorizedViewRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_authorized_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_delete_backup(
        self,
        request: bigtable_table_admin.DeleteBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DeleteBackupRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_delete_schema_bundle(
        self,
        request: bigtable_table_admin.DeleteSchemaBundleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DeleteSchemaBundleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_schema_bundle

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_delete_snapshot(
        self,
        request: bigtable_table_admin.DeleteSnapshotRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DeleteSnapshotRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_delete_table(
        self,
        request: bigtable_table_admin.DeleteTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DeleteTableRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_drop_row_range(
        self,
        request: bigtable_table_admin.DropRowRangeRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.DropRowRangeRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for drop_row_range

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def pre_generate_consistency_token(
        self,
        request: bigtable_table_admin.GenerateConsistencyTokenRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GenerateConsistencyTokenRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for generate_consistency_token

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_generate_consistency_token(
        self, response: bigtable_table_admin.GenerateConsistencyTokenResponse
    ) -> bigtable_table_admin.GenerateConsistencyTokenResponse:
        """Post-rpc interceptor for generate_consistency_token

        DEPRECATED. Please use the `post_generate_consistency_token_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_generate_consistency_token` interceptor runs
        before the `post_generate_consistency_token_with_metadata` interceptor.
        """
        return response

    def post_generate_consistency_token_with_metadata(
        self,
        response: bigtable_table_admin.GenerateConsistencyTokenResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GenerateConsistencyTokenResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for generate_consistency_token

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_generate_consistency_token_with_metadata`
        interceptor in new development instead of the `post_generate_consistency_token` interceptor.
        When both interceptors are used, this `post_generate_consistency_token_with_metadata` interceptor runs after the
        `post_generate_consistency_token` interceptor. The (possibly modified) response returned by
        `post_generate_consistency_token` will be passed to
        `post_generate_consistency_token_with_metadata`.
        """
        return response, metadata

    def pre_get_authorized_view(
        self,
        request: bigtable_table_admin.GetAuthorizedViewRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GetAuthorizedViewRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_authorized_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_authorized_view(
        self, response: table.AuthorizedView
    ) -> table.AuthorizedView:
        """Post-rpc interceptor for get_authorized_view

        DEPRECATED. Please use the `post_get_authorized_view_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_authorized_view` interceptor runs
        before the `post_get_authorized_view_with_metadata` interceptor.
        """
        return response

    def post_get_authorized_view_with_metadata(
        self,
        response: table.AuthorizedView,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[table.AuthorizedView, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_authorized_view

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_authorized_view_with_metadata`
        interceptor in new development instead of the `post_get_authorized_view` interceptor.
        When both interceptors are used, this `post_get_authorized_view_with_metadata` interceptor runs after the
        `post_get_authorized_view` interceptor. The (possibly modified) response returned by
        `post_get_authorized_view` will be passed to
        `post_get_authorized_view_with_metadata`.
        """
        return response, metadata

    def pre_get_backup(
        self,
        request: bigtable_table_admin.GetBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GetBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_backup(self, response: table.Backup) -> table.Backup:
        """Post-rpc interceptor for get_backup

        DEPRECATED. Please use the `post_get_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_backup` interceptor runs
        before the `post_get_backup_with_metadata` interceptor.
        """
        return response

    def post_get_backup_with_metadata(
        self, response: table.Backup, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[table.Backup, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_backup_with_metadata`
        interceptor in new development instead of the `post_get_backup` interceptor.
        When both interceptors are used, this `post_get_backup_with_metadata` interceptor runs after the
        `post_get_backup` interceptor. The (possibly modified) response returned by
        `post_get_backup` will be passed to
        `post_get_backup_with_metadata`.
        """
        return response, metadata

    def pre_get_iam_policy(
        self,
        request: iam_policy_pb2.GetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        DEPRECATED. Please use the `post_get_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_iam_policy` interceptor runs
        before the `post_get_iam_policy_with_metadata` interceptor.
        """
        return response

    def post_get_iam_policy_with_metadata(
        self,
        response: policy_pb2.Policy,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[policy_pb2.Policy, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_iam_policy_with_metadata`
        interceptor in new development instead of the `post_get_iam_policy` interceptor.
        When both interceptors are used, this `post_get_iam_policy_with_metadata` interceptor runs after the
        `post_get_iam_policy` interceptor. The (possibly modified) response returned by
        `post_get_iam_policy` will be passed to
        `post_get_iam_policy_with_metadata`.
        """
        return response, metadata

    def pre_get_schema_bundle(
        self,
        request: bigtable_table_admin.GetSchemaBundleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GetSchemaBundleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_schema_bundle

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_schema_bundle(
        self, response: table.SchemaBundle
    ) -> table.SchemaBundle:
        """Post-rpc interceptor for get_schema_bundle

        DEPRECATED. Please use the `post_get_schema_bundle_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_schema_bundle` interceptor runs
        before the `post_get_schema_bundle_with_metadata` interceptor.
        """
        return response

    def post_get_schema_bundle_with_metadata(
        self,
        response: table.SchemaBundle,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[table.SchemaBundle, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_schema_bundle

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_schema_bundle_with_metadata`
        interceptor in new development instead of the `post_get_schema_bundle` interceptor.
        When both interceptors are used, this `post_get_schema_bundle_with_metadata` interceptor runs after the
        `post_get_schema_bundle` interceptor. The (possibly modified) response returned by
        `post_get_schema_bundle` will be passed to
        `post_get_schema_bundle_with_metadata`.
        """
        return response, metadata

    def pre_get_snapshot(
        self,
        request: bigtable_table_admin.GetSnapshotRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GetSnapshotRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_snapshot(self, response: table.Snapshot) -> table.Snapshot:
        """Post-rpc interceptor for get_snapshot

        DEPRECATED. Please use the `post_get_snapshot_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_snapshot` interceptor runs
        before the `post_get_snapshot_with_metadata` interceptor.
        """
        return response

    def post_get_snapshot_with_metadata(
        self,
        response: table.Snapshot,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[table.Snapshot, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_snapshot

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_snapshot_with_metadata`
        interceptor in new development instead of the `post_get_snapshot` interceptor.
        When both interceptors are used, this `post_get_snapshot_with_metadata` interceptor runs after the
        `post_get_snapshot` interceptor. The (possibly modified) response returned by
        `post_get_snapshot` will be passed to
        `post_get_snapshot_with_metadata`.
        """
        return response, metadata

    def pre_get_table(
        self,
        request: bigtable_table_admin.GetTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.GetTableRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_get_table(self, response: table.Table) -> table.Table:
        """Post-rpc interceptor for get_table

        DEPRECATED. Please use the `post_get_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_get_table` interceptor runs
        before the `post_get_table_with_metadata` interceptor.
        """
        return response

    def post_get_table_with_metadata(
        self, response: table.Table, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[table.Table, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_get_table_with_metadata`
        interceptor in new development instead of the `post_get_table` interceptor.
        When both interceptors are used, this `post_get_table_with_metadata` interceptor runs after the
        `post_get_table` interceptor. The (possibly modified) response returned by
        `post_get_table` will be passed to
        `post_get_table_with_metadata`.
        """
        return response, metadata

    def pre_list_authorized_views(
        self,
        request: bigtable_table_admin.ListAuthorizedViewsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListAuthorizedViewsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_authorized_views

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_list_authorized_views(
        self, response: bigtable_table_admin.ListAuthorizedViewsResponse
    ) -> bigtable_table_admin.ListAuthorizedViewsResponse:
        """Post-rpc interceptor for list_authorized_views

        DEPRECATED. Please use the `post_list_authorized_views_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_list_authorized_views` interceptor runs
        before the `post_list_authorized_views_with_metadata` interceptor.
        """
        return response

    def post_list_authorized_views_with_metadata(
        self,
        response: bigtable_table_admin.ListAuthorizedViewsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListAuthorizedViewsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_authorized_views

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_list_authorized_views_with_metadata`
        interceptor in new development instead of the `post_list_authorized_views` interceptor.
        When both interceptors are used, this `post_list_authorized_views_with_metadata` interceptor runs after the
        `post_list_authorized_views` interceptor. The (possibly modified) response returned by
        `post_list_authorized_views` will be passed to
        `post_list_authorized_views_with_metadata`.
        """
        return response, metadata

    def pre_list_backups(
        self,
        request: bigtable_table_admin.ListBackupsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListBackupsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_backups

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_list_backups(
        self, response: bigtable_table_admin.ListBackupsResponse
    ) -> bigtable_table_admin.ListBackupsResponse:
        """Post-rpc interceptor for list_backups

        DEPRECATED. Please use the `post_list_backups_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_list_backups` interceptor runs
        before the `post_list_backups_with_metadata` interceptor.
        """
        return response

    def post_list_backups_with_metadata(
        self,
        response: bigtable_table_admin.ListBackupsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListBackupsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_backups

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_list_backups_with_metadata`
        interceptor in new development instead of the `post_list_backups` interceptor.
        When both interceptors are used, this `post_list_backups_with_metadata` interceptor runs after the
        `post_list_backups` interceptor. The (possibly modified) response returned by
        `post_list_backups` will be passed to
        `post_list_backups_with_metadata`.
        """
        return response, metadata

    def pre_list_schema_bundles(
        self,
        request: bigtable_table_admin.ListSchemaBundlesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListSchemaBundlesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_schema_bundles

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_list_schema_bundles(
        self, response: bigtable_table_admin.ListSchemaBundlesResponse
    ) -> bigtable_table_admin.ListSchemaBundlesResponse:
        """Post-rpc interceptor for list_schema_bundles

        DEPRECATED. Please use the `post_list_schema_bundles_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_list_schema_bundles` interceptor runs
        before the `post_list_schema_bundles_with_metadata` interceptor.
        """
        return response

    def post_list_schema_bundles_with_metadata(
        self,
        response: bigtable_table_admin.ListSchemaBundlesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListSchemaBundlesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_schema_bundles

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_list_schema_bundles_with_metadata`
        interceptor in new development instead of the `post_list_schema_bundles` interceptor.
        When both interceptors are used, this `post_list_schema_bundles_with_metadata` interceptor runs after the
        `post_list_schema_bundles` interceptor. The (possibly modified) response returned by
        `post_list_schema_bundles` will be passed to
        `post_list_schema_bundles_with_metadata`.
        """
        return response, metadata

    def pre_list_snapshots(
        self,
        request: bigtable_table_admin.ListSnapshotsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListSnapshotsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_snapshots

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_list_snapshots(
        self, response: bigtable_table_admin.ListSnapshotsResponse
    ) -> bigtable_table_admin.ListSnapshotsResponse:
        """Post-rpc interceptor for list_snapshots

        DEPRECATED. Please use the `post_list_snapshots_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_list_snapshots` interceptor runs
        before the `post_list_snapshots_with_metadata` interceptor.
        """
        return response

    def post_list_snapshots_with_metadata(
        self,
        response: bigtable_table_admin.ListSnapshotsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListSnapshotsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_snapshots

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_list_snapshots_with_metadata`
        interceptor in new development instead of the `post_list_snapshots` interceptor.
        When both interceptors are used, this `post_list_snapshots_with_metadata` interceptor runs after the
        `post_list_snapshots` interceptor. The (possibly modified) response returned by
        `post_list_snapshots` will be passed to
        `post_list_snapshots_with_metadata`.
        """
        return response, metadata

    def pre_list_tables(
        self,
        request: bigtable_table_admin.ListTablesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListTablesRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_tables

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_list_tables(
        self, response: bigtable_table_admin.ListTablesResponse
    ) -> bigtable_table_admin.ListTablesResponse:
        """Post-rpc interceptor for list_tables

        DEPRECATED. Please use the `post_list_tables_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_list_tables` interceptor runs
        before the `post_list_tables_with_metadata` interceptor.
        """
        return response

    def post_list_tables_with_metadata(
        self,
        response: bigtable_table_admin.ListTablesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ListTablesResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_tables

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_list_tables_with_metadata`
        interceptor in new development instead of the `post_list_tables` interceptor.
        When both interceptors are used, this `post_list_tables_with_metadata` interceptor runs after the
        `post_list_tables` interceptor. The (possibly modified) response returned by
        `post_list_tables` will be passed to
        `post_list_tables_with_metadata`.
        """
        return response, metadata

    def pre_modify_column_families(
        self,
        request: bigtable_table_admin.ModifyColumnFamiliesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.ModifyColumnFamiliesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for modify_column_families

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_modify_column_families(self, response: table.Table) -> table.Table:
        """Post-rpc interceptor for modify_column_families

        DEPRECATED. Please use the `post_modify_column_families_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_modify_column_families` interceptor runs
        before the `post_modify_column_families_with_metadata` interceptor.
        """
        return response

    def post_modify_column_families_with_metadata(
        self, response: table.Table, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[table.Table, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for modify_column_families

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_modify_column_families_with_metadata`
        interceptor in new development instead of the `post_modify_column_families` interceptor.
        When both interceptors are used, this `post_modify_column_families_with_metadata` interceptor runs after the
        `post_modify_column_families` interceptor. The (possibly modified) response returned by
        `post_modify_column_families` will be passed to
        `post_modify_column_families_with_metadata`.
        """
        return response, metadata

    def pre_restore_table(
        self,
        request: bigtable_table_admin.RestoreTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.RestoreTableRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for restore_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_restore_table(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for restore_table

        DEPRECATED. Please use the `post_restore_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_restore_table` interceptor runs
        before the `post_restore_table_with_metadata` interceptor.
        """
        return response

    def post_restore_table_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for restore_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_restore_table_with_metadata`
        interceptor in new development instead of the `post_restore_table` interceptor.
        When both interceptors are used, this `post_restore_table_with_metadata` interceptor runs after the
        `post_restore_table` interceptor. The (possibly modified) response returned by
        `post_restore_table` will be passed to
        `post_restore_table_with_metadata`.
        """
        return response, metadata

    def pre_set_iam_policy(
        self,
        request: iam_policy_pb2.SetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        DEPRECATED. Please use the `post_set_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_set_iam_policy` interceptor runs
        before the `post_set_iam_policy_with_metadata` interceptor.
        """
        return response

    def post_set_iam_policy_with_metadata(
        self,
        response: policy_pb2.Policy,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[policy_pb2.Policy, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_set_iam_policy_with_metadata`
        interceptor in new development instead of the `post_set_iam_policy` interceptor.
        When both interceptors are used, this `post_set_iam_policy_with_metadata` interceptor runs after the
        `post_set_iam_policy` interceptor. The (possibly modified) response returned by
        `post_set_iam_policy` will be passed to
        `post_set_iam_policy_with_metadata`.
        """
        return response, metadata

    def pre_snapshot_table(
        self,
        request: bigtable_table_admin.SnapshotTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.SnapshotTableRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for snapshot_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_snapshot_table(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for snapshot_table

        DEPRECATED. Please use the `post_snapshot_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_snapshot_table` interceptor runs
        before the `post_snapshot_table_with_metadata` interceptor.
        """
        return response

    def post_snapshot_table_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for snapshot_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_snapshot_table_with_metadata`
        interceptor in new development instead of the `post_snapshot_table` interceptor.
        When both interceptors are used, this `post_snapshot_table_with_metadata` interceptor runs after the
        `post_snapshot_table` interceptor. The (possibly modified) response returned by
        `post_snapshot_table` will be passed to
        `post_snapshot_table_with_metadata`.
        """
        return response, metadata

    def pre_test_iam_permissions(
        self,
        request: iam_policy_pb2.TestIamPermissionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.TestIamPermissionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        DEPRECATED. Please use the `post_test_iam_permissions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_test_iam_permissions` interceptor runs
        before the `post_test_iam_permissions_with_metadata` interceptor.
        """
        return response

    def post_test_iam_permissions_with_metadata(
        self,
        response: iam_policy_pb2.TestIamPermissionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.TestIamPermissionsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_test_iam_permissions_with_metadata`
        interceptor in new development instead of the `post_test_iam_permissions` interceptor.
        When both interceptors are used, this `post_test_iam_permissions_with_metadata` interceptor runs after the
        `post_test_iam_permissions` interceptor. The (possibly modified) response returned by
        `post_test_iam_permissions` will be passed to
        `post_test_iam_permissions_with_metadata`.
        """
        return response, metadata

    def pre_undelete_table(
        self,
        request: bigtable_table_admin.UndeleteTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.UndeleteTableRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for undelete_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_undelete_table(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for undelete_table

        DEPRECATED. Please use the `post_undelete_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_undelete_table` interceptor runs
        before the `post_undelete_table_with_metadata` interceptor.
        """
        return response

    def post_undelete_table_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for undelete_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_undelete_table_with_metadata`
        interceptor in new development instead of the `post_undelete_table` interceptor.
        When both interceptors are used, this `post_undelete_table_with_metadata` interceptor runs after the
        `post_undelete_table` interceptor. The (possibly modified) response returned by
        `post_undelete_table` will be passed to
        `post_undelete_table_with_metadata`.
        """
        return response, metadata

    def pre_update_authorized_view(
        self,
        request: bigtable_table_admin.UpdateAuthorizedViewRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.UpdateAuthorizedViewRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_authorized_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_update_authorized_view(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_authorized_view

        DEPRECATED. Please use the `post_update_authorized_view_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_update_authorized_view` interceptor runs
        before the `post_update_authorized_view_with_metadata` interceptor.
        """
        return response

    def post_update_authorized_view_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_authorized_view

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_update_authorized_view_with_metadata`
        interceptor in new development instead of the `post_update_authorized_view` interceptor.
        When both interceptors are used, this `post_update_authorized_view_with_metadata` interceptor runs after the
        `post_update_authorized_view` interceptor. The (possibly modified) response returned by
        `post_update_authorized_view` will be passed to
        `post_update_authorized_view_with_metadata`.
        """
        return response, metadata

    def pre_update_backup(
        self,
        request: bigtable_table_admin.UpdateBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.UpdateBackupRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_update_backup(self, response: table.Backup) -> table.Backup:
        """Post-rpc interceptor for update_backup

        DEPRECATED. Please use the `post_update_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_update_backup` interceptor runs
        before the `post_update_backup_with_metadata` interceptor.
        """
        return response

    def post_update_backup_with_metadata(
        self, response: table.Backup, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[table.Backup, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_update_backup_with_metadata`
        interceptor in new development instead of the `post_update_backup` interceptor.
        When both interceptors are used, this `post_update_backup_with_metadata` interceptor runs after the
        `post_update_backup` interceptor. The (possibly modified) response returned by
        `post_update_backup` will be passed to
        `post_update_backup_with_metadata`.
        """
        return response, metadata

    def pre_update_schema_bundle(
        self,
        request: bigtable_table_admin.UpdateSchemaBundleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.UpdateSchemaBundleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_schema_bundle

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_update_schema_bundle(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_schema_bundle

        DEPRECATED. Please use the `post_update_schema_bundle_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_update_schema_bundle` interceptor runs
        before the `post_update_schema_bundle_with_metadata` interceptor.
        """
        return response

    def post_update_schema_bundle_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_schema_bundle

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_update_schema_bundle_with_metadata`
        interceptor in new development instead of the `post_update_schema_bundle` interceptor.
        When both interceptors are used, this `post_update_schema_bundle_with_metadata` interceptor runs after the
        `post_update_schema_bundle` interceptor. The (possibly modified) response returned by
        `post_update_schema_bundle` will be passed to
        `post_update_schema_bundle_with_metadata`.
        """
        return response, metadata

    def pre_update_table(
        self,
        request: bigtable_table_admin.UpdateTableRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable_table_admin.UpdateTableRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_table

        Override in a subclass to manipulate the request or metadata
        before they are sent to the BigtableTableAdmin server.
        """
        return request, metadata

    def post_update_table(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_table

        DEPRECATED. Please use the `post_update_table_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the BigtableTableAdmin server but before
        it is returned to user code. This `post_update_table` interceptor runs
        before the `post_update_table_with_metadata` interceptor.
        """
        return response

    def post_update_table_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_table

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the BigtableTableAdmin server but before it is returned to user code.

        We recommend only using this `post_update_table_with_metadata`
        interceptor in new development instead of the `post_update_table` interceptor.
        When both interceptors are used, this `post_update_table_with_metadata` interceptor runs after the
        `post_update_table` interceptor. The (possibly modified) response returned by
        `post_update_table` will be passed to
        `post_update_table_with_metadata`.
        """
        return response, metadata


@dataclasses.dataclass
class BigtableTableAdminRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: BigtableTableAdminRestInterceptor


class BigtableTableAdminRestTransport(_BaseBigtableTableAdminRestTransport):
    """REST backend synchronous transport for BigtableTableAdmin.

    Service for creating, configuring, and deleting Cloud
    Bigtable tables.

    Provides access to the table schemas only, not the data stored
    within the tables.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "bigtableadmin.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[BigtableTableAdminRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'bigtableadmin.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        self._operations_client: Optional[operations_v1.AbstractOperationsClient] = None
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or BigtableTableAdminRestInterceptor()
        self._prep_wrapped_messages(client_info)

    @property
    def operations_client(self) -> operations_v1.AbstractOperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Only create a new client if we do not already have one.
        if self._operations_client is None:
            http_options: Dict[str, List[Dict[str, str]]] = {
                "google.longrunning.Operations.CancelOperation": [
                    {
                        "method": "post",
                        "uri": "/v2/{name=operations/**}:cancel",
                    },
                ],
                "google.longrunning.Operations.DeleteOperation": [
                    {
                        "method": "delete",
                        "uri": "/v2/{name=operations/**}",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v2/{name=operations/**}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/v2/{name=operations/projects/**}/operations",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v2",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _CheckConsistency(
        _BaseBigtableTableAdminRestTransport._BaseCheckConsistency,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CheckConsistency")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CheckConsistencyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.CheckConsistencyResponse:
            r"""Call the check consistency method over HTTP.

            Args:
                request (~.bigtable_table_admin.CheckConsistencyRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.CheckConsistencyResponse:
                    Response message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCheckConsistency._get_http_options()
            )

            request, metadata = self._interceptor.pre_check_consistency(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCheckConsistency._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCheckConsistency._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCheckConsistency._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CheckConsistency",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CheckConsistency",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._CheckConsistency._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.CheckConsistencyResponse()
            pb_resp = bigtable_table_admin.CheckConsistencyResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_check_consistency(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_check_consistency_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        bigtable_table_admin.CheckConsistencyResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.check_consistency",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CheckConsistency",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CopyBackup(
        _BaseBigtableTableAdminRestTransport._BaseCopyBackup, BigtableTableAdminRestStub
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CopyBackup")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CopyBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the copy backup method over HTTP.

            Args:
                request (~.bigtable_table_admin.CopyBackupRequest):
                    The request object. The request for
                [CopyBackup][google.bigtable.admin.v2.BigtableTableAdmin.CopyBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCopyBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_copy_backup(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCopyBackup._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCopyBackup._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCopyBackup._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CopyBackup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CopyBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._CopyBackup._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_copy_backup(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_copy_backup_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.copy_backup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CopyBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateAuthorizedView(
        _BaseBigtableTableAdminRestTransport._BaseCreateAuthorizedView,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CreateAuthorizedView")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CreateAuthorizedViewRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create authorized view method over HTTP.

            Args:
                request (~.bigtable_table_admin.CreateAuthorizedViewRequest):
                    The request object. The request for
                [CreateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.CreateAuthorizedView]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCreateAuthorizedView._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_authorized_view(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCreateAuthorizedView._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCreateAuthorizedView._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCreateAuthorizedView._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CreateAuthorizedView",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateAuthorizedView",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._CreateAuthorizedView._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_authorized_view(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_authorized_view_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.create_authorized_view",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateAuthorizedView",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateBackup(
        _BaseBigtableTableAdminRestTransport._BaseCreateBackup,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CreateBackup")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CreateBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create backup method over HTTP.

            Args:
                request (~.bigtable_table_admin.CreateBackupRequest):
                    The request object. The request for
                [CreateBackup][google.bigtable.admin.v2.BigtableTableAdmin.CreateBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCreateBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_backup(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCreateBackup._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCreateBackup._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCreateBackup._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CreateBackup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._CreateBackup._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_backup(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_backup_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.create_backup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateSchemaBundle(
        _BaseBigtableTableAdminRestTransport._BaseCreateSchemaBundle,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CreateSchemaBundle")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CreateSchemaBundleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create schema bundle method over HTTP.

            Args:
                request (~.bigtable_table_admin.CreateSchemaBundleRequest):
                    The request object. The request for
                [CreateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.CreateSchemaBundle].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCreateSchemaBundle._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_schema_bundle(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCreateSchemaBundle._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCreateSchemaBundle._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCreateSchemaBundle._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CreateSchemaBundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateSchemaBundle",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._CreateSchemaBundle._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_schema_bundle(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_schema_bundle_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.create_schema_bundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateSchemaBundle",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateTable(
        _BaseBigtableTableAdminRestTransport._BaseCreateTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CreateTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CreateTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gba_table.Table:
            r"""Call the create table method over HTTP.

            Args:
                request (~.bigtable_table_admin.CreateTableRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.CreateTable][google.bigtable.admin.v2.BigtableTableAdmin.CreateTable]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gba_table.Table:
                    A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCreateTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCreateTable._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCreateTable._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCreateTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CreateTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._CreateTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gba_table.Table()
            pb_resp = gba_table.Table.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gba_table.Table.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.create_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateTableFromSnapshot(
        _BaseBigtableTableAdminRestTransport._BaseCreateTableFromSnapshot,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.CreateTableFromSnapshot")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.CreateTableFromSnapshotRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create table from
            snapshot method over HTTP.

                Args:
                    request (~.bigtable_table_admin.CreateTableFromSnapshotRequest):
                        The request object. Request message for
                    [google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot]

                    Note: This is a private alpha release of Cloud Bigtable
                    snapshots. This feature is not currently available to
                    most Cloud Bigtable customers. This feature might be
                    changed in backward-incompatible ways and is not
                    recommended for production use. It is not subject to any
                    SLA or deprecation policy.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.operations_pb2.Operation:
                        This resource represents a
                    long-running operation that is the
                    result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseCreateTableFromSnapshot._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_table_from_snapshot(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseCreateTableFromSnapshot._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseCreateTableFromSnapshot._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseCreateTableFromSnapshot._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.CreateTableFromSnapshot",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateTableFromSnapshot",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._CreateTableFromSnapshot._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_table_from_snapshot(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_table_from_snapshot_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.create_table_from_snapshot",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "CreateTableFromSnapshot",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteAuthorizedView(
        _BaseBigtableTableAdminRestTransport._BaseDeleteAuthorizedView,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DeleteAuthorizedView")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DeleteAuthorizedViewRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete authorized view method over HTTP.

            Args:
                request (~.bigtable_table_admin.DeleteAuthorizedViewRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDeleteAuthorizedView._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_authorized_view(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDeleteAuthorizedView._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDeleteAuthorizedView._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DeleteAuthorizedView",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DeleteAuthorizedView",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._DeleteAuthorizedView._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteBackup(
        _BaseBigtableTableAdminRestTransport._BaseDeleteBackup,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DeleteBackup")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DeleteBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete backup method over HTTP.

            Args:
                request (~.bigtable_table_admin.DeleteBackupRequest):
                    The request object. The request for
                [DeleteBackup][google.bigtable.admin.v2.BigtableTableAdmin.DeleteBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDeleteBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_backup(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDeleteBackup._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDeleteBackup._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DeleteBackup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DeleteBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._DeleteBackup._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteSchemaBundle(
        _BaseBigtableTableAdminRestTransport._BaseDeleteSchemaBundle,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DeleteSchemaBundle")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DeleteSchemaBundleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete schema bundle method over HTTP.

            Args:
                request (~.bigtable_table_admin.DeleteSchemaBundleRequest):
                    The request object. The request for
                [DeleteSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSchemaBundle].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDeleteSchemaBundle._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_schema_bundle(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDeleteSchemaBundle._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDeleteSchemaBundle._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DeleteSchemaBundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DeleteSchemaBundle",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._DeleteSchemaBundle._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteSnapshot(
        _BaseBigtableTableAdminRestTransport._BaseDeleteSnapshot,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DeleteSnapshot")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DeleteSnapshotRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete snapshot method over HTTP.

            Args:
                request (~.bigtable_table_admin.DeleteSnapshotRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDeleteSnapshot._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_snapshot(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDeleteSnapshot._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDeleteSnapshot._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DeleteSnapshot",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DeleteSnapshot",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._DeleteSnapshot._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteTable(
        _BaseBigtableTableAdminRestTransport._BaseDeleteTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DeleteTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DeleteTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete table method over HTTP.

            Args:
                request (~.bigtable_table_admin.DeleteTableRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDeleteTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDeleteTable._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDeleteTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DeleteTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DeleteTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._DeleteTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DropRowRange(
        _BaseBigtableTableAdminRestTransport._BaseDropRowRange,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.DropRowRange")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.DropRowRangeRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the drop row range method over HTTP.

            Args:
                request (~.bigtable_table_admin.DropRowRangeRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange][google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseDropRowRange._get_http_options()
            )

            request, metadata = self._interceptor.pre_drop_row_range(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseDropRowRange._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseDropRowRange._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseDropRowRange._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.DropRowRange",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "DropRowRange",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._DropRowRange._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GenerateConsistencyToken(
        _BaseBigtableTableAdminRestTransport._BaseGenerateConsistencyToken,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GenerateConsistencyToken")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GenerateConsistencyTokenRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.GenerateConsistencyTokenResponse:
            r"""Call the generate consistency
            token method over HTTP.

                Args:
                    request (~.bigtable_table_admin.GenerateConsistencyTokenRequest):
                        The request object. Request message for
                    [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.bigtable_table_admin.GenerateConsistencyTokenResponse:
                        Response message for
                    [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGenerateConsistencyToken._get_http_options()
            )

            request, metadata = self._interceptor.pre_generate_consistency_token(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGenerateConsistencyToken._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseGenerateConsistencyToken._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGenerateConsistencyToken._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GenerateConsistencyToken",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GenerateConsistencyToken",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._GenerateConsistencyToken._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.GenerateConsistencyTokenResponse()
            pb_resp = bigtable_table_admin.GenerateConsistencyTokenResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_generate_consistency_token(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_generate_consistency_token_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        bigtable_table_admin.GenerateConsistencyTokenResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.generate_consistency_token",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GenerateConsistencyToken",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetAuthorizedView(
        _BaseBigtableTableAdminRestTransport._BaseGetAuthorizedView,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetAuthorizedView")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GetAuthorizedViewRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.AuthorizedView:
            r"""Call the get authorized view method over HTTP.

            Args:
                request (~.bigtable_table_admin.GetAuthorizedViewRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.AuthorizedView:
                    AuthorizedViews represent subsets of
                a particular Cloud Bigtable table. Users
                can configure access to each Authorized
                View independently from the table and
                use the existing Data APIs to access the
                subset of data.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetAuthorizedView._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_authorized_view(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetAuthorizedView._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetAuthorizedView._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetAuthorizedView",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetAuthorizedView",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetAuthorizedView._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.AuthorizedView()
            pb_resp = table.AuthorizedView.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_authorized_view(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_authorized_view_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.AuthorizedView.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_authorized_view",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetAuthorizedView",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetBackup(
        _BaseBigtableTableAdminRestTransport._BaseGetBackup, BigtableTableAdminRestStub
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetBackup")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GetBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.Backup:
            r"""Call the get backup method over HTTP.

            Args:
                request (~.bigtable_table_admin.GetBackupRequest):
                    The request object. The request for
                [GetBackup][google.bigtable.admin.v2.BigtableTableAdmin.GetBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.Backup:
                    A backup of a Cloud Bigtable table.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_backup(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetBackup._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetBackup._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetBackup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetBackup._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.Backup()
            pb_resp = table.Backup.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_backup(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_backup_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.Backup.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_backup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetIamPolicy(
        _BaseBigtableTableAdminRestTransport._BaseGetIamPolicy,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetIamPolicy")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: iam_policy_pb2.GetIamPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_pb2.Policy:
            r"""Call the get iam policy method over HTTP.

            Args:
                request (~.iam_policy_pb2.GetIamPolicyRequest):
                    The request object. Request message for ``GetIamPolicy`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_pb2.Policy:
                    An Identity and Access Management (IAM) policy, which
                specifies access controls for Google Cloud resources.

                A ``Policy`` is a collection of ``bindings``. A
                ``binding`` binds one or more ``members``, or
                principals, to a single ``role``. Principals can be user
                accounts, service accounts, Google groups, and domains
                (such as G Suite). A ``role`` is a named list of
                permissions; each ``role`` can be an IAM predefined role
                or a user-created custom role.

                For some types of Google Cloud resources, a ``binding``
                can also specify a ``condition``, which is a logical
                expression that allows access to a resource only if the
                expression evaluates to ``true``. A condition can add
                constraints based on attributes of the request, the
                resource, or both. To learn which resources support
                conditions in their IAM policies, see the `IAM
                documentation <https://cloud.google.com/iam/help/conditions/resource-policies>`__.

                **JSON example:**

                ::

                       {
                         "bindings": [
                           {
                             "role": "roles/resourcemanager.organizationAdmin",
                             "members": [
                               "user:mike@example.com",
                               "group:admins@example.com",
                               "domain:google.com",
                               "serviceAccount:my-project-id@appspot.gserviceaccount.com"
                             ]
                           },
                           {
                             "role": "roles/resourcemanager.organizationViewer",
                             "members": [
                               "user:eve@example.com"
                             ],
                             "condition": {
                               "title": "expirable access",
                               "description": "Does not grant access after Sep 2020",
                               "expression": "request.time <
                               timestamp('2020-10-01T00:00:00.000Z')",
                             }
                           }
                         ],
                         "etag": "BwWWja0YfJA=",
                         "version": 3
                       }

                **YAML example:**

                ::

                       bindings:
                       - members:
                         - user:mike@example.com
                         - group:admins@example.com
                         - domain:google.com
                         - serviceAccount:my-project-id@appspot.gserviceaccount.com
                         role: roles/resourcemanager.organizationAdmin
                       - members:
                         - user:eve@example.com
                         role: roles/resourcemanager.organizationViewer
                         condition:
                           title: expirable access
                           description: Does not grant access after Sep 2020
                           expression: request.time < timestamp('2020-10-01T00:00:00.000Z')
                       etag: BwWWja0YfJA=
                       version: 3

                For a description of IAM and its features, see the `IAM
                documentation <https://cloud.google.com/iam/docs/>`__.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_iam_policy(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseGetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetIamPolicy._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetIamPolicy._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = policy_pb2.Policy()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_iam_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_iam_policy_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_iam_policy",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetSchemaBundle(
        _BaseBigtableTableAdminRestTransport._BaseGetSchemaBundle,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetSchemaBundle")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GetSchemaBundleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.SchemaBundle:
            r"""Call the get schema bundle method over HTTP.

            Args:
                request (~.bigtable_table_admin.GetSchemaBundleRequest):
                    The request object. The request for
                [GetSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.GetSchemaBundle].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.SchemaBundle:
                    A named collection of related
                schemas.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetSchemaBundle._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_schema_bundle(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetSchemaBundle._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetSchemaBundle._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetSchemaBundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetSchemaBundle",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetSchemaBundle._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.SchemaBundle()
            pb_resp = table.SchemaBundle.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_schema_bundle(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_schema_bundle_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.SchemaBundle.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_schema_bundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetSchemaBundle",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetSnapshot(
        _BaseBigtableTableAdminRestTransport._BaseGetSnapshot,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetSnapshot")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GetSnapshotRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.Snapshot:
            r"""Call the get snapshot method over HTTP.

            Args:
                request (~.bigtable_table_admin.GetSnapshotRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.Snapshot:
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

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetSnapshot._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_snapshot(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetSnapshot._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetSnapshot._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetSnapshot",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetSnapshot",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetSnapshot._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.Snapshot()
            pb_resp = table.Snapshot.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_snapshot(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_snapshot_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.Snapshot.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_snapshot",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetSnapshot",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetTable(
        _BaseBigtableTableAdminRestTransport._BaseGetTable, BigtableTableAdminRestStub
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.GetTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.GetTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.Table:
            r"""Call the get table method over HTTP.

            Args:
                request (~.bigtable_table_admin.GetTableRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.GetTable][google.bigtable.admin.v2.BigtableTableAdmin.GetTable]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.Table:
                    A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseGetTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseGetTable._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseGetTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.GetTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._GetTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.Table()
            pb_resp = table.Table.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.Table.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.get_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "GetTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListAuthorizedViews(
        _BaseBigtableTableAdminRestTransport._BaseListAuthorizedViews,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ListAuthorizedViews")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ListAuthorizedViewsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.ListAuthorizedViewsResponse:
            r"""Call the list authorized views method over HTTP.

            Args:
                request (~.bigtable_table_admin.ListAuthorizedViewsRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.ListAuthorizedViewsResponse:
                    Response message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseListAuthorizedViews._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_authorized_views(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseListAuthorizedViews._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseListAuthorizedViews._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ListAuthorizedViews",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListAuthorizedViews",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._ListAuthorizedViews._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.ListAuthorizedViewsResponse()
            pb_resp = bigtable_table_admin.ListAuthorizedViewsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_authorized_views(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_authorized_views_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        bigtable_table_admin.ListAuthorizedViewsResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.list_authorized_views",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListAuthorizedViews",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListBackups(
        _BaseBigtableTableAdminRestTransport._BaseListBackups,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ListBackups")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ListBackupsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.ListBackupsResponse:
            r"""Call the list backups method over HTTP.

            Args:
                request (~.bigtable_table_admin.ListBackupsRequest):
                    The request object. The request for
                [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.ListBackupsResponse:
                    The response for
                [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseListBackups._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_backups(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseListBackups._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseListBackups._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ListBackups",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListBackups",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._ListBackups._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.ListBackupsResponse()
            pb_resp = bigtable_table_admin.ListBackupsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_backups(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_backups_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable_table_admin.ListBackupsResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.list_backups",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListBackups",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListSchemaBundles(
        _BaseBigtableTableAdminRestTransport._BaseListSchemaBundles,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ListSchemaBundles")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ListSchemaBundlesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.ListSchemaBundlesResponse:
            r"""Call the list schema bundles method over HTTP.

            Args:
                request (~.bigtable_table_admin.ListSchemaBundlesRequest):
                    The request object. The request for
                [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.ListSchemaBundlesResponse:
                    The response for
                [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseListSchemaBundles._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_schema_bundles(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseListSchemaBundles._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseListSchemaBundles._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ListSchemaBundles",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListSchemaBundles",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._ListSchemaBundles._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.ListSchemaBundlesResponse()
            pb_resp = bigtable_table_admin.ListSchemaBundlesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_schema_bundles(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_schema_bundles_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        bigtable_table_admin.ListSchemaBundlesResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.list_schema_bundles",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListSchemaBundles",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListSnapshots(
        _BaseBigtableTableAdminRestTransport._BaseListSnapshots,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ListSnapshots")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ListSnapshotsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.ListSnapshotsResponse:
            r"""Call the list snapshots method over HTTP.

            Args:
                request (~.bigtable_table_admin.ListSnapshotsRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.ListSnapshotsResponse:
                    Response message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseListSnapshots._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_snapshots(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseListSnapshots._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseListSnapshots._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ListSnapshots",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListSnapshots",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._ListSnapshots._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.ListSnapshotsResponse()
            pb_resp = bigtable_table_admin.ListSnapshotsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_snapshots(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_snapshots_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        bigtable_table_admin.ListSnapshotsResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.list_snapshots",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListSnapshots",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListTables(
        _BaseBigtableTableAdminRestTransport._BaseListTables, BigtableTableAdminRestStub
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ListTables")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ListTablesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable_table_admin.ListTablesResponse:
            r"""Call the list tables method over HTTP.

            Args:
                request (~.bigtable_table_admin.ListTablesRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable_table_admin.ListTablesResponse:
                    Response message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseListTables._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_tables(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseListTables._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseListTables._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ListTables",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListTables",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._ListTables._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable_table_admin.ListTablesResponse()
            pb_resp = bigtable_table_admin.ListTablesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_tables(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_tables_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable_table_admin.ListTablesResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.list_tables",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ListTables",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ModifyColumnFamilies(
        _BaseBigtableTableAdminRestTransport._BaseModifyColumnFamilies,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.ModifyColumnFamilies")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.ModifyColumnFamiliesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.Table:
            r"""Call the modify column families method over HTTP.

            Args:
                request (~.bigtable_table_admin.ModifyColumnFamiliesRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies][google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.Table:
                    A collection of user data indexed by
                row, column, and timestamp. Each table
                is served using the resources of its
                parent cluster.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseModifyColumnFamilies._get_http_options()
            )

            request, metadata = self._interceptor.pre_modify_column_families(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseModifyColumnFamilies._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseModifyColumnFamilies._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseModifyColumnFamilies._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.ModifyColumnFamilies",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ModifyColumnFamilies",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._ModifyColumnFamilies._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.Table()
            pb_resp = table.Table.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_modify_column_families(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_modify_column_families_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.Table.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.modify_column_families",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "ModifyColumnFamilies",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _RestoreTable(
        _BaseBigtableTableAdminRestTransport._BaseRestoreTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.RestoreTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.RestoreTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the restore table method over HTTP.

            Args:
                request (~.bigtable_table_admin.RestoreTableRequest):
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
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseRestoreTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_restore_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseRestoreTable._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseRestoreTable._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseRestoreTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.RestoreTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "RestoreTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._RestoreTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_restore_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_restore_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.restore_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "RestoreTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SetIamPolicy(
        _BaseBigtableTableAdminRestTransport._BaseSetIamPolicy,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.SetIamPolicy")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: iam_policy_pb2.SetIamPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_pb2.Policy:
            r"""Call the set iam policy method over HTTP.

            Args:
                request (~.iam_policy_pb2.SetIamPolicyRequest):
                    The request object. Request message for ``SetIamPolicy`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_pb2.Policy:
                    An Identity and Access Management (IAM) policy, which
                specifies access controls for Google Cloud resources.

                A ``Policy`` is a collection of ``bindings``. A
                ``binding`` binds one or more ``members``, or
                principals, to a single ``role``. Principals can be user
                accounts, service accounts, Google groups, and domains
                (such as G Suite). A ``role`` is a named list of
                permissions; each ``role`` can be an IAM predefined role
                or a user-created custom role.

                For some types of Google Cloud resources, a ``binding``
                can also specify a ``condition``, which is a logical
                expression that allows access to a resource only if the
                expression evaluates to ``true``. A condition can add
                constraints based on attributes of the request, the
                resource, or both. To learn which resources support
                conditions in their IAM policies, see the `IAM
                documentation <https://cloud.google.com/iam/help/conditions/resource-policies>`__.

                **JSON example:**

                ::

                       {
                         "bindings": [
                           {
                             "role": "roles/resourcemanager.organizationAdmin",
                             "members": [
                               "user:mike@example.com",
                               "group:admins@example.com",
                               "domain:google.com",
                               "serviceAccount:my-project-id@appspot.gserviceaccount.com"
                             ]
                           },
                           {
                             "role": "roles/resourcemanager.organizationViewer",
                             "members": [
                               "user:eve@example.com"
                             ],
                             "condition": {
                               "title": "expirable access",
                               "description": "Does not grant access after Sep 2020",
                               "expression": "request.time <
                               timestamp('2020-10-01T00:00:00.000Z')",
                             }
                           }
                         ],
                         "etag": "BwWWja0YfJA=",
                         "version": 3
                       }

                **YAML example:**

                ::

                       bindings:
                       - members:
                         - user:mike@example.com
                         - group:admins@example.com
                         - domain:google.com
                         - serviceAccount:my-project-id@appspot.gserviceaccount.com
                         role: roles/resourcemanager.organizationAdmin
                       - members:
                         - user:eve@example.com
                         role: roles/resourcemanager.organizationViewer
                         condition:
                           title: expirable access
                           description: Does not grant access after Sep 2020
                           expression: request.time < timestamp('2020-10-01T00:00:00.000Z')
                       etag: BwWWja0YfJA=
                       version: 3

                For a description of IAM and its features, see the `IAM
                documentation <https://cloud.google.com/iam/docs/>`__.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_set_iam_policy(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseSetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseSetIamPolicy._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._SetIamPolicy._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = policy_pb2.Policy()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_set_iam_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_set_iam_policy_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.set_iam_policy",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "SetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SnapshotTable(
        _BaseBigtableTableAdminRestTransport._BaseSnapshotTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.SnapshotTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.SnapshotTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the snapshot table method over HTTP.

            Args:
                request (~.bigtable_table_admin.SnapshotTableRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable][google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable]

                Note: This is a private alpha release of Cloud Bigtable
                snapshots. This feature is not currently available to
                most Cloud Bigtable customers. This feature might be
                changed in backward-incompatible ways and is not
                recommended for production use. It is not subject to any
                SLA or deprecation policy.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseSnapshotTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_snapshot_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseSnapshotTable._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseSnapshotTable._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseSnapshotTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.SnapshotTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "SnapshotTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._SnapshotTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_snapshot_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_snapshot_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.snapshot_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "SnapshotTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _TestIamPermissions(
        _BaseBigtableTableAdminRestTransport._BaseTestIamPermissions,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.TestIamPermissions")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: iam_policy_pb2.TestIamPermissionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> iam_policy_pb2.TestIamPermissionsResponse:
            r"""Call the test iam permissions method over HTTP.

            Args:
                request (~.iam_policy_pb2.TestIamPermissionsRequest):
                    The request object. Request message for ``TestIamPermissions`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.iam_policy_pb2.TestIamPermissionsResponse:
                    Response message for ``TestIamPermissions`` method.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseTestIamPermissions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseTestIamPermissions._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._TestIamPermissions._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = iam_policy_pb2.TestIamPermissionsResponse()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_test_iam_permissions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_test_iam_permissions_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.test_iam_permissions",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "TestIamPermissions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UndeleteTable(
        _BaseBigtableTableAdminRestTransport._BaseUndeleteTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.UndeleteTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.UndeleteTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the undelete table method over HTTP.

            Args:
                request (~.bigtable_table_admin.UndeleteTableRequest):
                    The request object. Request message for
                [google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseUndeleteTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_undelete_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseUndeleteTable._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseUndeleteTable._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseUndeleteTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.UndeleteTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UndeleteTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._UndeleteTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_undelete_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_undelete_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.undelete_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UndeleteTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateAuthorizedView(
        _BaseBigtableTableAdminRestTransport._BaseUpdateAuthorizedView,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.UpdateAuthorizedView")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.UpdateAuthorizedViewRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update authorized view method over HTTP.

            Args:
                request (~.bigtable_table_admin.UpdateAuthorizedViewRequest):
                    The request object. The request for
                [UpdateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.UpdateAuthorizedView].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseUpdateAuthorizedView._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_authorized_view(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseUpdateAuthorizedView._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseUpdateAuthorizedView._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseUpdateAuthorizedView._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.UpdateAuthorizedView",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateAuthorizedView",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._UpdateAuthorizedView._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_authorized_view(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_authorized_view_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.update_authorized_view",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateAuthorizedView",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateBackup(
        _BaseBigtableTableAdminRestTransport._BaseUpdateBackup,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.UpdateBackup")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.UpdateBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> table.Backup:
            r"""Call the update backup method over HTTP.

            Args:
                request (~.bigtable_table_admin.UpdateBackupRequest):
                    The request object. The request for
                [UpdateBackup][google.bigtable.admin.v2.BigtableTableAdmin.UpdateBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.table.Backup:
                    A backup of a Cloud Bigtable table.
            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseUpdateBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_backup(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseUpdateBackup._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseUpdateBackup._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseUpdateBackup._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.UpdateBackup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._UpdateBackup._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = table.Backup()
            pb_resp = table.Backup.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_backup(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_backup_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = table.Backup.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.update_backup",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateSchemaBundle(
        _BaseBigtableTableAdminRestTransport._BaseUpdateSchemaBundle,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.UpdateSchemaBundle")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.UpdateSchemaBundleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update schema bundle method over HTTP.

            Args:
                request (~.bigtable_table_admin.UpdateSchemaBundleRequest):
                    The request object. The request for
                [UpdateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.UpdateSchemaBundle].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseUpdateSchemaBundle._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_schema_bundle(
                request, metadata
            )
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseUpdateSchemaBundle._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseUpdateSchemaBundle._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseUpdateSchemaBundle._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.UpdateSchemaBundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateSchemaBundle",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                BigtableTableAdminRestTransport._UpdateSchemaBundle._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_schema_bundle(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_schema_bundle_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.update_schema_bundle",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateSchemaBundle",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateTable(
        _BaseBigtableTableAdminRestTransport._BaseUpdateTable,
        BigtableTableAdminRestStub,
    ):
        def __hash__(self):
            return hash("BigtableTableAdminRestTransport.UpdateTable")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable_table_admin.UpdateTableRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update table method over HTTP.

            Args:
                request (~.bigtable_table_admin.UpdateTableRequest):
                    The request object. The request for
                [UpdateTable][google.bigtable.admin.v2.BigtableTableAdmin.UpdateTable].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseBigtableTableAdminRestTransport._BaseUpdateTable._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_table(request, metadata)
            transcoded_request = _BaseBigtableTableAdminRestTransport._BaseUpdateTable._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableTableAdminRestTransport._BaseUpdateTable._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableTableAdminRestTransport._BaseUpdateTable._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable.admin_v2.BaseBigtableTableAdminClient.UpdateTable",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateTable",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableTableAdminRestTransport._UpdateTable._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_table(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_table_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable.admin_v2.BaseBigtableTableAdminClient.update_table",
                    extra={
                        "serviceName": "google.bigtable.admin.v2.BigtableTableAdmin",
                        "rpcName": "UpdateTable",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def check_consistency(
        self,
    ) -> Callable[
        [bigtable_table_admin.CheckConsistencyRequest],
        bigtable_table_admin.CheckConsistencyResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CheckConsistency(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def copy_backup(
        self,
    ) -> Callable[[bigtable_table_admin.CopyBackupRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CopyBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateAuthorizedViewRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateAuthorizedView(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_backup(
        self,
    ) -> Callable[[bigtable_table_admin.CreateBackupRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateSchemaBundleRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateSchemaBundle(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_table(
        self,
    ) -> Callable[[bigtable_table_admin.CreateTableRequest], gba_table.Table]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_table_from_snapshot(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateTableFromSnapshotRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateTableFromSnapshot(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_authorized_view(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteAuthorizedViewRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteAuthorizedView(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_backup(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteBackupRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_schema_bundle(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteSchemaBundleRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteSchemaBundle(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_snapshot(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteSnapshotRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteSnapshot(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_table(
        self,
    ) -> Callable[[bigtable_table_admin.DeleteTableRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def drop_row_range(
        self,
    ) -> Callable[[bigtable_table_admin.DropRowRangeRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DropRowRange(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def generate_consistency_token(
        self,
    ) -> Callable[
        [bigtable_table_admin.GenerateConsistencyTokenRequest],
        bigtable_table_admin.GenerateConsistencyTokenResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GenerateConsistencyToken(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetAuthorizedViewRequest], table.AuthorizedView
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetAuthorizedView(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_backup(
        self,
    ) -> Callable[[bigtable_table_admin.GetBackupRequest], table.Backup]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_schema_bundle(
        self,
    ) -> Callable[[bigtable_table_admin.GetSchemaBundleRequest], table.SchemaBundle]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetSchemaBundle(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_snapshot(
        self,
    ) -> Callable[[bigtable_table_admin.GetSnapshotRequest], table.Snapshot]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetSnapshot(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_table(
        self,
    ) -> Callable[[bigtable_table_admin.GetTableRequest], table.Table]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_authorized_views(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListAuthorizedViewsRequest],
        bigtable_table_admin.ListAuthorizedViewsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListAuthorizedViews(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_backups(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListBackupsRequest],
        bigtable_table_admin.ListBackupsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListBackups(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_schema_bundles(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSchemaBundlesRequest],
        bigtable_table_admin.ListSchemaBundlesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListSchemaBundles(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_snapshots(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSnapshotsRequest],
        bigtable_table_admin.ListSnapshotsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListSnapshots(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_tables(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListTablesRequest],
        bigtable_table_admin.ListTablesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListTables(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def modify_column_families(
        self,
    ) -> Callable[[bigtable_table_admin.ModifyColumnFamiliesRequest], table.Table]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ModifyColumnFamilies(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def restore_table(
        self,
    ) -> Callable[[bigtable_table_admin.RestoreTableRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._RestoreTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def snapshot_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.SnapshotTableRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SnapshotTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._TestIamPermissions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def undelete_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.UndeleteTableRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UndeleteTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateAuthorizedViewRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateAuthorizedView(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_backup(
        self,
    ) -> Callable[[bigtable_table_admin.UpdateBackupRequest], table.Backup]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateSchemaBundleRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateSchemaBundle(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_table(
        self,
    ) -> Callable[[bigtable_table_admin.UpdateTableRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateTable(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("BigtableTableAdminRestTransport",)

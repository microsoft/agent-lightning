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


from google.cloud.spanner_admin_database_v1.types import backup
from google.cloud.spanner_admin_database_v1.types import backup as gsad_backup
from google.cloud.spanner_admin_database_v1.types import backup_schedule
from google.cloud.spanner_admin_database_v1.types import (
    backup_schedule as gsad_backup_schedule,
)
from google.cloud.spanner_admin_database_v1.types import spanner_database_admin
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BaseDatabaseAdminRestTransport
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


class DatabaseAdminRestInterceptor:
    """Interceptor for DatabaseAdmin.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the DatabaseAdminRestTransport.

    .. code-block:: python
        class MyCustomDatabaseAdminInterceptor(DatabaseAdminRestInterceptor):
            def pre_add_split_points(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_add_split_points(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_copy_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_copy_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_backup_schedule(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_backup_schedule(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_database(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_database(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_backup_schedule(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_drop_database(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_backup_schedule(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_backup_schedule(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_database(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_database(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_database_ddl(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_database_ddl(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_internal_update_graph_operation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_internal_update_graph_operation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_backup_operations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_backup_operations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_backups(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_backups(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_backup_schedules(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_backup_schedules(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_database_operations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_database_operations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_database_roles(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_database_roles(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_databases(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_databases(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_restore_database(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_restore_database(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_set_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_set_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_test_iam_permissions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_test_iam_permissions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_backup(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_backup(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_backup_schedule(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_backup_schedule(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_database(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_database(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_database_ddl(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_database_ddl(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = DatabaseAdminRestTransport(interceptor=MyCustomDatabaseAdminInterceptor())
        client = DatabaseAdminClient(transport=transport)


    """

    def pre_add_split_points(
        self,
        request: spanner_database_admin.AddSplitPointsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.AddSplitPointsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for add_split_points

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_add_split_points(
        self, response: spanner_database_admin.AddSplitPointsResponse
    ) -> spanner_database_admin.AddSplitPointsResponse:
        """Post-rpc interceptor for add_split_points

        DEPRECATED. Please use the `post_add_split_points_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_add_split_points` interceptor runs
        before the `post_add_split_points_with_metadata` interceptor.
        """
        return response

    def post_add_split_points_with_metadata(
        self,
        response: spanner_database_admin.AddSplitPointsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.AddSplitPointsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for add_split_points

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_add_split_points_with_metadata`
        interceptor in new development instead of the `post_add_split_points` interceptor.
        When both interceptors are used, this `post_add_split_points_with_metadata` interceptor runs after the
        `post_add_split_points` interceptor. The (possibly modified) response returned by
        `post_add_split_points` will be passed to
        `post_add_split_points_with_metadata`.
        """
        return response, metadata

    def pre_copy_backup(
        self,
        request: backup.CopyBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup.CopyBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for copy_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_copy_backup(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for copy_backup

        DEPRECATED. Please use the `post_copy_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
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
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_copy_backup_with_metadata`
        interceptor in new development instead of the `post_copy_backup` interceptor.
        When both interceptors are used, this `post_copy_backup_with_metadata` interceptor runs after the
        `post_copy_backup` interceptor. The (possibly modified) response returned by
        `post_copy_backup` will be passed to
        `post_copy_backup_with_metadata`.
        """
        return response, metadata

    def pre_create_backup(
        self,
        request: gsad_backup.CreateBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup.CreateBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_create_backup(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_backup

        DEPRECATED. Please use the `post_create_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
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
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_create_backup_with_metadata`
        interceptor in new development instead of the `post_create_backup` interceptor.
        When both interceptors are used, this `post_create_backup_with_metadata` interceptor runs after the
        `post_create_backup` interceptor. The (possibly modified) response returned by
        `post_create_backup` will be passed to
        `post_create_backup_with_metadata`.
        """
        return response, metadata

    def pre_create_backup_schedule(
        self,
        request: gsad_backup_schedule.CreateBackupScheduleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup_schedule.CreateBackupScheduleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_backup_schedule

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_create_backup_schedule(
        self, response: gsad_backup_schedule.BackupSchedule
    ) -> gsad_backup_schedule.BackupSchedule:
        """Post-rpc interceptor for create_backup_schedule

        DEPRECATED. Please use the `post_create_backup_schedule_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_create_backup_schedule` interceptor runs
        before the `post_create_backup_schedule_with_metadata` interceptor.
        """
        return response

    def post_create_backup_schedule_with_metadata(
        self,
        response: gsad_backup_schedule.BackupSchedule,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup_schedule.BackupSchedule, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for create_backup_schedule

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_create_backup_schedule_with_metadata`
        interceptor in new development instead of the `post_create_backup_schedule` interceptor.
        When both interceptors are used, this `post_create_backup_schedule_with_metadata` interceptor runs after the
        `post_create_backup_schedule` interceptor. The (possibly modified) response returned by
        `post_create_backup_schedule` will be passed to
        `post_create_backup_schedule_with_metadata`.
        """
        return response, metadata

    def pre_create_database(
        self,
        request: spanner_database_admin.CreateDatabaseRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.CreateDatabaseRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_database

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_create_database(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_database

        DEPRECATED. Please use the `post_create_database_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_create_database` interceptor runs
        before the `post_create_database_with_metadata` interceptor.
        """
        return response

    def post_create_database_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_database

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_create_database_with_metadata`
        interceptor in new development instead of the `post_create_database` interceptor.
        When both interceptors are used, this `post_create_database_with_metadata` interceptor runs after the
        `post_create_database` interceptor. The (possibly modified) response returned by
        `post_create_database` will be passed to
        `post_create_database_with_metadata`.
        """
        return response, metadata

    def pre_delete_backup(
        self,
        request: backup.DeleteBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup.DeleteBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for delete_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def pre_delete_backup_schedule(
        self,
        request: backup_schedule.DeleteBackupScheduleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup_schedule.DeleteBackupScheduleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_backup_schedule

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def pre_drop_database(
        self,
        request: spanner_database_admin.DropDatabaseRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.DropDatabaseRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for drop_database

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def pre_get_backup(
        self,
        request: backup.GetBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup.GetBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_backup(self, response: backup.Backup) -> backup.Backup:
        """Post-rpc interceptor for get_backup

        DEPRECATED. Please use the `post_get_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_get_backup` interceptor runs
        before the `post_get_backup_with_metadata` interceptor.
        """
        return response

    def post_get_backup_with_metadata(
        self, response: backup.Backup, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[backup.Backup, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_get_backup_with_metadata`
        interceptor in new development instead of the `post_get_backup` interceptor.
        When both interceptors are used, this `post_get_backup_with_metadata` interceptor runs after the
        `post_get_backup` interceptor. The (possibly modified) response returned by
        `post_get_backup` will be passed to
        `post_get_backup_with_metadata`.
        """
        return response, metadata

    def pre_get_backup_schedule(
        self,
        request: backup_schedule.GetBackupScheduleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup_schedule.GetBackupScheduleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_backup_schedule

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_backup_schedule(
        self, response: backup_schedule.BackupSchedule
    ) -> backup_schedule.BackupSchedule:
        """Post-rpc interceptor for get_backup_schedule

        DEPRECATED. Please use the `post_get_backup_schedule_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_get_backup_schedule` interceptor runs
        before the `post_get_backup_schedule_with_metadata` interceptor.
        """
        return response

    def post_get_backup_schedule_with_metadata(
        self,
        response: backup_schedule.BackupSchedule,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup_schedule.BackupSchedule, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_backup_schedule

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_get_backup_schedule_with_metadata`
        interceptor in new development instead of the `post_get_backup_schedule` interceptor.
        When both interceptors are used, this `post_get_backup_schedule_with_metadata` interceptor runs after the
        `post_get_backup_schedule` interceptor. The (possibly modified) response returned by
        `post_get_backup_schedule` will be passed to
        `post_get_backup_schedule_with_metadata`.
        """
        return response, metadata

    def pre_get_database(
        self,
        request: spanner_database_admin.GetDatabaseRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.GetDatabaseRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_database

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_database(
        self, response: spanner_database_admin.Database
    ) -> spanner_database_admin.Database:
        """Post-rpc interceptor for get_database

        DEPRECATED. Please use the `post_get_database_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_get_database` interceptor runs
        before the `post_get_database_with_metadata` interceptor.
        """
        return response

    def post_get_database_with_metadata(
        self,
        response: spanner_database_admin.Database,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.Database, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for get_database

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_get_database_with_metadata`
        interceptor in new development instead of the `post_get_database` interceptor.
        When both interceptors are used, this `post_get_database_with_metadata` interceptor runs after the
        `post_get_database` interceptor. The (possibly modified) response returned by
        `post_get_database` will be passed to
        `post_get_database_with_metadata`.
        """
        return response, metadata

    def pre_get_database_ddl(
        self,
        request: spanner_database_admin.GetDatabaseDdlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.GetDatabaseDdlRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_database_ddl

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_database_ddl(
        self, response: spanner_database_admin.GetDatabaseDdlResponse
    ) -> spanner_database_admin.GetDatabaseDdlResponse:
        """Post-rpc interceptor for get_database_ddl

        DEPRECATED. Please use the `post_get_database_ddl_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_get_database_ddl` interceptor runs
        before the `post_get_database_ddl_with_metadata` interceptor.
        """
        return response

    def post_get_database_ddl_with_metadata(
        self,
        response: spanner_database_admin.GetDatabaseDdlResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.GetDatabaseDdlResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for get_database_ddl

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_get_database_ddl_with_metadata`
        interceptor in new development instead of the `post_get_database_ddl` interceptor.
        When both interceptors are used, this `post_get_database_ddl_with_metadata` interceptor runs after the
        `post_get_database_ddl` interceptor. The (possibly modified) response returned by
        `post_get_database_ddl` will be passed to
        `post_get_database_ddl_with_metadata`.
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
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        DEPRECATED. Please use the `post_get_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
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
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_get_iam_policy_with_metadata`
        interceptor in new development instead of the `post_get_iam_policy` interceptor.
        When both interceptors are used, this `post_get_iam_policy_with_metadata` interceptor runs after the
        `post_get_iam_policy` interceptor. The (possibly modified) response returned by
        `post_get_iam_policy` will be passed to
        `post_get_iam_policy_with_metadata`.
        """
        return response, metadata

    def pre_list_backup_operations(
        self,
        request: backup.ListBackupOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup.ListBackupOperationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_backup_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_backup_operations(
        self, response: backup.ListBackupOperationsResponse
    ) -> backup.ListBackupOperationsResponse:
        """Post-rpc interceptor for list_backup_operations

        DEPRECATED. Please use the `post_list_backup_operations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_backup_operations` interceptor runs
        before the `post_list_backup_operations_with_metadata` interceptor.
        """
        return response

    def post_list_backup_operations_with_metadata(
        self,
        response: backup.ListBackupOperationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup.ListBackupOperationsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_backup_operations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_backup_operations_with_metadata`
        interceptor in new development instead of the `post_list_backup_operations` interceptor.
        When both interceptors are used, this `post_list_backup_operations_with_metadata` interceptor runs after the
        `post_list_backup_operations` interceptor. The (possibly modified) response returned by
        `post_list_backup_operations` will be passed to
        `post_list_backup_operations_with_metadata`.
        """
        return response, metadata

    def pre_list_backups(
        self,
        request: backup.ListBackupsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup.ListBackupsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_backups

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_backups(
        self, response: backup.ListBackupsResponse
    ) -> backup.ListBackupsResponse:
        """Post-rpc interceptor for list_backups

        DEPRECATED. Please use the `post_list_backups_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_backups` interceptor runs
        before the `post_list_backups_with_metadata` interceptor.
        """
        return response

    def post_list_backups_with_metadata(
        self,
        response: backup.ListBackupsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[backup.ListBackupsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_backups

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_backups_with_metadata`
        interceptor in new development instead of the `post_list_backups` interceptor.
        When both interceptors are used, this `post_list_backups_with_metadata` interceptor runs after the
        `post_list_backups` interceptor. The (possibly modified) response returned by
        `post_list_backups` will be passed to
        `post_list_backups_with_metadata`.
        """
        return response, metadata

    def pre_list_backup_schedules(
        self,
        request: backup_schedule.ListBackupSchedulesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup_schedule.ListBackupSchedulesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_backup_schedules

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_backup_schedules(
        self, response: backup_schedule.ListBackupSchedulesResponse
    ) -> backup_schedule.ListBackupSchedulesResponse:
        """Post-rpc interceptor for list_backup_schedules

        DEPRECATED. Please use the `post_list_backup_schedules_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_backup_schedules` interceptor runs
        before the `post_list_backup_schedules_with_metadata` interceptor.
        """
        return response

    def post_list_backup_schedules_with_metadata(
        self,
        response: backup_schedule.ListBackupSchedulesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        backup_schedule.ListBackupSchedulesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_backup_schedules

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_backup_schedules_with_metadata`
        interceptor in new development instead of the `post_list_backup_schedules` interceptor.
        When both interceptors are used, this `post_list_backup_schedules_with_metadata` interceptor runs after the
        `post_list_backup_schedules` interceptor. The (possibly modified) response returned by
        `post_list_backup_schedules` will be passed to
        `post_list_backup_schedules_with_metadata`.
        """
        return response, metadata

    def pre_list_database_operations(
        self,
        request: spanner_database_admin.ListDatabaseOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabaseOperationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_database_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_database_operations(
        self, response: spanner_database_admin.ListDatabaseOperationsResponse
    ) -> spanner_database_admin.ListDatabaseOperationsResponse:
        """Post-rpc interceptor for list_database_operations

        DEPRECATED. Please use the `post_list_database_operations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_database_operations` interceptor runs
        before the `post_list_database_operations_with_metadata` interceptor.
        """
        return response

    def post_list_database_operations_with_metadata(
        self,
        response: spanner_database_admin.ListDatabaseOperationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabaseOperationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_database_operations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_database_operations_with_metadata`
        interceptor in new development instead of the `post_list_database_operations` interceptor.
        When both interceptors are used, this `post_list_database_operations_with_metadata` interceptor runs after the
        `post_list_database_operations` interceptor. The (possibly modified) response returned by
        `post_list_database_operations` will be passed to
        `post_list_database_operations_with_metadata`.
        """
        return response, metadata

    def pre_list_database_roles(
        self,
        request: spanner_database_admin.ListDatabaseRolesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabaseRolesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_database_roles

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_database_roles(
        self, response: spanner_database_admin.ListDatabaseRolesResponse
    ) -> spanner_database_admin.ListDatabaseRolesResponse:
        """Post-rpc interceptor for list_database_roles

        DEPRECATED. Please use the `post_list_database_roles_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_database_roles` interceptor runs
        before the `post_list_database_roles_with_metadata` interceptor.
        """
        return response

    def post_list_database_roles_with_metadata(
        self,
        response: spanner_database_admin.ListDatabaseRolesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabaseRolesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_database_roles

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_database_roles_with_metadata`
        interceptor in new development instead of the `post_list_database_roles` interceptor.
        When both interceptors are used, this `post_list_database_roles_with_metadata` interceptor runs after the
        `post_list_database_roles` interceptor. The (possibly modified) response returned by
        `post_list_database_roles` will be passed to
        `post_list_database_roles_with_metadata`.
        """
        return response, metadata

    def pre_list_databases(
        self,
        request: spanner_database_admin.ListDatabasesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabasesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_databases

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_databases(
        self, response: spanner_database_admin.ListDatabasesResponse
    ) -> spanner_database_admin.ListDatabasesResponse:
        """Post-rpc interceptor for list_databases

        DEPRECATED. Please use the `post_list_databases_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_list_databases` interceptor runs
        before the `post_list_databases_with_metadata` interceptor.
        """
        return response

    def post_list_databases_with_metadata(
        self,
        response: spanner_database_admin.ListDatabasesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.ListDatabasesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_databases

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_list_databases_with_metadata`
        interceptor in new development instead of the `post_list_databases` interceptor.
        When both interceptors are used, this `post_list_databases_with_metadata` interceptor runs after the
        `post_list_databases` interceptor. The (possibly modified) response returned by
        `post_list_databases` will be passed to
        `post_list_databases_with_metadata`.
        """
        return response, metadata

    def pre_restore_database(
        self,
        request: spanner_database_admin.RestoreDatabaseRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.RestoreDatabaseRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for restore_database

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_restore_database(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for restore_database

        DEPRECATED. Please use the `post_restore_database_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_restore_database` interceptor runs
        before the `post_restore_database_with_metadata` interceptor.
        """
        return response

    def post_restore_database_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for restore_database

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_restore_database_with_metadata`
        interceptor in new development instead of the `post_restore_database` interceptor.
        When both interceptors are used, this `post_restore_database_with_metadata` interceptor runs after the
        `post_restore_database` interceptor. The (possibly modified) response returned by
        `post_restore_database` will be passed to
        `post_restore_database_with_metadata`.
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
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        DEPRECATED. Please use the `post_set_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
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
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_set_iam_policy_with_metadata`
        interceptor in new development instead of the `post_set_iam_policy` interceptor.
        When both interceptors are used, this `post_set_iam_policy_with_metadata` interceptor runs after the
        `post_set_iam_policy` interceptor. The (possibly modified) response returned by
        `post_set_iam_policy` will be passed to
        `post_set_iam_policy_with_metadata`.
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
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        DEPRECATED. Please use the `post_test_iam_permissions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
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
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_test_iam_permissions_with_metadata`
        interceptor in new development instead of the `post_test_iam_permissions` interceptor.
        When both interceptors are used, this `post_test_iam_permissions_with_metadata` interceptor runs after the
        `post_test_iam_permissions` interceptor. The (possibly modified) response returned by
        `post_test_iam_permissions` will be passed to
        `post_test_iam_permissions_with_metadata`.
        """
        return response, metadata

    def pre_update_backup(
        self,
        request: gsad_backup.UpdateBackupRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup.UpdateBackupRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_backup

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_update_backup(self, response: gsad_backup.Backup) -> gsad_backup.Backup:
        """Post-rpc interceptor for update_backup

        DEPRECATED. Please use the `post_update_backup_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_update_backup` interceptor runs
        before the `post_update_backup_with_metadata` interceptor.
        """
        return response

    def post_update_backup_with_metadata(
        self,
        response: gsad_backup.Backup,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gsad_backup.Backup, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_backup

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_update_backup_with_metadata`
        interceptor in new development instead of the `post_update_backup` interceptor.
        When both interceptors are used, this `post_update_backup_with_metadata` interceptor runs after the
        `post_update_backup` interceptor. The (possibly modified) response returned by
        `post_update_backup` will be passed to
        `post_update_backup_with_metadata`.
        """
        return response, metadata

    def pre_update_backup_schedule(
        self,
        request: gsad_backup_schedule.UpdateBackupScheduleRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup_schedule.UpdateBackupScheduleRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_backup_schedule

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_update_backup_schedule(
        self, response: gsad_backup_schedule.BackupSchedule
    ) -> gsad_backup_schedule.BackupSchedule:
        """Post-rpc interceptor for update_backup_schedule

        DEPRECATED. Please use the `post_update_backup_schedule_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_update_backup_schedule` interceptor runs
        before the `post_update_backup_schedule_with_metadata` interceptor.
        """
        return response

    def post_update_backup_schedule_with_metadata(
        self,
        response: gsad_backup_schedule.BackupSchedule,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gsad_backup_schedule.BackupSchedule, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for update_backup_schedule

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_update_backup_schedule_with_metadata`
        interceptor in new development instead of the `post_update_backup_schedule` interceptor.
        When both interceptors are used, this `post_update_backup_schedule_with_metadata` interceptor runs after the
        `post_update_backup_schedule` interceptor. The (possibly modified) response returned by
        `post_update_backup_schedule` will be passed to
        `post_update_backup_schedule_with_metadata`.
        """
        return response, metadata

    def pre_update_database(
        self,
        request: spanner_database_admin.UpdateDatabaseRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.UpdateDatabaseRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_database

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_update_database(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_database

        DEPRECATED. Please use the `post_update_database_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_update_database` interceptor runs
        before the `post_update_database_with_metadata` interceptor.
        """
        return response

    def post_update_database_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_database

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_update_database_with_metadata`
        interceptor in new development instead of the `post_update_database` interceptor.
        When both interceptors are used, this `post_update_database_with_metadata` interceptor runs after the
        `post_update_database` interceptor. The (possibly modified) response returned by
        `post_update_database` will be passed to
        `post_update_database_with_metadata`.
        """
        return response, metadata

    def pre_update_database_ddl(
        self,
        request: spanner_database_admin.UpdateDatabaseDdlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_database_admin.UpdateDatabaseDdlRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_database_ddl

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_update_database_ddl(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_database_ddl

        DEPRECATED. Please use the `post_update_database_ddl_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code. This `post_update_database_ddl` interceptor runs
        before the `post_update_database_ddl_with_metadata` interceptor.
        """
        return response

    def post_update_database_ddl_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_database_ddl

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DatabaseAdmin server but before it is returned to user code.

        We recommend only using this `post_update_database_ddl_with_metadata`
        interceptor in new development instead of the `post_update_database_ddl` interceptor.
        When both interceptors are used, this `post_update_database_ddl_with_metadata` interceptor runs after the
        `post_update_database_ddl` interceptor. The (possibly modified) response returned by
        `post_update_database_ddl` will be passed to
        `post_update_database_ddl_with_metadata`.
        """
        return response, metadata

    def pre_cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.CancelOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code.
        """
        return response

    def pre_delete_operation(
        self,
        request: operations_pb2.DeleteOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.DeleteOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_delete_operation(self, response: None) -> None:
        """Post-rpc interceptor for delete_operation

        Override in a subclass to manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.ListOperationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DatabaseAdmin server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the DatabaseAdmin server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class DatabaseAdminRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: DatabaseAdminRestInterceptor


class DatabaseAdminRestTransport(_BaseDatabaseAdminRestTransport):
    """REST backend synchronous transport for DatabaseAdmin.

    Cloud Spanner Database Admin API

    The Cloud Spanner Database Admin API can be used to:

    - create, drop, and list databases
    - update the schema of pre-existing databases
    - create, delete, copy and list backups for a database
    - restore a database from an existing backup

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "spanner.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[DatabaseAdminRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'spanner.googleapis.com').
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
        self._interceptor = interceptor or DatabaseAdminRestInterceptor()
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
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}:cancel",
                    },
                ],
                "google.longrunning.Operations.DeleteOperation": [
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations}",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v1",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _AddSplitPoints(
        _BaseDatabaseAdminRestTransport._BaseAddSplitPoints, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.AddSplitPoints")

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
            request: spanner_database_admin.AddSplitPointsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.AddSplitPointsResponse:
            r"""Call the add split points method over HTTP.

            Args:
                request (~.spanner_database_admin.AddSplitPointsRequest):
                    The request object. The request for
                [AddSplitPoints][google.spanner.admin.database.v1.DatabaseAdmin.AddSplitPoints].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.AddSplitPointsResponse:
                    The response for
                [AddSplitPoints][google.spanner.admin.database.v1.DatabaseAdmin.AddSplitPoints].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseAddSplitPoints._get_http_options()
            )

            request, metadata = self._interceptor.pre_add_split_points(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseAddSplitPoints._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseAddSplitPoints._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseAddSplitPoints._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.AddSplitPoints",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "AddSplitPoints",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._AddSplitPoints._get_response(
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
            resp = spanner_database_admin.AddSplitPointsResponse()
            pb_resp = spanner_database_admin.AddSplitPointsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_add_split_points(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_add_split_points_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_database_admin.AddSplitPointsResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.add_split_points",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "AddSplitPoints",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CopyBackup(
        _BaseDatabaseAdminRestTransport._BaseCopyBackup, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.CopyBackup")

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
            request: backup.CopyBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the copy backup method over HTTP.

            Args:
                request (~.backup.CopyBackupRequest):
                    The request object. The request for
                [CopyBackup][google.spanner.admin.database.v1.DatabaseAdmin.CopyBackup].
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
                _BaseDatabaseAdminRestTransport._BaseCopyBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_copy_backup(request, metadata)
            transcoded_request = (
                _BaseDatabaseAdminRestTransport._BaseCopyBackup._get_transcoded_request(
                    http_options, request
                )
            )

            body = (
                _BaseDatabaseAdminRestTransport._BaseCopyBackup._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseDatabaseAdminRestTransport._BaseCopyBackup._get_query_params_json(
                    transcoded_request
                )
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.CopyBackup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CopyBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._CopyBackup._get_response(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.copy_backup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CopyBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateBackup(
        _BaseDatabaseAdminRestTransport._BaseCreateBackup, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.CreateBackup")

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
            request: gsad_backup.CreateBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create backup method over HTTP.

            Args:
                request (~.gsad_backup.CreateBackupRequest):
                    The request object. The request for
                [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup].
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
                _BaseDatabaseAdminRestTransport._BaseCreateBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_backup(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseCreateBackup._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseCreateBackup._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseCreateBackup._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.CreateBackup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._CreateBackup._get_response(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.create_backup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateBackupSchedule(
        _BaseDatabaseAdminRestTransport._BaseCreateBackupSchedule, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.CreateBackupSchedule")

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
            request: gsad_backup_schedule.CreateBackupScheduleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gsad_backup_schedule.BackupSchedule:
            r"""Call the create backup schedule method over HTTP.

            Args:
                request (~.gsad_backup_schedule.CreateBackupScheduleRequest):
                    The request object. The request for
                [CreateBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackupSchedule].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gsad_backup_schedule.BackupSchedule:
                    BackupSchedule expresses the
                automated backup creation specification
                for a Spanner database. Next ID: 10

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseCreateBackupSchedule._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_backup_schedule(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseCreateBackupSchedule._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseCreateBackupSchedule._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseCreateBackupSchedule._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.CreateBackupSchedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateBackupSchedule",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._CreateBackupSchedule._get_response(
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
            resp = gsad_backup_schedule.BackupSchedule()
            pb_resp = gsad_backup_schedule.BackupSchedule.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_backup_schedule(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_backup_schedule_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gsad_backup_schedule.BackupSchedule.to_json(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.create_backup_schedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateBackupSchedule",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateDatabase(
        _BaseDatabaseAdminRestTransport._BaseCreateDatabase, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.CreateDatabase")

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
            request: spanner_database_admin.CreateDatabaseRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create database method over HTTP.

            Args:
                request (~.spanner_database_admin.CreateDatabaseRequest):
                    The request object. The request for
                [CreateDatabase][google.spanner.admin.database.v1.DatabaseAdmin.CreateDatabase].
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
                _BaseDatabaseAdminRestTransport._BaseCreateDatabase._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_database(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseCreateDatabase._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseCreateDatabase._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseCreateDatabase._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.CreateDatabase",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateDatabase",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._CreateDatabase._get_response(
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

            resp = self._interceptor.post_create_database(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_database_with_metadata(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.create_database",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CreateDatabase",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteBackup(
        _BaseDatabaseAdminRestTransport._BaseDeleteBackup, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.DeleteBackup")

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
            request: backup.DeleteBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete backup method over HTTP.

            Args:
                request (~.backup.DeleteBackupRequest):
                    The request object. The request for
                [DeleteBackup][google.spanner.admin.database.v1.DatabaseAdmin.DeleteBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseDeleteBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_backup(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseDeleteBackup._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseDeleteBackup._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.DeleteBackup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "DeleteBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._DeleteBackup._get_response(
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

    class _DeleteBackupSchedule(
        _BaseDatabaseAdminRestTransport._BaseDeleteBackupSchedule, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.DeleteBackupSchedule")

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
            request: backup_schedule.DeleteBackupScheduleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete backup schedule method over HTTP.

            Args:
                request (~.backup_schedule.DeleteBackupScheduleRequest):
                    The request object. The request for
                [DeleteBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.DeleteBackupSchedule].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseDeleteBackupSchedule._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_backup_schedule(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseDeleteBackupSchedule._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseDeleteBackupSchedule._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.DeleteBackupSchedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "DeleteBackupSchedule",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._DeleteBackupSchedule._get_response(
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

    class _DropDatabase(
        _BaseDatabaseAdminRestTransport._BaseDropDatabase, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.DropDatabase")

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
            request: spanner_database_admin.DropDatabaseRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the drop database method over HTTP.

            Args:
                request (~.spanner_database_admin.DropDatabaseRequest):
                    The request object. The request for
                [DropDatabase][google.spanner.admin.database.v1.DatabaseAdmin.DropDatabase].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseDropDatabase._get_http_options()
            )

            request, metadata = self._interceptor.pre_drop_database(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseDropDatabase._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseDropDatabase._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.DropDatabase",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "DropDatabase",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._DropDatabase._get_response(
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

    class _GetBackup(
        _BaseDatabaseAdminRestTransport._BaseGetBackup, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetBackup")

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
            request: backup.GetBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> backup.Backup:
            r"""Call the get backup method over HTTP.

            Args:
                request (~.backup.GetBackupRequest):
                    The request object. The request for
                [GetBackup][google.spanner.admin.database.v1.DatabaseAdmin.GetBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.backup.Backup:
                    A backup of a Cloud Spanner database.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseGetBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_backup(request, metadata)
            transcoded_request = (
                _BaseDatabaseAdminRestTransport._BaseGetBackup._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseDatabaseAdminRestTransport._BaseGetBackup._get_query_params_json(
                    transcoded_request
                )
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetBackup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetBackup._get_response(
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
            resp = backup.Backup()
            pb_resp = backup.Backup.pb(resp)

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
                    response_payload = backup.Backup.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.get_backup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetBackupSchedule(
        _BaseDatabaseAdminRestTransport._BaseGetBackupSchedule, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetBackupSchedule")

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
            request: backup_schedule.GetBackupScheduleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> backup_schedule.BackupSchedule:
            r"""Call the get backup schedule method over HTTP.

            Args:
                request (~.backup_schedule.GetBackupScheduleRequest):
                    The request object. The request for
                [GetBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.GetBackupSchedule].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.backup_schedule.BackupSchedule:
                    BackupSchedule expresses the
                automated backup creation specification
                for a Spanner database. Next ID: 10

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseGetBackupSchedule._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_backup_schedule(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseGetBackupSchedule._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseGetBackupSchedule._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetBackupSchedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetBackupSchedule",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetBackupSchedule._get_response(
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
            resp = backup_schedule.BackupSchedule()
            pb_resp = backup_schedule.BackupSchedule.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_backup_schedule(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_backup_schedule_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = backup_schedule.BackupSchedule.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.get_backup_schedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetBackupSchedule",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetDatabase(
        _BaseDatabaseAdminRestTransport._BaseGetDatabase, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetDatabase")

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
            request: spanner_database_admin.GetDatabaseRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.Database:
            r"""Call the get database method over HTTP.

            Args:
                request (~.spanner_database_admin.GetDatabaseRequest):
                    The request object. The request for
                [GetDatabase][google.spanner.admin.database.v1.DatabaseAdmin.GetDatabase].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.Database:
                    A Cloud Spanner database.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseGetDatabase._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_database(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseGetDatabase._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseDatabaseAdminRestTransport._BaseGetDatabase._get_query_params_json(
                    transcoded_request
                )
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetDatabase",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetDatabase",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetDatabase._get_response(
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
            resp = spanner_database_admin.Database()
            pb_resp = spanner_database_admin.Database.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_database(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_database_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_database_admin.Database.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.get_database",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetDatabase",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetDatabaseDdl(
        _BaseDatabaseAdminRestTransport._BaseGetDatabaseDdl, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetDatabaseDdl")

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
            request: spanner_database_admin.GetDatabaseDdlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.GetDatabaseDdlResponse:
            r"""Call the get database ddl method over HTTP.

            Args:
                request (~.spanner_database_admin.GetDatabaseDdlRequest):
                    The request object. The request for
                [GetDatabaseDdl][google.spanner.admin.database.v1.DatabaseAdmin.GetDatabaseDdl].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.GetDatabaseDdlResponse:
                    The response for
                [GetDatabaseDdl][google.spanner.admin.database.v1.DatabaseAdmin.GetDatabaseDdl].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseGetDatabaseDdl._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_database_ddl(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseGetDatabaseDdl._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseGetDatabaseDdl._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetDatabaseDdl",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetDatabaseDdl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetDatabaseDdl._get_response(
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
            resp = spanner_database_admin.GetDatabaseDdlResponse()
            pb_resp = spanner_database_admin.GetDatabaseDdlResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_database_ddl(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_database_ddl_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_database_admin.GetDatabaseDdlResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.get_database_ddl",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetDatabaseDdl",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetIamPolicy(
        _BaseDatabaseAdminRestTransport._BaseGetIamPolicy, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetIamPolicy")

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
                _BaseDatabaseAdminRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_iam_policy(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseGetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseGetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetIamPolicy._get_response(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.get_iam_policy",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _InternalUpdateGraphOperation(
        _BaseDatabaseAdminRestTransport._BaseInternalUpdateGraphOperation,
        DatabaseAdminRestStub,
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.InternalUpdateGraphOperation")

        def __call__(
            self,
            request: spanner_database_admin.InternalUpdateGraphOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.InternalUpdateGraphOperationResponse:
            raise NotImplementedError(
                "Method InternalUpdateGraphOperation is not available over REST transport"
            )

    class _ListBackupOperations(
        _BaseDatabaseAdminRestTransport._BaseListBackupOperations, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListBackupOperations")

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
            request: backup.ListBackupOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> backup.ListBackupOperationsResponse:
            r"""Call the list backup operations method over HTTP.

            Args:
                request (~.backup.ListBackupOperationsRequest):
                    The request object. The request for
                [ListBackupOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupOperations].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.backup.ListBackupOperationsResponse:
                    The response for
                [ListBackupOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupOperations].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListBackupOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_backup_operations(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListBackupOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListBackupOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListBackupOperations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackupOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListBackupOperations._get_response(
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
            resp = backup.ListBackupOperationsResponse()
            pb_resp = backup.ListBackupOperationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_backup_operations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_backup_operations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = backup.ListBackupOperationsResponse.to_json(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_backup_operations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackupOperations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListBackups(
        _BaseDatabaseAdminRestTransport._BaseListBackups, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListBackups")

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
            request: backup.ListBackupsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> backup.ListBackupsResponse:
            r"""Call the list backups method over HTTP.

            Args:
                request (~.backup.ListBackupsRequest):
                    The request object. The request for
                [ListBackups][google.spanner.admin.database.v1.DatabaseAdmin.ListBackups].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.backup.ListBackupsResponse:
                    The response for
                [ListBackups][google.spanner.admin.database.v1.DatabaseAdmin.ListBackups].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListBackups._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_backups(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListBackups._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseDatabaseAdminRestTransport._BaseListBackups._get_query_params_json(
                    transcoded_request
                )
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListBackups",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackups",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListBackups._get_response(
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
            resp = backup.ListBackupsResponse()
            pb_resp = backup.ListBackupsResponse.pb(resp)

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
                    response_payload = backup.ListBackupsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_backups",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackups",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListBackupSchedules(
        _BaseDatabaseAdminRestTransport._BaseListBackupSchedules, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListBackupSchedules")

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
            request: backup_schedule.ListBackupSchedulesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> backup_schedule.ListBackupSchedulesResponse:
            r"""Call the list backup schedules method over HTTP.

            Args:
                request (~.backup_schedule.ListBackupSchedulesRequest):
                    The request object. The request for
                [ListBackupSchedules][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupSchedules].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.backup_schedule.ListBackupSchedulesResponse:
                    The response for
                [ListBackupSchedules][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupSchedules].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListBackupSchedules._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_backup_schedules(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListBackupSchedules._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListBackupSchedules._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListBackupSchedules",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackupSchedules",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListBackupSchedules._get_response(
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
            resp = backup_schedule.ListBackupSchedulesResponse()
            pb_resp = backup_schedule.ListBackupSchedulesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_backup_schedules(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_backup_schedules_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        backup_schedule.ListBackupSchedulesResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_backup_schedules",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListBackupSchedules",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListDatabaseOperations(
        _BaseDatabaseAdminRestTransport._BaseListDatabaseOperations,
        DatabaseAdminRestStub,
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListDatabaseOperations")

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
            request: spanner_database_admin.ListDatabaseOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.ListDatabaseOperationsResponse:
            r"""Call the list database operations method over HTTP.

            Args:
                request (~.spanner_database_admin.ListDatabaseOperationsRequest):
                    The request object. The request for
                [ListDatabaseOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabaseOperations].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.ListDatabaseOperationsResponse:
                    The response for
                [ListDatabaseOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabaseOperations].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListDatabaseOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_database_operations(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListDatabaseOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListDatabaseOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListDatabaseOperations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabaseOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListDatabaseOperations._get_response(
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
            resp = spanner_database_admin.ListDatabaseOperationsResponse()
            pb_resp = spanner_database_admin.ListDatabaseOperationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_database_operations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_database_operations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_database_admin.ListDatabaseOperationsResponse.to_json(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_database_operations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabaseOperations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListDatabaseRoles(
        _BaseDatabaseAdminRestTransport._BaseListDatabaseRoles, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListDatabaseRoles")

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
            request: spanner_database_admin.ListDatabaseRolesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.ListDatabaseRolesResponse:
            r"""Call the list database roles method over HTTP.

            Args:
                request (~.spanner_database_admin.ListDatabaseRolesRequest):
                    The request object. The request for
                [ListDatabaseRoles][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabaseRoles].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.ListDatabaseRolesResponse:
                    The response for
                [ListDatabaseRoles][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabaseRoles].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListDatabaseRoles._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_database_roles(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListDatabaseRoles._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListDatabaseRoles._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListDatabaseRoles",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabaseRoles",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListDatabaseRoles._get_response(
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
            resp = spanner_database_admin.ListDatabaseRolesResponse()
            pb_resp = spanner_database_admin.ListDatabaseRolesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_database_roles(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_database_roles_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_database_admin.ListDatabaseRolesResponse.to_json(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_database_roles",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabaseRoles",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListDatabases(
        _BaseDatabaseAdminRestTransport._BaseListDatabases, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListDatabases")

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
            request: spanner_database_admin.ListDatabasesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_database_admin.ListDatabasesResponse:
            r"""Call the list databases method over HTTP.

            Args:
                request (~.spanner_database_admin.ListDatabasesRequest):
                    The request object. The request for
                [ListDatabases][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabases].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_database_admin.ListDatabasesResponse:
                    The response for
                [ListDatabases][google.spanner.admin.database.v1.DatabaseAdmin.ListDatabases].

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListDatabases._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_databases(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListDatabases._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListDatabases._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListDatabases",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabases",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListDatabases._get_response(
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
            resp = spanner_database_admin.ListDatabasesResponse()
            pb_resp = spanner_database_admin.ListDatabasesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_databases(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_databases_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_database_admin.ListDatabasesResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.list_databases",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListDatabases",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _RestoreDatabase(
        _BaseDatabaseAdminRestTransport._BaseRestoreDatabase, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.RestoreDatabase")

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
            request: spanner_database_admin.RestoreDatabaseRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the restore database method over HTTP.

            Args:
                request (~.spanner_database_admin.RestoreDatabaseRequest):
                    The request object. The request for
                [RestoreDatabase][google.spanner.admin.database.v1.DatabaseAdmin.RestoreDatabase].
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
                _BaseDatabaseAdminRestTransport._BaseRestoreDatabase._get_http_options()
            )

            request, metadata = self._interceptor.pre_restore_database(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseRestoreDatabase._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseRestoreDatabase._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseRestoreDatabase._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.RestoreDatabase",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "RestoreDatabase",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._RestoreDatabase._get_response(
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

            resp = self._interceptor.post_restore_database(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_restore_database_with_metadata(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.restore_database",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "RestoreDatabase",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SetIamPolicy(
        _BaseDatabaseAdminRestTransport._BaseSetIamPolicy, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.SetIamPolicy")

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
                _BaseDatabaseAdminRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_set_iam_policy(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseSetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseSetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._SetIamPolicy._get_response(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.set_iam_policy",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "SetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _TestIamPermissions(
        _BaseDatabaseAdminRestTransport._BaseTestIamPermissions, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.TestIamPermissions")

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
                _BaseDatabaseAdminRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseTestIamPermissions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseTestIamPermissions._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._TestIamPermissions._get_response(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.test_iam_permissions",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "TestIamPermissions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateBackup(
        _BaseDatabaseAdminRestTransport._BaseUpdateBackup, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.UpdateBackup")

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
            request: gsad_backup.UpdateBackupRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gsad_backup.Backup:
            r"""Call the update backup method over HTTP.

            Args:
                request (~.gsad_backup.UpdateBackupRequest):
                    The request object. The request for
                [UpdateBackup][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackup].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gsad_backup.Backup:
                    A backup of a Cloud Spanner database.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseUpdateBackup._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_backup(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseUpdateBackup._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseUpdateBackup._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseUpdateBackup._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.UpdateBackup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateBackup",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._UpdateBackup._get_response(
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
            resp = gsad_backup.Backup()
            pb_resp = gsad_backup.Backup.pb(resp)

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
                    response_payload = gsad_backup.Backup.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.update_backup",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateBackup",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateBackupSchedule(
        _BaseDatabaseAdminRestTransport._BaseUpdateBackupSchedule, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.UpdateBackupSchedule")

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
            request: gsad_backup_schedule.UpdateBackupScheduleRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gsad_backup_schedule.BackupSchedule:
            r"""Call the update backup schedule method over HTTP.

            Args:
                request (~.gsad_backup_schedule.UpdateBackupScheduleRequest):
                    The request object. The request for
                [UpdateBackupScheduleRequest][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackupSchedule].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gsad_backup_schedule.BackupSchedule:
                    BackupSchedule expresses the
                automated backup creation specification
                for a Spanner database. Next ID: 10

            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseUpdateBackupSchedule._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_backup_schedule(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseUpdateBackupSchedule._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseUpdateBackupSchedule._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseUpdateBackupSchedule._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.UpdateBackupSchedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateBackupSchedule",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._UpdateBackupSchedule._get_response(
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
            resp = gsad_backup_schedule.BackupSchedule()
            pb_resp = gsad_backup_schedule.BackupSchedule.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_backup_schedule(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_backup_schedule_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gsad_backup_schedule.BackupSchedule.to_json(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.update_backup_schedule",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateBackupSchedule",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateDatabase(
        _BaseDatabaseAdminRestTransport._BaseUpdateDatabase, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.UpdateDatabase")

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
            request: spanner_database_admin.UpdateDatabaseRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update database method over HTTP.

            Args:
                request (~.spanner_database_admin.UpdateDatabaseRequest):
                    The request object. The request for
                [UpdateDatabase][google.spanner.admin.database.v1.DatabaseAdmin.UpdateDatabase].
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
                _BaseDatabaseAdminRestTransport._BaseUpdateDatabase._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_database(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseUpdateDatabase._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseUpdateDatabase._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseUpdateDatabase._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.UpdateDatabase",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateDatabase",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._UpdateDatabase._get_response(
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

            resp = self._interceptor.post_update_database(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_database_with_metadata(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.update_database",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateDatabase",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateDatabaseDdl(
        _BaseDatabaseAdminRestTransport._BaseUpdateDatabaseDdl, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.UpdateDatabaseDdl")

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
            request: spanner_database_admin.UpdateDatabaseDdlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update database ddl method over HTTP.

            Args:
                request (~.spanner_database_admin.UpdateDatabaseDdlRequest):
                    The request object. Enqueues the given DDL statements to be applied, in
                order but not necessarily all at once, to the database
                schema at some point (or points) in the future. The
                server checks that the statements are executable
                (syntactically valid, name tables that exist, etc.)
                before enqueueing them, but they may still fail upon
                later execution (e.g., if a statement from another batch
                of statements is applied first and it conflicts in some
                way, or if there is some data-related problem like a
                ``NULL`` value in a column to which ``NOT NULL`` would
                be added). If a statement fails, all subsequent
                statements in the batch are automatically cancelled.

                Each batch of statements is assigned a name which can be
                used with the
                [Operations][google.longrunning.Operations] API to
                monitor progress. See the
                [operation_id][google.spanner.admin.database.v1.UpdateDatabaseDdlRequest.operation_id]
                field for more details.
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
                _BaseDatabaseAdminRestTransport._BaseUpdateDatabaseDdl._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_database_ddl(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseUpdateDatabaseDdl._get_transcoded_request(
                http_options, request
            )

            body = _BaseDatabaseAdminRestTransport._BaseUpdateDatabaseDdl._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseUpdateDatabaseDdl._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.UpdateDatabaseDdl",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateDatabaseDdl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._UpdateDatabaseDdl._get_response(
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

            resp = self._interceptor.post_update_database_ddl(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_database_ddl_with_metadata(
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminClient.update_database_ddl",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "UpdateDatabaseDdl",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def add_split_points(
        self,
    ) -> Callable[
        [spanner_database_admin.AddSplitPointsRequest],
        spanner_database_admin.AddSplitPointsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._AddSplitPoints(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def copy_backup(
        self,
    ) -> Callable[[backup.CopyBackupRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CopyBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_backup(
        self,
    ) -> Callable[[gsad_backup.CreateBackupRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_backup_schedule(
        self,
    ) -> Callable[
        [gsad_backup_schedule.CreateBackupScheduleRequest],
        gsad_backup_schedule.BackupSchedule,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateBackupSchedule(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_database(
        self,
    ) -> Callable[
        [spanner_database_admin.CreateDatabaseRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateDatabase(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_backup(self) -> Callable[[backup.DeleteBackupRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_backup_schedule(
        self,
    ) -> Callable[[backup_schedule.DeleteBackupScheduleRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteBackupSchedule(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def drop_database(
        self,
    ) -> Callable[[spanner_database_admin.DropDatabaseRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DropDatabase(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_backup(self) -> Callable[[backup.GetBackupRequest], backup.Backup]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_backup_schedule(
        self,
    ) -> Callable[
        [backup_schedule.GetBackupScheduleRequest], backup_schedule.BackupSchedule
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetBackupSchedule(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_database(
        self,
    ) -> Callable[
        [spanner_database_admin.GetDatabaseRequest], spanner_database_admin.Database
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDatabase(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_database_ddl(
        self,
    ) -> Callable[
        [spanner_database_admin.GetDatabaseDdlRequest],
        spanner_database_admin.GetDatabaseDdlResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDatabaseDdl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def internal_update_graph_operation(
        self,
    ) -> Callable[
        [spanner_database_admin.InternalUpdateGraphOperationRequest],
        spanner_database_admin.InternalUpdateGraphOperationResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._InternalUpdateGraphOperation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_backup_operations(
        self,
    ) -> Callable[
        [backup.ListBackupOperationsRequest], backup.ListBackupOperationsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListBackupOperations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_backups(
        self,
    ) -> Callable[[backup.ListBackupsRequest], backup.ListBackupsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListBackups(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_backup_schedules(
        self,
    ) -> Callable[
        [backup_schedule.ListBackupSchedulesRequest],
        backup_schedule.ListBackupSchedulesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListBackupSchedules(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_database_operations(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabaseOperationsRequest],
        spanner_database_admin.ListDatabaseOperationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDatabaseOperations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_database_roles(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabaseRolesRequest],
        spanner_database_admin.ListDatabaseRolesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDatabaseRoles(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_databases(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabasesRequest],
        spanner_database_admin.ListDatabasesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDatabases(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def restore_database(
        self,
    ) -> Callable[
        [spanner_database_admin.RestoreDatabaseRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._RestoreDatabase(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

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
    def update_backup(
        self,
    ) -> Callable[[gsad_backup.UpdateBackupRequest], gsad_backup.Backup]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateBackup(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_backup_schedule(
        self,
    ) -> Callable[
        [gsad_backup_schedule.UpdateBackupScheduleRequest],
        gsad_backup_schedule.BackupSchedule,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateBackupSchedule(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_database(
        self,
    ) -> Callable[
        [spanner_database_admin.UpdateDatabaseRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDatabase(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_database_ddl(
        self,
    ) -> Callable[
        [spanner_database_admin.UpdateDatabaseDdlRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDatabaseDdl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseDatabaseAdminRestTransport._BaseCancelOperation, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.CancelOperation")

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
            request: operations_pb2.CancelOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> None:
            r"""Call the cancel operation method over HTTP.

            Args:
                request (operations_pb2.CancelOperationRequest):
                    The request object for CancelOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.CancelOperation",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._CancelOperation._get_response(
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

            return self._interceptor.post_cancel_operation(None)

    @property
    def delete_operation(self):
        return self._DeleteOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _DeleteOperation(
        _BaseDatabaseAdminRestTransport._BaseDeleteOperation, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.DeleteOperation")

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
            request: operations_pb2.DeleteOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> None:
            r"""Call the delete operation method over HTTP.

            Args:
                request (operations_pb2.DeleteOperationRequest):
                    The request object for DeleteOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseDeleteOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_operation(
                request, metadata
            )
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseDeleteOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseDeleteOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.DeleteOperation",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "DeleteOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._DeleteOperation._get_response(
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

            return self._interceptor.post_delete_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseDatabaseAdminRestTransport._BaseGetOperation, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.GetOperation")

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
            request: operations_pb2.GetOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the get operation method over HTTP.

            Args:
                request (operations_pb2.GetOperationRequest):
                    The request object for GetOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.Operation: Response from GetOperation method.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.GetOperation",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._GetOperation._get_response(
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

            content = response.content.decode("utf-8")
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_operation(resp)
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "GetOperation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def list_operations(self):
        return self._ListOperations(self._session, self._host, self._interceptor)  # type: ignore

    class _ListOperations(
        _BaseDatabaseAdminRestTransport._BaseListOperations, DatabaseAdminRestStub
    ):
        def __hash__(self):
            return hash("DatabaseAdminRestTransport.ListOperations")

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
            request: operations_pb2.ListOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.ListOperationsResponse:
            r"""Call the list operations method over HTTP.

            Args:
                request (operations_pb2.ListOperationsRequest):
                    The request object for ListOperations method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.ListOperationsResponse: Response from ListOperations method.
            """

            http_options = (
                _BaseDatabaseAdminRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseDatabaseAdminRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDatabaseAdminRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.database_v1.DatabaseAdminClient.ListOperations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DatabaseAdminRestTransport._ListOperations._get_response(
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

            content = response.content.decode("utf-8")
            resp = operations_pb2.ListOperationsResponse()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_list_operations(resp)
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
                    "Received response for google.spanner.admin.database_v1.DatabaseAdminAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.spanner.admin.database.v1.DatabaseAdmin",
                        "rpcName": "ListOperations",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("DatabaseAdminRestTransport",)

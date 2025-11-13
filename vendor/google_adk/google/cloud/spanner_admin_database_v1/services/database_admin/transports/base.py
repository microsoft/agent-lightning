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
import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

from google.cloud.spanner_admin_database_v1 import gapic_version as package_version

import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.spanner_admin_database_v1.types import backup
from google.cloud.spanner_admin_database_v1.types import backup as gsad_backup
from google.cloud.spanner_admin_database_v1.types import backup_schedule
from google.cloud.spanner_admin_database_v1.types import (
    backup_schedule as gsad_backup_schedule,
)
from google.cloud.spanner_admin_database_v1.types import spanner_database_admin
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class DatabaseAdminTransport(abc.ABC):
    """Abstract transport class for DatabaseAdmin."""

    AUTH_SCOPES = (
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/spanner.admin",
    )

    DEFAULT_HOST: str = "spanner.googleapis.com"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        api_audience: Optional[str] = None,
        **kwargs,
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
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes
        if not hasattr(self, "_ignore_credentials"):
            self._ignore_credentials: bool = False

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )
        elif credentials is None and not self._ignore_credentials:
            credentials, _ = google.auth.default(
                **scopes_kwargs, quota_project_id=quota_project_id
            )
            # Don't apply audience if the credentials file passed from user.
            if hasattr(credentials, "with_gdch_audience"):
                credentials = credentials.with_gdch_audience(
                    api_audience if api_audience else host
                )

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ":" not in host:
            host += ":443"
        self._host = host

    @property
    def host(self):
        return self._host

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.list_databases: gapic_v1.method.wrap_method(
                self.list_databases,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.create_database: gapic_v1.method.wrap_method(
                self.create_database,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.get_database: gapic_v1.method.wrap_method(
                self.get_database,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.update_database: gapic_v1.method.wrap_method(
                self.update_database,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.update_database_ddl: gapic_v1.method.wrap_method(
                self.update_database_ddl,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.drop_database: gapic_v1.method.wrap_method(
                self.drop_database,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.get_database_ddl: gapic_v1.method.wrap_method(
                self.get_database_ddl,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.set_iam_policy: gapic_v1.method.wrap_method(
                self.set_iam_policy,
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.get_iam_policy: gapic_v1.method.wrap_method(
                self.get_iam_policy,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=30.0,
                ),
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.test_iam_permissions: gapic_v1.method.wrap_method(
                self.test_iam_permissions,
                default_timeout=30.0,
                client_info=client_info,
            ),
            self.create_backup: gapic_v1.method.wrap_method(
                self.create_backup,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.copy_backup: gapic_v1.method.wrap_method(
                self.copy_backup,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.get_backup: gapic_v1.method.wrap_method(
                self.get_backup,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.update_backup: gapic_v1.method.wrap_method(
                self.update_backup,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.delete_backup: gapic_v1.method.wrap_method(
                self.delete_backup,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_backups: gapic_v1.method.wrap_method(
                self.list_backups,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.restore_database: gapic_v1.method.wrap_method(
                self.restore_database,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_database_operations: gapic_v1.method.wrap_method(
                self.list_database_operations,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_backup_operations: gapic_v1.method.wrap_method(
                self.list_backup_operations,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_database_roles: gapic_v1.method.wrap_method(
                self.list_database_roles,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.add_split_points: gapic_v1.method.wrap_method(
                self.add_split_points,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.create_backup_schedule: gapic_v1.method.wrap_method(
                self.create_backup_schedule,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.get_backup_schedule: gapic_v1.method.wrap_method(
                self.get_backup_schedule,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.update_backup_schedule: gapic_v1.method.wrap_method(
                self.update_backup_schedule,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.delete_backup_schedule: gapic_v1.method.wrap_method(
                self.delete_backup_schedule,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.list_backup_schedules: gapic_v1.method.wrap_method(
                self.list_backup_schedules,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=32.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.internal_update_graph_operation: gapic_v1.method.wrap_method(
                self.internal_update_graph_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.cancel_operation: gapic_v1.method.wrap_method(
                self.cancel_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_operation: gapic_v1.method.wrap_method(
                self.delete_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_operation: gapic_v1.method.wrap_method(
                self.get_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_operations: gapic_v1.method.wrap_method(
                self.list_operations,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def operations_client(self):
        """Return the client designed to process long-running operations."""
        raise NotImplementedError()

    @property
    def list_databases(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabasesRequest],
        Union[
            spanner_database_admin.ListDatabasesResponse,
            Awaitable[spanner_database_admin.ListDatabasesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_database(
        self,
    ) -> Callable[
        [spanner_database_admin.CreateDatabaseRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_database(
        self,
    ) -> Callable[
        [spanner_database_admin.GetDatabaseRequest],
        Union[
            spanner_database_admin.Database, Awaitable[spanner_database_admin.Database]
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_database(
        self,
    ) -> Callable[
        [spanner_database_admin.UpdateDatabaseRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def update_database_ddl(
        self,
    ) -> Callable[
        [spanner_database_admin.UpdateDatabaseDdlRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def drop_database(
        self,
    ) -> Callable[
        [spanner_database_admin.DropDatabaseRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def get_database_ddl(
        self,
    ) -> Callable[
        [spanner_database_admin.GetDatabaseDdlRequest],
        Union[
            spanner_database_admin.GetDatabaseDdlResponse,
            Awaitable[spanner_database_admin.GetDatabaseDdlResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def set_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.SetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def get_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.GetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        Union[
            iam_policy_pb2.TestIamPermissionsResponse,
            Awaitable[iam_policy_pb2.TestIamPermissionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_backup(
        self,
    ) -> Callable[
        [gsad_backup.CreateBackupRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def copy_backup(
        self,
    ) -> Callable[
        [backup.CopyBackupRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_backup(
        self,
    ) -> Callable[
        [backup.GetBackupRequest], Union[backup.Backup, Awaitable[backup.Backup]]
    ]:
        raise NotImplementedError()

    @property
    def update_backup(
        self,
    ) -> Callable[
        [gsad_backup.UpdateBackupRequest],
        Union[gsad_backup.Backup, Awaitable[gsad_backup.Backup]],
    ]:
        raise NotImplementedError()

    @property
    def delete_backup(
        self,
    ) -> Callable[
        [backup.DeleteBackupRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]
    ]:
        raise NotImplementedError()

    @property
    def list_backups(
        self,
    ) -> Callable[
        [backup.ListBackupsRequest],
        Union[backup.ListBackupsResponse, Awaitable[backup.ListBackupsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def restore_database(
        self,
    ) -> Callable[
        [spanner_database_admin.RestoreDatabaseRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def list_database_operations(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabaseOperationsRequest],
        Union[
            spanner_database_admin.ListDatabaseOperationsResponse,
            Awaitable[spanner_database_admin.ListDatabaseOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_backup_operations(
        self,
    ) -> Callable[
        [backup.ListBackupOperationsRequest],
        Union[
            backup.ListBackupOperationsResponse,
            Awaitable[backup.ListBackupOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_database_roles(
        self,
    ) -> Callable[
        [spanner_database_admin.ListDatabaseRolesRequest],
        Union[
            spanner_database_admin.ListDatabaseRolesResponse,
            Awaitable[spanner_database_admin.ListDatabaseRolesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def add_split_points(
        self,
    ) -> Callable[
        [spanner_database_admin.AddSplitPointsRequest],
        Union[
            spanner_database_admin.AddSplitPointsResponse,
            Awaitable[spanner_database_admin.AddSplitPointsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_backup_schedule(
        self,
    ) -> Callable[
        [gsad_backup_schedule.CreateBackupScheduleRequest],
        Union[
            gsad_backup_schedule.BackupSchedule,
            Awaitable[gsad_backup_schedule.BackupSchedule],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_backup_schedule(
        self,
    ) -> Callable[
        [backup_schedule.GetBackupScheduleRequest],
        Union[
            backup_schedule.BackupSchedule, Awaitable[backup_schedule.BackupSchedule]
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_backup_schedule(
        self,
    ) -> Callable[
        [gsad_backup_schedule.UpdateBackupScheduleRequest],
        Union[
            gsad_backup_schedule.BackupSchedule,
            Awaitable[gsad_backup_schedule.BackupSchedule],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_backup_schedule(
        self,
    ) -> Callable[
        [backup_schedule.DeleteBackupScheduleRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_backup_schedules(
        self,
    ) -> Callable[
        [backup_schedule.ListBackupSchedulesRequest],
        Union[
            backup_schedule.ListBackupSchedulesResponse,
            Awaitable[backup_schedule.ListBackupSchedulesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def internal_update_graph_operation(
        self,
    ) -> Callable[
        [spanner_database_admin.InternalUpdateGraphOperationRequest],
        Union[
            spanner_database_admin.InternalUpdateGraphOperationResponse,
            Awaitable[spanner_database_admin.InternalUpdateGraphOperationResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Union[
            operations_pb2.ListOperationsResponse,
            Awaitable[operations_pb2.ListOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("DatabaseAdminTransport",)

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

from google.cloud.bigtable_admin_v2 import gapic_version as package_version

import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.bigtable_admin_v2.types import bigtable_table_admin
from google.cloud.bigtable_admin_v2.types import table
from google.cloud.bigtable_admin_v2.types import table as gba_table
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class BigtableTableAdminTransport(abc.ABC):
    """Abstract transport class for BigtableTableAdmin."""

    AUTH_SCOPES = (
        "https://www.googleapis.com/auth/bigtable.admin",
        "https://www.googleapis.com/auth/bigtable.admin.table",
        "https://www.googleapis.com/auth/cloud-bigtable.admin",
        "https://www.googleapis.com/auth/cloud-bigtable.admin.table",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/cloud-platform.read-only",
    )

    DEFAULT_HOST: str = "bigtableadmin.googleapis.com"

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
                 The hostname to connect to (default: 'bigtableadmin.googleapis.com').
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
            self.create_table: gapic_v1.method.wrap_method(
                self.create_table,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.create_table_from_snapshot: gapic_v1.method.wrap_method(
                self.create_table_from_snapshot,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_tables: gapic_v1.method.wrap_method(
                self.list_tables,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_table: gapic_v1.method.wrap_method(
                self.get_table,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_table: gapic_v1.method.wrap_method(
                self.update_table,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_table: gapic_v1.method.wrap_method(
                self.delete_table,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.undelete_table: gapic_v1.method.wrap_method(
                self.undelete_table,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_authorized_view: gapic_v1.method.wrap_method(
                self.create_authorized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_authorized_views: gapic_v1.method.wrap_method(
                self.list_authorized_views,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_authorized_view: gapic_v1.method.wrap_method(
                self.get_authorized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_authorized_view: gapic_v1.method.wrap_method(
                self.update_authorized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_authorized_view: gapic_v1.method.wrap_method(
                self.delete_authorized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.modify_column_families: gapic_v1.method.wrap_method(
                self.modify_column_families,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.drop_row_range: gapic_v1.method.wrap_method(
                self.drop_row_range,
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.generate_consistency_token: gapic_v1.method.wrap_method(
                self.generate_consistency_token,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.check_consistency: gapic_v1.method.wrap_method(
                self.check_consistency,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=3600.0,
                ),
                default_timeout=3600.0,
                client_info=client_info,
            ),
            self.snapshot_table: gapic_v1.method.wrap_method(
                self.snapshot_table,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_snapshot: gapic_v1.method.wrap_method(
                self.get_snapshot,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.list_snapshots: gapic_v1.method.wrap_method(
                self.list_snapshots,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.delete_snapshot: gapic_v1.method.wrap_method(
                self.delete_snapshot,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.create_backup: gapic_v1.method.wrap_method(
                self.create_backup,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_backup: gapic_v1.method.wrap_method(
                self.get_backup,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_backup: gapic_v1.method.wrap_method(
                self.update_backup,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.delete_backup: gapic_v1.method.wrap_method(
                self.delete_backup,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.list_backups: gapic_v1.method.wrap_method(
                self.list_backups,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.restore_table: gapic_v1.method.wrap_method(
                self.restore_table,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.copy_backup: gapic_v1.method.wrap_method(
                self.copy_backup,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_iam_policy: gapic_v1.method.wrap_method(
                self.get_iam_policy,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.set_iam_policy: gapic_v1.method.wrap_method(
                self.set_iam_policy,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.test_iam_permissions: gapic_v1.method.wrap_method(
                self.test_iam_permissions,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_schema_bundle: gapic_v1.method.wrap_method(
                self.create_schema_bundle,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_schema_bundle: gapic_v1.method.wrap_method(
                self.update_schema_bundle,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_schema_bundle: gapic_v1.method.wrap_method(
                self.get_schema_bundle,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_schema_bundles: gapic_v1.method.wrap_method(
                self.list_schema_bundles,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_schema_bundle: gapic_v1.method.wrap_method(
                self.delete_schema_bundle,
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
    def create_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateTableRequest],
        Union[gba_table.Table, Awaitable[gba_table.Table]],
    ]:
        raise NotImplementedError()

    @property
    def create_table_from_snapshot(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateTableFromSnapshotRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def list_tables(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListTablesRequest],
        Union[
            bigtable_table_admin.ListTablesResponse,
            Awaitable[bigtable_table_admin.ListTablesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetTableRequest],
        Union[table.Table, Awaitable[table.Table]],
    ]:
        raise NotImplementedError()

    @property
    def update_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateTableRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.DeleteTableRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def undelete_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.UndeleteTableRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def create_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateAuthorizedViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def list_authorized_views(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListAuthorizedViewsRequest],
        Union[
            bigtable_table_admin.ListAuthorizedViewsResponse,
            Awaitable[bigtable_table_admin.ListAuthorizedViewsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetAuthorizedViewRequest],
        Union[table.AuthorizedView, Awaitable[table.AuthorizedView]],
    ]:
        raise NotImplementedError()

    @property
    def update_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateAuthorizedViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_authorized_view(
        self,
    ) -> Callable[
        [bigtable_table_admin.DeleteAuthorizedViewRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def modify_column_families(
        self,
    ) -> Callable[
        [bigtable_table_admin.ModifyColumnFamiliesRequest],
        Union[table.Table, Awaitable[table.Table]],
    ]:
        raise NotImplementedError()

    @property
    def drop_row_range(
        self,
    ) -> Callable[
        [bigtable_table_admin.DropRowRangeRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def generate_consistency_token(
        self,
    ) -> Callable[
        [bigtable_table_admin.GenerateConsistencyTokenRequest],
        Union[
            bigtable_table_admin.GenerateConsistencyTokenResponse,
            Awaitable[bigtable_table_admin.GenerateConsistencyTokenResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def check_consistency(
        self,
    ) -> Callable[
        [bigtable_table_admin.CheckConsistencyRequest],
        Union[
            bigtable_table_admin.CheckConsistencyResponse,
            Awaitable[bigtable_table_admin.CheckConsistencyResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def snapshot_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.SnapshotTableRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_snapshot(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetSnapshotRequest],
        Union[table.Snapshot, Awaitable[table.Snapshot]],
    ]:
        raise NotImplementedError()

    @property
    def list_snapshots(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSnapshotsRequest],
        Union[
            bigtable_table_admin.ListSnapshotsResponse,
            Awaitable[bigtable_table_admin.ListSnapshotsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_snapshot(
        self,
    ) -> Callable[
        [bigtable_table_admin.DeleteSnapshotRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def create_backup(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateBackupRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_backup(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetBackupRequest],
        Union[table.Backup, Awaitable[table.Backup]],
    ]:
        raise NotImplementedError()

    @property
    def update_backup(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateBackupRequest],
        Union[table.Backup, Awaitable[table.Backup]],
    ]:
        raise NotImplementedError()

    @property
    def delete_backup(
        self,
    ) -> Callable[
        [bigtable_table_admin.DeleteBackupRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_backups(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListBackupsRequest],
        Union[
            bigtable_table_admin.ListBackupsResponse,
            Awaitable[bigtable_table_admin.ListBackupsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def restore_table(
        self,
    ) -> Callable[
        [bigtable_table_admin.RestoreTableRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def copy_backup(
        self,
    ) -> Callable[
        [bigtable_table_admin.CopyBackupRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
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
    def set_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.SetIamPolicyRequest],
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
    def create_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.CreateSchemaBundleRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def update_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.UpdateSchemaBundleRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.GetSchemaBundleRequest],
        Union[table.SchemaBundle, Awaitable[table.SchemaBundle]],
    ]:
        raise NotImplementedError()

    @property
    def list_schema_bundles(
        self,
    ) -> Callable[
        [bigtable_table_admin.ListSchemaBundlesRequest],
        Union[
            bigtable_table_admin.ListSchemaBundlesResponse,
            Awaitable[bigtable_table_admin.ListSchemaBundlesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_schema_bundle(
        self,
    ) -> Callable[
        [bigtable_table_admin.DeleteSchemaBundleRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("BigtableTableAdminTransport",)

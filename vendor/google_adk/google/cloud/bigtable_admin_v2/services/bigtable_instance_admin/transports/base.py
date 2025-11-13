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

from google.cloud.bigtable_admin_v2.types import bigtable_instance_admin
from google.cloud.bigtable_admin_v2.types import instance
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class BigtableInstanceAdminTransport(abc.ABC):
    """Abstract transport class for BigtableInstanceAdmin."""

    AUTH_SCOPES = (
        "https://www.googleapis.com/auth/bigtable.admin",
        "https://www.googleapis.com/auth/bigtable.admin.cluster",
        "https://www.googleapis.com/auth/bigtable.admin.instance",
        "https://www.googleapis.com/auth/cloud-bigtable.admin",
        "https://www.googleapis.com/auth/cloud-bigtable.admin.cluster",
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
            self.create_instance: gapic_v1.method.wrap_method(
                self.create_instance,
                default_timeout=300.0,
                client_info=client_info,
            ),
            self.get_instance: gapic_v1.method.wrap_method(
                self.get_instance,
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
            self.list_instances: gapic_v1.method.wrap_method(
                self.list_instances,
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
            self.update_instance: gapic_v1.method.wrap_method(
                self.update_instance,
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
            self.partial_update_instance: gapic_v1.method.wrap_method(
                self.partial_update_instance,
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
            self.delete_instance: gapic_v1.method.wrap_method(
                self.delete_instance,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_cluster: gapic_v1.method.wrap_method(
                self.create_cluster,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_cluster: gapic_v1.method.wrap_method(
                self.get_cluster,
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
            self.list_clusters: gapic_v1.method.wrap_method(
                self.list_clusters,
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
            self.update_cluster: gapic_v1.method.wrap_method(
                self.update_cluster,
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
            self.partial_update_cluster: gapic_v1.method.wrap_method(
                self.partial_update_cluster,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_cluster: gapic_v1.method.wrap_method(
                self.delete_cluster,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_app_profile: gapic_v1.method.wrap_method(
                self.create_app_profile,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_app_profile: gapic_v1.method.wrap_method(
                self.get_app_profile,
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
            self.list_app_profiles: gapic_v1.method.wrap_method(
                self.list_app_profiles,
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
            self.update_app_profile: gapic_v1.method.wrap_method(
                self.update_app_profile,
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
            self.delete_app_profile: gapic_v1.method.wrap_method(
                self.delete_app_profile,
                default_timeout=60.0,
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
            self.list_hot_tablets: gapic_v1.method.wrap_method(
                self.list_hot_tablets,
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
            self.create_logical_view: gapic_v1.method.wrap_method(
                self.create_logical_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_logical_view: gapic_v1.method.wrap_method(
                self.get_logical_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_logical_views: gapic_v1.method.wrap_method(
                self.list_logical_views,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_logical_view: gapic_v1.method.wrap_method(
                self.update_logical_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_logical_view: gapic_v1.method.wrap_method(
                self.delete_logical_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_materialized_view: gapic_v1.method.wrap_method(
                self.create_materialized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_materialized_view: gapic_v1.method.wrap_method(
                self.get_materialized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_materialized_views: gapic_v1.method.wrap_method(
                self.list_materialized_views,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_materialized_view: gapic_v1.method.wrap_method(
                self.update_materialized_view,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_materialized_view: gapic_v1.method.wrap_method(
                self.delete_materialized_view,
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
    def create_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateInstanceRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetInstanceRequest],
        Union[instance.Instance, Awaitable[instance.Instance]],
    ]:
        raise NotImplementedError()

    @property
    def list_instances(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListInstancesRequest],
        Union[
            bigtable_instance_admin.ListInstancesResponse,
            Awaitable[bigtable_instance_admin.ListInstancesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_instance(
        self,
    ) -> Callable[
        [instance.Instance], Union[instance.Instance, Awaitable[instance.Instance]]
    ]:
        raise NotImplementedError()

    @property
    def partial_update_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.PartialUpdateInstanceRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_instance(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteInstanceRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def create_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateClusterRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetClusterRequest],
        Union[instance.Cluster, Awaitable[instance.Cluster]],
    ]:
        raise NotImplementedError()

    @property
    def list_clusters(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListClustersRequest],
        Union[
            bigtable_instance_admin.ListClustersResponse,
            Awaitable[bigtable_instance_admin.ListClustersResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_cluster(
        self,
    ) -> Callable[
        [instance.Cluster],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def partial_update_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.PartialUpdateClusterRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_cluster(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteClusterRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def create_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateAppProfileRequest],
        Union[instance.AppProfile, Awaitable[instance.AppProfile]],
    ]:
        raise NotImplementedError()

    @property
    def get_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetAppProfileRequest],
        Union[instance.AppProfile, Awaitable[instance.AppProfile]],
    ]:
        raise NotImplementedError()

    @property
    def list_app_profiles(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListAppProfilesRequest],
        Union[
            bigtable_instance_admin.ListAppProfilesResponse,
            Awaitable[bigtable_instance_admin.ListAppProfilesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateAppProfileRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_app_profile(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteAppProfileRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
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
    def list_hot_tablets(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListHotTabletsRequest],
        Union[
            bigtable_instance_admin.ListHotTabletsResponse,
            Awaitable[bigtable_instance_admin.ListHotTabletsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateLogicalViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetLogicalViewRequest],
        Union[instance.LogicalView, Awaitable[instance.LogicalView]],
    ]:
        raise NotImplementedError()

    @property
    def list_logical_views(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListLogicalViewsRequest],
        Union[
            bigtable_instance_admin.ListLogicalViewsResponse,
            Awaitable[bigtable_instance_admin.ListLogicalViewsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateLogicalViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_logical_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteLogicalViewRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def create_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.CreateMaterializedViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.GetMaterializedViewRequest],
        Union[instance.MaterializedView, Awaitable[instance.MaterializedView]],
    ]:
        raise NotImplementedError()

    @property
    def list_materialized_views(
        self,
    ) -> Callable[
        [bigtable_instance_admin.ListMaterializedViewsRequest],
        Union[
            bigtable_instance_admin.ListMaterializedViewsResponse,
            Awaitable[bigtable_instance_admin.ListMaterializedViewsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.UpdateMaterializedViewRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_materialized_view(
        self,
    ) -> Callable[
        [bigtable_instance_admin.DeleteMaterializedViewRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("BigtableInstanceAdminTransport",)

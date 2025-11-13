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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.bigtable_admin_v2.types import instance as gba_instance
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.admin.v2",
    manifest={
        "CreateInstanceRequest",
        "GetInstanceRequest",
        "ListInstancesRequest",
        "ListInstancesResponse",
        "PartialUpdateInstanceRequest",
        "DeleteInstanceRequest",
        "CreateClusterRequest",
        "GetClusterRequest",
        "ListClustersRequest",
        "ListClustersResponse",
        "DeleteClusterRequest",
        "CreateInstanceMetadata",
        "UpdateInstanceMetadata",
        "CreateClusterMetadata",
        "UpdateClusterMetadata",
        "PartialUpdateClusterMetadata",
        "PartialUpdateClusterRequest",
        "CreateAppProfileRequest",
        "GetAppProfileRequest",
        "ListAppProfilesRequest",
        "ListAppProfilesResponse",
        "UpdateAppProfileRequest",
        "DeleteAppProfileRequest",
        "UpdateAppProfileMetadata",
        "ListHotTabletsRequest",
        "ListHotTabletsResponse",
        "CreateLogicalViewRequest",
        "CreateLogicalViewMetadata",
        "GetLogicalViewRequest",
        "ListLogicalViewsRequest",
        "ListLogicalViewsResponse",
        "UpdateLogicalViewRequest",
        "UpdateLogicalViewMetadata",
        "DeleteLogicalViewRequest",
        "CreateMaterializedViewRequest",
        "CreateMaterializedViewMetadata",
        "GetMaterializedViewRequest",
        "ListMaterializedViewsRequest",
        "ListMaterializedViewsResponse",
        "UpdateMaterializedViewRequest",
        "UpdateMaterializedViewMetadata",
        "DeleteMaterializedViewRequest",
    },
)


class CreateInstanceRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.CreateInstance.

    Attributes:
        parent (str):
            Required. The unique name of the project in which to create
            the new instance. Values are of the form
            ``projects/{project}``.
        instance_id (str):
            Required. The ID to be used when referring to the new
            instance within its project, e.g., just ``myinstance``
            rather than ``projects/myproject/instances/myinstance``.
        instance (google.cloud.bigtable_admin_v2.types.Instance):
            Required. The instance to create. Fields marked
            ``OutputOnly`` must be left blank.
        clusters (MutableMapping[str, google.cloud.bigtable_admin_v2.types.Cluster]):
            Required. The clusters to be created within the instance,
            mapped by desired cluster ID, e.g., just ``mycluster``
            rather than
            ``projects/myproject/instances/myinstance/clusters/mycluster``.
            Fields marked ``OutputOnly`` must be left blank.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instance_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    instance: gba_instance.Instance = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_instance.Instance,
    )
    clusters: MutableMapping[str, gba_instance.Cluster] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=4,
        message=gba_instance.Cluster,
    )


class GetInstanceRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.GetInstance.

    Attributes:
        name (str):
            Required. The unique name of the requested instance. Values
            are of the form ``projects/{project}/instances/{instance}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListInstancesRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.ListInstances.

    Attributes:
        parent (str):
            Required. The unique name of the project for which a list of
            instances is requested. Values are of the form
            ``projects/{project}``.
        page_token (str):
            DEPRECATED: This field is unused and ignored.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListInstancesResponse(proto.Message):
    r"""Response message for BigtableInstanceAdmin.ListInstances.

    Attributes:
        instances (MutableSequence[google.cloud.bigtable_admin_v2.types.Instance]):
            The list of requested instances.
        failed_locations (MutableSequence[str]):
            Locations from which Instance information could not be
            retrieved, due to an outage or some other transient
            condition. Instances whose Clusters are all in one of the
            failed locations may be missing from ``instances``, and
            Instances with at least one Cluster in a failed location may
            only have partial information returned. Values are of the
            form ``projects/<project>/locations/<zone_id>``
        next_page_token (str):
            DEPRECATED: This field is unused and ignored.
    """

    @property
    def raw_page(self):
        return self

    instances: MutableSequence[gba_instance.Instance] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.Instance,
    )
    failed_locations: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class PartialUpdateInstanceRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.PartialUpdateInstance.

    Attributes:
        instance (google.cloud.bigtable_admin_v2.types.Instance):
            Required. The Instance which will (partially)
            replace the current value.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The subset of Instance fields which
            should be replaced. Must be explicitly set.
    """

    instance: gba_instance.Instance = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.Instance,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteInstanceRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.DeleteInstance.

    Attributes:
        name (str):
            Required. The unique name of the instance to be deleted.
            Values are of the form
            ``projects/{project}/instances/{instance}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateClusterRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.CreateCluster.

    Attributes:
        parent (str):
            Required. The unique name of the instance in which to create
            the new cluster. Values are of the form
            ``projects/{project}/instances/{instance}``.
        cluster_id (str):
            Required. The ID to be used when referring to the new
            cluster within its instance, e.g., just ``mycluster`` rather
            than
            ``projects/myproject/instances/myinstance/clusters/mycluster``.
        cluster (google.cloud.bigtable_admin_v2.types.Cluster):
            Required. The cluster to be created. Fields marked
            ``OutputOnly`` must be left blank.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    cluster_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    cluster: gba_instance.Cluster = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_instance.Cluster,
    )


class GetClusterRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.GetCluster.

    Attributes:
        name (str):
            Required. The unique name of the requested cluster. Values
            are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListClustersRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.ListClusters.

    Attributes:
        parent (str):
            Required. The unique name of the instance for which a list
            of clusters is requested. Values are of the form
            ``projects/{project}/instances/{instance}``. Use
            ``{instance} = '-'`` to list Clusters for all Instances in a
            project, e.g., ``projects/myproject/instances/-``.
        page_token (str):
            DEPRECATED: This field is unused and ignored.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListClustersResponse(proto.Message):
    r"""Response message for BigtableInstanceAdmin.ListClusters.

    Attributes:
        clusters (MutableSequence[google.cloud.bigtable_admin_v2.types.Cluster]):
            The list of requested clusters.
        failed_locations (MutableSequence[str]):
            Locations from which Cluster information could not be
            retrieved, due to an outage or some other transient
            condition. Clusters from these locations may be missing from
            ``clusters``, or may only have partial information returned.
            Values are of the form
            ``projects/<project>/locations/<zone_id>``
        next_page_token (str):
            DEPRECATED: This field is unused and ignored.
    """

    @property
    def raw_page(self):
        return self

    clusters: MutableSequence[gba_instance.Cluster] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.Cluster,
    )
    failed_locations: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DeleteClusterRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.DeleteCluster.

    Attributes:
        name (str):
            Required. The unique name of the cluster to be deleted.
            Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateInstanceMetadata(proto.Message):
    r"""The metadata for the Operation returned by CreateInstance.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateInstanceRequest):
            The request that prompted the initiation of
            this CreateInstance operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "CreateInstanceRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateInstanceRequest",
    )
    request_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    finish_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class UpdateInstanceMetadata(proto.Message):
    r"""The metadata for the Operation returned by UpdateInstance.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.PartialUpdateInstanceRequest):
            The request that prompted the initiation of
            this UpdateInstance operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "PartialUpdateInstanceRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PartialUpdateInstanceRequest",
    )
    request_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    finish_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class CreateClusterMetadata(proto.Message):
    r"""The metadata for the Operation returned by CreateCluster.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateClusterRequest):
            The request that prompted the initiation of
            this CreateCluster operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
        tables (MutableMapping[str, google.cloud.bigtable_admin_v2.types.CreateClusterMetadata.TableProgress]):
            Keys: the full ``name`` of each table that existed in the
            instance when CreateCluster was first called, i.e.
            ``projects/<project>/instances/<instance>/tables/<table>``.
            Any table added to the instance by a later API call will be
            created in the new cluster by that API call, not this one.

            Values: information on how much of a table's data has been
            copied to the newly-created cluster so far.
    """

    class TableProgress(proto.Message):
        r"""Progress info for copying a table's data to the new cluster.

        Attributes:
            estimated_size_bytes (int):
                Estimate of the size of the table to be
                copied.
            estimated_copied_bytes (int):
                Estimate of the number of bytes copied so far for this
                table. This will eventually reach 'estimated_size_bytes'
                unless the table copy is CANCELLED.
            state (google.cloud.bigtable_admin_v2.types.CreateClusterMetadata.TableProgress.State):

        """

        class State(proto.Enum):
            r"""

            Values:
                STATE_UNSPECIFIED (0):
                    No description available.
                PENDING (1):
                    The table has not yet begun copying to the
                    new cluster.
                COPYING (2):
                    The table is actively being copied to the new
                    cluster.
                COMPLETED (3):
                    The table has been fully copied to the new
                    cluster.
                CANCELLED (4):
                    The table was deleted before it finished
                    copying to the new cluster. Note that tables
                    deleted after completion will stay marked as
                    COMPLETED, not CANCELLED.
            """
            STATE_UNSPECIFIED = 0
            PENDING = 1
            COPYING = 2
            COMPLETED = 3
            CANCELLED = 4

        estimated_size_bytes: int = proto.Field(
            proto.INT64,
            number=2,
        )
        estimated_copied_bytes: int = proto.Field(
            proto.INT64,
            number=3,
        )
        state: "CreateClusterMetadata.TableProgress.State" = proto.Field(
            proto.ENUM,
            number=4,
            enum="CreateClusterMetadata.TableProgress.State",
        )

    original_request: "CreateClusterRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateClusterRequest",
    )
    request_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    finish_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    tables: MutableMapping[str, TableProgress] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=4,
        message=TableProgress,
    )


class UpdateClusterMetadata(proto.Message):
    r"""The metadata for the Operation returned by UpdateCluster.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.Cluster):
            The request that prompted the initiation of
            this UpdateCluster operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: gba_instance.Cluster = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.Cluster,
    )
    request_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    finish_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class PartialUpdateClusterMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    PartialUpdateCluster.

    Attributes:
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
        original_request (google.cloud.bigtable_admin_v2.types.PartialUpdateClusterRequest):
            The original request for
            PartialUpdateCluster.
    """

    request_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    finish_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    original_request: "PartialUpdateClusterRequest" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="PartialUpdateClusterRequest",
    )


class PartialUpdateClusterRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.PartialUpdateCluster.

    Attributes:
        cluster (google.cloud.bigtable_admin_v2.types.Cluster):
            Required. The Cluster which contains the partial updates to
            be applied, subject to the update_mask.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The subset of Cluster fields which
            should be replaced.
    """

    cluster: gba_instance.Cluster = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.Cluster,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class CreateAppProfileRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.CreateAppProfile.

    Attributes:
        parent (str):
            Required. The unique name of the instance in which to create
            the new app profile. Values are of the form
            ``projects/{project}/instances/{instance}``.
        app_profile_id (str):
            Required. The ID to be used when referring to the new app
            profile within its instance, e.g., just ``myprofile`` rather
            than
            ``projects/myproject/instances/myinstance/appProfiles/myprofile``.
        app_profile (google.cloud.bigtable_admin_v2.types.AppProfile):
            Required. The app profile to be created. Fields marked
            ``OutputOnly`` will be ignored.
        ignore_warnings (bool):
            If true, ignore safety checks when creating
            the app profile.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    app_profile: gba_instance.AppProfile = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_instance.AppProfile,
    )
    ignore_warnings: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class GetAppProfileRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.GetAppProfile.

    Attributes:
        name (str):
            Required. The unique name of the requested app profile.
            Values are of the form
            ``projects/{project}/instances/{instance}/appProfiles/{app_profile}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListAppProfilesRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.ListAppProfiles.

    Attributes:
        parent (str):
            Required. The unique name of the instance for which a list
            of app profiles is requested. Values are of the form
            ``projects/{project}/instances/{instance}``. Use
            ``{instance} = '-'`` to list AppProfiles for all Instances
            in a project, e.g., ``projects/myproject/instances/-``.
        page_size (int):
            Maximum number of results per page.

            A page_size of zero lets the server choose the number of
            items to return. A page_size which is strictly positive will
            return at most that many items. A negative page_size will
            cause an error.

            Following the first request, subsequent paginated calls are
            not required to pass a page_size. If a page_size is set in
            subsequent calls, it must match the page_size given in the
            first request.
        page_token (str):
            The value of ``next_page_token`` returned by a previous
            call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListAppProfilesResponse(proto.Message):
    r"""Response message for BigtableInstanceAdmin.ListAppProfiles.

    Attributes:
        app_profiles (MutableSequence[google.cloud.bigtable_admin_v2.types.AppProfile]):
            The list of requested app profiles.
        next_page_token (str):
            Set if not all app profiles could be returned in a single
            response. Pass this value to ``page_token`` in another
            request to get the next page of results.
        failed_locations (MutableSequence[str]):
            Locations from which AppProfile information could not be
            retrieved, due to an outage or some other transient
            condition. AppProfiles from these locations may be missing
            from ``app_profiles``. Values are of the form
            ``projects/<project>/locations/<zone_id>``
    """

    @property
    def raw_page(self):
        return self

    app_profiles: MutableSequence[gba_instance.AppProfile] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.AppProfile,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    failed_locations: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class UpdateAppProfileRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.UpdateAppProfile.

    Attributes:
        app_profile (google.cloud.bigtable_admin_v2.types.AppProfile):
            Required. The app profile which will
            (partially) replace the current value.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The subset of app profile fields
            which should be replaced. If unset, all fields
            will be replaced.
        ignore_warnings (bool):
            If true, ignore safety checks when updating
            the app profile.
    """

    app_profile: gba_instance.AppProfile = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.AppProfile,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )
    ignore_warnings: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class DeleteAppProfileRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.DeleteAppProfile.

    Attributes:
        name (str):
            Required. The unique name of the app profile to be deleted.
            Values are of the form
            ``projects/{project}/instances/{instance}/appProfiles/{app_profile}``.
        ignore_warnings (bool):
            Required. If true, ignore safety checks when
            deleting the app profile.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    ignore_warnings: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class UpdateAppProfileMetadata(proto.Message):
    r"""The metadata for the Operation returned by UpdateAppProfile."""


class ListHotTabletsRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.ListHotTablets.

    Attributes:
        parent (str):
            Required. The cluster name to list hot tablets. Value is in
            the following form:
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The start time to list hot tablets. The hot
            tablets in the response will have start times
            between the requested start time and end time.
            Start time defaults to Now if it is unset, and
            end time defaults to Now - 24 hours if it is
            unset. The start time should be less than the
            end time, and the maximum allowed time range
            between start time and end time is 48 hours.
            Start time and end time should have values
            between Now and Now - 14 days.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            The end time to list hot tablets.
        page_size (int):
            Maximum number of results per page.

            A page_size that is empty or zero lets the server choose the
            number of items to return. A page_size which is strictly
            positive will return at most that many items. A negative
            page_size will cause an error.

            Following the first request, subsequent paginated calls do
            not need a page_size field. If a page_size is set in
            subsequent calls, it must match the page_size given in the
            first request.
        page_token (str):
            The value of ``next_page_token`` returned by a previous
            call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=4,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListHotTabletsResponse(proto.Message):
    r"""Response message for BigtableInstanceAdmin.ListHotTablets.

    Attributes:
        hot_tablets (MutableSequence[google.cloud.bigtable_admin_v2.types.HotTablet]):
            List of hot tablets in the tables of the
            requested cluster that fall within the requested
            time range. Hot tablets are ordered by node cpu
            usage percent. If there are multiple hot tablets
            that correspond to the same tablet within a
            15-minute interval, only the hot tablet with the
            highest node cpu usage will be included in the
            response.
        next_page_token (str):
            Set if not all hot tablets could be returned in a single
            response. Pass this value to ``page_token`` in another
            request to get the next page of results.
    """

    @property
    def raw_page(self):
        return self

    hot_tablets: MutableSequence[gba_instance.HotTablet] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.HotTablet,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateLogicalViewRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.CreateLogicalView.

    Attributes:
        parent (str):
            Required. The parent instance where this logical view will
            be created. Format:
            ``projects/{project}/instances/{instance}``.
        logical_view_id (str):
            Required. The ID to use for the logical view,
            which will become the final component of the
            logical view's resource name.
        logical_view (google.cloud.bigtable_admin_v2.types.LogicalView):
            Required. The logical view to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    logical_view_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    logical_view: gba_instance.LogicalView = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_instance.LogicalView,
    )


class CreateLogicalViewMetadata(proto.Message):
    r"""The metadata for the Operation returned by CreateLogicalView.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateLogicalViewRequest):
            The request that prompted the initiation of
            this CreateLogicalView operation.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    original_request: "CreateLogicalViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateLogicalViewRequest",
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class GetLogicalViewRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.GetLogicalView.

    Attributes:
        name (str):
            Required. The unique name of the requested logical view.
            Values are of the form
            ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListLogicalViewsRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.ListLogicalViews.

    Attributes:
        parent (str):
            Required. The unique name of the instance for which the list
            of logical views is requested. Values are of the form
            ``projects/{project}/instances/{instance}``.
        page_size (int):
            Optional. The maximum number of logical views
            to return. The service may return fewer than
            this value
        page_token (str):
            Optional. A page token, received from a previous
            ``ListLogicalViews`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListLogicalViews`` must match the call that provided the
            page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListLogicalViewsResponse(proto.Message):
    r"""Response message for BigtableInstanceAdmin.ListLogicalViews.

    Attributes:
        logical_views (MutableSequence[google.cloud.bigtable_admin_v2.types.LogicalView]):
            The list of requested logical views.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    logical_views: MutableSequence[gba_instance.LogicalView] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.LogicalView,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateLogicalViewRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.UpdateLogicalView.

    Attributes:
        logical_view (google.cloud.bigtable_admin_v2.types.LogicalView):
            Required. The logical view to update.

            The logical view's ``name`` field is used to identify the
            view to update. Format:
            ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update.
    """

    logical_view: gba_instance.LogicalView = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.LogicalView,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateLogicalViewMetadata(proto.Message):
    r"""The metadata for the Operation returned by UpdateLogicalView.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.UpdateLogicalViewRequest):
            The request that prompted the initiation of
            this UpdateLogicalView operation.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation was started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    original_request: "UpdateLogicalViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="UpdateLogicalViewRequest",
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class DeleteLogicalViewRequest(proto.Message):
    r"""Request message for BigtableInstanceAdmin.DeleteLogicalView.

    Attributes:
        name (str):
            Required. The unique name of the logical view to be deleted.
            Format:
            ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``.
        etag (str):
            Optional. The current etag of the logical
            view. If an etag is provided and does not match
            the current etag of the logical view, deletion
            will be blocked and an ABORTED error will be
            returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateMaterializedViewRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.CreateMaterializedView.

    Attributes:
        parent (str):
            Required. The parent instance where this materialized view
            will be created. Format:
            ``projects/{project}/instances/{instance}``.
        materialized_view_id (str):
            Required. The ID to use for the materialized
            view, which will become the final component of
            the materialized view's resource name.
        materialized_view (google.cloud.bigtable_admin_v2.types.MaterializedView):
            Required. The materialized view to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    materialized_view_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    materialized_view: gba_instance.MaterializedView = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_instance.MaterializedView,
    )


class CreateMaterializedViewMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    CreateMaterializedView.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateMaterializedViewRequest):
            The request that prompted the initiation of
            this CreateMaterializedView operation.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    original_request: "CreateMaterializedViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateMaterializedViewRequest",
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class GetMaterializedViewRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.GetMaterializedView.

    Attributes:
        name (str):
            Required. The unique name of the requested materialized
            view. Values are of the form
            ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListMaterializedViewsRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.ListMaterializedViews.

    Attributes:
        parent (str):
            Required. The unique name of the instance for which the list
            of materialized views is requested. Values are of the form
            ``projects/{project}/instances/{instance}``.
        page_size (int):
            Optional. The maximum number of materialized
            views to return. The service may return fewer
            than this value
        page_token (str):
            Optional. A page token, received from a previous
            ``ListMaterializedViews`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListMaterializedViews`` must match the call that provided
            the page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListMaterializedViewsResponse(proto.Message):
    r"""Response message for
    BigtableInstanceAdmin.ListMaterializedViews.

    Attributes:
        materialized_views (MutableSequence[google.cloud.bigtable_admin_v2.types.MaterializedView]):
            The list of requested materialized views.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    materialized_views: MutableSequence[
        gba_instance.MaterializedView
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_instance.MaterializedView,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateMaterializedViewRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.UpdateMaterializedView.

    Attributes:
        materialized_view (google.cloud.bigtable_admin_v2.types.MaterializedView):
            Required. The materialized view to update.

            The materialized view's ``name`` field is used to identify
            the view to update. Format:
            ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update.
    """

    materialized_view: gba_instance.MaterializedView = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_instance.MaterializedView,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateMaterializedViewMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    UpdateMaterializedView.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.UpdateMaterializedViewRequest):
            The request that prompted the initiation of
            this UpdateMaterializedView operation.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation was started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    original_request: "UpdateMaterializedViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="UpdateMaterializedViewRequest",
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class DeleteMaterializedViewRequest(proto.Message):
    r"""Request message for
    BigtableInstanceAdmin.DeleteMaterializedView.

    Attributes:
        name (str):
            Required. The unique name of the materialized view to be
            deleted. Format:
            ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``.
        etag (str):
            Optional. The current etag of the
            materialized view. If an etag is provided and
            does not match the current etag of the
            materialized view, deletion will be blocked and
            an ABORTED error will be returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

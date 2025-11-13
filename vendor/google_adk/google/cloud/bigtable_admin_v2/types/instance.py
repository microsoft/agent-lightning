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

from google.cloud.bigtable_admin_v2.types import common
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.admin.v2",
    manifest={
        "Instance",
        "AutoscalingTargets",
        "AutoscalingLimits",
        "Cluster",
        "AppProfile",
        "HotTablet",
        "LogicalView",
        "MaterializedView",
    },
)


class Instance(proto.Message):
    r"""A collection of Bigtable [Tables][google.bigtable.admin.v2.Table]
    and the resources that serve them. All tables in an instance are
    served from all [Clusters][google.bigtable.admin.v2.Cluster] in the
    instance.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            The unique name of the instance. Values are of the form
            ``projects/{project}/instances/[a-z][a-z0-9\\-]+[a-z0-9]``.
        display_name (str):
            Required. The descriptive name for this
            instance as it appears in UIs. Can be changed at
            any time, but should be kept globally unique to
            avoid confusion.
        state (google.cloud.bigtable_admin_v2.types.Instance.State):
            Output only. The current state of the
            instance.
        type_ (google.cloud.bigtable_admin_v2.types.Instance.Type):
            The type of the instance. Defaults to ``PRODUCTION``.
        labels (MutableMapping[str, str]):
            Labels are a flexible and lightweight mechanism for
            organizing cloud resources into groups that reflect a
            customer's organizational needs and deployment strategies.
            They can be used to filter resources and aggregate metrics.

            -  Label keys must be between 1 and 63 characters long and
               must conform to the regular expression:
               ``[\p{Ll}\p{Lo}][\p{Ll}\p{Lo}\p{N}_-]{0,62}``.
            -  Label values must be between 0 and 63 characters long and
               must conform to the regular expression:
               ``[\p{Ll}\p{Lo}\p{N}_-]{0,63}``.
            -  No more than 64 labels can be associated with a given
               resource.
            -  Keys and values must both be under 128 bytes.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. A commit timestamp representing when this
            Instance was created. For instances created before this
            field was added (August 2021), this value is
            ``seconds: 0, nanos: 1``.
        satisfies_pzs (bool):
            Output only. Reserved for future use.

            This field is a member of `oneof`_ ``_satisfies_pzs``.
        satisfies_pzi (bool):
            Output only. Reserved for future use.

            This field is a member of `oneof`_ ``_satisfies_pzi``.
        tags (MutableMapping[str, str]):
            Optional. Input only. Immutable. Tag
            keys/values directly bound to this resource. For
            example:

            - "123/environment": "production",
            - "123/costCenter": "marketing"

            Tags and Labels (above) are both used to bind
            metadata to resources, with different use-cases.
            See
            https://cloud.google.com/resource-manager/docs/tags/tags-overview
            for an in-depth overview on the difference
            between tags and labels.
    """

    class State(proto.Enum):
        r"""Possible states of an instance.

        Values:
            STATE_NOT_KNOWN (0):
                The state of the instance could not be
                determined.
            READY (1):
                The instance has been successfully created
                and can serve requests to its tables.
            CREATING (2):
                The instance is currently being created, and
                may be destroyed if the creation process
                encounters an error.
        """
        STATE_NOT_KNOWN = 0
        READY = 1
        CREATING = 2

    class Type(proto.Enum):
        r"""The type of the instance.

        Values:
            TYPE_UNSPECIFIED (0):
                The type of the instance is unspecified. If set when
                creating an instance, a ``PRODUCTION`` instance will be
                created. If set when updating an instance, the type will be
                left unchanged.
            PRODUCTION (1):
                An instance meant for production use. ``serve_nodes`` must
                be set on the cluster.
            DEVELOPMENT (2):
                DEPRECATED: Prefer PRODUCTION for all use
                cases, as it no longer enforces a higher minimum
                node count than DEVELOPMENT.
        """
        TYPE_UNSPECIFIED = 0
        PRODUCTION = 1
        DEVELOPMENT = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=3,
        enum=State,
    )
    type_: Type = proto.Field(
        proto.ENUM,
        number=4,
        enum=Type,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    satisfies_pzs: bool = proto.Field(
        proto.BOOL,
        number=8,
        optional=True,
    )
    satisfies_pzi: bool = proto.Field(
        proto.BOOL,
        number=11,
        optional=True,
    )
    tags: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )


class AutoscalingTargets(proto.Message):
    r"""The Autoscaling targets for a Cluster. These determine the
    recommended nodes.

    Attributes:
        cpu_utilization_percent (int):
            The cpu utilization that the Autoscaler should be trying to
            achieve. This number is on a scale from 0 (no utilization)
            to 100 (total utilization), and is limited between 10 and
            80, otherwise it will return INVALID_ARGUMENT error.
        storage_utilization_gib_per_node (int):
            The storage utilization that the Autoscaler should be trying
            to achieve. This number is limited between 2560 (2.5TiB) and
            5120 (5TiB) for a SSD cluster and between 8192 (8TiB) and
            16384 (16TiB) for an HDD cluster, otherwise it will return
            INVALID_ARGUMENT error. If this value is set to 0, it will
            be treated as if it were set to the default value: 2560 for
            SSD, 8192 for HDD.
    """

    cpu_utilization_percent: int = proto.Field(
        proto.INT32,
        number=2,
    )
    storage_utilization_gib_per_node: int = proto.Field(
        proto.INT32,
        number=3,
    )


class AutoscalingLimits(proto.Message):
    r"""Limits for the number of nodes a Cluster can autoscale
    up/down to.

    Attributes:
        min_serve_nodes (int):
            Required. Minimum number of nodes to scale
            down to.
        max_serve_nodes (int):
            Required. Maximum number of nodes to scale up
            to.
    """

    min_serve_nodes: int = proto.Field(
        proto.INT32,
        number=1,
    )
    max_serve_nodes: int = proto.Field(
        proto.INT32,
        number=2,
    )


class Cluster(proto.Message):
    r"""A resizable group of nodes in a particular cloud location, capable
    of serving all [Tables][google.bigtable.admin.v2.Table] in the
    parent [Instance][google.bigtable.admin.v2.Instance].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            The unique name of the cluster. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/[a-z][-a-z0-9]*``.
        location (str):
            Immutable. The location where this cluster's nodes and
            storage reside. For best performance, clients should be
            located as close as possible to this cluster. Currently only
            zones are supported, so values should be of the form
            ``projects/{project}/locations/{zone}``.
        state (google.cloud.bigtable_admin_v2.types.Cluster.State):
            Output only. The current state of the
            cluster.
        serve_nodes (int):
            The number of nodes in the cluster. If no
            value is set, Cloud Bigtable automatically
            allocates nodes based on your data footprint and
            optimized for 50% storage utilization.
        node_scaling_factor (google.cloud.bigtable_admin_v2.types.Cluster.NodeScalingFactor):
            Immutable. The node scaling factor of this
            cluster.
        cluster_config (google.cloud.bigtable_admin_v2.types.Cluster.ClusterConfig):
            Configuration for this cluster.

            This field is a member of `oneof`_ ``config``.
        default_storage_type (google.cloud.bigtable_admin_v2.types.StorageType):
            Immutable. The type of storage used by this
            cluster to serve its parent instance's tables,
            unless explicitly overridden.
        encryption_config (google.cloud.bigtable_admin_v2.types.Cluster.EncryptionConfig):
            Immutable. The encryption configuration for
            CMEK-protected clusters.
    """

    class State(proto.Enum):
        r"""Possible states of a cluster.

        Values:
            STATE_NOT_KNOWN (0):
                The state of the cluster could not be
                determined.
            READY (1):
                The cluster has been successfully created and
                is ready to serve requests.
            CREATING (2):
                The cluster is currently being created, and
                may be destroyed if the creation process
                encounters an error. A cluster may not be able
                to serve requests while being created.
            RESIZING (3):
                The cluster is currently being resized, and
                may revert to its previous node count if the
                process encounters an error. A cluster is still
                capable of serving requests while being resized,
                but may exhibit performance as if its number of
                allocated nodes is between the starting and
                requested states.
            DISABLED (4):
                The cluster has no backing nodes. The data
                (tables) still exist, but no operations can be
                performed on the cluster.
        """
        STATE_NOT_KNOWN = 0
        READY = 1
        CREATING = 2
        RESIZING = 3
        DISABLED = 4

    class NodeScalingFactor(proto.Enum):
        r"""Possible node scaling factors of the clusters. Node scaling
        delivers better latency and more throughput by removing node
        boundaries.

        Values:
            NODE_SCALING_FACTOR_UNSPECIFIED (0):
                No node scaling specified. Defaults to
                NODE_SCALING_FACTOR_1X.
            NODE_SCALING_FACTOR_1X (1):
                The cluster is running with a scaling factor
                of 1.
            NODE_SCALING_FACTOR_2X (2):
                The cluster is running with a scaling factor of 2. All node
                count values must be in increments of 2 with this scaling
                factor enabled, otherwise an INVALID_ARGUMENT error will be
                returned.
        """
        NODE_SCALING_FACTOR_UNSPECIFIED = 0
        NODE_SCALING_FACTOR_1X = 1
        NODE_SCALING_FACTOR_2X = 2

    class ClusterAutoscalingConfig(proto.Message):
        r"""Autoscaling config for a cluster.

        Attributes:
            autoscaling_limits (google.cloud.bigtable_admin_v2.types.AutoscalingLimits):
                Required. Autoscaling limits for this
                cluster.
            autoscaling_targets (google.cloud.bigtable_admin_v2.types.AutoscalingTargets):
                Required. Autoscaling targets for this
                cluster.
        """

        autoscaling_limits: "AutoscalingLimits" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="AutoscalingLimits",
        )
        autoscaling_targets: "AutoscalingTargets" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="AutoscalingTargets",
        )

    class ClusterConfig(proto.Message):
        r"""Configuration for a cluster.

        Attributes:
            cluster_autoscaling_config (google.cloud.bigtable_admin_v2.types.Cluster.ClusterAutoscalingConfig):
                Autoscaling configuration for this cluster.
        """

        cluster_autoscaling_config: "Cluster.ClusterAutoscalingConfig" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Cluster.ClusterAutoscalingConfig",
        )

    class EncryptionConfig(proto.Message):
        r"""Cloud Key Management Service (Cloud KMS) settings for a
        CMEK-protected cluster.

        Attributes:
            kms_key_name (str):
                Describes the Cloud KMS encryption key that will be used to
                protect the destination Bigtable cluster. The requirements
                for this key are:

                1) The Cloud Bigtable service account associated with the
                   project that contains this cluster must be granted the
                   ``cloudkms.cryptoKeyEncrypterDecrypter`` role on the CMEK
                   key.
                2) Only regional keys can be used and the region of the CMEK
                   key must match the region of the cluster. Values are of
                   the form
                   ``projects/{project}/locations/{location}/keyRings/{keyring}/cryptoKeys/{key}``
        """

        kms_key_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    location: str = proto.Field(
        proto.STRING,
        number=2,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=3,
        enum=State,
    )
    serve_nodes: int = proto.Field(
        proto.INT32,
        number=4,
    )
    node_scaling_factor: NodeScalingFactor = proto.Field(
        proto.ENUM,
        number=9,
        enum=NodeScalingFactor,
    )
    cluster_config: ClusterConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="config",
        message=ClusterConfig,
    )
    default_storage_type: common.StorageType = proto.Field(
        proto.ENUM,
        number=5,
        enum=common.StorageType,
    )
    encryption_config: EncryptionConfig = proto.Field(
        proto.MESSAGE,
        number=6,
        message=EncryptionConfig,
    )


class AppProfile(proto.Message):
    r"""A configuration object describing how Cloud Bigtable should
    treat traffic from a particular end user application.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            The unique name of the app profile. Values are of the form
            ``projects/{project}/instances/{instance}/appProfiles/[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
        etag (str):
            Strongly validated etag for optimistic concurrency control.
            Preserve the value returned from ``GetAppProfile`` when
            calling ``UpdateAppProfile`` to fail the request if there
            has been a modification in the mean time. The
            ``update_mask`` of the request need not include ``etag`` for
            this protection to apply. See
            `Wikipedia <https://en.wikipedia.org/wiki/HTTP_ETag>`__ and
            `RFC
            7232 <https://tools.ietf.org/html/rfc7232#section-2.3>`__
            for more details.
        description (str):
            Long form description of the use case for
            this AppProfile.
        multi_cluster_routing_use_any (google.cloud.bigtable_admin_v2.types.AppProfile.MultiClusterRoutingUseAny):
            Use a multi-cluster routing policy.

            This field is a member of `oneof`_ ``routing_policy``.
        single_cluster_routing (google.cloud.bigtable_admin_v2.types.AppProfile.SingleClusterRouting):
            Use a single-cluster routing policy.

            This field is a member of `oneof`_ ``routing_policy``.
        priority (google.cloud.bigtable_admin_v2.types.AppProfile.Priority):
            This field has been deprecated in favor of
            ``standard_isolation.priority``. If you set this field,
            ``standard_isolation.priority`` will be set instead.

            The priority of requests sent using this app profile.

            This field is a member of `oneof`_ ``isolation``.
        standard_isolation (google.cloud.bigtable_admin_v2.types.AppProfile.StandardIsolation):
            The standard options used for isolating this
            app profile's traffic from other use cases.

            This field is a member of `oneof`_ ``isolation``.
        data_boost_isolation_read_only (google.cloud.bigtable_admin_v2.types.AppProfile.DataBoostIsolationReadOnly):
            Specifies that this app profile is intended
            for read-only usage via the Data Boost feature.

            This field is a member of `oneof`_ ``isolation``.
    """

    class Priority(proto.Enum):
        r"""Possible priorities for an app profile. Note that higher
        priority writes can sometimes queue behind lower priority writes
        to the same tablet, as writes must be strictly sequenced in the
        durability log.

        Values:
            PRIORITY_UNSPECIFIED (0):
                Default value. Mapped to PRIORITY_HIGH (the legacy behavior)
                on creation.
            PRIORITY_LOW (1):
                No description available.
            PRIORITY_MEDIUM (2):
                No description available.
            PRIORITY_HIGH (3):
                No description available.
        """
        PRIORITY_UNSPECIFIED = 0
        PRIORITY_LOW = 1
        PRIORITY_MEDIUM = 2
        PRIORITY_HIGH = 3

    class MultiClusterRoutingUseAny(proto.Message):
        r"""Read/write requests are routed to the nearest cluster in the
        instance, and will fail over to the nearest cluster that is
        available in the event of transient errors or delays. Clusters
        in a region are considered equidistant. Choosing this option
        sacrifices read-your-writes consistency to improve availability.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            cluster_ids (MutableSequence[str]):
                The set of clusters to route to. The order is
                ignored; clusters will be tried in order of
                distance. If left empty, all clusters are
                eligible.
            row_affinity (google.cloud.bigtable_admin_v2.types.AppProfile.MultiClusterRoutingUseAny.RowAffinity):
                Row affinity sticky routing based on the row
                key of the request. Requests that span multiple
                rows are routed non-deterministically.

                This field is a member of `oneof`_ ``affinity``.
        """

        class RowAffinity(proto.Message):
            r"""If enabled, Bigtable will route the request based on the row
            key of the request, rather than randomly. Instead, each row key
            will be assigned to a cluster, and will stick to that cluster.
            If clusters are added or removed, then this may affect which row
            keys stick to which clusters. To avoid this, users can use a
            cluster group to specify which clusters are to be used. In this
            case, new clusters that are not a part of the cluster group will
            not be routed to, and routing will be unaffected by the new
            cluster. Moreover, clusters specified in the cluster group
            cannot be deleted unless removed from the cluster group.

            """

        cluster_ids: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )
        row_affinity: "AppProfile.MultiClusterRoutingUseAny.RowAffinity" = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="affinity",
            message="AppProfile.MultiClusterRoutingUseAny.RowAffinity",
        )

    class SingleClusterRouting(proto.Message):
        r"""Unconditionally routes all read/write requests to a specific
        cluster. This option preserves read-your-writes consistency but
        does not improve availability.

        Attributes:
            cluster_id (str):
                The cluster to which read/write requests
                should be routed.
            allow_transactional_writes (bool):
                Whether or not ``CheckAndMutateRow`` and
                ``ReadModifyWriteRow`` requests are allowed by this app
                profile. It is unsafe to send these requests to the same
                table/row/column in multiple clusters.
        """

        cluster_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        allow_transactional_writes: bool = proto.Field(
            proto.BOOL,
            number=2,
        )

    class StandardIsolation(proto.Message):
        r"""Standard options for isolating this app profile's traffic
        from other use cases.

        Attributes:
            priority (google.cloud.bigtable_admin_v2.types.AppProfile.Priority):
                The priority of requests sent using this app
                profile.
        """

        priority: "AppProfile.Priority" = proto.Field(
            proto.ENUM,
            number=1,
            enum="AppProfile.Priority",
        )

    class DataBoostIsolationReadOnly(proto.Message):
        r"""Data Boost is a serverless compute capability that lets you
        run high-throughput read jobs and queries on your Bigtable data,
        without impacting the performance of the clusters that handle
        your application traffic. Data Boost supports read-only use
        cases with single-cluster routing.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            compute_billing_owner (google.cloud.bigtable_admin_v2.types.AppProfile.DataBoostIsolationReadOnly.ComputeBillingOwner):
                The Compute Billing Owner for this Data Boost
                App Profile.

                This field is a member of `oneof`_ ``_compute_billing_owner``.
        """

        class ComputeBillingOwner(proto.Enum):
            r"""Compute Billing Owner specifies how usage should be accounted
            when using Data Boost. Compute Billing Owner also configures
            which Cloud Project is charged for relevant quota.

            Values:
                COMPUTE_BILLING_OWNER_UNSPECIFIED (0):
                    Unspecified value.
                HOST_PAYS (1):
                    The host Cloud Project containing the
                    targeted Bigtable Instance / Table pays for
                    compute.
            """
            COMPUTE_BILLING_OWNER_UNSPECIFIED = 0
            HOST_PAYS = 1

        compute_billing_owner: "AppProfile.DataBoostIsolationReadOnly.ComputeBillingOwner" = proto.Field(
            proto.ENUM,
            number=1,
            optional=True,
            enum="AppProfile.DataBoostIsolationReadOnly.ComputeBillingOwner",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    multi_cluster_routing_use_any: MultiClusterRoutingUseAny = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="routing_policy",
        message=MultiClusterRoutingUseAny,
    )
    single_cluster_routing: SingleClusterRouting = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="routing_policy",
        message=SingleClusterRouting,
    )
    priority: Priority = proto.Field(
        proto.ENUM,
        number=7,
        oneof="isolation",
        enum=Priority,
    )
    standard_isolation: StandardIsolation = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="isolation",
        message=StandardIsolation,
    )
    data_boost_isolation_read_only: DataBoostIsolationReadOnly = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="isolation",
        message=DataBoostIsolationReadOnly,
    )


class HotTablet(proto.Message):
    r"""A tablet is a defined by a start and end key and is explained
    in https://cloud.google.com/bigtable/docs/overview#architecture
    and
    https://cloud.google.com/bigtable/docs/performance#optimization.
    A Hot tablet is a tablet that exhibits high average cpu usage
    during the time interval from start time to end time.

    Attributes:
        name (str):
            The unique name of the hot tablet. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/hotTablets/[a-zA-Z0-9_-]*``.
        table_name (str):
            Name of the table that contains the tablet. Values are of
            the form
            ``projects/{project}/instances/{instance}/tables/[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The start time of the hot
            tablet.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The end time of the hot tablet.
        start_key (str):
            Tablet Start Key (inclusive).
        end_key (str):
            Tablet End Key (inclusive).
        node_cpu_usage_percent (float):
            Output only. The average CPU usage spent by a node on this
            tablet over the start_time to end_time time range. The
            percentage is the amount of CPU used by the node to serve
            the tablet, from 0% (tablet was not interacted with) to 100%
            (the node spent all cycles serving the hot tablet).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    table_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    start_key: str = proto.Field(
        proto.STRING,
        number=5,
    )
    end_key: str = proto.Field(
        proto.STRING,
        number=6,
    )
    node_cpu_usage_percent: float = proto.Field(
        proto.FLOAT,
        number=7,
    )


class LogicalView(proto.Message):
    r"""A SQL logical view object that can be referenced in SQL
    queries.

    Attributes:
        name (str):
            Identifier. The unique name of the logical view. Format:
            ``projects/{project}/instances/{instance}/logicalViews/{logical_view}``
        query (str):
            Required. The logical view's select query.
        etag (str):
            Optional. The etag for this logical view.
            This may be sent on update requests to ensure
            that the client has an up-to-date value before
            proceeding. The server returns an ABORTED error
            on a mismatched etag.
        deletion_protection (bool):
            Optional. Set to true to make the LogicalView
            protected against deletion.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )
    deletion_protection: bool = proto.Field(
        proto.BOOL,
        number=6,
    )


class MaterializedView(proto.Message):
    r"""A materialized view object that can be referenced in SQL
    queries.

    Attributes:
        name (str):
            Identifier. The unique name of the materialized view.
            Format:
            ``projects/{project}/instances/{instance}/materializedViews/{materialized_view}``
        query (str):
            Required. Immutable. The materialized view's
            select query.
        etag (str):
            Optional. The etag for this materialized
            view. This may be sent on update requests to
            ensure that the client has an up-to-date value
            before proceeding. The server returns an ABORTED
            error on a mismatched etag.
        deletion_protection (bool):
            Set to true to make the MaterializedView
            protected against deletion.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )
    deletion_protection: bool = proto.Field(
        proto.BOOL,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrappers for gapic enum types."""

from google.cloud.bigtable_admin_v2.types import common
from google.cloud.bigtable_admin_v2.types import instance
from google.cloud.bigtable_admin_v2.types import table


class StorageType(object):
    """
    Storage media types for persisting Bigtable data.

    Attributes:
      UNSPECIFIED (int): The user did not specify a storage type.
      SSD (int): Flash (SSD) storage should be used.
      HDD (int): Magnetic drive (HDD) storage should be used.
    """

    UNSPECIFIED = common.StorageType.STORAGE_TYPE_UNSPECIFIED
    SSD = common.StorageType.SSD
    HDD = common.StorageType.HDD


class Instance(object):
    class State(object):
        """
        Possible states of an instance.

        Attributes:
          STATE_NOT_KNOWN (int): The state of the instance could not be
          determined.
          READY (int): The instance has been successfully created and can
          serve requests to its tables.
          CREATING (int): The instance is currently being created, and may be
          destroyed if the creation process encounters an error.
        """

        NOT_KNOWN = instance.Instance.State.STATE_NOT_KNOWN
        READY = instance.Instance.State.READY
        CREATING = instance.Instance.State.CREATING

    class Type(object):
        """
        The type of the instance.

        Attributes:
          UNSPECIFIED (int): The type of the instance is unspecified.
          If set when creating an instance, a ``PRODUCTION`` instance will
          be created. If set when updating an instance, the type will be
          left unchanged.
          PRODUCTION (int): An instance meant for production use.
          ``serve_nodes`` must be set on the cluster.
          DEVELOPMENT (int): The instance is meant for development and testing
          purposes only; it has no performance or uptime guarantees and is not
          covered by SLA.
          After a development instance is created, it can be upgraded by
          updating the instance to type ``PRODUCTION``. An instance created
          as a production instance cannot be changed to a development instance.
          When creating a development instance, ``serve_nodes`` on the cluster
          must not be set.
        """

        UNSPECIFIED = instance.Instance.Type.TYPE_UNSPECIFIED
        PRODUCTION = instance.Instance.Type.PRODUCTION
        DEVELOPMENT = instance.Instance.Type.DEVELOPMENT


class Cluster(object):
    class State(object):
        """
        Possible states of a cluster.

        Attributes:
          NOT_KNOWN (int): The state of the cluster could not be determined.
          READY (int): The cluster has been successfully created and is ready
          to serve requests.
          CREATING (int): The cluster is currently being created, and may be
          destroyed if the creation process encounters an error.
          A cluster may not be able to serve requests while being created.
          RESIZING (int): The cluster is currently being resized, and may
          revert to its previous node count if the process encounters an error.
          A cluster is still capable of serving requests while being resized,
          but may exhibit performance as if its number of allocated nodes is
          between the starting and requested states.
          DISABLED (int): The cluster has no backing nodes. The data (tables)
          still exist, but no operations can be performed on the cluster.
        """

        NOT_KNOWN = instance.Cluster.State.STATE_NOT_KNOWN
        READY = instance.Cluster.State.READY
        CREATING = instance.Cluster.State.CREATING
        RESIZING = instance.Cluster.State.RESIZING
        DISABLED = instance.Cluster.State.DISABLED


class RoutingPolicyType(object):
    """
    The type of the routing policy for app_profile.

    Attributes:
      ANY (int): Read/write requests may be routed to any cluster in the
      instance, and will fail over to another cluster in the event of
      transient errors or delays.
      Choosing this option sacrifices read-your-writes consistency to
      improve availability.
      See
      https://cloud.google.com/bigtable/docs/reference/admin/rpc/google.bigtable.admin.v2#google.bigtable.admin.v2.AppProfile.MultiClusterRoutingUseAny

      SINGLE (int): Unconditionally routes all read/write requests to a
      specific cluster.
      This option preserves read-your-writes consistency, but does not improve
      availability.
      See
      https://cloud.google.com/bigtable/docs/reference/admin/rpc/google.bigtable.admin.v2#google.bigtable.admin.v2.AppProfile.SingleClusterRouting
    """

    ANY = 1
    SINGLE = 2


class Table(object):
    class View(object):
        """
        Defines a view over a table's fields.

        Attributes:
          VIEW_UNSPECIFIED (int): Uses the default view for each method
          as documented in its request.
          NAME_ONLY (int): Only populates ``name``.
          SCHEMA_VIEW (int): Only populates ``name`` and fields related
          to the table's schema.
          REPLICATION_VIEW (int): This is a private alpha release of
          Cloud Bigtable replication. This feature is not currently available
          to most Cloud Bigtable customers. This feature might be changed in
          backward-incompatible ways and is not recommended for production use.
          It is not subject to any SLA or deprecation policy.

          Only populates ``name`` and fields related to the table's
          replication state.
          FULL (int): Populates all fields.
        """

        VIEW_UNSPECIFIED = table.Table.View.VIEW_UNSPECIFIED
        NAME_ONLY = table.Table.View.NAME_ONLY
        SCHEMA_VIEW = table.Table.View.SCHEMA_VIEW
        REPLICATION_VIEW = table.Table.View.REPLICATION_VIEW
        ENCRYPTION_VIEW = table.Table.View.ENCRYPTION_VIEW
        FULL = table.Table.View.FULL

    class ReplicationState(object):
        """
        Table replication states.

        Attributes:
          STATE_NOT_KNOWN (int): The replication state of the table is unknown
           in this cluster.
          INITIALIZING (int): The cluster was recently created, and the table
           must finish copying
          over pre-existing data from other clusters before it can begin
          receiving live replication updates and serving
          ``Data API`` requests.
          PLANNED_MAINTENANCE (int): The table is temporarily unable to serve
          ``Data API`` requests from this
          cluster due to planned internal maintenance.
          UNPLANNED_MAINTENANCE (int): The table is temporarily unable to serve
          ``Data API`` requests from this
          cluster due to unplanned or emergency maintenance.
          READY (int): The table can serve
          ``Data API`` requests from this
          cluster. Depending on replication delay, reads may not immediately
          reflect the state of the table in other clusters.
        """

        STATE_NOT_KNOWN = table.Table.ClusterState.ReplicationState.STATE_NOT_KNOWN
        INITIALIZING = table.Table.ClusterState.ReplicationState.INITIALIZING
        PLANNED_MAINTENANCE = (
            table.Table.ClusterState.ReplicationState.PLANNED_MAINTENANCE
        )
        UNPLANNED_MAINTENANCE = (
            table.Table.ClusterState.ReplicationState.UNPLANNED_MAINTENANCE
        )
        READY = table.Table.ClusterState.ReplicationState.READY


class EncryptionInfo:
    class EncryptionType:
        """Possible encryption types for a resource.

        Attributes:
            ENCRYPTION_TYPE_UNSPECIFIED (int): Encryption type was not specified, though
                data at rest remains encrypted.
            GOOGLE_DEFAULT_ENCRYPTION (int): The data backing this resource is encrypted
                at rest with a key that is fully managed by Google. No key version or
                status will be populated. This is the default state.
            CUSTOMER_MANAGED_ENCRYPTION (int): The data backing this resource is
                encrypted at rest with a key that is managed by the customer. The in-use
                version of the key and its status are populated for CMEK-protected
                tables. CMEK-protected backups are pinned to the key version that was in
                use at the time the backup was taken. This key version is populated but
                its status is not tracked and is reported as `UNKNOWN`.
        """

        ENCRYPTION_TYPE_UNSPECIFIED = (
            table.EncryptionInfo.EncryptionType.ENCRYPTION_TYPE_UNSPECIFIED
        )
        GOOGLE_DEFAULT_ENCRYPTION = (
            table.EncryptionInfo.EncryptionType.GOOGLE_DEFAULT_ENCRYPTION
        )
        CUSTOMER_MANAGED_ENCRYPTION = (
            table.EncryptionInfo.EncryptionType.CUSTOMER_MANAGED_ENCRYPTION
        )

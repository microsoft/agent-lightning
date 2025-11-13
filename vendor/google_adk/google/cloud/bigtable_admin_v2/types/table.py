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

from google.cloud.bigtable_admin_v2.types import types
from google.cloud.bigtable_admin_v2.utils import oneof_message
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.admin.v2",
    manifest={
        "RestoreSourceType",
        "RestoreInfo",
        "ChangeStreamConfig",
        "Table",
        "AuthorizedView",
        "ColumnFamily",
        "GcRule",
        "EncryptionInfo",
        "Snapshot",
        "Backup",
        "BackupInfo",
        "ProtoSchema",
        "SchemaBundle",
    },
)


class RestoreSourceType(proto.Enum):
    r"""Indicates the type of the restore source.

    Values:
        RESTORE_SOURCE_TYPE_UNSPECIFIED (0):
            No restore associated.
        BACKUP (1):
            A backup was used as the source of the
            restore.
    """
    RESTORE_SOURCE_TYPE_UNSPECIFIED = 0
    BACKUP = 1


class RestoreInfo(proto.Message):
    r"""Information about a table restore.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        source_type (google.cloud.bigtable_admin_v2.types.RestoreSourceType):
            The type of the restore source.
        backup_info (google.cloud.bigtable_admin_v2.types.BackupInfo):
            Information about the backup used to restore
            the table. The backup may no longer exist.

            This field is a member of `oneof`_ ``source_info``.
    """

    source_type: "RestoreSourceType" = proto.Field(
        proto.ENUM,
        number=1,
        enum="RestoreSourceType",
    )
    backup_info: "BackupInfo" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source_info",
        message="BackupInfo",
    )


class ChangeStreamConfig(proto.Message):
    r"""Change stream configuration.

    Attributes:
        retention_period (google.protobuf.duration_pb2.Duration):
            How long the change stream should be
            retained. Change stream data older than the
            retention period will not be returned when
            reading the change stream from the table.
            Values must be at least 1 day and at most 7
            days, and will be truncated to microsecond
            granularity.
    """

    retention_period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )


class Table(proto.Message):
    r"""A collection of user data indexed by row, column, and
    timestamp. Each table is served using the resources of its
    parent cluster.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            The unique name of the table. Values are of the form
            ``projects/{project}/instances/{instance}/tables/[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
            Views: ``NAME_ONLY``, ``SCHEMA_VIEW``, ``REPLICATION_VIEW``,
            ``FULL``
        cluster_states (MutableMapping[str, google.cloud.bigtable_admin_v2.types.Table.ClusterState]):
            Output only. Map from cluster ID to per-cluster table state.
            If it could not be determined whether or not the table has
            data in a particular cluster (for example, if its zone is
            unavailable), then there will be an entry for the cluster
            with UNKNOWN ``replication_status``. Views:
            ``REPLICATION_VIEW``, ``ENCRYPTION_VIEW``, ``FULL``
        column_families (MutableMapping[str, google.cloud.bigtable_admin_v2.types.ColumnFamily]):
            The column families configured for this table, mapped by
            column family ID. Views: ``SCHEMA_VIEW``, ``STATS_VIEW``,
            ``FULL``
        granularity (google.cloud.bigtable_admin_v2.types.Table.TimestampGranularity):
            Immutable. The granularity (i.e. ``MILLIS``) at which
            timestamps are stored in this table. Timestamps not matching
            the granularity will be rejected. If unspecified at creation
            time, the value will be set to ``MILLIS``. Views:
            ``SCHEMA_VIEW``, ``FULL``.
        restore_info (google.cloud.bigtable_admin_v2.types.RestoreInfo):
            Output only. If this table was restored from
            another data source (e.g. a backup), this field
            will be populated with information about the
            restore.
        change_stream_config (google.cloud.bigtable_admin_v2.types.ChangeStreamConfig):
            If specified, enable the change stream on
            this table. Otherwise, the change stream is
            disabled and the change stream is not retained.
        deletion_protection (bool):
            Set to true to make the table protected against data loss.
            i.e. deleting the following resources through Admin APIs are
            prohibited:

            -  The table.
            -  The column families in the table.
            -  The instance containing the table.

            Note one can still delete the data stored in the table
            through Data APIs.
        automated_backup_policy (google.cloud.bigtable_admin_v2.types.Table.AutomatedBackupPolicy):
            If specified, automated backups are enabled
            for this table. Otherwise, automated backups are
            disabled.

            This field is a member of `oneof`_ ``automated_backup_config``.
        row_key_schema (google.cloud.bigtable_admin_v2.types.Type.Struct):
            The row key schema for this table. The schema is used to
            decode the raw row key bytes into a structured format. The
            order of field declarations in this schema is important, as
            it reflects how the raw row key bytes are structured.
            Currently, this only affects how the key is read via a
            GoogleSQL query from the ExecuteQuery API.

            For a SQL query, the \_key column is still read as raw
            bytes. But queries can reference the key fields by name,
            which will be decoded from \_key using provided type and
            encoding. Queries that reference key fields will fail if
            they encounter an invalid row key.

            For example, if \_key =
            "some_id#2024-04-30#\x00\x13\x00\xf3" with the following
            schema:

            .. code-block::

                {
                  fields {
                    field_name: "id"
                    type { string { encoding: utf8_bytes {} } }
                  }
                  fields {
                    field_name: "date"
                    type { string { encoding: utf8_bytes {} } }
                  }
                  fields {
                    field_name: "product_code"
                    type { int64 { encoding: big_endian_bytes {} } }
                  }
                  encoding { delimited_bytes { delimiter: "#" } }
                }

            The decoded key parts would be:
            id = "some_id", date = "2024-04-30", product_code = 1245427
            The query "SELECT \_key, product_code FROM table" will return
            two columns:

            +========================================+==============+
            | \_key                                  | product_code |
            +========================================+==============+
            | "some_id#2024-04-30#\x00\x13\x00\xf3"  |    1245427   |
            +----------------------------------------+--------------+

            The schema has the following invariants: (1) The decoded
            field values are order-preserved. For read, the field values
            will be decoded in sorted mode from the raw bytes. (2) Every
            field in the schema must specify a non-empty name. (3) Every
            field must specify a type with an associated encoding. The
            type is limited to scalar types only: Array, Map, Aggregate,
            and Struct are not allowed. (4) The field names must not
            collide with existing column family names and reserved
            keywords "_key" and "_timestamp".

            The following update operations are allowed for
            row_key_schema:

            -  Update from an empty schema to a new schema.
            -  Remove the existing schema. This operation requires
               setting the ``ignore_warnings`` flag to ``true``, since
               it might be a backward incompatible change. Without the
               flag, the update request will fail with an
               INVALID_ARGUMENT error. Any other row key schema update
               operation (e.g. update existing schema columns names or
               types) is currently unsupported.
    """

    class TimestampGranularity(proto.Enum):
        r"""Possible timestamp granularities to use when keeping multiple
        versions of data in a table.

        Values:
            TIMESTAMP_GRANULARITY_UNSPECIFIED (0):
                The user did not specify a granularity.
                Should not be returned. When specified during
                table creation, MILLIS will be used.
            MILLIS (1):
                The table keeps data versioned at a
                granularity of 1ms.
        """
        TIMESTAMP_GRANULARITY_UNSPECIFIED = 0
        MILLIS = 1

    class View(proto.Enum):
        r"""Defines a view over a table's fields.

        Values:
            VIEW_UNSPECIFIED (0):
                Uses the default view for each method as
                documented in its request.
            NAME_ONLY (1):
                Only populates ``name``.
            SCHEMA_VIEW (2):
                Only populates ``name`` and fields related to the table's
                schema.
            REPLICATION_VIEW (3):
                Only populates ``name`` and fields related to the table's
                replication state.
            ENCRYPTION_VIEW (5):
                Only populates ``name`` and fields related to the table's
                encryption state.
            FULL (4):
                Populates all fields.
        """
        VIEW_UNSPECIFIED = 0
        NAME_ONLY = 1
        SCHEMA_VIEW = 2
        REPLICATION_VIEW = 3
        ENCRYPTION_VIEW = 5
        FULL = 4

    class ClusterState(proto.Message):
        r"""The state of a table's data in a particular cluster.

        Attributes:
            replication_state (google.cloud.bigtable_admin_v2.types.Table.ClusterState.ReplicationState):
                Output only. The state of replication for the
                table in this cluster.
            encryption_info (MutableSequence[google.cloud.bigtable_admin_v2.types.EncryptionInfo]):
                Output only. The encryption information for
                the table in this cluster. If the encryption key
                protecting this resource is customer managed,
                then its version can be rotated in Cloud Key
                Management Service (Cloud KMS). The primary
                version of the key and its status will be
                reflected here when changes propagate from Cloud
                KMS.
        """

        class ReplicationState(proto.Enum):
            r"""Table replication states.

            Values:
                STATE_NOT_KNOWN (0):
                    The replication state of the table is unknown
                    in this cluster.
                INITIALIZING (1):
                    The cluster was recently created, and the
                    table must finish copying over pre-existing data
                    from other clusters before it can begin
                    receiving live replication updates and serving
                    Data API requests.
                PLANNED_MAINTENANCE (2):
                    The table is temporarily unable to serve Data
                    API requests from this cluster due to planned
                    internal maintenance.
                UNPLANNED_MAINTENANCE (3):
                    The table is temporarily unable to serve Data
                    API requests from this cluster due to unplanned
                    or emergency maintenance.
                READY (4):
                    The table can serve Data API requests from
                    this cluster. Depending on replication delay,
                    reads may not immediately reflect the state of
                    the table in other clusters.
                READY_OPTIMIZING (5):
                    The table is fully created and ready for use after a
                    restore, and is being optimized for performance. When
                    optimizations are complete, the table will transition to
                    ``READY`` state.
            """
            STATE_NOT_KNOWN = 0
            INITIALIZING = 1
            PLANNED_MAINTENANCE = 2
            UNPLANNED_MAINTENANCE = 3
            READY = 4
            READY_OPTIMIZING = 5

        replication_state: "Table.ClusterState.ReplicationState" = proto.Field(
            proto.ENUM,
            number=1,
            enum="Table.ClusterState.ReplicationState",
        )
        encryption_info: MutableSequence["EncryptionInfo"] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="EncryptionInfo",
        )

    class AutomatedBackupPolicy(proto.Message):
        r"""Defines an automated backup policy for a table

        Attributes:
            retention_period (google.protobuf.duration_pb2.Duration):
                Required. How long the automated backups
                should be retained. The only supported value at
                this time is 3 days.
            frequency (google.protobuf.duration_pb2.Duration):
                Required. How frequently automated backups
                should occur. The only supported value at this
                time is 24 hours.
        """

        retention_period: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=1,
            message=duration_pb2.Duration,
        )
        frequency: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=2,
            message=duration_pb2.Duration,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    cluster_states: MutableMapping[str, ClusterState] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=2,
        message=ClusterState,
    )
    column_families: MutableMapping[str, "ColumnFamily"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=3,
        message="ColumnFamily",
    )
    granularity: TimestampGranularity = proto.Field(
        proto.ENUM,
        number=4,
        enum=TimestampGranularity,
    )
    restore_info: "RestoreInfo" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="RestoreInfo",
    )
    change_stream_config: "ChangeStreamConfig" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="ChangeStreamConfig",
    )
    deletion_protection: bool = proto.Field(
        proto.BOOL,
        number=9,
    )
    automated_backup_policy: AutomatedBackupPolicy = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="automated_backup_config",
        message=AutomatedBackupPolicy,
    )
    row_key_schema: types.Type.Struct = proto.Field(
        proto.MESSAGE,
        number=15,
        message=types.Type.Struct,
    )


class AuthorizedView(proto.Message):
    r"""AuthorizedViews represent subsets of a particular Cloud
    Bigtable table. Users can configure access to each Authorized
    View independently from the table and use the existing Data APIs
    to access the subset of data.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. The name of this AuthorizedView. Values are of
            the form
            ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``
        subset_view (google.cloud.bigtable_admin_v2.types.AuthorizedView.SubsetView):
            An AuthorizedView permitting access to an
            explicit subset of a Table.

            This field is a member of `oneof`_ ``authorized_view``.
        etag (str):
            The etag for this AuthorizedView.
            If this is provided on update, it must match the
            server's etag. The server returns ABORTED error
            on a mismatched etag.
        deletion_protection (bool):
            Set to true to make the AuthorizedView
            protected against deletion. The parent Table and
            containing Instance cannot be deleted if an
            AuthorizedView has this bit set.
    """

    class ResponseView(proto.Enum):
        r"""Defines a subset of an AuthorizedView's fields.

        Values:
            RESPONSE_VIEW_UNSPECIFIED (0):
                Uses the default view for each method as
                documented in the request.
            NAME_ONLY (1):
                Only populates ``name``.
            BASIC (2):
                Only populates the AuthorizedView's basic metadata. This
                includes: name, deletion_protection, etag.
            FULL (3):
                Populates every fields.
        """
        RESPONSE_VIEW_UNSPECIFIED = 0
        NAME_ONLY = 1
        BASIC = 2
        FULL = 3

    class FamilySubsets(proto.Message):
        r"""Subsets of a column family that are included in this
        AuthorizedView.

        Attributes:
            qualifiers (MutableSequence[bytes]):
                Individual exact column qualifiers to be
                included in the AuthorizedView.
            qualifier_prefixes (MutableSequence[bytes]):
                Prefixes for qualifiers to be included in the
                AuthorizedView. Every qualifier starting with
                one of these prefixes is included in the
                AuthorizedView. To provide access to all
                qualifiers, include the empty string as a prefix
                ("").
        """

        qualifiers: MutableSequence[bytes] = proto.RepeatedField(
            proto.BYTES,
            number=1,
        )
        qualifier_prefixes: MutableSequence[bytes] = proto.RepeatedField(
            proto.BYTES,
            number=2,
        )

    class SubsetView(proto.Message):
        r"""Defines a simple AuthorizedView that is a subset of the
        underlying Table.

        Attributes:
            row_prefixes (MutableSequence[bytes]):
                Row prefixes to be included in the
                AuthorizedView. To provide access to all rows,
                include the empty string as a prefix ("").
            family_subsets (MutableMapping[str, google.cloud.bigtable_admin_v2.types.AuthorizedView.FamilySubsets]):
                Map from column family name to the columns in
                this family to be included in the
                AuthorizedView.
        """

        row_prefixes: MutableSequence[bytes] = proto.RepeatedField(
            proto.BYTES,
            number=1,
        )
        family_subsets: MutableMapping[
            str, "AuthorizedView.FamilySubsets"
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=2,
            message="AuthorizedView.FamilySubsets",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    subset_view: SubsetView = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="authorized_view",
        message=SubsetView,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )
    deletion_protection: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class ColumnFamily(proto.Message):
    r"""A set of columns within a table which share a common
    configuration.

    Attributes:
        gc_rule (google.cloud.bigtable_admin_v2.types.GcRule):
            Garbage collection rule specified as a
            protobuf. Must serialize to at most 500 bytes.

            NOTE: Garbage collection executes
            opportunistically in the background, and so it's
            possible for reads to return a cell even if it
            matches the active GC expression for its family.
        value_type (google.cloud.bigtable_admin_v2.types.Type):
            The type of data stored in each of this family's cell
            values, including its full encoding. If omitted, the family
            only serves raw untyped bytes.

            For now, only the ``Aggregate`` type is supported.

            ``Aggregate`` can only be set at family creation and is
            immutable afterwards.

            If ``value_type`` is ``Aggregate``, written data must be
            compatible with:

            -  ``value_type.input_type`` for ``AddInput`` mutations
    """

    gc_rule: "GcRule" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="GcRule",
    )
    value_type: types.Type = proto.Field(
        proto.MESSAGE,
        number=3,
        message=types.Type,
    )


class GcRule(oneof_message.OneofMessage):
    r"""Rule for determining which cells to delete during garbage
    collection.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        max_num_versions (int):
            Delete all cells in a column except the most
            recent N.

            This field is a member of `oneof`_ ``rule``.
        max_age (google.protobuf.duration_pb2.Duration):
            Delete cells in a column older than the given
            age. Values must be at least one millisecond,
            and will be truncated to microsecond
            granularity.

            This field is a member of `oneof`_ ``rule``.
        intersection (google.cloud.bigtable_admin_v2.types.GcRule.Intersection):
            Delete cells that would be deleted by every
            nested rule.

            This field is a member of `oneof`_ ``rule``.
        union (google.cloud.bigtable_admin_v2.types.GcRule.Union):
            Delete cells that would be deleted by any
            nested rule.

            This field is a member of `oneof`_ ``rule``.
    """

    class Intersection(proto.Message):
        r"""A GcRule which deletes cells matching all of the given rules.

        Attributes:
            rules (MutableSequence[google.cloud.bigtable_admin_v2.types.GcRule]):
                Only delete cells which would be deleted by every element of
                ``rules``.
        """

        rules: MutableSequence["GcRule"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="GcRule",
        )

    class Union(proto.Message):
        r"""A GcRule which deletes cells matching any of the given rules.

        Attributes:
            rules (MutableSequence[google.cloud.bigtable_admin_v2.types.GcRule]):
                Delete cells which would be deleted by any element of
                ``rules``.
        """

        rules: MutableSequence["GcRule"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="GcRule",
        )

    max_num_versions: int = proto.Field(
        proto.INT32,
        number=1,
        oneof="rule",
    )
    max_age: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="rule",
        message=duration_pb2.Duration,
    )
    intersection: Intersection = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="rule",
        message=Intersection,
    )
    union: Union = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="rule",
        message=Union,
    )


class EncryptionInfo(proto.Message):
    r"""Encryption information for a given resource.
    If this resource is protected with customer managed encryption,
    the in-use Cloud Key Management Service (Cloud KMS) key version
    is specified along with its status.

    Attributes:
        encryption_type (google.cloud.bigtable_admin_v2.types.EncryptionInfo.EncryptionType):
            Output only. The type of encryption used to
            protect this resource.
        encryption_status (google.rpc.status_pb2.Status):
            Output only. The status of encrypt/decrypt
            calls on underlying data for this resource.
            Regardless of status, the existing data is
            always encrypted at rest.
        kms_key_version (str):
            Output only. The version of the Cloud KMS key
            specified in the parent cluster that is in use
            for the data underlying this table.
    """

    class EncryptionType(proto.Enum):
        r"""Possible encryption types for a resource.

        Values:
            ENCRYPTION_TYPE_UNSPECIFIED (0):
                Encryption type was not specified, though
                data at rest remains encrypted.
            GOOGLE_DEFAULT_ENCRYPTION (1):
                The data backing this resource is encrypted
                at rest with a key that is fully managed by
                Google. No key version or status will be
                populated. This is the default state.
            CUSTOMER_MANAGED_ENCRYPTION (2):
                The data backing this resource is encrypted at rest with a
                key that is managed by the customer. The in-use version of
                the key and its status are populated for CMEK-protected
                tables. CMEK-protected backups are pinned to the key version
                that was in use at the time the backup was taken. This key
                version is populated but its status is not tracked and is
                reported as ``UNKNOWN``.
        """
        ENCRYPTION_TYPE_UNSPECIFIED = 0
        GOOGLE_DEFAULT_ENCRYPTION = 1
        CUSTOMER_MANAGED_ENCRYPTION = 2

    encryption_type: EncryptionType = proto.Field(
        proto.ENUM,
        number=3,
        enum=EncryptionType,
    )
    encryption_status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        message=status_pb2.Status,
    )
    kms_key_version: str = proto.Field(
        proto.STRING,
        number=2,
    )


class Snapshot(proto.Message):
    r"""A snapshot of a table at a particular time. A snapshot can be
    used as a checkpoint for data restoration or a data source for a
    new table.

    Note: This is a private alpha release of Cloud Bigtable
    snapshots. This feature is not currently available to most Cloud
    Bigtable customers. This feature might be changed in
    backward-incompatible ways and is not recommended for production
    use. It is not subject to any SLA or deprecation policy.

    Attributes:
        name (str):
            The unique name of the snapshot. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.
        source_table (google.cloud.bigtable_admin_v2.types.Table):
            Output only. The source table at the time the
            snapshot was taken.
        data_size_bytes (int):
            Output only. The size of the data in the
            source table at the time the snapshot was taken.
            In some cases, this value may be computed
            asynchronously via a background process and a
            placeholder of 0 will be used in the meantime.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the snapshot is
            created.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            The time when the snapshot will be deleted.
            The maximum amount of time a snapshot can stay
            active is 365 days. If 'ttl' is not specified,
            the default maximum of 365 days will be used.
        state (google.cloud.bigtable_admin_v2.types.Snapshot.State):
            Output only. The current state of the
            snapshot.
        description (str):
            Description of the snapshot.
    """

    class State(proto.Enum):
        r"""Possible states of a snapshot.

        Values:
            STATE_NOT_KNOWN (0):
                The state of the snapshot could not be
                determined.
            READY (1):
                The snapshot has been successfully created
                and can serve all requests.
            CREATING (2):
                The snapshot is currently being created, and
                may be destroyed if the creation process
                encounters an error. A snapshot may not be
                restored to a table while it is being created.
        """
        STATE_NOT_KNOWN = 0
        READY = 1
        CREATING = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_table: "Table" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Table",
    )
    data_size_bytes: int = proto.Field(
        proto.INT64,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=6,
        enum=State,
    )
    description: str = proto.Field(
        proto.STRING,
        number=7,
    )


class Backup(proto.Message):
    r"""A backup of a Cloud Bigtable table.

    Attributes:
        name (str):
            A globally unique identifier for the backup which cannot be
            changed. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/ backups/[_a-zA-Z0-9][-_.a-zA-Z0-9]*``
            The final segment of the name must be between 1 and 50
            characters in length.

            The backup is stored in the cluster identified by the prefix
            of the backup name of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
        source_table (str):
            Required. Immutable. Name of the table from which this
            backup was created. This needs to be in the same instance as
            the backup. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{source_table}``.
        source_backup (str):
            Output only. Name of the backup from which
            this backup was copied. If a backup is not
            created by copying a backup, this field will be
            empty. Values are of the form:

            projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Required. The expiration time of the backup. When creating a
            backup or updating its ``expire_time``, the value must be
            greater than the backup creation time by:

            -  At least 6 hours
            -  At most 90 days

            Once the ``expire_time`` has passed, Cloud Bigtable will
            delete the backup.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. ``start_time`` is the time that the backup was
            started (i.e. approximately the time the
            [CreateBackup][google.bigtable.admin.v2.BigtableTableAdmin.CreateBackup]
            request is received). The row data in this backup will be no
            older than this timestamp.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. ``end_time`` is the time that the backup was
            finished. The row data in the backup will be no newer than
            this timestamp.
        size_bytes (int):
            Output only. Size of the backup in bytes.
        state (google.cloud.bigtable_admin_v2.types.Backup.State):
            Output only. The current state of the backup.
        encryption_info (google.cloud.bigtable_admin_v2.types.EncryptionInfo):
            Output only. The encryption information for
            the backup.
        backup_type (google.cloud.bigtable_admin_v2.types.Backup.BackupType):
            Indicates the backup type of the backup.
        hot_to_standard_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the hot backup will be converted to a
            standard backup. Once the ``hot_to_standard_time`` has
            passed, Cloud Bigtable will convert the hot backup to a
            standard backup. This value must be greater than the backup
            creation time by:

            -  At least 24 hours

            This field only applies for hot backups. When creating or
            updating a standard backup, attempting to set this field
            will fail the request.
    """

    class State(proto.Enum):
        r"""Indicates the current state of the backup.

        Values:
            STATE_UNSPECIFIED (0):
                Not specified.
            CREATING (1):
                The pending backup is still being created. Operations on the
                backup may fail with ``FAILED_PRECONDITION`` in this state.
            READY (2):
                The backup is complete and ready for use.
        """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2

    class BackupType(proto.Enum):
        r"""The type of the backup.

        Values:
            BACKUP_TYPE_UNSPECIFIED (0):
                Not specified.
            STANDARD (1):
                The default type for Cloud Bigtable managed
                backups. Supported for backups created in both
                HDD and SSD instances. Requires optimization
                when restored to a table in an SSD instance.
            HOT (2):
                A backup type with faster restore to SSD
                performance. Only supported for backups created
                in SSD instances. A new SSD table restored from
                a hot backup reaches production performance more
                quickly than a standard backup.
        """
        BACKUP_TYPE_UNSPECIFIED = 0
        STANDARD = 1
        HOT = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_table: str = proto.Field(
        proto.STRING,
        number=2,
    )
    source_backup: str = proto.Field(
        proto.STRING,
        number=10,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    size_bytes: int = proto.Field(
        proto.INT64,
        number=6,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=7,
        enum=State,
    )
    encryption_info: "EncryptionInfo" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="EncryptionInfo",
    )
    backup_type: BackupType = proto.Field(
        proto.ENUM,
        number=11,
        enum=BackupType,
    )
    hot_to_standard_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )


class BackupInfo(proto.Message):
    r"""Information about a backup.

    Attributes:
        backup (str):
            Output only. Name of the backup.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time that the backup was
            started. Row data in the backup will be no older
            than this timestamp.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. This time that the backup was
            finished. Row data in the backup will be no
            newer than this timestamp.
        source_table (str):
            Output only. Name of the table the backup was
            created from.
        source_backup (str):
            Output only. Name of the backup from which
            this backup was copied. If a backup is not
            created by copying a backup, this field will be
            empty. Values are of the form:

            projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>
    """

    backup: str = proto.Field(
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
    source_table: str = proto.Field(
        proto.STRING,
        number=4,
    )
    source_backup: str = proto.Field(
        proto.STRING,
        number=10,
    )


class ProtoSchema(proto.Message):
    r"""Represents a protobuf schema.

    Attributes:
        proto_descriptors (bytes):
            Required. Contains a protobuf-serialized
            `google.protobuf.FileDescriptorSet <https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/descriptor.proto>`__,
            which could include multiple proto files. To generate it,
            `install <https://grpc.io/docs/protoc-installation/>`__ and
            run ``protoc`` with ``--include_imports`` and
            ``--descriptor_set_out``. For example, to generate for
            moon/shot/app.proto, run

            ::

               $protoc  --proto_path=/app_path --proto_path=/lib_path \
                        --include_imports \
                        --descriptor_set_out=descriptors.pb \
                        moon/shot/app.proto

            For more details, see protobuffer `self
            description <https://developers.google.com/protocol-buffers/docs/techniques#self-description>`__.
    """

    proto_descriptors: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )


class SchemaBundle(proto.Message):
    r"""A named collection of related schemas.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. The unique name identifying this schema bundle.
            Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
        proto_schema (google.cloud.bigtable_admin_v2.types.ProtoSchema):
            Schema for Protobufs.

            This field is a member of `oneof`_ ``type``.
        etag (str):
            Optional. The etag for this schema bundle.
            This may be sent on update and delete requests
            to ensure the client has an up-to-date value
            before proceeding. The server returns an ABORTED
            error on a mismatched etag.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    proto_schema: "ProtoSchema" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="type",
        message="ProtoSchema",
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

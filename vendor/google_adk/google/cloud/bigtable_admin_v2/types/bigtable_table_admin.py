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
from google.cloud.bigtable_admin_v2.types import table as gba_table
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.admin.v2",
    manifest={
        "RestoreTableRequest",
        "RestoreTableMetadata",
        "OptimizeRestoredTableMetadata",
        "CreateTableRequest",
        "CreateTableFromSnapshotRequest",
        "DropRowRangeRequest",
        "ListTablesRequest",
        "ListTablesResponse",
        "GetTableRequest",
        "UpdateTableRequest",
        "UpdateTableMetadata",
        "DeleteTableRequest",
        "UndeleteTableRequest",
        "UndeleteTableMetadata",
        "ModifyColumnFamiliesRequest",
        "GenerateConsistencyTokenRequest",
        "GenerateConsistencyTokenResponse",
        "CheckConsistencyRequest",
        "StandardReadRemoteWrites",
        "DataBoostReadLocalWrites",
        "CheckConsistencyResponse",
        "SnapshotTableRequest",
        "GetSnapshotRequest",
        "ListSnapshotsRequest",
        "ListSnapshotsResponse",
        "DeleteSnapshotRequest",
        "SnapshotTableMetadata",
        "CreateTableFromSnapshotMetadata",
        "CreateBackupRequest",
        "CreateBackupMetadata",
        "UpdateBackupRequest",
        "GetBackupRequest",
        "DeleteBackupRequest",
        "ListBackupsRequest",
        "ListBackupsResponse",
        "CopyBackupRequest",
        "CopyBackupMetadata",
        "CreateAuthorizedViewRequest",
        "CreateAuthorizedViewMetadata",
        "ListAuthorizedViewsRequest",
        "ListAuthorizedViewsResponse",
        "GetAuthorizedViewRequest",
        "UpdateAuthorizedViewRequest",
        "UpdateAuthorizedViewMetadata",
        "DeleteAuthorizedViewRequest",
        "CreateSchemaBundleRequest",
        "CreateSchemaBundleMetadata",
        "UpdateSchemaBundleRequest",
        "UpdateSchemaBundleMetadata",
        "GetSchemaBundleRequest",
        "ListSchemaBundlesRequest",
        "ListSchemaBundlesResponse",
        "DeleteSchemaBundleRequest",
    },
)


class RestoreTableRequest(proto.Message):
    r"""The request for
    [RestoreTable][google.bigtable.admin.v2.BigtableTableAdmin.RestoreTable].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. The name of the instance in which to create the
            restored table. Values are of the form
            ``projects/<project>/instances/<instance>``.
        table_id (str):
            Required. The id of the table to create and restore to. This
            table must not already exist. The ``table_id`` appended to
            ``parent`` forms the full table name of the form
            ``projects/<project>/instances/<instance>/tables/<table_id>``.
        backup (str):
            Name of the backup from which to restore. Values are of the
            form
            ``projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>``.

            This field is a member of `oneof`_ ``source``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    backup: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="source",
    )


class RestoreTableMetadata(proto.Message):
    r"""Metadata type for the long-running operation returned by
    [RestoreTable][google.bigtable.admin.v2.BigtableTableAdmin.RestoreTable].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Name of the table being created and restored
            to.
        source_type (google.cloud.bigtable_admin_v2.types.RestoreSourceType):
            The type of the restore source.
        backup_info (google.cloud.bigtable_admin_v2.types.BackupInfo):

            This field is a member of `oneof`_ ``source_info``.
        optimize_table_operation_name (str):
            If exists, the name of the long-running operation that will
            be used to track the post-restore optimization process to
            optimize the performance of the restored table. The metadata
            type of the long-running operation is
            [OptimizeRestoreTableMetadata][]. The response type is
            [Empty][google.protobuf.Empty]. This long-running operation
            may be automatically created by the system if applicable
            after the RestoreTable long-running operation completes
            successfully. This operation may not be created if the table
            is already optimized or the restore was not successful.
        progress (google.cloud.bigtable_admin_v2.types.OperationProgress):
            The progress of the
            [RestoreTable][google.bigtable.admin.v2.BigtableTableAdmin.RestoreTable]
            operation.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_type: gba_table.RestoreSourceType = proto.Field(
        proto.ENUM,
        number=2,
        enum=gba_table.RestoreSourceType,
    )
    backup_info: gba_table.BackupInfo = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source_info",
        message=gba_table.BackupInfo,
    )
    optimize_table_operation_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    progress: common.OperationProgress = proto.Field(
        proto.MESSAGE,
        number=5,
        message=common.OperationProgress,
    )


class OptimizeRestoredTableMetadata(proto.Message):
    r"""Metadata type for the long-running operation used to track
    the progress of optimizations performed on a newly restored
    table. This long-running operation is automatically created by
    the system after the successful completion of a table restore,
    and cannot be cancelled.

    Attributes:
        name (str):
            Name of the restored table being optimized.
        progress (google.cloud.bigtable_admin_v2.types.OperationProgress):
            The progress of the post-restore
            optimizations.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    progress: common.OperationProgress = proto.Field(
        proto.MESSAGE,
        number=2,
        message=common.OperationProgress,
    )


class CreateTableRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.CreateTable][google.bigtable.admin.v2.BigtableTableAdmin.CreateTable]

    Attributes:
        parent (str):
            Required. The unique name of the instance in which to create
            the table. Values are of the form
            ``projects/{project}/instances/{instance}``.
        table_id (str):
            Required. The name by which the new table should be referred
            to within the parent instance, e.g., ``foobar`` rather than
            ``{parent}/tables/foobar``. Maximum 50 characters.
        table (google.cloud.bigtable_admin_v2.types.Table):
            Required. The Table to create.
        initial_splits (MutableSequence[google.cloud.bigtable_admin_v2.types.CreateTableRequest.Split]):
            The optional list of row keys that will be used to initially
            split the table into several tablets (tablets are similar to
            HBase regions). Given two split keys, ``s1`` and ``s2``,
            three tablets will be created, spanning the key ranges:
            ``[, s1), [s1, s2), [s2, )``.

            Example:

            -  Row keys :=
               ``["a", "apple", "custom", "customer_1", "customer_2",``
               ``"other", "zz"]``
            -  initial_split_keys :=
               ``["apple", "customer_1", "customer_2", "other"]``
            -  Key assignment:

               -  Tablet 1 ``[, apple) => {"a"}.``
               -  Tablet 2
                  ``[apple, customer_1) => {"apple", "custom"}.``
               -  Tablet 3
                  ``[customer_1, customer_2) => {"customer_1"}.``
               -  Tablet 4 ``[customer_2, other) => {"customer_2"}.``
               -  Tablet 5 ``[other, ) => {"other", "zz"}.``
    """

    class Split(proto.Message):
        r"""An initial split point for a newly created table.

        Attributes:
            key (bytes):
                Row key to use as an initial tablet boundary.
        """

        key: bytes = proto.Field(
            proto.BYTES,
            number=1,
        )

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    table: gba_table.Table = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_table.Table,
    )
    initial_splits: MutableSequence[Split] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Split,
    )


class CreateTableFromSnapshotRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.CreateTableFromSnapshot]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        parent (str):
            Required. The unique name of the instance in which to create
            the table. Values are of the form
            ``projects/{project}/instances/{instance}``.
        table_id (str):
            Required. The name by which the new table should be referred
            to within the parent instance, e.g., ``foobar`` rather than
            ``{parent}/tables/foobar``.
        source_snapshot (str):
            Required. The unique name of the snapshot from which to
            restore the table. The snapshot and the table must be in the
            same instance. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    source_snapshot: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DropRowRangeRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange][google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange]

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The unique name of the table on which to drop a
            range of rows. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        row_key_prefix (bytes):
            Delete all rows that start with this row key
            prefix. Prefix cannot be zero length.

            This field is a member of `oneof`_ ``target``.
        delete_all_data_from_table (bool):
            Delete all rows in the table. Setting this to
            false is a no-op.

            This field is a member of `oneof`_ ``target``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    row_key_prefix: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="target",
    )
    delete_all_data_from_table: bool = proto.Field(
        proto.BOOL,
        number=3,
        oneof="target",
    )


class ListTablesRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]

    Attributes:
        parent (str):
            Required. The unique name of the instance for which tables
            should be listed. Values are of the form
            ``projects/{project}/instances/{instance}``.
        view (google.cloud.bigtable_admin_v2.types.Table.View):
            The view to be applied to the returned tables' fields.
            NAME_ONLY view (default) and REPLICATION_VIEW are supported.
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
    view: gba_table.Table.View = proto.Field(
        proto.ENUM,
        number=2,
        enum=gba_table.Table.View,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=4,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListTablesResponse(proto.Message):
    r"""Response message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListTables][google.bigtable.admin.v2.BigtableTableAdmin.ListTables]

    Attributes:
        tables (MutableSequence[google.cloud.bigtable_admin_v2.types.Table]):
            The tables present in the requested instance.
        next_page_token (str):
            Set if not all tables could be returned in a single
            response. Pass this value to ``page_token`` in another
            request to get the next page of results.
    """

    @property
    def raw_page(self):
        return self

    tables: MutableSequence[gba_table.Table] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_table.Table,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetTableRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.GetTable][google.bigtable.admin.v2.BigtableTableAdmin.GetTable]

    Attributes:
        name (str):
            Required. The unique name of the requested table. Values are
            of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        view (google.cloud.bigtable_admin_v2.types.Table.View):
            The view to be applied to the returned table's fields.
            Defaults to ``SCHEMA_VIEW`` if unspecified.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    view: gba_table.Table.View = proto.Field(
        proto.ENUM,
        number=2,
        enum=gba_table.Table.View,
    )


class UpdateTableRequest(proto.Message):
    r"""The request for
    [UpdateTable][google.bigtable.admin.v2.BigtableTableAdmin.UpdateTable].

    Attributes:
        table (google.cloud.bigtable_admin_v2.types.Table):
            Required. The table to update. The table's ``name`` field is
            used to identify the table to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update. A mask specifying
            which fields (e.g. ``change_stream_config``) in the
            ``table`` field should be updated. This mask is relative to
            the ``table`` field, not to the request message. The
            wildcard (*) path is currently not supported. Currently
            UpdateTable is only supported for the following fields:

            -  ``change_stream_config``
            -  ``change_stream_config.retention_period``
            -  ``deletion_protection``
            -  ``row_key_schema``

            If ``column_families`` is set in ``update_mask``, it will
            return an UNIMPLEMENTED error.
        ignore_warnings (bool):
            Optional. If true, ignore safety checks when
            updating the table.
    """

    table: gba_table.Table = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_table.Table,
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


class UpdateTableMetadata(proto.Message):
    r"""Metadata type for the operation returned by
    [UpdateTable][google.bigtable.admin.v2.BigtableTableAdmin.UpdateTable].

    Attributes:
        name (str):
            The name of the table being updated.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    name: str = proto.Field(
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


class DeleteTableRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.DeleteTable]

    Attributes:
        name (str):
            Required. The unique name of the table to be deleted. Values
            are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UndeleteTableRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable]

    Attributes:
        name (str):
            Required. The unique name of the table to be restored.
            Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UndeleteTableMetadata(proto.Message):
    r"""Metadata type for the operation returned by
    [google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable][google.bigtable.admin.v2.BigtableTableAdmin.UndeleteTable].

    Attributes:
        name (str):
            The name of the table being restored.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was cancelled.
    """

    name: str = proto.Field(
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


class ModifyColumnFamiliesRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies][google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies]

    Attributes:
        name (str):
            Required. The unique name of the table whose families should
            be modified. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        modifications (MutableSequence[google.cloud.bigtable_admin_v2.types.ModifyColumnFamiliesRequest.Modification]):
            Required. Modifications to be atomically
            applied to the specified table's families.
            Entries are applied in order, meaning that
            earlier modifications can be masked by later
            ones (in the case of repeated updates to the
            same family, for example).
        ignore_warnings (bool):
            Optional. If true, ignore safety checks when
            modifying the column families.
    """

    class Modification(proto.Message):
        r"""A create, update, or delete of a particular column family.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            id (str):
                The ID of the column family to be modified.
            create (google.cloud.bigtable_admin_v2.types.ColumnFamily):
                Create a new column family with the specified
                schema, or fail if one already exists with the
                given ID.

                This field is a member of `oneof`_ ``mod``.
            update (google.cloud.bigtable_admin_v2.types.ColumnFamily):
                Update an existing column family to the
                specified schema, or fail if no column family
                exists with the given ID.

                This field is a member of `oneof`_ ``mod``.
            drop (bool):
                Drop (delete) the column family with the
                given ID, or fail if no such family exists.

                This field is a member of `oneof`_ ``mod``.
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Optional. A mask specifying which fields (e.g. ``gc_rule``)
                in the ``update`` mod should be updated, ignored for other
                modification types. If unset or empty, we treat it as
                updating ``gc_rule`` to be backward compatible.
        """

        id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        create: gba_table.ColumnFamily = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="mod",
            message=gba_table.ColumnFamily,
        )
        update: gba_table.ColumnFamily = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="mod",
            message=gba_table.ColumnFamily,
        )
        drop: bool = proto.Field(
            proto.BOOL,
            number=4,
            oneof="mod",
        )
        update_mask: field_mask_pb2.FieldMask = proto.Field(
            proto.MESSAGE,
            number=6,
            message=field_mask_pb2.FieldMask,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    modifications: MutableSequence[Modification] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=Modification,
    )
    ignore_warnings: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class GenerateConsistencyTokenRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]

    Attributes:
        name (str):
            Required. The unique name of the Table for which to create a
            consistency token. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GenerateConsistencyTokenResponse(proto.Message):
    r"""Response message for
    [google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken][google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken]

    Attributes:
        consistency_token (str):
            The generated consistency token.
    """

    consistency_token: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CheckConsistencyRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The unique name of the Table for which to check
            replication consistency. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        consistency_token (str):
            Required. The token created using
            GenerateConsistencyToken for the Table.
        standard_read_remote_writes (google.cloud.bigtable_admin_v2.types.StandardReadRemoteWrites):
            Checks that reads using an app profile with
            ``StandardIsolation`` can see all writes committed before
            the token was created, even if the read and write target
            different clusters.

            This field is a member of `oneof`_ ``mode``.
        data_boost_read_local_writes (google.cloud.bigtable_admin_v2.types.DataBoostReadLocalWrites):
            Checks that reads using an app profile with
            ``DataBoostIsolationReadOnly`` can see all writes committed
            before the token was created, but only if the read and write
            target the same cluster.

            This field is a member of `oneof`_ ``mode``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    consistency_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    standard_read_remote_writes: "StandardReadRemoteWrites" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="mode",
        message="StandardReadRemoteWrites",
    )
    data_boost_read_local_writes: "DataBoostReadLocalWrites" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="mode",
        message="DataBoostReadLocalWrites",
    )


class StandardReadRemoteWrites(proto.Message):
    r"""Checks that all writes before the consistency token was
    generated are replicated in every cluster and readable.

    """


class DataBoostReadLocalWrites(proto.Message):
    r"""Checks that all writes before the consistency token was
    generated in the same cluster are readable by Databoost.

    """


class CheckConsistencyResponse(proto.Message):
    r"""Response message for
    [google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency][google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency]

    Attributes:
        consistent (bool):
            True only if the token is consistent. A token
            is consistent if replication has caught up with
            the restrictions specified in the request.
    """

    consistent: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class SnapshotTableRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable][google.bigtable.admin.v2.BigtableTableAdmin.SnapshotTable]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        name (str):
            Required. The unique name of the table to have the snapshot
            taken. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        cluster (str):
            Required. The name of the cluster where the snapshot will be
            created in. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
        snapshot_id (str):
            Required. The ID by which the new snapshot should be
            referred to within the parent cluster, e.g., ``mysnapshot``
            of the form: ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*`` rather than
            ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/mysnapshot``.
        ttl (google.protobuf.duration_pb2.Duration):
            The amount of time that the new snapshot can
            stay active after it is created. Once 'ttl'
            expires, the snapshot will get deleted. The
            maximum amount of time a snapshot can stay
            active is 7 days. If 'ttl' is not specified, the
            default value of 24 hours will be used.
        description (str):
            Description of the snapshot.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    cluster: str = proto.Field(
        proto.STRING,
        number=2,
    )
    snapshot_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    ttl: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=4,
        message=duration_pb2.Duration,
    )
    description: str = proto.Field(
        proto.STRING,
        number=5,
    )


class GetSnapshotRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.GetSnapshot]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        name (str):
            Required. The unique name of the requested snapshot. Values
            are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSnapshotsRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        parent (str):
            Required. The unique name of the cluster for which snapshots
            should be listed. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
            Use ``{cluster} = '-'`` to list snapshots for all clusters
            in an instance, e.g.,
            ``projects/{project}/instances/{instance}/clusters/-``.
        page_size (int):
            The maximum number of snapshots to return per
            page. CURRENTLY UNIMPLEMENTED AND IGNORED.
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
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListSnapshotsResponse(proto.Message):
    r"""Response message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots][google.bigtable.admin.v2.BigtableTableAdmin.ListSnapshots]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        snapshots (MutableSequence[google.cloud.bigtable_admin_v2.types.Snapshot]):
            The snapshots present in the requested
            cluster.
        next_page_token (str):
            Set if not all snapshots could be returned in a single
            response. Pass this value to ``page_token`` in another
            request to get the next page of results.
    """

    @property
    def raw_page(self):
        return self

    snapshots: MutableSequence[gba_table.Snapshot] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_table.Snapshot,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteSnapshotRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSnapshot]

    Note: This is a private alpha release of Cloud Bigtable snapshots.
    This feature is not currently available to most Cloud Bigtable
    customers. This feature might be changed in backward-incompatible
    ways and is not recommended for production use. It is not subject to
    any SLA or deprecation policy.

    Attributes:
        name (str):
            Required. The unique name of the snapshot to be deleted.
            Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/snapshots/{snapshot}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class SnapshotTableMetadata(proto.Message):
    r"""The metadata for the Operation returned by SnapshotTable.

    Note: This is a private alpha release of Cloud Bigtable
    snapshots. This feature is not currently available to most Cloud
    Bigtable customers. This feature might be changed in
    backward-incompatible ways and is not recommended for production
    use. It is not subject to any SLA or deprecation policy.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.SnapshotTableRequest):
            The request that prompted the initiation of
            this SnapshotTable operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "SnapshotTableRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SnapshotTableRequest",
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


class CreateTableFromSnapshotMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    CreateTableFromSnapshot.
    Note: This is a private alpha release of Cloud Bigtable
    snapshots. This feature is not currently available to most Cloud
    Bigtable customers. This feature might be changed in
    backward-incompatible ways and is not recommended for production
    use. It is not subject to any SLA or deprecation policy.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateTableFromSnapshotRequest):
            The request that prompted the initiation of
            this CreateTableFromSnapshot operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "CreateTableFromSnapshotRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateTableFromSnapshotRequest",
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


class CreateBackupRequest(proto.Message):
    r"""The request for
    [CreateBackup][google.bigtable.admin.v2.BigtableTableAdmin.CreateBackup].

    Attributes:
        parent (str):
            Required. This must be one of the clusters in the instance
            in which this table is located. The backup will be stored in
            this cluster. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
        backup_id (str):
            Required. The id of the backup to be created. The
            ``backup_id`` along with the parent ``parent`` are combined
            as {parent}/backups/{backup_id} to create the full backup
            name, of the form:
            ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup_id}``.
            This string must be between 1 and 50 characters in length
            and match the regex [*a-zA-Z0-9][-*.a-zA-Z0-9]*.
        backup (google.cloud.bigtable_admin_v2.types.Backup):
            Required. The backup to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    backup_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    backup: gba_table.Backup = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_table.Backup,
    )


class CreateBackupMetadata(proto.Message):
    r"""Metadata type for the operation returned by
    [CreateBackup][google.bigtable.admin.v2.BigtableTableAdmin.CreateBackup].

    Attributes:
        name (str):
            The name of the backup being created.
        source_table (str):
            The name of the table the backup is created
            from.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was cancelled.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_table: str = proto.Field(
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


class UpdateBackupRequest(proto.Message):
    r"""The request for
    [UpdateBackup][google.bigtable.admin.v2.BigtableTableAdmin.UpdateBackup].

    Attributes:
        backup (google.cloud.bigtable_admin_v2.types.Backup):
            Required. The backup to update. ``backup.name``, and the
            fields to be updated as specified by ``update_mask`` are
            required. Other fields are ignored. Update is only supported
            for the following fields:

            -  ``backup.expire_time``.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. A mask specifying which fields (e.g.
            ``expire_time``) in the Backup resource should be updated.
            This mask is relative to the Backup resource, not to the
            request message. The field mask must always be specified;
            this prevents any future fields from being erased
            accidentally by clients that do not know about them.
    """

    backup: gba_table.Backup = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_table.Backup,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class GetBackupRequest(proto.Message):
    r"""The request for
    [GetBackup][google.bigtable.admin.v2.BigtableTableAdmin.GetBackup].

    Attributes:
        name (str):
            Required. Name of the backup. Values are of the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteBackupRequest(proto.Message):
    r"""The request for
    [DeleteBackup][google.bigtable.admin.v2.BigtableTableAdmin.DeleteBackup].

    Attributes:
        name (str):
            Required. Name of the backup to delete. Values are of the
            form
            ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListBackupsRequest(proto.Message):
    r"""The request for
    [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].

    Attributes:
        parent (str):
            Required. The cluster to list backups from. Values are of
            the form
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
            Use ``{cluster} = '-'`` to list backups for all clusters in
            an instance, e.g.,
            ``projects/{project}/instances/{instance}/clusters/-``.
        filter (str):
            A filter expression that filters backups listed in the
            response. The expression must specify the field name, a
            comparison operator, and the value that you want to use for
            filtering. The value must be a string, a number, or a
            boolean. The comparison operator must be <, >, <=, >=, !=,
            =, or :. Colon ':' represents a HAS operator which is
            roughly synonymous with equality. Filter rules are case
            insensitive.

            The fields eligible for filtering are:

            -  ``name``
            -  ``source_table``
            -  ``state``
            -  ``start_time`` (and values are of the format
               YYYY-MM-DDTHH:MM:SSZ)
            -  ``end_time`` (and values are of the format
               YYYY-MM-DDTHH:MM:SSZ)
            -  ``expire_time`` (and values are of the format
               YYYY-MM-DDTHH:MM:SSZ)
            -  ``size_bytes``

            To filter on multiple expressions, provide each separate
            expression within parentheses. By default, each expression
            is an AND expression. However, you can include AND, OR, and
            NOT expressions explicitly.

            Some examples of using filters are:

            -  ``name:"exact"`` --> The backup's name is the string
               "exact".
            -  ``name:howl`` --> The backup's name contains the string
               "howl".
            -  ``source_table:prod`` --> The source_table's name
               contains the string "prod".
            -  ``state:CREATING`` --> The backup is pending creation.
            -  ``state:READY`` --> The backup is fully created and ready
               for use.
            -  ``(name:howl) AND (start_time < \"2018-03-28T14:50:00Z\")``
               --> The backup name contains the string "howl" and
               start_time of the backup is before 2018-03-28T14:50:00Z.
            -  ``size_bytes > 10000000000`` --> The backup's size is
               greater than 10GB
        order_by (str):
            An expression for specifying the sort order of the results
            of the request. The string value should specify one or more
            fields in [Backup][google.bigtable.admin.v2.Backup]. The
            full syntax is described at https://aip.dev/132#ordering.

            Fields supported are:

            -  name
            -  source_table
            -  expire_time
            -  start_time
            -  end_time
            -  size_bytes
            -  state

            For example, "start_time". The default sorting order is
            ascending. To specify descending order for the field, a
            suffix " desc" should be appended to the field name. For
            example, "start_time desc". Redundant space characters in
            the syntax are insigificant.

            If order_by is empty, results will be sorted by
            ``start_time`` in descending order starting from the most
            recently created backup.
        page_size (int):
            Number of backups to be returned in the
            response. If 0 or less, defaults to the server's
            maximum allowed page size.
        page_token (str):
            If non-empty, ``page_token`` should contain a
            [next_page_token][google.bigtable.admin.v2.ListBackupsResponse.next_page_token]
            from a previous
            [ListBackupsResponse][google.bigtable.admin.v2.ListBackupsResponse]
            to the same ``parent`` and with the same ``filter``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=3,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=4,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListBackupsResponse(proto.Message):
    r"""The response for
    [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups].

    Attributes:
        backups (MutableSequence[google.cloud.bigtable_admin_v2.types.Backup]):
            The list of matching backups.
        next_page_token (str):
            ``next_page_token`` can be sent in a subsequent
            [ListBackups][google.bigtable.admin.v2.BigtableTableAdmin.ListBackups]
            call to fetch more of the matching backups.
    """

    @property
    def raw_page(self):
        return self

    backups: MutableSequence[gba_table.Backup] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_table.Backup,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CopyBackupRequest(proto.Message):
    r"""The request for
    [CopyBackup][google.bigtable.admin.v2.BigtableTableAdmin.CopyBackup].

    Attributes:
        parent (str):
            Required. The name of the destination cluster that will
            contain the backup copy. The cluster must already exist.
            Values are of the form:
            ``projects/{project}/instances/{instance}/clusters/{cluster}``.
        backup_id (str):
            Required. The id of the new backup. The ``backup_id`` along
            with ``parent`` are combined as {parent}/backups/{backup_id}
            to create the full backup name, of the form:
            ``projects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup_id}``.
            This string must be between 1 and 50 characters in length
            and match the regex [*a-zA-Z0-9][-*.a-zA-Z0-9]*.
        source_backup (str):
            Required. The source backup to be copied from. The source
            backup needs to be in READY state for it to be copied.
            Copying a copied backup is not allowed. Once CopyBackup is
            in progress, the source backup cannot be deleted or cleaned
            up on expiration until CopyBackup is finished. Values are of
            the form:
            ``projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>``.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Required. Required. The expiration time of the copied backup
            with microsecond granularity that must be at least 6 hours
            and at most 30 days from the time the request is received.
            Once the ``expire_time`` has passed, Cloud Bigtable will
            delete the backup and free the resources used by the backup.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    backup_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    source_backup: str = proto.Field(
        proto.STRING,
        number=3,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


class CopyBackupMetadata(proto.Message):
    r"""Metadata type for the google.longrunning.Operation returned by
    [CopyBackup][google.bigtable.admin.v2.BigtableTableAdmin.CopyBackup].

    Attributes:
        name (str):
            The name of the backup being created through the copy
            operation. Values are of the form
            ``projects/<project>/instances/<instance>/clusters/<cluster>/backups/<backup>``.
        source_backup_info (google.cloud.bigtable_admin_v2.types.BackupInfo):
            Information about the source backup that is
            being copied from.
        progress (google.cloud.bigtable_admin_v2.types.OperationProgress):
            The progress of the
            [CopyBackup][google.bigtable.admin.v2.BigtableTableAdmin.CopyBackup]
            operation.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_backup_info: gba_table.BackupInfo = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gba_table.BackupInfo,
    )
    progress: common.OperationProgress = proto.Field(
        proto.MESSAGE,
        number=3,
        message=common.OperationProgress,
    )


class CreateAuthorizedViewRequest(proto.Message):
    r"""The request for
    [CreateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.CreateAuthorizedView]

    Attributes:
        parent (str):
            Required. This is the name of the table the AuthorizedView
            belongs to. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        authorized_view_id (str):
            Required. The id of the AuthorizedView to create. This
            AuthorizedView must not already exist. The
            ``authorized_view_id`` appended to ``parent`` forms the full
            AuthorizedView name of the form
            ``projects/{project}/instances/{instance}/tables/{table}/authorizedView/{authorized_view}``.
        authorized_view (google.cloud.bigtable_admin_v2.types.AuthorizedView):
            Required. The AuthorizedView to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    authorized_view: gba_table.AuthorizedView = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_table.AuthorizedView,
    )


class CreateAuthorizedViewMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    CreateAuthorizedView.

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.CreateAuthorizedViewRequest):
            The request that prompted the initiation of
            this CreateAuthorizedView operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "CreateAuthorizedViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CreateAuthorizedViewRequest",
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


class ListAuthorizedViewsRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]

    Attributes:
        parent (str):
            Required. The unique name of the table for which
            AuthorizedViews should be listed. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        page_size (int):
            Optional. Maximum number of results per page.

            A page_size of zero lets the server choose the number of
            items to return. A page_size which is strictly positive will
            return at most that many items. A negative page_size will
            cause an error.

            Following the first request, subsequent paginated calls are
            not required to pass a page_size. If a page_size is set in
            subsequent calls, it must match the page_size given in the
            first request.
        page_token (str):
            Optional. The value of ``next_page_token`` returned by a
            previous call.
        view (google.cloud.bigtable_admin_v2.types.AuthorizedView.ResponseView):
            Optional. The resource_view to be applied to the returned
            AuthorizedViews' fields. Default to NAME_ONLY.
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
    view: gba_table.AuthorizedView.ResponseView = proto.Field(
        proto.ENUM,
        number=4,
        enum=gba_table.AuthorizedView.ResponseView,
    )


class ListAuthorizedViewsResponse(proto.Message):
    r"""Response message for
    [google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews][google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews]

    Attributes:
        authorized_views (MutableSequence[google.cloud.bigtable_admin_v2.types.AuthorizedView]):
            The AuthorizedViews present in the requested
            table.
        next_page_token (str):
            Set if not all tables could be returned in a single
            response. Pass this value to ``page_token`` in another
            request to get the next page of results.
    """

    @property
    def raw_page(self):
        return self

    authorized_views: MutableSequence[gba_table.AuthorizedView] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_table.AuthorizedView,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetAuthorizedViewRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.GetAuthorizedView]

    Attributes:
        name (str):
            Required. The unique name of the requested AuthorizedView.
            Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.
        view (google.cloud.bigtable_admin_v2.types.AuthorizedView.ResponseView):
            Optional. The resource_view to be applied to the returned
            AuthorizedView's fields. Default to BASIC.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    view: gba_table.AuthorizedView.ResponseView = proto.Field(
        proto.ENUM,
        number=2,
        enum=gba_table.AuthorizedView.ResponseView,
    )


class UpdateAuthorizedViewRequest(proto.Message):
    r"""The request for
    [UpdateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.UpdateAuthorizedView].

    Attributes:
        authorized_view (google.cloud.bigtable_admin_v2.types.AuthorizedView):
            Required. The AuthorizedView to update. The ``name`` in
            ``authorized_view`` is used to identify the AuthorizedView.
            AuthorizedView name must in this format:
            ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update. A mask specifying
            which fields in the AuthorizedView resource should be
            updated. This mask is relative to the AuthorizedView
            resource, not to the request message. A field will be
            overwritten if it is in the mask. If empty, all fields set
            in the request will be overwritten. A special value ``*``
            means to overwrite all fields (including fields not set in
            the request).
        ignore_warnings (bool):
            Optional. If true, ignore the safety checks
            when updating the AuthorizedView.
    """

    authorized_view: gba_table.AuthorizedView = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_table.AuthorizedView,
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


class UpdateAuthorizedViewMetadata(proto.Message):
    r"""Metadata for the google.longrunning.Operation returned by
    [UpdateAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.UpdateAuthorizedView].

    Attributes:
        original_request (google.cloud.bigtable_admin_v2.types.UpdateAuthorizedViewRequest):
            The request that prompted the initiation of
            this UpdateAuthorizedView operation.
        request_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the original request was
            received.
        finish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation failed or was
            completed successfully.
    """

    original_request: "UpdateAuthorizedViewRequest" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="UpdateAuthorizedViewRequest",
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


class DeleteAuthorizedViewRequest(proto.Message):
    r"""Request message for
    [google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView][google.bigtable.admin.v2.BigtableTableAdmin.DeleteAuthorizedView]

    Attributes:
        name (str):
            Required. The unique name of the AuthorizedView to be
            deleted. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/authorizedViews/{authorized_view}``.
        etag (str):
            Optional. The current etag of the
            AuthorizedView. If an etag is provided and does
            not match the current etag of the
            AuthorizedView, deletion will be blocked and an
            ABORTED error will be returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateSchemaBundleRequest(proto.Message):
    r"""The request for
    [CreateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.CreateSchemaBundle].

    Attributes:
        parent (str):
            Required. The parent resource where this schema bundle will
            be created. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        schema_bundle_id (str):
            Required. The unique ID to use for the schema
            bundle, which will become the final component of
            the schema bundle's resource name.
        schema_bundle (google.cloud.bigtable_admin_v2.types.SchemaBundle):
            Required. The schema bundle to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    schema_bundle_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    schema_bundle: gba_table.SchemaBundle = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gba_table.SchemaBundle,
    )


class CreateSchemaBundleMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    [CreateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.CreateSchemaBundle].

    Attributes:
        name (str):
            The unique name identifying this schema bundle. Values are
            of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    name: str = proto.Field(
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


class UpdateSchemaBundleRequest(proto.Message):
    r"""The request for
    [UpdateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.UpdateSchemaBundle].

    Attributes:
        schema_bundle (google.cloud.bigtable_admin_v2.types.SchemaBundle):
            Required. The schema bundle to update.

            The schema bundle's ``name`` field is used to identify the
            schema bundle to update. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update.
        ignore_warnings (bool):
            Optional. If set, ignore the safety checks
            when updating the Schema Bundle. The safety
            checks are:

            - The new Schema Bundle is backwards compatible
              with the existing Schema Bundle.
    """

    schema_bundle: gba_table.SchemaBundle = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gba_table.SchemaBundle,
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


class UpdateSchemaBundleMetadata(proto.Message):
    r"""The metadata for the Operation returned by
    [UpdateSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.UpdateSchemaBundle].

    Attributes:
        name (str):
            The unique name identifying this schema bundle. Values are
            of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which this operation started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            finished or was canceled.
    """

    name: str = proto.Field(
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


class GetSchemaBundleRequest(proto.Message):
    r"""The request for
    [GetSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.GetSchemaBundle].

    Attributes:
        name (str):
            Required. The unique name of the schema bundle to retrieve.
            Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSchemaBundlesRequest(proto.Message):
    r"""The request for
    [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].

    Attributes:
        parent (str):
            Required. The parent, which owns this collection of schema
            bundles. Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}``.
        page_size (int):
            The maximum number of schema bundles to
            return. If the value is positive, the server may
            return at most this value. If unspecified, the
            server will return the maximum allowed page
            size.
        page_token (str):
            A page token, received from a previous ``ListSchemaBundles``
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListSchemaBundles`` must match the call that provided the
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


class ListSchemaBundlesResponse(proto.Message):
    r"""The response for
    [ListSchemaBundles][google.bigtable.admin.v2.BigtableTableAdmin.ListSchemaBundles].

    Attributes:
        schema_bundles (MutableSequence[google.cloud.bigtable_admin_v2.types.SchemaBundle]):
            The schema bundles from the specified table.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    schema_bundles: MutableSequence[gba_table.SchemaBundle] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gba_table.SchemaBundle,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteSchemaBundleRequest(proto.Message):
    r"""The request for
    [DeleteSchemaBundle][google.bigtable.admin.v2.BigtableTableAdmin.DeleteSchemaBundle].

    Attributes:
        name (str):
            Required. The unique name of the schema bundle to delete.
            Values are of the form
            ``projects/{project}/instances/{instance}/tables/{table}/schemaBundles/{schema_bundle}``
        etag (str):
            Optional. The etag of the schema bundle.
            If this is provided, it must match the server's
            etag. The server returns an ABORTED error on a
            mismatched etag.
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

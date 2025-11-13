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

from google.cloud.spanner_admin_database_v1.types import common
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.spanner.admin.database.v1",
    manifest={
        "Backup",
        "CreateBackupRequest",
        "CreateBackupMetadata",
        "CopyBackupRequest",
        "CopyBackupMetadata",
        "UpdateBackupRequest",
        "GetBackupRequest",
        "DeleteBackupRequest",
        "ListBackupsRequest",
        "ListBackupsResponse",
        "ListBackupOperationsRequest",
        "ListBackupOperationsResponse",
        "BackupInfo",
        "CreateBackupEncryptionConfig",
        "CopyBackupEncryptionConfig",
        "FullBackupSpec",
        "IncrementalBackupSpec",
        "BackupInstancePartition",
    },
)


class Backup(proto.Message):
    r"""A backup of a Cloud Spanner database.

    Attributes:
        database (str):
            Required for the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            operation. Name of the database from which this backup was
            created. This needs to be in the same instance as the
            backup. Values are of the form
            ``projects/<project>/instances/<instance>/databases/<database>``.
        version_time (google.protobuf.timestamp_pb2.Timestamp):
            The backup will contain an externally consistent copy of the
            database at the timestamp specified by ``version_time``. If
            ``version_time`` is not specified, the system will set
            ``version_time`` to the ``create_time`` of the backup.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Required for the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            operation. The expiration time of the backup, with
            microseconds granularity that must be at least 6 hours and
            at most 366 days from the time the CreateBackup request is
            processed. Once the ``expire_time`` has passed, the backup
            is eligible to be automatically deleted by Cloud Spanner to
            free the resources used by the backup.
        name (str):
            Output only for the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            operation. Required for the
            [UpdateBackup][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackup]
            operation.

            A globally unique identifier for the backup which cannot be
            changed. Values are of the form
            ``projects/<project>/instances/<instance>/backups/[a-z][a-z0-9_\-]*[a-z0-9]``
            The final segment of the name must be between 2 and 60
            characters in length.

            The backup is stored in the location(s) specified in the
            instance configuration of the instance containing the
            backup, identified by the prefix of the backup name of the
            form ``projects/<project>/instances/<instance>``.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            request is received. If the request does not specify
            ``version_time``, the ``version_time`` of the backup will be
            equivalent to the ``create_time``.
        size_bytes (int):
            Output only. Size of the backup in bytes.
        freeable_size_bytes (int):
            Output only. The number of bytes that will be
            freed by deleting this backup. This value will
            be zero if, for example, this backup is part of
            an incremental backup chain and younger backups
            in the chain require that we keep its data. For
            backups not in an incremental backup chain, this
            is always the size of the backup. This value may
            change if backups on the same chain get created,
            deleted or expired.
        exclusive_size_bytes (int):
            Output only. For a backup in an incremental
            backup chain, this is the storage space needed
            to keep the data that has changed since the
            previous backup. For all other backups, this is
            always the size of the backup. This value may
            change if backups on the same chain get deleted
            or expired.

            This field can be used to calculate the total
            storage space used by a set of backups. For
            example, the total space used by all backups of
            a database can be computed by summing up this
            field.
        state (google.cloud.spanner_admin_database_v1.types.Backup.State):
            Output only. The current state of the backup.
        referencing_databases (MutableSequence[str]):
            Output only. The names of the restored databases that
            reference the backup. The database names are of the form
            ``projects/<project>/instances/<instance>/databases/<database>``.
            Referencing databases may exist in different instances. The
            existence of any referencing database prevents the backup
            from being deleted. When a restored database from the backup
            enters the ``READY`` state, the reference to the backup is
            removed.
        encryption_info (google.cloud.spanner_admin_database_v1.types.EncryptionInfo):
            Output only. The encryption information for
            the backup.
        encryption_information (MutableSequence[google.cloud.spanner_admin_database_v1.types.EncryptionInfo]):
            Output only. The encryption information for the backup,
            whether it is protected by one or more KMS keys. The
            information includes all Cloud KMS key versions used to
            encrypt the backup. The
            ``encryption_status' field inside of each``\ EncryptionInfo\`
            is not populated. At least one of the key versions must be
            available for the backup to be restored. If a key version is
            revoked in the middle of a restore, the restore behavior is
            undefined.
        database_dialect (google.cloud.spanner_admin_database_v1.types.DatabaseDialect):
            Output only. The database dialect information
            for the backup.
        referencing_backups (MutableSequence[str]):
            Output only. The names of the destination backups being
            created by copying this source backup. The backup names are
            of the form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
            Referencing backups may exist in different instances. The
            existence of any referencing backup prevents the backup from
            being deleted. When the copy operation is done (either
            successfully completed or cancelled or the destination
            backup is deleted), the reference to the backup is removed.
        max_expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The max allowed expiration time of the backup,
            with microseconds granularity. A backup's expiration time
            can be configured in multiple APIs: CreateBackup,
            UpdateBackup, CopyBackup. When updating or copying an
            existing backup, the expiration time specified must be less
            than ``Backup.max_expire_time``.
        backup_schedules (MutableSequence[str]):
            Output only. List of backup schedule URIs
            that are associated with creating this backup.
            This is only applicable for scheduled backups,
            and is empty for on-demand backups.

            To optimize for storage, whenever possible,
            multiple schedules are collapsed together to
            create one backup. In such cases, this field
            captures the list of all backup schedule URIs
            that are associated with creating this backup.
            If collapsing is not done, then this field
            captures the single backup schedule URI
            associated with creating this backup.
        incremental_backup_chain_id (str):
            Output only. Populated only for backups in an incremental
            backup chain. Backups share the same chain id if and only if
            they belong to the same incremental backup chain. Use this
            field to determine which backups are part of the same
            incremental backup chain. The ordering of backups in the
            chain can be determined by ordering the backup
            ``version_time``.
        oldest_version_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Data deleted at a time older
            than this is guaranteed not to be retained in
            order to support this backup. For a backup in an
            incremental backup chain, this is the version
            time of the oldest backup that exists or ever
            existed in the chain. For all other backups,
            this is the version time of the backup. This
            field can be used to understand what data is
            being retained by the backup system.
        instance_partitions (MutableSequence[google.cloud.spanner_admin_database_v1.types.BackupInstancePartition]):
            Output only. The instance partition(s) storing the backup.

            This is the same as the list of the instance partition(s)
            that the database had footprint in at the backup's
            ``version_time``.
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

    database: str = proto.Field(
        proto.STRING,
        number=2,
    )
    version_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    size_bytes: int = proto.Field(
        proto.INT64,
        number=5,
    )
    freeable_size_bytes: int = proto.Field(
        proto.INT64,
        number=15,
    )
    exclusive_size_bytes: int = proto.Field(
        proto.INT64,
        number=16,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=6,
        enum=State,
    )
    referencing_databases: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=7,
    )
    encryption_info: common.EncryptionInfo = proto.Field(
        proto.MESSAGE,
        number=8,
        message=common.EncryptionInfo,
    )
    encryption_information: MutableSequence[
        common.EncryptionInfo
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=13,
        message=common.EncryptionInfo,
    )
    database_dialect: common.DatabaseDialect = proto.Field(
        proto.ENUM,
        number=10,
        enum=common.DatabaseDialect,
    )
    referencing_backups: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    max_expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    backup_schedules: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=14,
    )
    incremental_backup_chain_id: str = proto.Field(
        proto.STRING,
        number=17,
    )
    oldest_version_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=18,
        message=timestamp_pb2.Timestamp,
    )
    instance_partitions: MutableSequence[
        "BackupInstancePartition"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=19,
        message="BackupInstancePartition",
    )


class CreateBackupRequest(proto.Message):
    r"""The request for
    [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup].

    Attributes:
        parent (str):
            Required. The name of the instance in which the backup will
            be created. This must be the same instance that contains the
            database the backup will be created from. The backup will be
            stored in the location(s) specified in the instance
            configuration of this instance. Values are of the form
            ``projects/<project>/instances/<instance>``.
        backup_id (str):
            Required. The id of the backup to be created. The
            ``backup_id`` appended to ``parent`` forms the full backup
            name of the form
            ``projects/<project>/instances/<instance>/backups/<backup_id>``.
        backup (google.cloud.spanner_admin_database_v1.types.Backup):
            Required. The backup to create.
        encryption_config (google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig):
            Optional. The encryption configuration used to encrypt the
            backup. If this field is not specified, the backup will use
            the same encryption configuration as the database by
            default, namely
            [encryption_type][google.spanner.admin.database.v1.CreateBackupEncryptionConfig.encryption_type]
            = ``USE_DATABASE_ENCRYPTION``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    backup_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    backup: "Backup" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="Backup",
    )
    encryption_config: "CreateBackupEncryptionConfig" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="CreateBackupEncryptionConfig",
    )


class CreateBackupMetadata(proto.Message):
    r"""Metadata type for the operation returned by
    [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup].

    Attributes:
        name (str):
            The name of the backup being created.
        database (str):
            The name of the database the backup is
            created from.
        progress (google.cloud.spanner_admin_database_v1.types.OperationProgress):
            The progress of the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            operation.
        cancel_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which cancellation of this operation was
            received.
            [Operations.CancelOperation][google.longrunning.Operations.CancelOperation]
            starts asynchronous cancellation on a long-running
            operation. The server makes a best effort to cancel the
            operation, but success is not guaranteed. Clients can use
            [Operations.GetOperation][google.longrunning.Operations.GetOperation]
            or other methods to check whether the cancellation succeeded
            or whether the operation completed despite cancellation. On
            successful cancellation, the operation is not deleted;
            instead, it becomes an operation with an
            [Operation.error][google.longrunning.Operation.error] value
            with a [google.rpc.Status.code][google.rpc.Status.code] of
            1, corresponding to ``Code.CANCELLED``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    database: str = proto.Field(
        proto.STRING,
        number=2,
    )
    progress: common.OperationProgress = proto.Field(
        proto.MESSAGE,
        number=3,
        message=common.OperationProgress,
    )
    cancel_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


class CopyBackupRequest(proto.Message):
    r"""The request for
    [CopyBackup][google.spanner.admin.database.v1.DatabaseAdmin.CopyBackup].

    Attributes:
        parent (str):
            Required. The name of the destination instance that will
            contain the backup copy. Values are of the form:
            ``projects/<project>/instances/<instance>``.
        backup_id (str):
            Required. The id of the backup copy. The ``backup_id``
            appended to ``parent`` forms the full backup_uri of the form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
        source_backup (str):
            Required. The source backup to be copied. The source backup
            needs to be in READY state for it to be copied. Once
            CopyBackup is in progress, the source backup cannot be
            deleted or cleaned up on expiration until CopyBackup is
            finished. Values are of the form:
            ``projects/<project>/instances/<instance>/backups/<backup>``.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Required. The expiration time of the backup in microsecond
            granularity. The expiration time must be at least 6 hours
            and at most 366 days from the ``create_time`` of the source
            backup. Once the ``expire_time`` has passed, the backup is
            eligible to be automatically deleted by Cloud Spanner to
            free the resources used by the backup.
        encryption_config (google.cloud.spanner_admin_database_v1.types.CopyBackupEncryptionConfig):
            Optional. The encryption configuration used to encrypt the
            backup. If this field is not specified, the backup will use
            the same encryption configuration as the source backup by
            default, namely
            [encryption_type][google.spanner.admin.database.v1.CopyBackupEncryptionConfig.encryption_type]
            = ``USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION``.
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
    encryption_config: "CopyBackupEncryptionConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="CopyBackupEncryptionConfig",
    )


class CopyBackupMetadata(proto.Message):
    r"""Metadata type for the operation returned by
    [CopyBackup][google.spanner.admin.database.v1.DatabaseAdmin.CopyBackup].

    Attributes:
        name (str):
            The name of the backup being created through the copy
            operation. Values are of the form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
        source_backup (str):
            The name of the source backup that is being copied. Values
            are of the form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
        progress (google.cloud.spanner_admin_database_v1.types.OperationProgress):
            The progress of the
            [CopyBackup][google.spanner.admin.database.v1.DatabaseAdmin.CopyBackup]
            operation.
        cancel_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which cancellation of CopyBackup operation was
            received.
            [Operations.CancelOperation][google.longrunning.Operations.CancelOperation]
            starts asynchronous cancellation on a long-running
            operation. The server makes a best effort to cancel the
            operation, but success is not guaranteed. Clients can use
            [Operations.GetOperation][google.longrunning.Operations.GetOperation]
            or other methods to check whether the cancellation succeeded
            or whether the operation completed despite cancellation. On
            successful cancellation, the operation is not deleted;
            instead, it becomes an operation with an
            [Operation.error][google.longrunning.Operation.error] value
            with a [google.rpc.Status.code][google.rpc.Status.code] of
            1, corresponding to ``Code.CANCELLED``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_backup: str = proto.Field(
        proto.STRING,
        number=2,
    )
    progress: common.OperationProgress = proto.Field(
        proto.MESSAGE,
        number=3,
        message=common.OperationProgress,
    )
    cancel_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


class UpdateBackupRequest(proto.Message):
    r"""The request for
    [UpdateBackup][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackup].

    Attributes:
        backup (google.cloud.spanner_admin_database_v1.types.Backup):
            Required. The backup to update. ``backup.name``, and the
            fields to be updated as specified by ``update_mask`` are
            required. Other fields are ignored. Update is only supported
            for the following fields:

            - ``backup.expire_time``.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. A mask specifying which fields (e.g.
            ``expire_time``) in the Backup resource should be updated.
            This mask is relative to the Backup resource, not to the
            request message. The field mask must always be specified;
            this prevents any future fields from being erased
            accidentally by clients that do not know about them.
    """

    backup: "Backup" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Backup",
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class GetBackupRequest(proto.Message):
    r"""The request for
    [GetBackup][google.spanner.admin.database.v1.DatabaseAdmin.GetBackup].

    Attributes:
        name (str):
            Required. Name of the backup. Values are of the form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteBackupRequest(proto.Message):
    r"""The request for
    [DeleteBackup][google.spanner.admin.database.v1.DatabaseAdmin.DeleteBackup].

    Attributes:
        name (str):
            Required. Name of the backup to delete. Values are of the
            form
            ``projects/<project>/instances/<instance>/backups/<backup>``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListBackupsRequest(proto.Message):
    r"""The request for
    [ListBackups][google.spanner.admin.database.v1.DatabaseAdmin.ListBackups].

    Attributes:
        parent (str):
            Required. The instance to list backups from. Values are of
            the form ``projects/<project>/instances/<instance>``.
        filter (str):
            An expression that filters the list of returned backups.

            A filter expression consists of a field name, a comparison
            operator, and a value for filtering. The value must be a
            string, a number, or a boolean. The comparison operator must
            be one of: ``<``, ``>``, ``<=``, ``>=``, ``!=``, ``=``, or
            ``:``. Colon ``:`` is the contains operator. Filter rules
            are not case sensitive.

            The following fields in the
            [Backup][google.spanner.admin.database.v1.Backup] are
            eligible for filtering:

            - ``name``
            - ``database``
            - ``state``
            - ``create_time`` (and values are of the format
              YYYY-MM-DDTHH:MM:SSZ)
            - ``expire_time`` (and values are of the format
              YYYY-MM-DDTHH:MM:SSZ)
            - ``version_time`` (and values are of the format
              YYYY-MM-DDTHH:MM:SSZ)
            - ``size_bytes``
            - ``backup_schedules``

            You can combine multiple expressions by enclosing each
            expression in parentheses. By default, expressions are
            combined with AND logic, but you can specify AND, OR, and
            NOT logic explicitly.

            Here are a few examples:

            - ``name:Howl`` - The backup's name contains the string
              "howl".
            - ``database:prod`` - The database's name contains the
              string "prod".
            - ``state:CREATING`` - The backup is pending creation.
            - ``state:READY`` - The backup is fully created and ready
              for use.
            - ``(name:howl) AND (create_time < \"2018-03-28T14:50:00Z\")``
              - The backup name contains the string "howl" and
              ``create_time`` of the backup is before
              2018-03-28T14:50:00Z.
            - ``expire_time < \"2018-03-28T14:50:00Z\"`` - The backup
              ``expire_time`` is before 2018-03-28T14:50:00Z.
            - ``size_bytes > 10000000000`` - The backup's size is
              greater than 10GB
            - ``backup_schedules:daily`` - The backup is created from a
              schedule with "daily" in its name.
        page_size (int):
            Number of backups to be returned in the
            response. If 0 or less, defaults to the server's
            maximum allowed page size.
        page_token (str):
            If non-empty, ``page_token`` should contain a
            [next_page_token][google.spanner.admin.database.v1.ListBackupsResponse.next_page_token]
            from a previous
            [ListBackupsResponse][google.spanner.admin.database.v1.ListBackupsResponse]
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
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListBackupsResponse(proto.Message):
    r"""The response for
    [ListBackups][google.spanner.admin.database.v1.DatabaseAdmin.ListBackups].

    Attributes:
        backups (MutableSequence[google.cloud.spanner_admin_database_v1.types.Backup]):
            The list of matching backups. Backups returned are ordered
            by ``create_time`` in descending order, starting from the
            most recent ``create_time``.
        next_page_token (str):
            ``next_page_token`` can be sent in a subsequent
            [ListBackups][google.spanner.admin.database.v1.DatabaseAdmin.ListBackups]
            call to fetch more of the matching backups.
    """

    @property
    def raw_page(self):
        return self

    backups: MutableSequence["Backup"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Backup",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListBackupOperationsRequest(proto.Message):
    r"""The request for
    [ListBackupOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupOperations].

    Attributes:
        parent (str):
            Required. The instance of the backup operations. Values are
            of the form ``projects/<project>/instances/<instance>``.
        filter (str):
            An expression that filters the list of returned backup
            operations.

            A filter expression consists of a field name, a comparison
            operator, and a value for filtering. The value must be a
            string, a number, or a boolean. The comparison operator must
            be one of: ``<``, ``>``, ``<=``, ``>=``, ``!=``, ``=``, or
            ``:``. Colon ``:`` is the contains operator. Filter rules
            are not case sensitive.

            The following fields in the
            [operation][google.longrunning.Operation] are eligible for
            filtering:

            - ``name`` - The name of the long-running operation
            - ``done`` - False if the operation is in progress, else
              true.
            - ``metadata.@type`` - the type of metadata. For example,
              the type string for
              [CreateBackupMetadata][google.spanner.admin.database.v1.CreateBackupMetadata]
              is
              ``type.googleapis.com/google.spanner.admin.database.v1.CreateBackupMetadata``.
            - ``metadata.<field_name>`` - any field in metadata.value.
              ``metadata.@type`` must be specified first if filtering on
              metadata fields.
            - ``error`` - Error associated with the long-running
              operation.
            - ``response.@type`` - the type of response.
            - ``response.<field_name>`` - any field in response.value.

            You can combine multiple expressions by enclosing each
            expression in parentheses. By default, expressions are
            combined with AND logic, but you can specify AND, OR, and
            NOT logic explicitly.

            Here are a few examples:

            - ``done:true`` - The operation is complete.
            - ``(metadata.@type=type.googleapis.com/google.spanner.admin.database.v1.CreateBackupMetadata) AND``
              ``metadata.database:prod`` - Returns operations where:

              - The operation's metadata type is
                [CreateBackupMetadata][google.spanner.admin.database.v1.CreateBackupMetadata].
              - The source database name of backup contains the string
                "prod".

            - ``(metadata.@type=type.googleapis.com/google.spanner.admin.database.v1.CreateBackupMetadata) AND``
              ``(metadata.name:howl) AND``
              ``(metadata.progress.start_time < \"2018-03-28T14:50:00Z\") AND``
              ``(error:*)`` - Returns operations where:

              - The operation's metadata type is
                [CreateBackupMetadata][google.spanner.admin.database.v1.CreateBackupMetadata].
              - The backup name contains the string "howl".
              - The operation started before 2018-03-28T14:50:00Z.
              - The operation resulted in an error.

            - ``(metadata.@type=type.googleapis.com/google.spanner.admin.database.v1.CopyBackupMetadata) AND``
              ``(metadata.source_backup:test) AND``
              ``(metadata.progress.start_time < \"2022-01-18T14:50:00Z\") AND``
              ``(error:*)`` - Returns operations where:

              - The operation's metadata type is
                [CopyBackupMetadata][google.spanner.admin.database.v1.CopyBackupMetadata].
              - The source backup name contains the string "test".
              - The operation started before 2022-01-18T14:50:00Z.
              - The operation resulted in an error.

            - ``((metadata.@type=type.googleapis.com/google.spanner.admin.database.v1.CreateBackupMetadata) AND``
              ``(metadata.database:test_db)) OR``
              ``((metadata.@type=type.googleapis.com/google.spanner.admin.database.v1.CopyBackupMetadata) AND``
              ``(metadata.source_backup:test_bkp)) AND``
              ``(error:*)`` - Returns operations where:

              - The operation's metadata matches either of criteria:

                - The operation's metadata type is
                  [CreateBackupMetadata][google.spanner.admin.database.v1.CreateBackupMetadata]
                  AND the source database name of the backup contains
                  the string "test_db"
                - The operation's metadata type is
                  [CopyBackupMetadata][google.spanner.admin.database.v1.CopyBackupMetadata]
                  AND the source backup name contains the string
                  "test_bkp"

              - The operation resulted in an error.
        page_size (int):
            Number of operations to be returned in the
            response. If 0 or less, defaults to the server's
            maximum allowed page size.
        page_token (str):
            If non-empty, ``page_token`` should contain a
            [next_page_token][google.spanner.admin.database.v1.ListBackupOperationsResponse.next_page_token]
            from a previous
            [ListBackupOperationsResponse][google.spanner.admin.database.v1.ListBackupOperationsResponse]
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
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListBackupOperationsResponse(proto.Message):
    r"""The response for
    [ListBackupOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupOperations].

    Attributes:
        operations (MutableSequence[google.longrunning.operations_pb2.Operation]):
            The list of matching backup [long-running
            operations][google.longrunning.Operation]. Each operation's
            name will be prefixed by the backup's name. The operation's
            [metadata][google.longrunning.Operation.metadata] field type
            ``metadata.type_url`` describes the type of the metadata.
            Operations returned include those that are pending or have
            completed/failed/canceled within the last 7 days. Operations
            returned are ordered by
            ``operation.metadata.value.progress.start_time`` in
            descending order starting from the most recently started
            operation.
        next_page_token (str):
            ``next_page_token`` can be sent in a subsequent
            [ListBackupOperations][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupOperations]
            call to fetch more of the matching metadata.
    """

    @property
    def raw_page(self):
        return self

    operations: MutableSequence[operations_pb2.Operation] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=operations_pb2.Operation,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class BackupInfo(proto.Message):
    r"""Information about a backup.

    Attributes:
        backup (str):
            Name of the backup.
        version_time (google.protobuf.timestamp_pb2.Timestamp):
            The backup contains an externally consistent copy of
            ``source_database`` at the timestamp specified by
            ``version_time``. If the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            request did not specify ``version_time``, the
            ``version_time`` of the backup is equivalent to the
            ``create_time``.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            The time the
            [CreateBackup][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackup]
            request was received.
        source_database (str):
            Name of the database the backup was created
            from.
    """

    backup: str = proto.Field(
        proto.STRING,
        number=1,
    )
    version_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    source_database: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateBackupEncryptionConfig(proto.Message):
    r"""Encryption configuration for the backup to create.

    Attributes:
        encryption_type (google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig.EncryptionType):
            Required. The encryption type of the backup.
        kms_key_name (str):
            Optional. The Cloud KMS key that will be used to protect the
            backup. This field should be set only when
            [encryption_type][google.spanner.admin.database.v1.CreateBackupEncryptionConfig.encryption_type]
            is ``CUSTOMER_MANAGED_ENCRYPTION``. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.
        kms_key_names (MutableSequence[str]):
            Optional. Specifies the KMS configuration for the one or
            more keys used to protect the backup. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.

            The keys referenced by kms_key_names must fully cover all
            regions of the backup's instance configuration. Some
            examples:

            - For single region instance configs, specify a single
              regional location KMS key.
            - For multi-regional instance configs of type
              GOOGLE_MANAGED, either specify a multi-regional location
              KMS key or multiple regional location KMS keys that cover
              all regions in the instance config.
            - For an instance config of type USER_MANAGED, please
              specify only regional location KMS keys to cover each
              region in the instance config. Multi-regional location KMS
              keys are not supported for USER_MANAGED instance configs.
    """

    class EncryptionType(proto.Enum):
        r"""Encryption types for the backup.

        Values:
            ENCRYPTION_TYPE_UNSPECIFIED (0):
                Unspecified. Do not use.
            USE_DATABASE_ENCRYPTION (1):
                Use the same encryption configuration as the database. This
                is the default option when
                [encryption_config][google.spanner.admin.database.v1.CreateBackupEncryptionConfig]
                is empty. For example, if the database is using
                ``Customer_Managed_Encryption``, the backup will be using
                the same Cloud KMS key as the database.
            GOOGLE_DEFAULT_ENCRYPTION (2):
                Use Google default encryption.
            CUSTOMER_MANAGED_ENCRYPTION (3):
                Use customer managed encryption. If specified,
                ``kms_key_name`` must contain a valid Cloud KMS key.
        """
        ENCRYPTION_TYPE_UNSPECIFIED = 0
        USE_DATABASE_ENCRYPTION = 1
        GOOGLE_DEFAULT_ENCRYPTION = 2
        CUSTOMER_MANAGED_ENCRYPTION = 3

    encryption_type: EncryptionType = proto.Field(
        proto.ENUM,
        number=1,
        enum=EncryptionType,
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    kms_key_names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class CopyBackupEncryptionConfig(proto.Message):
    r"""Encryption configuration for the copied backup.

    Attributes:
        encryption_type (google.cloud.spanner_admin_database_v1.types.CopyBackupEncryptionConfig.EncryptionType):
            Required. The encryption type of the backup.
        kms_key_name (str):
            Optional. The Cloud KMS key that will be used to protect the
            backup. This field should be set only when
            [encryption_type][google.spanner.admin.database.v1.CopyBackupEncryptionConfig.encryption_type]
            is ``CUSTOMER_MANAGED_ENCRYPTION``. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.
        kms_key_names (MutableSequence[str]):
            Optional. Specifies the KMS configuration for the one or
            more keys used to protect the backup. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.
            Kms keys specified can be in any order.

            The keys referenced by kms_key_names must fully cover all
            regions of the backup's instance configuration. Some
            examples:

            - For single region instance configs, specify a single
              regional location KMS key.
            - For multi-regional instance configs of type
              GOOGLE_MANAGED, either specify a multi-regional location
              KMS key or multiple regional location KMS keys that cover
              all regions in the instance config.
            - For an instance config of type USER_MANAGED, please
              specify only regional location KMS keys to cover each
              region in the instance config. Multi-regional location KMS
              keys are not supported for USER_MANAGED instance configs.
    """

    class EncryptionType(proto.Enum):
        r"""Encryption types for the backup.

        Values:
            ENCRYPTION_TYPE_UNSPECIFIED (0):
                Unspecified. Do not use.
            USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION (1):
                This is the default option for
                [CopyBackup][google.spanner.admin.database.v1.DatabaseAdmin.CopyBackup]
                when
                [encryption_config][google.spanner.admin.database.v1.CopyBackupEncryptionConfig]
                is not specified. For example, if the source backup is using
                ``Customer_Managed_Encryption``, the backup will be using
                the same Cloud KMS key as the source backup.
            GOOGLE_DEFAULT_ENCRYPTION (2):
                Use Google default encryption.
            CUSTOMER_MANAGED_ENCRYPTION (3):
                Use customer managed encryption. If specified, either
                ``kms_key_name`` or ``kms_key_names`` must contain valid
                Cloud KMS key(s).
        """
        ENCRYPTION_TYPE_UNSPECIFIED = 0
        USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION = 1
        GOOGLE_DEFAULT_ENCRYPTION = 2
        CUSTOMER_MANAGED_ENCRYPTION = 3

    encryption_type: EncryptionType = proto.Field(
        proto.ENUM,
        number=1,
        enum=EncryptionType,
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    kms_key_names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class FullBackupSpec(proto.Message):
    r"""The specification for full backups.
    A full backup stores the entire contents of the database at a
    given version time.

    """


class IncrementalBackupSpec(proto.Message):
    r"""The specification for incremental backup chains.
    An incremental backup stores the delta of changes between a
    previous backup and the database contents at a given version
    time. An incremental backup chain consists of a full backup and
    zero or more successive incremental backups. The first backup
    created for an incremental backup chain is always a full backup.

    """


class BackupInstancePartition(proto.Message):
    r"""Instance partition information for the backup.

    Attributes:
        instance_partition (str):
            A unique identifier for the instance partition. Values are
            of the form
            ``projects/<project>/instances/<instance>/instancePartitions/<instance_partition_id>``
    """

    instance_partition: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

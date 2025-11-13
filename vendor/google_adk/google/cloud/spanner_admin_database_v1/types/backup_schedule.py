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

from google.cloud.spanner_admin_database_v1.types import backup
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.spanner.admin.database.v1",
    manifest={
        "BackupScheduleSpec",
        "BackupSchedule",
        "CrontabSpec",
        "CreateBackupScheduleRequest",
        "GetBackupScheduleRequest",
        "DeleteBackupScheduleRequest",
        "ListBackupSchedulesRequest",
        "ListBackupSchedulesResponse",
        "UpdateBackupScheduleRequest",
    },
)


class BackupScheduleSpec(proto.Message):
    r"""Defines specifications of the backup schedule.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        cron_spec (google.cloud.spanner_admin_database_v1.types.CrontabSpec):
            Cron style schedule specification.

            This field is a member of `oneof`_ ``schedule_spec``.
    """

    cron_spec: "CrontabSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="schedule_spec",
        message="CrontabSpec",
    )


class BackupSchedule(proto.Message):
    r"""BackupSchedule expresses the automated backup creation
    specification for a Spanner database.
    Next ID: 10

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. Output only for the
            [CreateBackupSchedule][DatabaseAdmin.CreateBackupSchededule]
            operation. Required for the
            [UpdateBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackupSchedule]
            operation. A globally unique identifier for the backup
            schedule which cannot be changed. Values are of the form
            ``projects/<project>/instances/<instance>/databases/<database>/backupSchedules/[a-z][a-z0-9_\-]*[a-z0-9]``
            The final segment of the name must be between 2 and 60
            characters in length.
        spec (google.cloud.spanner_admin_database_v1.types.BackupScheduleSpec):
            Optional. The schedule specification based on
            which the backup creations are triggered.
        retention_duration (google.protobuf.duration_pb2.Duration):
            Optional. The retention duration of a backup
            that must be at least 6 hours and at most 366
            days. The backup is eligible to be automatically
            deleted once the retention period has elapsed.
        encryption_config (google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig):
            Optional. The encryption configuration that
            will be used to encrypt the backup. If this
            field is not specified, the backup will use the
            same encryption configuration as the database.
        full_backup_spec (google.cloud.spanner_admin_database_v1.types.FullBackupSpec):
            The schedule creates only full backups.

            This field is a member of `oneof`_ ``backup_type_spec``.
        incremental_backup_spec (google.cloud.spanner_admin_database_v1.types.IncrementalBackupSpec):
            The schedule creates incremental backup
            chains.

            This field is a member of `oneof`_ ``backup_type_spec``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp at which the
            schedule was last updated. If the schedule has
            never been updated, this field contains the
            timestamp when the schedule was first created.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    spec: "BackupScheduleSpec" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="BackupScheduleSpec",
    )
    retention_duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=duration_pb2.Duration,
    )
    encryption_config: backup.CreateBackupEncryptionConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=backup.CreateBackupEncryptionConfig,
    )
    full_backup_spec: backup.FullBackupSpec = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="backup_type_spec",
        message=backup.FullBackupSpec,
    )
    incremental_backup_spec: backup.IncrementalBackupSpec = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="backup_type_spec",
        message=backup.IncrementalBackupSpec,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )


class CrontabSpec(proto.Message):
    r"""CrontabSpec can be used to specify the version time and
    frequency at which the backup should be created.

    Attributes:
        text (str):
            Required. Textual representation of the crontab. User can
            customize the backup frequency and the backup version time
            using the cron expression. The version time must be in UTC
            timezone.

            The backup will contain an externally consistent copy of the
            database at the version time. Allowed frequencies are 12
            hour, 1 day, 1 week and 1 month. Examples of valid cron
            specifications:

            - ``0 2/12 * * *`` : every 12 hours at (2, 14) hours past
              midnight in UTC.
            - ``0 2,14 * * *`` : every 12 hours at (2,14) hours past
              midnight in UTC.
            - ``0 2 * * *`` : once a day at 2 past midnight in UTC.
            - ``0 2 * * 0`` : once a week every Sunday at 2 past
              midnight in UTC.
            - ``0 2 8 * *`` : once a month on 8th day at 2 past midnight
              in UTC.
        time_zone (str):
            Output only. The time zone of the times in
            ``CrontabSpec.text``. Currently only UTC is supported.
        creation_window (google.protobuf.duration_pb2.Duration):
            Output only. Schedule backups will contain an externally
            consistent copy of the database at the version time
            specified in ``schedule_spec.cron_spec``. However, Spanner
            may not initiate the creation of the scheduled backups at
            that version time. Spanner will initiate the creation of
            scheduled backups within the time window bounded by the
            version_time specified in ``schedule_spec.cron_spec`` and
            version_time + ``creation_window``.
    """

    text: str = proto.Field(
        proto.STRING,
        number=1,
    )
    time_zone: str = proto.Field(
        proto.STRING,
        number=2,
    )
    creation_window: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=duration_pb2.Duration,
    )


class CreateBackupScheduleRequest(proto.Message):
    r"""The request for
    [CreateBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.CreateBackupSchedule].

    Attributes:
        parent (str):
            Required. The name of the database that this
            backup schedule applies to.
        backup_schedule_id (str):
            Required. The Id to use for the backup schedule. The
            ``backup_schedule_id`` appended to ``parent`` forms the full
            backup schedule name of the form
            ``projects/<project>/instances/<instance>/databases/<database>/backupSchedules/<backup_schedule_id>``.
        backup_schedule (google.cloud.spanner_admin_database_v1.types.BackupSchedule):
            Required. The backup schedule to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    backup_schedule_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    backup_schedule: "BackupSchedule" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="BackupSchedule",
    )


class GetBackupScheduleRequest(proto.Message):
    r"""The request for
    [GetBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.GetBackupSchedule].

    Attributes:
        name (str):
            Required. The name of the schedule to retrieve. Values are
            of the form
            ``projects/<project>/instances/<instance>/databases/<database>/backupSchedules/<backup_schedule_id>``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteBackupScheduleRequest(proto.Message):
    r"""The request for
    [DeleteBackupSchedule][google.spanner.admin.database.v1.DatabaseAdmin.DeleteBackupSchedule].

    Attributes:
        name (str):
            Required. The name of the schedule to delete. Values are of
            the form
            ``projects/<project>/instances/<instance>/databases/<database>/backupSchedules/<backup_schedule_id>``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListBackupSchedulesRequest(proto.Message):
    r"""The request for
    [ListBackupSchedules][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupSchedules].

    Attributes:
        parent (str):
            Required. Database is the parent resource
            whose backup schedules should be listed. Values
            are of the form
            projects/<project>/instances/<instance>/databases/<database>
        page_size (int):
            Optional. Number of backup schedules to be
            returned in the response. If 0 or less, defaults
            to the server's maximum allowed page size.
        page_token (str):
            Optional. If non-empty, ``page_token`` should contain a
            [next_page_token][google.spanner.admin.database.v1.ListBackupSchedulesResponse.next_page_token]
            from a previous
            [ListBackupSchedulesResponse][google.spanner.admin.database.v1.ListBackupSchedulesResponse]
            to the same ``parent``.
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
        number=4,
    )


class ListBackupSchedulesResponse(proto.Message):
    r"""The response for
    [ListBackupSchedules][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupSchedules].

    Attributes:
        backup_schedules (MutableSequence[google.cloud.spanner_admin_database_v1.types.BackupSchedule]):
            The list of backup schedules for a database.
        next_page_token (str):
            ``next_page_token`` can be sent in a subsequent
            [ListBackupSchedules][google.spanner.admin.database.v1.DatabaseAdmin.ListBackupSchedules]
            call to fetch more of the schedules.
    """

    @property
    def raw_page(self):
        return self

    backup_schedules: MutableSequence["BackupSchedule"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="BackupSchedule",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateBackupScheduleRequest(proto.Message):
    r"""The request for
    [UpdateBackupScheduleRequest][google.spanner.admin.database.v1.DatabaseAdmin.UpdateBackupSchedule].

    Attributes:
        backup_schedule (google.cloud.spanner_admin_database_v1.types.BackupSchedule):
            Required. The backup schedule to update.
            ``backup_schedule.name``, and the fields to be updated as
            specified by ``update_mask`` are required. Other fields are
            ignored.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. A mask specifying which fields in
            the BackupSchedule resource should be updated.
            This mask is relative to the BackupSchedule
            resource, not to the request message. The field
            mask must always be specified; this prevents any
            future fields from being erased accidentally.
    """

    backup_schedule: "BackupSchedule" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="BackupSchedule",
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

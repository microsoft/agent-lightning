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

from google.cloud.spanner_v1.types import type as gs_type
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.spanner.v1",
    manifest={
        "ChangeStreamRecord",
    },
)


class ChangeStreamRecord(proto.Message):
    r"""Spanner Change Streams enable customers to capture and stream out
    changes to their Spanner databases in real-time. A change stream can
    be created with option partition_mode='IMMUTABLE_KEY_RANGE' or
    partition_mode='MUTABLE_KEY_RANGE'.

    This message is only used in Change Streams created with the option
    partition_mode='MUTABLE_KEY_RANGE'. Spanner automatically creates a
    special Table-Valued Function (TVF) along with each Change Streams.
    The function provides access to the change stream's records. The
    function is named READ\_<change_stream_name> (where
    <change_stream_name> is the name of the change stream), and it
    returns a table with only one column called ChangeRecord.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        data_change_record (google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord):
            Data change record describing a data change
            for a change stream partition.

            This field is a member of `oneof`_ ``record``.
        heartbeat_record (google.cloud.spanner_v1.types.ChangeStreamRecord.HeartbeatRecord):
            Heartbeat record describing a heartbeat for a
            change stream partition.

            This field is a member of `oneof`_ ``record``.
        partition_start_record (google.cloud.spanner_v1.types.ChangeStreamRecord.PartitionStartRecord):
            Partition start record describing a new
            change stream partition.

            This field is a member of `oneof`_ ``record``.
        partition_end_record (google.cloud.spanner_v1.types.ChangeStreamRecord.PartitionEndRecord):
            Partition end record describing a terminated
            change stream partition.

            This field is a member of `oneof`_ ``record``.
        partition_event_record (google.cloud.spanner_v1.types.ChangeStreamRecord.PartitionEventRecord):
            Partition event record describing key range
            changes for a change stream partition.

            This field is a member of `oneof`_ ``record``.
    """

    class DataChangeRecord(proto.Message):
        r"""A data change record contains a set of changes to a table
        with the same modification type (insert, update, or delete)
        committed at the same commit timestamp in one change stream
        partition for the same transaction. Multiple data change records
        can be returned for the same transaction across multiple change
        stream partitions.

        Attributes:
            commit_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Indicates the timestamp in which the change was committed.
                DataChangeRecord.commit_timestamps,
                PartitionStartRecord.start_timestamps,
                PartitionEventRecord.commit_timestamps, and
                PartitionEndRecord.end_timestamps can have the same value in
                the same partition.
            record_sequence (str):
                Record sequence numbers are unique and monotonically
                increasing (but not necessarily contiguous) for a specific
                timestamp across record types in the same partition. To
                guarantee ordered processing, the reader should process
                records (of potentially different types) in record_sequence
                order for a specific timestamp in the same partition.

                The record sequence number ordering across partitions is
                only meaningful in the context of a specific transaction.
                Record sequence numbers are unique across partitions for a
                specific transaction. Sort the DataChangeRecords for the
                same
                [server_transaction_id][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.server_transaction_id]
                by
                [record_sequence][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.record_sequence]
                to reconstruct the ordering of the changes within the
                transaction.
            server_transaction_id (str):
                Provides a globally unique string that represents the
                transaction in which the change was committed. Multiple
                transactions can have the same commit timestamp, but each
                transaction has a unique server_transaction_id.
            is_last_record_in_transaction_in_partition (bool):
                Indicates whether this is the last record for
                a transaction in the  current partition. Clients
                can use this field to determine when all
                records for a transaction in the current
                partition have been received.
            table (str):
                Name of the table affected by the change.
            column_metadata (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ColumnMetadata]):
                Provides metadata describing the columns associated with the
                [mods][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.mods]
                listed below.
            mods (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.Mod]):
                Describes the changes that were made.
            mod_type (google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ModType):
                Describes the type of change.
            value_capture_type (google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ValueCaptureType):
                Describes the value capture type that was
                specified in the change stream configuration
                when this change was captured.
            number_of_records_in_transaction (int):
                Indicates the number of data change records
                that are part of this transaction across all
                change stream partitions. This value can be used
                to assemble all the records associated with a
                particular transaction.
            number_of_partitions_in_transaction (int):
                Indicates the number of partitions that
                return data change records for this transaction.
                This value can be helpful in assembling all
                records associated with a particular
                transaction.
            transaction_tag (str):
                Indicates the transaction tag associated with
                this transaction.
            is_system_transaction (bool):
                Indicates whether the transaction is a system
                transaction. System transactions include those
                issued by time-to-live (TTL), column backfill,
                etc.
        """

        class ModType(proto.Enum):
            r"""Mod type describes the type of change Spanner applied to the data.
            For example, if the client submits an INSERT_OR_UPDATE request,
            Spanner will perform an insert if there is no existing row and
            return ModType INSERT. Alternatively, if there is an existing row,
            Spanner will perform an update and return ModType UPDATE.

            Values:
                MOD_TYPE_UNSPECIFIED (0):
                    Not specified.
                INSERT (10):
                    Indicates data was inserted.
                UPDATE (20):
                    Indicates existing data was updated.
                DELETE (30):
                    Indicates existing data was deleted.
            """
            MOD_TYPE_UNSPECIFIED = 0
            INSERT = 10
            UPDATE = 20
            DELETE = 30

        class ValueCaptureType(proto.Enum):
            r"""Value capture type describes which values are recorded in the
            data change record.

            Values:
                VALUE_CAPTURE_TYPE_UNSPECIFIED (0):
                    Not specified.
                OLD_AND_NEW_VALUES (10):
                    Records both old and new values of the
                    modified watched columns.
                NEW_VALUES (20):
                    Records only new values of the modified
                    watched columns.
                NEW_ROW (30):
                    Records new values of all watched columns,
                    including modified and unmodified columns.
                NEW_ROW_AND_OLD_VALUES (40):
                    Records the new values of all watched
                    columns, including modified and unmodified
                    columns. Also records the old values of the
                    modified columns.
            """
            VALUE_CAPTURE_TYPE_UNSPECIFIED = 0
            OLD_AND_NEW_VALUES = 10
            NEW_VALUES = 20
            NEW_ROW = 30
            NEW_ROW_AND_OLD_VALUES = 40

        class ColumnMetadata(proto.Message):
            r"""Metadata for a column.

            Attributes:
                name (str):
                    Name of the column.
                type_ (google.cloud.spanner_v1.types.Type):
                    Type of the column.
                is_primary_key (bool):
                    Indicates whether the column is a primary key
                    column.
                ordinal_position (int):
                    Ordinal position of the column based on the
                    original table definition in the schema starting
                    with a value of 1.
            """

            name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            type_: gs_type.Type = proto.Field(
                proto.MESSAGE,
                number=2,
                message=gs_type.Type,
            )
            is_primary_key: bool = proto.Field(
                proto.BOOL,
                number=3,
            )
            ordinal_position: int = proto.Field(
                proto.INT64,
                number=4,
            )

        class ModValue(proto.Message):
            r"""Returns the value and associated metadata for a particular field of
            the
            [Mod][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.Mod].

            Attributes:
                column_metadata_index (int):
                    Index within the repeated
                    [column_metadata][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.column_metadata]
                    field, to obtain the column metadata for the column that was
                    modified.
                value (google.protobuf.struct_pb2.Value):
                    The value of the column.
            """

            column_metadata_index: int = proto.Field(
                proto.INT32,
                number=1,
            )
            value: struct_pb2.Value = proto.Field(
                proto.MESSAGE,
                number=2,
                message=struct_pb2.Value,
            )

        class Mod(proto.Message):
            r"""A mod describes all data changes in a watched table row.

            Attributes:
                keys (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ModValue]):
                    Returns the value of the primary key of the
                    modified row.
                old_values (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ModValue]):
                    Returns the old values before the change for the modified
                    columns. Always empty for
                    [INSERT][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.ModType.INSERT],
                    or if old values are not being captured specified by
                    [value_capture_type][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.ValueCaptureType].
                new_values (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.DataChangeRecord.ModValue]):
                    Returns the new values after the change for the modified
                    columns. Always empty for
                    [DELETE][google.spanner.v1.ChangeStreamRecord.DataChangeRecord.ModType.DELETE].
            """

            keys: MutableSequence[
                "ChangeStreamRecord.DataChangeRecord.ModValue"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message="ChangeStreamRecord.DataChangeRecord.ModValue",
            )
            old_values: MutableSequence[
                "ChangeStreamRecord.DataChangeRecord.ModValue"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=2,
                message="ChangeStreamRecord.DataChangeRecord.ModValue",
            )
            new_values: MutableSequence[
                "ChangeStreamRecord.DataChangeRecord.ModValue"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=3,
                message="ChangeStreamRecord.DataChangeRecord.ModValue",
            )

        commit_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        record_sequence: str = proto.Field(
            proto.STRING,
            number=2,
        )
        server_transaction_id: str = proto.Field(
            proto.STRING,
            number=3,
        )
        is_last_record_in_transaction_in_partition: bool = proto.Field(
            proto.BOOL,
            number=4,
        )
        table: str = proto.Field(
            proto.STRING,
            number=5,
        )
        column_metadata: MutableSequence[
            "ChangeStreamRecord.DataChangeRecord.ColumnMetadata"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=6,
            message="ChangeStreamRecord.DataChangeRecord.ColumnMetadata",
        )
        mods: MutableSequence[
            "ChangeStreamRecord.DataChangeRecord.Mod"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=7,
            message="ChangeStreamRecord.DataChangeRecord.Mod",
        )
        mod_type: "ChangeStreamRecord.DataChangeRecord.ModType" = proto.Field(
            proto.ENUM,
            number=8,
            enum="ChangeStreamRecord.DataChangeRecord.ModType",
        )
        value_capture_type: "ChangeStreamRecord.DataChangeRecord.ValueCaptureType" = (
            proto.Field(
                proto.ENUM,
                number=9,
                enum="ChangeStreamRecord.DataChangeRecord.ValueCaptureType",
            )
        )
        number_of_records_in_transaction: int = proto.Field(
            proto.INT32,
            number=10,
        )
        number_of_partitions_in_transaction: int = proto.Field(
            proto.INT32,
            number=11,
        )
        transaction_tag: str = proto.Field(
            proto.STRING,
            number=12,
        )
        is_system_transaction: bool = proto.Field(
            proto.BOOL,
            number=13,
        )

    class HeartbeatRecord(proto.Message):
        r"""A heartbeat record is returned as a progress indicator, when
        there are no data changes or any other partition record types in
        the change stream partition.

        Attributes:
            timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Indicates the timestamp at which the query
                has returned all the records in the change
                stream partition with timestamp <= heartbeat
                timestamp. The heartbeat timestamp will not be
                the same as the timestamps of other record types
                in the same partition.
        """

        timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )

    class PartitionStartRecord(proto.Message):
        r"""A partition start record serves as a notification that the
        client should schedule the partitions to be queried.
        PartitionStartRecord returns information about one or more
        partitions.

        Attributes:
            start_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Start timestamp at which the partitions should be queried to
                return change stream records with timestamps >=
                start_timestamp. DataChangeRecord.commit_timestamps,
                PartitionStartRecord.start_timestamps,
                PartitionEventRecord.commit_timestamps, and
                PartitionEndRecord.end_timestamps can have the same value in
                the same partition.
            record_sequence (str):
                Record sequence numbers are unique and monotonically
                increasing (but not necessarily contiguous) for a specific
                timestamp across record types in the same partition. To
                guarantee ordered processing, the reader should process
                records (of potentially different types) in record_sequence
                order for a specific timestamp in the same partition.
            partition_tokens (MutableSequence[str]):
                Unique partition identifiers to be used in
                queries.
        """

        start_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        record_sequence: str = proto.Field(
            proto.STRING,
            number=2,
        )
        partition_tokens: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )

    class PartitionEndRecord(proto.Message):
        r"""A partition end record serves as a notification that the
        client should stop reading the partition. No further records are
        expected to be retrieved on it.

        Attributes:
            end_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                End timestamp at which the change stream partition is
                terminated. All changes generated by this partition will
                have timestamps <= end_timestamp.
                DataChangeRecord.commit_timestamps,
                PartitionStartRecord.start_timestamps,
                PartitionEventRecord.commit_timestamps, and
                PartitionEndRecord.end_timestamps can have the same value in
                the same partition. PartitionEndRecord is the last record
                returned for a partition.
            record_sequence (str):
                Record sequence numbers are unique and monotonically
                increasing (but not necessarily contiguous) for a specific
                timestamp across record types in the same partition. To
                guarantee ordered processing, the reader should process
                records (of potentially different types) in record_sequence
                order for a specific timestamp in the same partition.
            partition_token (str):
                Unique partition identifier describing the terminated change
                stream partition.
                [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEndRecord.partition_token]
                is equal to the partition token of the change stream
                partition currently queried to return this
                PartitionEndRecord.
        """

        end_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        record_sequence: str = proto.Field(
            proto.STRING,
            number=2,
        )
        partition_token: str = proto.Field(
            proto.STRING,
            number=3,
        )

    class PartitionEventRecord(proto.Message):
        r"""A partition event record describes key range changes for a change
        stream partition. The changes to a row defined by its primary key
        can be captured in one change stream partition for a specific time
        range, and then be captured in a different change stream partition
        for a different time range. This movement of key ranges across
        change stream partitions is a reflection of activities, such as
        Spanner's dynamic splitting and load balancing, etc. Processing this
        event is needed if users want to guarantee processing of the changes
        for any key in timestamp order. If time ordered processing of
        changes for a primary key is not needed, this event can be ignored.
        To guarantee time ordered processing for each primary key, if the
        event describes move-ins, the reader of this partition needs to wait
        until the readers of the source partitions have processed all
        records with timestamps <= this
        PartitionEventRecord.commit_timestamp, before advancing beyond this
        PartitionEventRecord. If the event describes move-outs, the reader
        can notify the readers of the destination partitions that they can
        continue processing.

        Attributes:
            commit_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Indicates the commit timestamp at which the key range change
                occurred. DataChangeRecord.commit_timestamps,
                PartitionStartRecord.start_timestamps,
                PartitionEventRecord.commit_timestamps, and
                PartitionEndRecord.end_timestamps can have the same value in
                the same partition.
            record_sequence (str):
                Record sequence numbers are unique and monotonically
                increasing (but not necessarily contiguous) for a specific
                timestamp across record types in the same partition. To
                guarantee ordered processing, the reader should process
                records (of potentially different types) in record_sequence
                order for a specific timestamp in the same partition.
            partition_token (str):
                Unique partition identifier describing the partition this
                event occurred on.
                [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token]
                is equal to the partition token of the change stream
                partition currently queried to return this
                PartitionEventRecord.
            move_in_events (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.PartitionEventRecord.MoveInEvent]):
                Set when one or more key ranges are moved into the change
                stream partition identified by
                [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token].

                Example: Two key ranges are moved into partition (P1) from
                partition (P2) and partition (P3) in a single transaction at
                timestamp T.

                The PartitionEventRecord returned in P1 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P1" move_in_events { source_partition_token: "P2" }
                move_in_events { source_partition_token: "P3" } }

                The PartitionEventRecord returned in P2 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P2" move_out_events { destination_partition_token: "P1" } }

                The PartitionEventRecord returned in P3 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P3" move_out_events { destination_partition_token: "P1" } }
            move_out_events (MutableSequence[google.cloud.spanner_v1.types.ChangeStreamRecord.PartitionEventRecord.MoveOutEvent]):
                Set when one or more key ranges are moved out of the change
                stream partition identified by
                [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token].

                Example: Two key ranges are moved out of partition (P1) to
                partition (P2) and partition (P3) in a single transaction at
                timestamp T.

                The PartitionEventRecord returned in P1 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P1" move_out_events { destination_partition_token: "P2" }
                move_out_events { destination_partition_token: "P3" } }

                The PartitionEventRecord returned in P2 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P2" move_in_events { source_partition_token: "P1" } }

                The PartitionEventRecord returned in P3 will reflect the
                move as:

                PartitionEventRecord { commit_timestamp: T partition_token:
                "P3" move_in_events { source_partition_token: "P1" } }
        """

        class MoveInEvent(proto.Message):
            r"""Describes move-in of the key ranges into the change stream partition
            identified by
            [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token].

            To maintain processing the changes for a particular key in timestamp
            order, the query processing the change stream partition identified
            by
            [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token]
            should not advance beyond the partition event record commit
            timestamp until the queries processing the source change stream
            partitions have processed all change stream records with timestamps
            <= the partition event record commit timestamp.

            Attributes:
                source_partition_token (str):
                    An unique partition identifier describing the
                    source change stream partition that recorded
                    changes for the key range that is moving into
                    this partition.
            """

            source_partition_token: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class MoveOutEvent(proto.Message):
            r"""Describes move-out of the key ranges out of the change stream
            partition identified by
            [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token].

            To maintain processing the changes for a particular key in timestamp
            order, the query processing the
            [MoveOutEvent][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.MoveOutEvent]
            in the partition identified by
            [partition_token][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.partition_token]
            should inform the queries processing the destination partitions that
            they can unblock and proceed processing records past the
            [commit_timestamp][google.spanner.v1.ChangeStreamRecord.PartitionEventRecord.commit_timestamp].

            Attributes:
                destination_partition_token (str):
                    An unique partition identifier describing the
                    destination change stream partition that will
                    record changes for the key range that is moving
                    out of this partition.
            """

            destination_partition_token: str = proto.Field(
                proto.STRING,
                number=1,
            )

        commit_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        record_sequence: str = proto.Field(
            proto.STRING,
            number=2,
        )
        partition_token: str = proto.Field(
            proto.STRING,
            number=3,
        )
        move_in_events: MutableSequence[
            "ChangeStreamRecord.PartitionEventRecord.MoveInEvent"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="ChangeStreamRecord.PartitionEventRecord.MoveInEvent",
        )
        move_out_events: MutableSequence[
            "ChangeStreamRecord.PartitionEventRecord.MoveOutEvent"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=5,
            message="ChangeStreamRecord.PartitionEventRecord.MoveOutEvent",
        )

    data_change_record: DataChangeRecord = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="record",
        message=DataChangeRecord,
    )
    heartbeat_record: HeartbeatRecord = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="record",
        message=HeartbeatRecord,
    )
    partition_start_record: PartitionStartRecord = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="record",
        message=PartitionStartRecord,
    )
    partition_end_record: PartitionEndRecord = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="record",
        message=PartitionEndRecord,
    )
    partition_event_record: PartitionEventRecord = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="record",
        message=PartitionEventRecord,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

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

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.spanner.v1",
    manifest={
        "TransactionOptions",
        "Transaction",
        "TransactionSelector",
        "MultiplexedSessionPrecommitToken",
    },
)


class TransactionOptions(proto.Message):
    r"""Options to use for transactions.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        read_write (google.cloud.spanner_v1.types.TransactionOptions.ReadWrite):
            Transaction may write.

            Authorization to begin a read-write transaction requires
            ``spanner.databases.beginOrRollbackReadWriteTransaction``
            permission on the ``session`` resource.

            This field is a member of `oneof`_ ``mode``.
        partitioned_dml (google.cloud.spanner_v1.types.TransactionOptions.PartitionedDml):
            Partitioned DML transaction.

            Authorization to begin a Partitioned DML transaction
            requires
            ``spanner.databases.beginPartitionedDmlTransaction``
            permission on the ``session`` resource.

            This field is a member of `oneof`_ ``mode``.
        read_only (google.cloud.spanner_v1.types.TransactionOptions.ReadOnly):
            Transaction does not write.

            Authorization to begin a read-only transaction requires
            ``spanner.databases.beginReadOnlyTransaction`` permission on
            the ``session`` resource.

            This field is a member of `oneof`_ ``mode``.
        exclude_txn_from_change_streams (bool):
            When ``exclude_txn_from_change_streams`` is set to ``true``,
            it prevents read or write transactions from being tracked in
            change streams.

            - If the DDL option ``allow_txn_exclusion`` is set to
              ``true``, then the updates made within this transaction
              aren't recorded in the change stream.

            - If you don't set the DDL option ``allow_txn_exclusion`` or
              if it's set to ``false``, then the updates made within
              this transaction are recorded in the change stream.

            When ``exclude_txn_from_change_streams`` is set to ``false``
            or not set, modifications from this transaction are recorded
            in all change streams that are tracking columns modified by
            these transactions.

            The ``exclude_txn_from_change_streams`` option can only be
            specified for read-write or partitioned DML transactions,
            otherwise the API returns an ``INVALID_ARGUMENT`` error.
        isolation_level (google.cloud.spanner_v1.types.TransactionOptions.IsolationLevel):
            Isolation level for the transaction.
    """

    class IsolationLevel(proto.Enum):
        r"""``IsolationLevel`` is used when setting ``isolation_level`` for a
        transaction.

        Values:
            ISOLATION_LEVEL_UNSPECIFIED (0):
                Default value.

                If the value is not specified, the ``SERIALIZABLE``
                isolation level is used.
            SERIALIZABLE (1):
                All transactions appear as if they executed in a serial
                order, even if some of the reads, writes, and other
                operations of distinct transactions actually occurred in
                parallel. Spanner assigns commit timestamps that reflect the
                order of committed transactions to implement this property.
                Spanner offers a stronger guarantee than serializability
                called external consistency. For more information, see
                `TrueTime and external
                consistency <https://cloud.google.com/spanner/docs/true-time-external-consistency#serializability>`__.
            REPEATABLE_READ (2):
                All reads performed during the transaction observe a
                consistent snapshot of the database, and the transaction is
                only successfully committed in the absence of conflicts
                between its updates and any concurrent updates that have
                occurred since that snapshot. Consequently, in contrast to
                ``SERIALIZABLE`` transactions, only write-write conflicts
                are detected in snapshot transactions.

                This isolation level does not support Read-only and
                Partitioned DML transactions.

                When ``REPEATABLE_READ`` is specified on a read-write
                transaction, the locking semantics default to
                ``OPTIMISTIC``.
        """
        ISOLATION_LEVEL_UNSPECIFIED = 0
        SERIALIZABLE = 1
        REPEATABLE_READ = 2

    class ReadWrite(proto.Message):
        r"""Message type to initiate a read-write transaction. Currently
        this transaction type has no options.

        Attributes:
            read_lock_mode (google.cloud.spanner_v1.types.TransactionOptions.ReadWrite.ReadLockMode):
                Read lock mode for the transaction.
            multiplexed_session_previous_transaction_id (bytes):
                Optional. Clients should pass the transaction
                ID of the previous transaction attempt that was
                aborted if this transaction is being executed on
                a multiplexed session.
        """

        class ReadLockMode(proto.Enum):
            r"""``ReadLockMode`` is used to set the read lock mode for read-write
            transactions.

            Values:
                READ_LOCK_MODE_UNSPECIFIED (0):
                    Default value.

                    - If isolation level is
                      [REPEATABLE_READ][google.spanner.v1.TransactionOptions.IsolationLevel.REPEATABLE_READ],
                      then it is an error to specify ``read_lock_mode``. Locking
                      semantics default to ``OPTIMISTIC``. No validation checks
                      are done for reads, except to validate that the data that
                      was served at the snapshot time is unchanged at commit
                      time in the following cases:

                      1. reads done as part of queries that use
                         ``SELECT FOR UPDATE``
                      2. reads done as part of statements with a
                         ``LOCK_SCANNED_RANGES`` hint
                      3. reads done as part of DML statements

                    - At all other isolation levels, if ``read_lock_mode`` is
                      the default value, then pessimistic read locks are used.
                PESSIMISTIC (1):
                    Pessimistic lock mode.

                    Read locks are acquired immediately on read. Semantics
                    described only applies to
                    [SERIALIZABLE][google.spanner.v1.TransactionOptions.IsolationLevel.SERIALIZABLE]
                    isolation.
                OPTIMISTIC (2):
                    Optimistic lock mode.

                    Locks for reads within the transaction are not acquired on
                    read. Instead the locks are acquired on a commit to validate
                    that read/queried data has not changed since the transaction
                    started. Semantics described only applies to
                    [SERIALIZABLE][google.spanner.v1.TransactionOptions.IsolationLevel.SERIALIZABLE]
                    isolation.
            """
            READ_LOCK_MODE_UNSPECIFIED = 0
            PESSIMISTIC = 1
            OPTIMISTIC = 2

        read_lock_mode: "TransactionOptions.ReadWrite.ReadLockMode" = proto.Field(
            proto.ENUM,
            number=1,
            enum="TransactionOptions.ReadWrite.ReadLockMode",
        )
        multiplexed_session_previous_transaction_id: bytes = proto.Field(
            proto.BYTES,
            number=2,
        )

    class PartitionedDml(proto.Message):
        r"""Message type to initiate a Partitioned DML transaction."""

    class ReadOnly(proto.Message):
        r"""Message type to initiate a read-only transaction.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            strong (bool):
                Read at a timestamp where all previously
                committed transactions are visible.

                This field is a member of `oneof`_ ``timestamp_bound``.
            min_read_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Executes all reads at a timestamp >= ``min_read_timestamp``.

                This is useful for requesting fresher data than some
                previous read, or data that is fresh enough to observe the
                effects of some previously committed transaction whose
                timestamp is known.

                Note that this option can only be used in single-use
                transactions.

                A timestamp in RFC3339 UTC "Zulu" format, accurate to
                nanoseconds. Example: ``"2014-10-02T15:01:23.045123456Z"``.

                This field is a member of `oneof`_ ``timestamp_bound``.
            max_staleness (google.protobuf.duration_pb2.Duration):
                Read data at a timestamp >= ``NOW - max_staleness`` seconds.
                Guarantees that all writes that have committed more than the
                specified number of seconds ago are visible. Because Cloud
                Spanner chooses the exact timestamp, this mode works even if
                the client's local clock is substantially skewed from Cloud
                Spanner commit timestamps.

                Useful for reading the freshest data available at a nearby
                replica, while bounding the possible staleness if the local
                replica has fallen behind.

                Note that this option can only be used in single-use
                transactions.

                This field is a member of `oneof`_ ``timestamp_bound``.
            read_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                Executes all reads at the given timestamp. Unlike other
                modes, reads at a specific timestamp are repeatable; the
                same read at the same timestamp always returns the same
                data. If the timestamp is in the future, the read is blocked
                until the specified timestamp, modulo the read's deadline.

                Useful for large scale consistent reads such as mapreduces,
                or for coordinating many reads against a consistent snapshot
                of the data.

                A timestamp in RFC3339 UTC "Zulu" format, accurate to
                nanoseconds. Example: ``"2014-10-02T15:01:23.045123456Z"``.

                This field is a member of `oneof`_ ``timestamp_bound``.
            exact_staleness (google.protobuf.duration_pb2.Duration):
                Executes all reads at a timestamp that is
                ``exact_staleness`` old. The timestamp is chosen soon after
                the read is started.

                Guarantees that all writes that have committed more than the
                specified number of seconds ago are visible. Because Cloud
                Spanner chooses the exact timestamp, this mode works even if
                the client's local clock is substantially skewed from Cloud
                Spanner commit timestamps.

                Useful for reading at nearby replicas without the
                distributed timestamp negotiation overhead of
                ``max_staleness``.

                This field is a member of `oneof`_ ``timestamp_bound``.
            return_read_timestamp (bool):
                If true, the Cloud Spanner-selected read timestamp is
                included in the [Transaction][google.spanner.v1.Transaction]
                message that describes the transaction.
        """

        strong: bool = proto.Field(
            proto.BOOL,
            number=1,
            oneof="timestamp_bound",
        )
        min_read_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="timestamp_bound",
            message=timestamp_pb2.Timestamp,
        )
        max_staleness: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="timestamp_bound",
            message=duration_pb2.Duration,
        )
        read_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof="timestamp_bound",
            message=timestamp_pb2.Timestamp,
        )
        exact_staleness: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof="timestamp_bound",
            message=duration_pb2.Duration,
        )
        return_read_timestamp: bool = proto.Field(
            proto.BOOL,
            number=6,
        )

    read_write: ReadWrite = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="mode",
        message=ReadWrite,
    )
    partitioned_dml: PartitionedDml = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="mode",
        message=PartitionedDml,
    )
    read_only: ReadOnly = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="mode",
        message=ReadOnly,
    )
    exclude_txn_from_change_streams: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    isolation_level: IsolationLevel = proto.Field(
        proto.ENUM,
        number=6,
        enum=IsolationLevel,
    )


class Transaction(proto.Message):
    r"""A transaction.

    Attributes:
        id (bytes):
            ``id`` may be used to identify the transaction in subsequent
            [Read][google.spanner.v1.Spanner.Read],
            [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql],
            [Commit][google.spanner.v1.Spanner.Commit], or
            [Rollback][google.spanner.v1.Spanner.Rollback] calls.

            Single-use read-only transactions do not have IDs, because
            single-use transactions do not support multiple requests.
        read_timestamp (google.protobuf.timestamp_pb2.Timestamp):
            For snapshot read-only transactions, the read timestamp
            chosen for the transaction. Not returned by default: see
            [TransactionOptions.ReadOnly.return_read_timestamp][google.spanner.v1.TransactionOptions.ReadOnly.return_read_timestamp].

            A timestamp in RFC3339 UTC "Zulu" format, accurate to
            nanoseconds. Example: ``"2014-10-02T15:01:23.045123456Z"``.
        precommit_token (google.cloud.spanner_v1.types.MultiplexedSessionPrecommitToken):
            A precommit token is included in the response of a
            BeginTransaction request if the read-write transaction is on
            a multiplexed session and a mutation_key was specified in
            the
            [BeginTransaction][google.spanner.v1.BeginTransactionRequest].
            The precommit token with the highest sequence number from
            this transaction attempt should be passed to the
            [Commit][google.spanner.v1.Spanner.Commit] request for this
            transaction.
    """

    id: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    read_timestamp: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    precommit_token: "MultiplexedSessionPrecommitToken" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="MultiplexedSessionPrecommitToken",
    )


class TransactionSelector(proto.Message):
    r"""This message is used to select the transaction in which a
    [Read][google.spanner.v1.Spanner.Read] or
    [ExecuteSql][google.spanner.v1.Spanner.ExecuteSql] call runs.

    See [TransactionOptions][google.spanner.v1.TransactionOptions] for
    more information about transactions.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        single_use (google.cloud.spanner_v1.types.TransactionOptions):
            Execute the read or SQL query in a temporary
            transaction. This is the most efficient way to
            execute a transaction that consists of a single
            SQL query.

            This field is a member of `oneof`_ ``selector``.
        id (bytes):
            Execute the read or SQL query in a
            previously-started transaction.

            This field is a member of `oneof`_ ``selector``.
        begin (google.cloud.spanner_v1.types.TransactionOptions):
            Begin a new transaction and execute this read or SQL query
            in it. The transaction ID of the new transaction is returned
            in
            [ResultSetMetadata.transaction][google.spanner.v1.ResultSetMetadata.transaction],
            which is a [Transaction][google.spanner.v1.Transaction].

            This field is a member of `oneof`_ ``selector``.
    """

    single_use: "TransactionOptions" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="selector",
        message="TransactionOptions",
    )
    id: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="selector",
    )
    begin: "TransactionOptions" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="selector",
        message="TransactionOptions",
    )


class MultiplexedSessionPrecommitToken(proto.Message):
    r"""When a read-write transaction is executed on a multiplexed session,
    this precommit token is sent back to the client as a part of the
    [Transaction][google.spanner.v1.Transaction] message in the
    [BeginTransaction][google.spanner.v1.BeginTransactionRequest]
    response and also as a part of the
    [ResultSet][google.spanner.v1.ResultSet] and
    [PartialResultSet][google.spanner.v1.PartialResultSet] responses.

    Attributes:
        precommit_token (bytes):
            Opaque precommit token.
        seq_num (int):
            An incrementing seq number is generated on
            every precommit token that is returned. Clients
            should remember the precommit token with the
            highest sequence number from the current
            transaction attempt.
    """

    precommit_token: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    seq_num: int = proto.Field(
        proto.INT32,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

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

from google.cloud.bigtable_v2.types import data
from google.cloud.bigtable_v2.types import request_stats as gb_request_stats
from google.cloud.bigtable_v2.types import types
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.v2",
    manifest={
        "ReadRowsRequest",
        "ReadRowsResponse",
        "SampleRowKeysRequest",
        "SampleRowKeysResponse",
        "MutateRowRequest",
        "MutateRowResponse",
        "MutateRowsRequest",
        "MutateRowsResponse",
        "RateLimitInfo",
        "CheckAndMutateRowRequest",
        "CheckAndMutateRowResponse",
        "PingAndWarmRequest",
        "PingAndWarmResponse",
        "ReadModifyWriteRowRequest",
        "ReadModifyWriteRowResponse",
        "GenerateInitialChangeStreamPartitionsRequest",
        "GenerateInitialChangeStreamPartitionsResponse",
        "ReadChangeStreamRequest",
        "ReadChangeStreamResponse",
        "ExecuteQueryRequest",
        "ExecuteQueryResponse",
        "PrepareQueryRequest",
        "PrepareQueryResponse",
    },
)


class ReadRowsRequest(proto.Message):
    r"""Request message for Bigtable.ReadRows.

    Attributes:
        table_name (str):
            Optional. The unique name of the table from which to read.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView from which
            to read.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        materialized_view_name (str):
            Optional. The unique name of the MaterializedView from which
            to read.

            Values are of the form
            ``projects/<project>/instances/<instance>/materializedViews/<materialized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
        rows (google.cloud.bigtable_v2.types.RowSet):
            The row keys and/or ranges to read
            sequentially. If not specified, reads from all
            rows.
        filter (google.cloud.bigtable_v2.types.RowFilter):
            The filter to apply to the contents of the
            specified row(s). If unset, reads the entirety
            of each row.
        rows_limit (int):
            The read will stop after committing to N
            rows' worth of results. The default (zero) is to
            return all results.
        request_stats_view (google.cloud.bigtable_v2.types.ReadRowsRequest.RequestStatsView):
            The view into RequestStats, as described
            above.
        reversed (bool):
            Experimental API - Please note that this API is currently
            experimental and can change in the future.

            Return rows in lexiographical descending order of the row
            keys. The row contents will not be affected by this flag.

            Example result set:

            ::

                [
                  {key: "k2", "f:col1": "v1", "f:col2": "v1"},
                  {key: "k1", "f:col1": "v2", "f:col2": "v2"}
                ]
    """

    class RequestStatsView(proto.Enum):
        r"""The desired view into RequestStats that should be returned in
        the response.
        See also: RequestStats message.

        Values:
            REQUEST_STATS_VIEW_UNSPECIFIED (0):
                The default / unset value. The API will
                default to the NONE option below.
            REQUEST_STATS_NONE (1):
                Do not include any RequestStats in the
                response. This will leave the RequestStats
                embedded message unset in the response.
            REQUEST_STATS_FULL (2):
                Include the full set of available
                RequestStats in the response, applicable to this
                read.
        """
        REQUEST_STATS_VIEW_UNSPECIFIED = 0
        REQUEST_STATS_NONE = 1
        REQUEST_STATS_FULL = 2

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=9,
    )
    materialized_view_name: str = proto.Field(
        proto.STRING,
        number=11,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=5,
    )
    rows: data.RowSet = proto.Field(
        proto.MESSAGE,
        number=2,
        message=data.RowSet,
    )
    filter: data.RowFilter = proto.Field(
        proto.MESSAGE,
        number=3,
        message=data.RowFilter,
    )
    rows_limit: int = proto.Field(
        proto.INT64,
        number=4,
    )
    request_stats_view: RequestStatsView = proto.Field(
        proto.ENUM,
        number=6,
        enum=RequestStatsView,
    )
    reversed: bool = proto.Field(
        proto.BOOL,
        number=7,
    )


class ReadRowsResponse(proto.Message):
    r"""Response message for Bigtable.ReadRows.

    Attributes:
        chunks (MutableSequence[google.cloud.bigtable_v2.types.ReadRowsResponse.CellChunk]):
            A collection of a row's contents as part of
            the read request.
        last_scanned_row_key (bytes):
            Optionally the server might return the row
            key of the last row it has scanned.  The client
            can use this to construct a more efficient retry
            request if needed: any row keys or portions of
            ranges less than this row key can be dropped
            from the request. This is primarily useful for
            cases where the server has read a lot of data
            that was filtered out since the last committed
            row key, allowing the client to skip that work
            on a retry.
        request_stats (google.cloud.bigtable_v2.types.RequestStats):
            If requested, return enhanced query performance statistics.
            The field request_stats is empty in a streamed response
            unless the ReadRowsResponse message contains request_stats
            in the last message of the stream. Always returned when
            requested, even when the read request returns an empty
            response.
    """

    class CellChunk(proto.Message):
        r"""Specifies a piece of a row's contents returned as part of the
        read response stream.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            row_key (bytes):
                The row key for this chunk of data.  If the
                row key is empty, this CellChunk is a
                continuation of the same row as the previous
                CellChunk in the response stream, even if that
                CellChunk was in a previous ReadRowsResponse
                message.
            family_name (google.protobuf.wrappers_pb2.StringValue):
                The column family name for this chunk of data. If this
                message is not present this CellChunk is a continuation of
                the same column family as the previous CellChunk. The empty
                string can occur as a column family name in a response so
                clients must check explicitly for the presence of this
                message, not just for ``family_name.value`` being non-empty.
            qualifier (google.protobuf.wrappers_pb2.BytesValue):
                The column qualifier for this chunk of data. If this message
                is not present, this CellChunk is a continuation of the same
                column as the previous CellChunk. Column qualifiers may be
                empty so clients must check for the presence of this
                message, not just for ``qualifier.value`` being non-empty.
            timestamp_micros (int):
                The cell's stored timestamp, which also uniquely identifies
                it within its column. Values are always expressed in
                microseconds, but individual tables may set a coarser
                granularity to further restrict the allowed values. For
                example, a table which specifies millisecond granularity
                will only allow values of ``timestamp_micros`` which are
                multiples of 1000. Timestamps are only set in the first
                CellChunk per cell (for cells split into multiple chunks).
            labels (MutableSequence[str]):
                Labels applied to the cell by a
                [RowFilter][google.bigtable.v2.RowFilter]. Labels are only
                set on the first CellChunk per cell.
            value (bytes):
                The value stored in the cell.  Cell values
                can be split across multiple CellChunks.  In
                that case only the value field will be set in
                CellChunks after the first: the timestamp and
                labels will only be present in the first
                CellChunk, even if the first CellChunk came in a
                previous ReadRowsResponse.
            value_size (int):
                If this CellChunk is part of a chunked cell value and this
                is not the final chunk of that cell, value_size will be set
                to the total length of the cell value. The client can use
                this size to pre-allocate memory to hold the full cell
                value.
            reset_row (bool):
                Indicates that the client should drop all previous chunks
                for ``row_key``, as it will be re-read from the beginning.

                This field is a member of `oneof`_ ``row_status``.
            commit_row (bool):
                Indicates that the client can safely process all previous
                chunks for ``row_key``, as its data has been fully read.

                This field is a member of `oneof`_ ``row_status``.
        """

        row_key: bytes = proto.Field(
            proto.BYTES,
            number=1,
        )
        family_name: wrappers_pb2.StringValue = proto.Field(
            proto.MESSAGE,
            number=2,
            message=wrappers_pb2.StringValue,
        )
        qualifier: wrappers_pb2.BytesValue = proto.Field(
            proto.MESSAGE,
            number=3,
            message=wrappers_pb2.BytesValue,
        )
        timestamp_micros: int = proto.Field(
            proto.INT64,
            number=4,
        )
        labels: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=5,
        )
        value: bytes = proto.Field(
            proto.BYTES,
            number=6,
        )
        value_size: int = proto.Field(
            proto.INT32,
            number=7,
        )
        reset_row: bool = proto.Field(
            proto.BOOL,
            number=8,
            oneof="row_status",
        )
        commit_row: bool = proto.Field(
            proto.BOOL,
            number=9,
            oneof="row_status",
        )

    chunks: MutableSequence[CellChunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=CellChunk,
    )
    last_scanned_row_key: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    request_stats: gb_request_stats.RequestStats = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gb_request_stats.RequestStats,
    )


class SampleRowKeysRequest(proto.Message):
    r"""Request message for Bigtable.SampleRowKeys.

    Attributes:
        table_name (str):
            Optional. The unique name of the table from which to sample
            row keys.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView from which
            to sample row keys.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        materialized_view_name (str):
            Optional. The unique name of the MaterializedView from which
            to read.

            Values are of the form
            ``projects/<project>/instances/<instance>/materializedViews/<materialized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    materialized_view_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SampleRowKeysResponse(proto.Message):
    r"""Response message for Bigtable.SampleRowKeys.

    Attributes:
        row_key (bytes):
            Sorted streamed sequence of sample row keys
            in the table. The table might have contents
            before the first row key in the list and after
            the last one, but a key containing the empty
            string indicates "end of table" and will be the
            last response given, if present.
            Note that row keys in this list may not have
            ever been written to or read from, and users
            should therefore not make any assumptions about
            the row key structure that are specific to their
            use case.
        offset_bytes (int):
            Approximate total storage space used by all rows in the
            table which precede ``row_key``. Buffering the contents of
            all rows between two subsequent samples would require space
            roughly equal to the difference in their ``offset_bytes``
            fields.
    """

    row_key: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    offset_bytes: int = proto.Field(
        proto.INT64,
        number=2,
    )


class MutateRowRequest(proto.Message):
    r"""Request message for Bigtable.MutateRow.

    Attributes:
        table_name (str):
            Optional. The unique name of the table to which the mutation
            should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView to which the
            mutation should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
        row_key (bytes):
            Required. The key of the row to which the
            mutation should be applied.
        mutations (MutableSequence[google.cloud.bigtable_v2.types.Mutation]):
            Required. Changes to be atomically applied to
            the specified row. Entries are applied in order,
            meaning that earlier mutations can be masked by
            later ones. Must contain at least one entry and
            at most 100000.
        idempotency (google.cloud.bigtable_v2.types.Idempotency):
            If set consistently across retries, prevents
            this mutation from being double applied to
            aggregate column families within a 15m window.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    row_key: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    mutations: MutableSequence[data.Mutation] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=data.Mutation,
    )
    idempotency: data.Idempotency = proto.Field(
        proto.MESSAGE,
        number=8,
        message=data.Idempotency,
    )


class MutateRowResponse(proto.Message):
    r"""Response message for Bigtable.MutateRow."""


class MutateRowsRequest(proto.Message):
    r"""Request message for BigtableService.MutateRows.

    Attributes:
        table_name (str):
            Optional. The unique name of the table to which the
            mutations should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView to which the
            mutations should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
        entries (MutableSequence[google.cloud.bigtable_v2.types.MutateRowsRequest.Entry]):
            Required. The row keys and corresponding
            mutations to be applied in bulk. Each entry is
            applied as an atomic mutation, but the entries
            may be applied in arbitrary order (even between
            entries for the same row). At least one entry
            must be specified, and in total the entries can
            contain at most 100000 mutations.
    """

    class Entry(proto.Message):
        r"""A mutation for a given row.

        Attributes:
            row_key (bytes):
                The key of the row to which the ``mutations`` should be
                applied.
            mutations (MutableSequence[google.cloud.bigtable_v2.types.Mutation]):
                Required. Changes to be atomically applied to
                the specified row. Mutations are applied in
                order, meaning that earlier mutations can be
                masked by later ones. You must specify at least
                one mutation.
            idempotency (google.cloud.bigtable_v2.types.Idempotency):
                If set consistently across retries, prevents
                this mutation from being double applied to
                aggregate column families within a 15m window.
        """

        row_key: bytes = proto.Field(
            proto.BYTES,
            number=1,
        )
        mutations: MutableSequence[data.Mutation] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=data.Mutation,
        )
        idempotency: data.Idempotency = proto.Field(
            proto.MESSAGE,
            number=3,
            message=data.Idempotency,
        )

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    entries: MutableSequence[Entry] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=Entry,
    )


class MutateRowsResponse(proto.Message):
    r"""Response message for BigtableService.MutateRows.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        entries (MutableSequence[google.cloud.bigtable_v2.types.MutateRowsResponse.Entry]):
            One or more results for Entries from the
            batch request.
        rate_limit_info (google.cloud.bigtable_v2.types.RateLimitInfo):
            Information about how client should limit the
            rate (QPS). Primirily used by supported official
            Cloud Bigtable clients. If unset, the rate limit
            info is not provided by the server.

            This field is a member of `oneof`_ ``_rate_limit_info``.
    """

    class Entry(proto.Message):
        r"""The result of applying a passed mutation in the original
        request.

        Attributes:
            index (int):
                The index into the original request's ``entries`` list of
                the Entry for which a result is being reported.
            status (google.rpc.status_pb2.Status):
                The result of the request Entry identified by ``index``.
                Depending on how requests are batched during execution, it
                is possible for one Entry to fail due to an error with
                another Entry. In the event that this occurs, the same error
                will be reported for both entries.
        """

        index: int = proto.Field(
            proto.INT64,
            number=1,
        )
        status: status_pb2.Status = proto.Field(
            proto.MESSAGE,
            number=2,
            message=status_pb2.Status,
        )

    entries: MutableSequence[Entry] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=Entry,
    )
    rate_limit_info: "RateLimitInfo" = proto.Field(
        proto.MESSAGE,
        number=3,
        optional=True,
        message="RateLimitInfo",
    )


class RateLimitInfo(proto.Message):
    r"""Information about how client should adjust the load to
    Bigtable.

    Attributes:
        period (google.protobuf.duration_pb2.Duration):
            Time that clients should wait before
            adjusting the target rate again. If clients
            adjust rate too frequently, the impact of the
            previous adjustment may not have been taken into
            account and may over-throttle or under-throttle.
            If clients adjust rate too slowly, they will not
            be responsive to load changes on server side,
            and may over-throttle or under-throttle.
        factor (float):
            If it has been at least one ``period`` since the last load
            adjustment, the client should multiply the current load by
            this value to get the new target load. For example, if the
            current load is 100 and ``factor`` is 0.8, the new target
            load should be 80. After adjusting, the client should ignore
            ``factor`` until another ``period`` has passed.

            The client can measure its load using any unit that's
            comparable over time. For example, QPS can be used as long
            as each request involves a similar amount of work.
    """

    period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    factor: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )


class CheckAndMutateRowRequest(proto.Message):
    r"""Request message for Bigtable.CheckAndMutateRow.

    Attributes:
        table_name (str):
            Optional. The unique name of the table to which the
            conditional mutation should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView to which the
            conditional mutation should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
        row_key (bytes):
            Required. The key of the row to which the
            conditional mutation should be applied.
        predicate_filter (google.cloud.bigtable_v2.types.RowFilter):
            The filter to be applied to the contents of the specified
            row. Depending on whether or not any results are yielded,
            either ``true_mutations`` or ``false_mutations`` will be
            executed. If unset, checks that the row contains any values
            at all.
        true_mutations (MutableSequence[google.cloud.bigtable_v2.types.Mutation]):
            Changes to be atomically applied to the specified row if
            ``predicate_filter`` yields at least one cell when applied
            to ``row_key``. Entries are applied in order, meaning that
            earlier mutations can be masked by later ones. Must contain
            at least one entry if ``false_mutations`` is empty, and at
            most 100000.
        false_mutations (MutableSequence[google.cloud.bigtable_v2.types.Mutation]):
            Changes to be atomically applied to the specified row if
            ``predicate_filter`` does not yield any cells when applied
            to ``row_key``. Entries are applied in order, meaning that
            earlier mutations can be masked by later ones. Must contain
            at least one entry if ``true_mutations`` is empty, and at
            most 100000.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=9,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=7,
    )
    row_key: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    predicate_filter: data.RowFilter = proto.Field(
        proto.MESSAGE,
        number=6,
        message=data.RowFilter,
    )
    true_mutations: MutableSequence[data.Mutation] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=data.Mutation,
    )
    false_mutations: MutableSequence[data.Mutation] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=data.Mutation,
    )


class CheckAndMutateRowResponse(proto.Message):
    r"""Response message for Bigtable.CheckAndMutateRow.

    Attributes:
        predicate_matched (bool):
            Whether or not the request's ``predicate_filter`` yielded
            any results for the specified row.
    """

    predicate_matched: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class PingAndWarmRequest(proto.Message):
    r"""Request message for client connection keep-alive and warming.

    Attributes:
        name (str):
            Required. The unique name of the instance to check
            permissions for as well as respond. Values are of the form
            ``projects/<project>/instances/<instance>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class PingAndWarmResponse(proto.Message):
    r"""Response message for Bigtable.PingAndWarm connection
    keepalive and warming.

    """


class ReadModifyWriteRowRequest(proto.Message):
    r"""Request message for Bigtable.ReadModifyWriteRow.

    Attributes:
        table_name (str):
            Optional. The unique name of the table to which the
            read/modify/write rules should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
        authorized_view_name (str):
            Optional. The unique name of the AuthorizedView to which the
            read/modify/write rules should be applied.

            Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>/authorizedViews/<authorized_view>``.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used.
        row_key (bytes):
            Required. The key of the row to which the
            read/modify/write rules should be applied.
        rules (MutableSequence[google.cloud.bigtable_v2.types.ReadModifyWriteRule]):
            Required. Rules specifying how the specified
            row's contents are to be transformed into
            writes. Entries are applied in order, meaning
            that earlier rules will affect the results of
            later ones. At least one entry must be
            specified, and there can be at most 100000
            rules.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    authorized_view_name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    row_key: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    rules: MutableSequence[data.ReadModifyWriteRule] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=data.ReadModifyWriteRule,
    )


class ReadModifyWriteRowResponse(proto.Message):
    r"""Response message for Bigtable.ReadModifyWriteRow.

    Attributes:
        row (google.cloud.bigtable_v2.types.Row):
            A Row containing the new contents of all
            cells modified by the request.
    """

    row: data.Row = proto.Field(
        proto.MESSAGE,
        number=1,
        message=data.Row,
    )


class GenerateInitialChangeStreamPartitionsRequest(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam
    BigtableIO. Request message for
    Bigtable.GenerateInitialChangeStreamPartitions.

    Attributes:
        table_name (str):
            Required. The unique name of the table from which to get
            change stream partitions. Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
            Change streaming must be enabled on the table.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used. Single cluster routing
            must be configured on the profile.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GenerateInitialChangeStreamPartitionsResponse(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam
    BigtableIO. Response message for
    Bigtable.GenerateInitialChangeStreamPartitions.

    Attributes:
        partition (google.cloud.bigtable_v2.types.StreamPartition):
            A partition of the change stream.
    """

    partition: data.StreamPartition = proto.Field(
        proto.MESSAGE,
        number=1,
        message=data.StreamPartition,
    )


class ReadChangeStreamRequest(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam
    BigtableIO. Request message for Bigtable.ReadChangeStream.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        table_name (str):
            Required. The unique name of the table from which to read a
            change stream. Values are of the form
            ``projects/<project>/instances/<instance>/tables/<table>``.
            Change streaming must be enabled on the table.
        app_profile_id (str):
            This value specifies routing for replication.
            If not specified, the "default" application
            profile will be used. Single cluster routing
            must be configured on the profile.
        partition (google.cloud.bigtable_v2.types.StreamPartition):
            The partition to read changes from.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Start reading the stream at the specified
            timestamp. This timestamp must be within the
            change stream retention period, less than or
            equal to the current time, and after change
            stream creation, whichever is greater. This
            value is inclusive and will be truncated to
            microsecond granularity.

            This field is a member of `oneof`_ ``start_from``.
        continuation_tokens (google.cloud.bigtable_v2.types.StreamContinuationTokens):
            Tokens that describe how to resume reading a stream where
            reading previously left off. If specified, changes will be
            read starting at the the position. Tokens are delivered on
            the stream as part of ``Heartbeat`` and ``CloseStream``
            messages.

            If a single token is provided, the token's partition must
            exactly match the request's partition. If multiple tokens
            are provided, as in the case of a partition merge, the union
            of the token partitions must exactly cover the request's
            partition. Otherwise, INVALID_ARGUMENT will be returned.

            This field is a member of `oneof`_ ``start_from``.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If specified, OK will be returned when the
            stream advances beyond this time. Otherwise,
            changes will be continuously delivered on the
            stream. This value is inclusive and will be
            truncated to microsecond granularity.
        heartbeat_duration (google.protobuf.duration_pb2.Duration):
            If specified, the duration between ``Heartbeat`` messages on
            the stream. Otherwise, defaults to 5 seconds.
    """

    table_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    partition: data.StreamPartition = proto.Field(
        proto.MESSAGE,
        number=3,
        message=data.StreamPartition,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="start_from",
        message=timestamp_pb2.Timestamp,
    )
    continuation_tokens: data.StreamContinuationTokens = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="start_from",
        message=data.StreamContinuationTokens,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    heartbeat_duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=7,
        message=duration_pb2.Duration,
    )


class ReadChangeStreamResponse(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam
    BigtableIO. Response message for Bigtable.ReadChangeStream.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        data_change (google.cloud.bigtable_v2.types.ReadChangeStreamResponse.DataChange):
            A mutation to the partition.

            This field is a member of `oneof`_ ``stream_record``.
        heartbeat (google.cloud.bigtable_v2.types.ReadChangeStreamResponse.Heartbeat):
            A periodic heartbeat message.

            This field is a member of `oneof`_ ``stream_record``.
        close_stream (google.cloud.bigtable_v2.types.ReadChangeStreamResponse.CloseStream):
            An indication that the stream should be
            closed.

            This field is a member of `oneof`_ ``stream_record``.
    """

    class MutationChunk(proto.Message):
        r"""A partial or complete mutation.

        Attributes:
            chunk_info (google.cloud.bigtable_v2.types.ReadChangeStreamResponse.MutationChunk.ChunkInfo):
                If set, then the mutation is a ``SetCell`` with a chunked
                value across multiple messages.
            mutation (google.cloud.bigtable_v2.types.Mutation):
                If this is a continuation of a chunked message
                (``chunked_value_offset`` > 0), ignore all fields except the
                ``SetCell``'s value and merge it with the previous message
                by concatenating the value fields.
        """

        class ChunkInfo(proto.Message):
            r"""Information about the chunking of this mutation. Only ``SetCell``
            mutations can be chunked, and all chunks for a ``SetCell`` will be
            delivered contiguously with no other mutation types interleaved.

            Attributes:
                chunked_value_size (int):
                    The total value size of all the chunks that make up the
                    ``SetCell``.
                chunked_value_offset (int):
                    The byte offset of this chunk into the total
                    value size of the mutation.
                last_chunk (bool):
                    When true, this is the last chunk of a chunked ``SetCell``.
            """

            chunked_value_size: int = proto.Field(
                proto.INT32,
                number=1,
            )
            chunked_value_offset: int = proto.Field(
                proto.INT32,
                number=2,
            )
            last_chunk: bool = proto.Field(
                proto.BOOL,
                number=3,
            )

        chunk_info: "ReadChangeStreamResponse.MutationChunk.ChunkInfo" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="ReadChangeStreamResponse.MutationChunk.ChunkInfo",
        )
        mutation: data.Mutation = proto.Field(
            proto.MESSAGE,
            number=2,
            message=data.Mutation,
        )

    class DataChange(proto.Message):
        r"""A message corresponding to one or more mutations to the partition
        being streamed. A single logical ``DataChange`` message may also be
        split across a sequence of multiple individual messages. Messages
        other than the first in a sequence will only have the ``type`` and
        ``chunks`` fields populated, with the final message in the sequence
        also containing ``done`` set to true.

        Attributes:
            type_ (google.cloud.bigtable_v2.types.ReadChangeStreamResponse.DataChange.Type):
                The type of the mutation.
            source_cluster_id (str):
                The cluster where the mutation was applied. Not set when
                ``type`` is ``GARBAGE_COLLECTION``.
            row_key (bytes):
                The row key for all mutations that are part of this
                ``DataChange``. If the ``DataChange`` is chunked across
                multiple messages, then this field will only be set for the
                first message.
            commit_timestamp (google.protobuf.timestamp_pb2.Timestamp):
                The timestamp at which the mutation was
                applied on the Bigtable server.
            tiebreaker (int):
                A value that lets stream consumers reconstruct Bigtable's
                conflict resolution semantics.
                https://cloud.google.com/bigtable/docs/writes#conflict-resolution
                In the event that the same row key, column family, column
                qualifier, timestamp are modified on different clusters at
                the same ``commit_timestamp``, the mutation with the larger
                ``tiebreaker`` will be the one chosen for the eventually
                consistent state of the system.
            chunks (MutableSequence[google.cloud.bigtable_v2.types.ReadChangeStreamResponse.MutationChunk]):
                The mutations associated with this change to the partition.
                May contain complete mutations or chunks of a multi-message
                chunked ``DataChange`` record.
            done (bool):
                When true, indicates that the entire ``DataChange`` has been
                read and the client can safely process the message.
            token (str):
                An encoded position for this stream's
                partition to restart reading from. This token is
                for the StreamPartition from the request.
            estimated_low_watermark (google.protobuf.timestamp_pb2.Timestamp):
                An estimate of the commit timestamp that is
                usually lower than or equal to any timestamp for
                a record that will be delivered in the future on
                the stream. It is possible that, under
                particular circumstances that a future record
                has a timestamp that is lower than a previously
                seen timestamp. For an example usage see
                https://beam.apache.org/documentation/basics/#watermarks
        """

        class Type(proto.Enum):
            r"""The type of mutation.

            Values:
                TYPE_UNSPECIFIED (0):
                    The type is unspecified.
                USER (1):
                    A user-initiated mutation.
                GARBAGE_COLLECTION (2):
                    A system-initiated mutation as part of
                    garbage collection.
                    https://cloud.google.com/bigtable/docs/garbage-collection
                CONTINUATION (3):
                    This is a continuation of a multi-message
                    change.
            """
            TYPE_UNSPECIFIED = 0
            USER = 1
            GARBAGE_COLLECTION = 2
            CONTINUATION = 3

        type_: "ReadChangeStreamResponse.DataChange.Type" = proto.Field(
            proto.ENUM,
            number=1,
            enum="ReadChangeStreamResponse.DataChange.Type",
        )
        source_cluster_id: str = proto.Field(
            proto.STRING,
            number=2,
        )
        row_key: bytes = proto.Field(
            proto.BYTES,
            number=3,
        )
        commit_timestamp: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            message=timestamp_pb2.Timestamp,
        )
        tiebreaker: int = proto.Field(
            proto.INT32,
            number=5,
        )
        chunks: MutableSequence[
            "ReadChangeStreamResponse.MutationChunk"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=6,
            message="ReadChangeStreamResponse.MutationChunk",
        )
        done: bool = proto.Field(
            proto.BOOL,
            number=8,
        )
        token: str = proto.Field(
            proto.STRING,
            number=9,
        )
        estimated_low_watermark: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=10,
            message=timestamp_pb2.Timestamp,
        )

    class Heartbeat(proto.Message):
        r"""A periodic message with information that can be used to
        checkpoint the state of a stream.

        Attributes:
            continuation_token (google.cloud.bigtable_v2.types.StreamContinuationToken):
                A token that can be provided to a subsequent
                ``ReadChangeStream`` call to pick up reading at the current
                stream position.
            estimated_low_watermark (google.protobuf.timestamp_pb2.Timestamp):
                An estimate of the commit timestamp that is
                usually lower than or equal to any timestamp for
                a record that will be delivered in the future on
                the stream. It is possible that, under
                particular circumstances that a future record
                has a timestamp that is lower than a previously
                seen timestamp. For an example usage see
                https://beam.apache.org/documentation/basics/#watermarks
        """

        continuation_token: data.StreamContinuationToken = proto.Field(
            proto.MESSAGE,
            number=1,
            message=data.StreamContinuationToken,
        )
        estimated_low_watermark: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=2,
            message=timestamp_pb2.Timestamp,
        )

    class CloseStream(proto.Message):
        r"""A message indicating that the client should stop reading from the
        stream. If status is OK and ``continuation_tokens`` &
        ``new_partitions`` are empty, the stream has finished (for example
        if there was an ``end_time`` specified). If ``continuation_tokens``
        & ``new_partitions`` are present, then a change in partitioning
        requires the client to open a new stream for each token to resume
        reading. Example:

        ::

                                             [B,      D) ends
                                                  |
                                                  v
                          new_partitions:  [A,  C) [C,  E)
            continuation_tokens.partitions:  [B,C) [C,D)
                                             ^---^ ^---^
                                             ^     ^
                                             |     |
                                             |     StreamContinuationToken 2
                                             |
                                             StreamContinuationToken 1

        To read the new partition [A,C), supply the continuation tokens
        whose ranges cover the new partition, for example
        ContinuationToken[A,B) & ContinuationToken[B,C).

        Attributes:
            status (google.rpc.status_pb2.Status):
                The status of the stream.
            continuation_tokens (MutableSequence[google.cloud.bigtable_v2.types.StreamContinuationToken]):
                If non-empty, contains the information needed
                to resume reading their associated partitions.
            new_partitions (MutableSequence[google.cloud.bigtable_v2.types.StreamPartition]):
                If non-empty, contains the new partitions to start reading
                from, which are related to but not necessarily identical to
                the partitions for the above ``continuation_tokens``.
        """

        status: status_pb2.Status = proto.Field(
            proto.MESSAGE,
            number=1,
            message=status_pb2.Status,
        )
        continuation_tokens: MutableSequence[
            data.StreamContinuationToken
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=data.StreamContinuationToken,
        )
        new_partitions: MutableSequence[data.StreamPartition] = proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message=data.StreamPartition,
        )

    data_change: DataChange = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="stream_record",
        message=DataChange,
    )
    heartbeat: Heartbeat = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="stream_record",
        message=Heartbeat,
    )
    close_stream: CloseStream = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="stream_record",
        message=CloseStream,
    )


class ExecuteQueryRequest(proto.Message):
    r"""Request message for Bigtable.ExecuteQuery

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        instance_name (str):
            Required. The unique name of the instance against which the
            query should be executed. Values are of the form
            ``projects/<project>/instances/<instance>``
        app_profile_id (str):
            Optional. This value specifies routing for replication. If
            not specified, the ``default`` application profile will be
            used.
        query (str):
            Required. The query string.

            Exactly one of ``query`` and ``prepared_query`` is required.
            Setting both or neither is an ``INVALID_ARGUMENT``.
        prepared_query (bytes):
            A prepared query that was returned from
            ``PrepareQueryResponse``.

            Exactly one of ``query`` and ``prepared_query`` is required.
            Setting both or neither is an ``INVALID_ARGUMENT``.

            Setting this field also places restrictions on several other
            fields:

            -  ``data_format`` must be empty.
            -  ``validate_only`` must be false.
            -  ``params`` must match the ``param_types`` set in the
               ``PrepareQueryRequest``.
        proto_format (google.cloud.bigtable_v2.types.ProtoFormat):
            Protocol buffer format as described by
            ProtoSchema and ProtoRows messages.

            This field is a member of `oneof`_ ``data_format``.
        resume_token (bytes):
            Optional. If this request is resuming a previously
            interrupted query execution, ``resume_token`` should be
            copied from the last PartialResultSet yielded before the
            interruption. Doing this enables the query execution to
            resume where the last one left off. The rest of the request
            parameters must exactly match the request that yielded this
            token. Otherwise the request will fail.
        params (MutableMapping[str, google.cloud.bigtable_v2.types.Value]):
            Required. params contains string type keys and Bigtable type
            values that bind to placeholders in the query string. In
            query string, a parameter placeholder consists of the ``@``
            character followed by the parameter name (for example,
            ``@firstName``) in the query string.

            For example, if
            ``params["firstName"] = bytes_value: "foo" type {bytes_type {}}``
            then ``@firstName`` will be replaced with googlesql bytes
            value "foo" in the query string during query evaluation.

            If ``Value.kind`` is not set, the value is treated as a NULL
            value of the given type. For example, if
            ``params["firstName"] = type {string_type {}}`` then
            ``@firstName`` will be replaced with googlesql null string.

            If ``query`` is set, any empty ``Value.type`` in the map
            will be rejected with ``INVALID_ARGUMENT``.

            If ``prepared_query`` is set, any empty ``Value.type`` in
            the map will be inferred from the ``param_types`` in the
            ``PrepareQueryRequest``. Any non-empty ``Value.type`` must
            match the corresponding ``param_types`` entry, or be
            rejected with ``INVALID_ARGUMENT``.
    """

    instance_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    query: str = proto.Field(
        proto.STRING,
        number=3,
    )
    prepared_query: bytes = proto.Field(
        proto.BYTES,
        number=9,
    )
    proto_format: data.ProtoFormat = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="data_format",
        message=data.ProtoFormat,
    )
    resume_token: bytes = proto.Field(
        proto.BYTES,
        number=8,
    )
    params: MutableMapping[str, data.Value] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=7,
        message=data.Value,
    )


class ExecuteQueryResponse(proto.Message):
    r"""Response message for Bigtable.ExecuteQuery

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        metadata (google.cloud.bigtable_v2.types.ResultSetMetadata):
            Structure of rows in this response stream.
            The first (and only the first) response streamed
            from the server will be of this type.

            This field is a member of `oneof`_ ``response``.
        results (google.cloud.bigtable_v2.types.PartialResultSet):
            A partial result set with row data
            potentially including additional instructions on
            how recent past and future partial responses
            should be interpreted.

            This field is a member of `oneof`_ ``response``.
    """

    metadata: data.ResultSetMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="response",
        message=data.ResultSetMetadata,
    )
    results: data.PartialResultSet = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="response",
        message=data.PartialResultSet,
    )


class PrepareQueryRequest(proto.Message):
    r"""Request message for Bigtable.PrepareQuery

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        instance_name (str):
            Required. The unique name of the instance against which the
            query should be executed. Values are of the form
            ``projects/<project>/instances/<instance>``
        app_profile_id (str):
            Optional. This value specifies routing for preparing the
            query. Note that this ``app_profile_id`` is only used for
            preparing the query. The actual query execution will use the
            app profile specified in the ``ExecuteQueryRequest``. If not
            specified, the ``default`` application profile will be used.
        query (str):
            Required. The query string.
        proto_format (google.cloud.bigtable_v2.types.ProtoFormat):
            Protocol buffer format as described by
            ProtoSchema and ProtoRows messages.

            This field is a member of `oneof`_ ``data_format``.
        param_types (MutableMapping[str, google.cloud.bigtable_v2.types.Type]):
            Required. ``param_types`` is a map of parameter identifier
            strings to their ``Type``\ s.

            In query string, a parameter placeholder consists of the
            ``@`` character followed by the parameter name (for example,
            ``@firstName``) in the query string.

            For example, if param_types["firstName"] = Bytes then
            @firstName will be a query parameter of type Bytes. The
            specific ``Value`` to be used for the query execution must
            be sent in ``ExecuteQueryRequest`` in the ``params`` map.
    """

    instance_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    app_profile_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    query: str = proto.Field(
        proto.STRING,
        number=3,
    )
    proto_format: data.ProtoFormat = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="data_format",
        message=data.ProtoFormat,
    )
    param_types: MutableMapping[str, types.Type] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=6,
        message=types.Type,
    )


class PrepareQueryResponse(proto.Message):
    r"""Response message for Bigtable.PrepareQueryResponse

    Attributes:
        metadata (google.cloud.bigtable_v2.types.ResultSetMetadata):
            Structure of rows in the response stream of
            ``ExecuteQueryResponse`` for the returned
            ``prepared_query``.
        prepared_query (bytes):
            A serialized prepared query. Clients should treat this as an
            opaque blob of bytes to send in ``ExecuteQueryRequest``.
        valid_until (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the prepared query token
            becomes invalid. A token may become invalid
            early due to changes in the data being read, but
            it provides a guideline to refresh query plans
            asynchronously.
    """

    metadata: data.ResultSetMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=data.ResultSetMetadata,
    )
    prepared_query: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    valid_until: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

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

from google.cloud.bigtable_v2.types import types
from google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.bigtable.v2",
    manifest={
        "Row",
        "Family",
        "Column",
        "Cell",
        "Value",
        "ArrayValue",
        "RowRange",
        "RowSet",
        "ColumnRange",
        "TimestampRange",
        "ValueRange",
        "RowFilter",
        "Mutation",
        "ReadModifyWriteRule",
        "StreamPartition",
        "StreamContinuationTokens",
        "StreamContinuationToken",
        "ProtoFormat",
        "ColumnMetadata",
        "ProtoSchema",
        "ResultSetMetadata",
        "ProtoRows",
        "ProtoRowsBatch",
        "PartialResultSet",
        "Idempotency",
    },
)


class Row(proto.Message):
    r"""Specifies the complete (requested) contents of a single row
    of a table. Rows which exceed 256MiB in size cannot be read in
    full.

    Attributes:
        key (bytes):
            The unique key which identifies this row
            within its table. This is the same key that's
            used to identify the row in, for example, a
            MutateRowRequest. May contain any non-empty byte
            string up to 4KiB in length.
        families (MutableSequence[google.cloud.bigtable_v2.types.Family]):
            May be empty, but only if the entire row is
            empty. The mutual ordering of column families is
            not specified.
    """

    key: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    families: MutableSequence["Family"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Family",
    )


class Family(proto.Message):
    r"""Specifies (some of) the contents of a single row/column
    family intersection of a table.

    Attributes:
        name (str):
            The unique key which identifies this family within its row.
            This is the same key that's used to identify the family in,
            for example, a RowFilter which sets its
            "family_name_regex_filter" field. Must match
            ``[-_.a-zA-Z0-9]+``, except that AggregatingRowProcessors
            may produce cells in a sentinel family with an empty name.
            Must be no greater than 64 characters in length.
        columns (MutableSequence[google.cloud.bigtable_v2.types.Column]):
            Must not be empty. Sorted in order of
            increasing "qualifier".
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    columns: MutableSequence["Column"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Column",
    )


class Column(proto.Message):
    r"""Specifies (some of) the contents of a single row/column
    intersection of a table.

    Attributes:
        qualifier (bytes):
            The unique key which identifies this column within its
            family. This is the same key that's used to identify the
            column in, for example, a RowFilter which sets its
            ``column_qualifier_regex_filter`` field. May contain any
            byte string, including the empty string, up to 16kiB in
            length.
        cells (MutableSequence[google.cloud.bigtable_v2.types.Cell]):
            Must not be empty. Sorted in order of decreasing
            "timestamp_micros".
    """

    qualifier: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    cells: MutableSequence["Cell"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Cell",
    )


class Cell(proto.Message):
    r"""Specifies (some of) the contents of a single
    row/column/timestamp of a table.

    Attributes:
        timestamp_micros (int):
            The cell's stored timestamp, which also uniquely identifies
            it within its column. Values are always expressed in
            microseconds, but individual tables may set a coarser
            granularity to further restrict the allowed values. For
            example, a table which specifies millisecond granularity
            will only allow values of ``timestamp_micros`` which are
            multiples of 1000.
        value (bytes):
            The value stored in the cell.
            May contain any byte string, including the empty
            string, up to 100MiB in length.
        labels (MutableSequence[str]):
            Labels applied to the cell by a
            [RowFilter][google.bigtable.v2.RowFilter].
    """

    timestamp_micros: int = proto.Field(
        proto.INT64,
        number=1,
    )
    value: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    labels: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class Value(proto.Message):
    r"""``Value`` represents a dynamically typed value. The typed fields in
    ``Value`` are used as a transport encoding for the actual value
    (which may be of a more complex type). See the documentation of the
    ``Type`` message for more details.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        type_ (google.cloud.bigtable_v2.types.Type):
            The verified ``Type`` of this ``Value``, if it cannot be
            inferred.

            Read results will never specify the encoding for ``type``
            since the value will already have been decoded by the
            server. Furthermore, the ``type`` will be omitted entirely
            if it can be inferred from a previous response. The exact
            semantics for inferring ``type`` will vary, and are
            therefore documented separately for each read method.

            When using composite types (Struct, Array, Map) only the
            outermost ``Value`` will specify the ``type``. This
            top-level ``type`` will define the types for any nested
            ``Struct' fields,``\ Array\ ``elements, or``\ Map\ ``key/value pairs. If a nested``\ Value\ ``provides a``\ type\`
            on write, the request will be rejected with
            INVALID_ARGUMENT.
        raw_value (bytes):
            Represents a raw byte sequence with no type information. The
            ``type`` field must be omitted.

            This field is a member of `oneof`_ ``kind``.
        raw_timestamp_micros (int):
            Represents a raw cell timestamp with no type information.
            The ``type`` field must be omitted.

            This field is a member of `oneof`_ ``kind``.
        bytes_value (bytes):
            Represents a typed value transported as a
            byte sequence.

            This field is a member of `oneof`_ ``kind``.
        string_value (str):
            Represents a typed value transported as a
            string.

            This field is a member of `oneof`_ ``kind``.
        int_value (int):
            Represents a typed value transported as an
            integer.

            This field is a member of `oneof`_ ``kind``.
        bool_value (bool):
            Represents a typed value transported as a
            boolean.

            This field is a member of `oneof`_ ``kind``.
        float_value (float):
            Represents a typed value transported as a
            floating point number. Does not support NaN or
            infinities.

            This field is a member of `oneof`_ ``kind``.
        timestamp_value (google.protobuf.timestamp_pb2.Timestamp):
            Represents a typed value transported as a
            timestamp.

            This field is a member of `oneof`_ ``kind``.
        date_value (google.type.date_pb2.Date):
            Represents a typed value transported as a
            date.

            This field is a member of `oneof`_ ``kind``.
        array_value (google.cloud.bigtable_v2.types.ArrayValue):
            Represents a typed value transported as a sequence of
            values. To differentiate between ``Struct``, ``Array``, and
            ``Map``, the outermost ``Value`` must provide an explicit
            ``type`` on write. This ``type`` will apply recursively to
            the nested ``Struct`` fields, ``Array`` elements, or ``Map``
            key/value pairs, which *must not* supply their own ``type``.

            This field is a member of `oneof`_ ``kind``.
    """

    type_: types.Type = proto.Field(
        proto.MESSAGE,
        number=7,
        message=types.Type,
    )
    raw_value: bytes = proto.Field(
        proto.BYTES,
        number=8,
        oneof="kind",
    )
    raw_timestamp_micros: int = proto.Field(
        proto.INT64,
        number=9,
        oneof="kind",
    )
    bytes_value: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="kind",
    )
    string_value: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="kind",
    )
    int_value: int = proto.Field(
        proto.INT64,
        number=6,
        oneof="kind",
    )
    bool_value: bool = proto.Field(
        proto.BOOL,
        number=10,
        oneof="kind",
    )
    float_value: float = proto.Field(
        proto.DOUBLE,
        number=11,
        oneof="kind",
    )
    timestamp_value: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="kind",
        message=timestamp_pb2.Timestamp,
    )
    date_value: date_pb2.Date = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="kind",
        message=date_pb2.Date,
    )
    array_value: "ArrayValue" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="kind",
        message="ArrayValue",
    )


class ArrayValue(proto.Message):
    r"""``ArrayValue`` is an ordered list of ``Value``.

    Attributes:
        values (MutableSequence[google.cloud.bigtable_v2.types.Value]):
            The ordered elements in the array.
    """

    values: MutableSequence["Value"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Value",
    )


class RowRange(proto.Message):
    r"""Specifies a contiguous range of rows.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        start_key_closed (bytes):
            Used when giving an inclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_key``.
        start_key_open (bytes):
            Used when giving an exclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_key``.
        end_key_open (bytes):
            Used when giving an exclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_key``.
        end_key_closed (bytes):
            Used when giving an inclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_key``.
    """

    start_key_closed: bytes = proto.Field(
        proto.BYTES,
        number=1,
        oneof="start_key",
    )
    start_key_open: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="start_key",
    )
    end_key_open: bytes = proto.Field(
        proto.BYTES,
        number=3,
        oneof="end_key",
    )
    end_key_closed: bytes = proto.Field(
        proto.BYTES,
        number=4,
        oneof="end_key",
    )


class RowSet(proto.Message):
    r"""Specifies a non-contiguous set of rows.

    Attributes:
        row_keys (MutableSequence[bytes]):
            Single rows included in the set.
        row_ranges (MutableSequence[google.cloud.bigtable_v2.types.RowRange]):
            Contiguous row ranges included in the set.
    """

    row_keys: MutableSequence[bytes] = proto.RepeatedField(
        proto.BYTES,
        number=1,
    )
    row_ranges: MutableSequence["RowRange"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="RowRange",
    )


class ColumnRange(proto.Message):
    r"""Specifies a contiguous range of columns within a single column
    family. The range spans from <column_family>:<start_qualifier> to
    <column_family>:<end_qualifier>, where both bounds can be either
    inclusive or exclusive.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        family_name (str):
            The name of the column family within which
            this range falls.
        start_qualifier_closed (bytes):
            Used when giving an inclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_qualifier``.
        start_qualifier_open (bytes):
            Used when giving an exclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_qualifier``.
        end_qualifier_closed (bytes):
            Used when giving an inclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_qualifier``.
        end_qualifier_open (bytes):
            Used when giving an exclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_qualifier``.
    """

    family_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    start_qualifier_closed: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="start_qualifier",
    )
    start_qualifier_open: bytes = proto.Field(
        proto.BYTES,
        number=3,
        oneof="start_qualifier",
    )
    end_qualifier_closed: bytes = proto.Field(
        proto.BYTES,
        number=4,
        oneof="end_qualifier",
    )
    end_qualifier_open: bytes = proto.Field(
        proto.BYTES,
        number=5,
        oneof="end_qualifier",
    )


class TimestampRange(proto.Message):
    r"""Specified a contiguous range of microsecond timestamps.

    Attributes:
        start_timestamp_micros (int):
            Inclusive lower bound. If left empty,
            interpreted as 0.
        end_timestamp_micros (int):
            Exclusive upper bound. If left empty,
            interpreted as infinity.
    """

    start_timestamp_micros: int = proto.Field(
        proto.INT64,
        number=1,
    )
    end_timestamp_micros: int = proto.Field(
        proto.INT64,
        number=2,
    )


class ValueRange(proto.Message):
    r"""Specifies a contiguous range of raw byte values.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        start_value_closed (bytes):
            Used when giving an inclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_value``.
        start_value_open (bytes):
            Used when giving an exclusive lower bound for
            the range.

            This field is a member of `oneof`_ ``start_value``.
        end_value_closed (bytes):
            Used when giving an inclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_value``.
        end_value_open (bytes):
            Used when giving an exclusive upper bound for
            the range.

            This field is a member of `oneof`_ ``end_value``.
    """

    start_value_closed: bytes = proto.Field(
        proto.BYTES,
        number=1,
        oneof="start_value",
    )
    start_value_open: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="start_value",
    )
    end_value_closed: bytes = proto.Field(
        proto.BYTES,
        number=3,
        oneof="end_value",
    )
    end_value_open: bytes = proto.Field(
        proto.BYTES,
        number=4,
        oneof="end_value",
    )


class RowFilter(proto.Message):
    r"""Takes a row as input and produces an alternate view of the row based
    on specified rules. For example, a RowFilter might trim down a row
    to include just the cells from columns matching a given regular
    expression, or might return all the cells of a row but not their
    values. More complicated filters can be composed out of these
    components to express requests such as, "within every column of a
    particular family, give just the two most recent cells which are
    older than timestamp X."

    There are two broad categories of RowFilters (true filters and
    transformers), as well as two ways to compose simple filters into
    more complex ones (chains and interleaves). They work as follows:

    -  True filters alter the input row by excluding some of its cells
       wholesale from the output row. An example of a true filter is the
       ``value_regex_filter``, which excludes cells whose values don't
       match the specified pattern. All regex true filters use RE2
       syntax (https://github.com/google/re2/wiki/Syntax) in raw byte
       mode (RE2::Latin1), and are evaluated as full matches. An
       important point to keep in mind is that ``RE2(.)`` is equivalent
       by default to ``RE2([^\n])``, meaning that it does not match
       newlines. When attempting to match an arbitrary byte, you should
       therefore use the escape sequence ``\C``, which may need to be
       further escaped as ``\\C`` in your client language.

    -  Transformers alter the input row by changing the values of some
       of its cells in the output, without excluding them completely.
       Currently, the only supported transformer is the
       ``strip_value_transformer``, which replaces every cell's value
       with the empty string.

    -  Chains and interleaves are described in more detail in the
       RowFilter.Chain and RowFilter.Interleave documentation.

    The total serialized size of a RowFilter message must not exceed
    20480 bytes, and RowFilters may not be nested within each other (in
    Chains or Interleaves) to a depth of more than 20.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        chain (google.cloud.bigtable_v2.types.RowFilter.Chain):
            Applies several RowFilters to the data in
            sequence, progressively narrowing the results.

            This field is a member of `oneof`_ ``filter``.
        interleave (google.cloud.bigtable_v2.types.RowFilter.Interleave):
            Applies several RowFilters to the data in
            parallel and combines the results.

            This field is a member of `oneof`_ ``filter``.
        condition (google.cloud.bigtable_v2.types.RowFilter.Condition):
            Applies one of two possible RowFilters to the
            data based on the output of a predicate
            RowFilter.

            This field is a member of `oneof`_ ``filter``.
        sink (bool):
            ADVANCED USE ONLY. Hook for introspection into the
            RowFilter. Outputs all cells directly to the output of the
            read rather than to any parent filter. Consider the
            following example:

            ::

                Chain(
                  FamilyRegex("A"),
                  Interleave(
                    All(),
                    Chain(Label("foo"), Sink())
                  ),
                  QualifierRegex("B")
                )

                                    A,A,1,w
                                    A,B,2,x
                                    B,B,4,z
                                       |
                                FamilyRegex("A")
                                       |
                                    A,A,1,w
                                    A,B,2,x
                                       |
                          +------------+-------------+
                          |                          |
                        All()                    Label(foo)
                          |                          |
                       A,A,1,w              A,A,1,w,labels:[foo]
                       A,B,2,x              A,B,2,x,labels:[foo]
                          |                          |
                          |                        Sink() --------------+
                          |                          |                  |
                          +------------+      x------+          A,A,1,w,labels:[foo]
                                       |                        A,B,2,x,labels:[foo]
                                    A,A,1,w                             |
                                    A,B,2,x                             |
                                       |                                |
                               QualifierRegex("B")                      |
                                       |                                |
                                    A,B,2,x                             |
                                       |                                |
                                       +--------------------------------+
                                       |
                                    A,A,1,w,labels:[foo]
                                    A,B,2,x,labels:[foo]  // could be switched
                                    A,B,2,x               // could be switched

            Despite being excluded by the qualifier filter, a copy of
            every cell that reaches the sink is present in the final
            result.

            As with an
            [Interleave][google.bigtable.v2.RowFilter.Interleave],
            duplicate cells are possible, and appear in an unspecified
            mutual order. In this case we have a duplicate with column
            "A:B" and timestamp 2, because one copy passed through the
            all filter while the other was passed through the label and
            sink. Note that one copy has label "foo", while the other
            does not.

            Cannot be used within the ``predicate_filter``,
            ``true_filter``, or ``false_filter`` of a
            [Condition][google.bigtable.v2.RowFilter.Condition].

            This field is a member of `oneof`_ ``filter``.
        pass_all_filter (bool):
            Matches all cells, regardless of input. Functionally
            equivalent to leaving ``filter`` unset, but included for
            completeness.

            This field is a member of `oneof`_ ``filter``.
        block_all_filter (bool):
            Does not match any cells, regardless of
            input. Useful for temporarily disabling just
            part of a filter.

            This field is a member of `oneof`_ ``filter``.
        row_key_regex_filter (bytes):
            Matches only cells from rows whose keys satisfy the given
            RE2 regex. In other words, passes through the entire row
            when the key matches, and otherwise produces an empty row.
            Note that, since row keys can contain arbitrary bytes, the
            ``\C`` escape sequence must be used if a true wildcard is
            desired. The ``.`` character will not match the new line
            character ``\n``, which may be present in a binary key.

            This field is a member of `oneof`_ ``filter``.
        row_sample_filter (float):
            Matches all cells from a row with probability
            p, and matches no cells from the row with
            probability 1-p.

            This field is a member of `oneof`_ ``filter``.
        family_name_regex_filter (str):
            Matches only cells from columns whose families satisfy the
            given RE2 regex. For technical reasons, the regex must not
            contain the ``:`` character, even if it is not being used as
            a literal. Note that, since column families cannot contain
            the new line character ``\n``, it is sufficient to use ``.``
            as a full wildcard when matching column family names.

            This field is a member of `oneof`_ ``filter``.
        column_qualifier_regex_filter (bytes):
            Matches only cells from columns whose qualifiers satisfy the
            given RE2 regex. Note that, since column qualifiers can
            contain arbitrary bytes, the ``\C`` escape sequence must be
            used if a true wildcard is desired. The ``.`` character will
            not match the new line character ``\n``, which may be
            present in a binary qualifier.

            This field is a member of `oneof`_ ``filter``.
        column_range_filter (google.cloud.bigtable_v2.types.ColumnRange):
            Matches only cells from columns within the
            given range.

            This field is a member of `oneof`_ ``filter``.
        timestamp_range_filter (google.cloud.bigtable_v2.types.TimestampRange):
            Matches only cells with timestamps within the
            given range.

            This field is a member of `oneof`_ ``filter``.
        value_regex_filter (bytes):
            Matches only cells with values that satisfy the given
            regular expression. Note that, since cell values can contain
            arbitrary bytes, the ``\C`` escape sequence must be used if
            a true wildcard is desired. The ``.`` character will not
            match the new line character ``\n``, which may be present in
            a binary value.

            This field is a member of `oneof`_ ``filter``.
        value_range_filter (google.cloud.bigtable_v2.types.ValueRange):
            Matches only cells with values that fall
            within the given range.

            This field is a member of `oneof`_ ``filter``.
        cells_per_row_offset_filter (int):
            Skips the first N cells of each row, matching
            all subsequent cells. If duplicate cells are
            present, as is possible when using an
            Interleave, each copy of the cell is counted
            separately.

            This field is a member of `oneof`_ ``filter``.
        cells_per_row_limit_filter (int):
            Matches only the first N cells of each row.
            If duplicate cells are present, as is possible
            when using an Interleave, each copy of the cell
            is counted separately.

            This field is a member of `oneof`_ ``filter``.
        cells_per_column_limit_filter (int):
            Matches only the most recent N cells within each column. For
            example, if N=2, this filter would match column ``foo:bar``
            at timestamps 10 and 9, skip all earlier cells in
            ``foo:bar``, and then begin matching again in column
            ``foo:bar2``. If duplicate cells are present, as is possible
            when using an Interleave, each copy of the cell is counted
            separately.

            This field is a member of `oneof`_ ``filter``.
        strip_value_transformer (bool):
            Replaces each cell's value with the empty
            string.

            This field is a member of `oneof`_ ``filter``.
        apply_label_transformer (str):
            Applies the given label to all cells in the output row. This
            allows the client to determine which results were produced
            from which part of the filter.

            Values must be at most 15 characters in length, and match
            the RE2 pattern ``[a-z0-9\\-]+``

            Due to a technical limitation, it is not currently possible
            to apply multiple labels to a cell. As a result, a Chain may
            have no more than one sub-filter which contains a
            ``apply_label_transformer``. It is okay for an Interleave to
            contain multiple ``apply_label_transformers``, as they will
            be applied to separate copies of the input. This may be
            relaxed in the future.

            This field is a member of `oneof`_ ``filter``.
    """

    class Chain(proto.Message):
        r"""A RowFilter which sends rows through several RowFilters in
        sequence.

        Attributes:
            filters (MutableSequence[google.cloud.bigtable_v2.types.RowFilter]):
                The elements of "filters" are chained
                together to process the input row: in row ->
                f(0) -> intermediate row -> f(1) -> ... -> f(N)
                -> out row The full chain is executed
                atomically.
        """

        filters: MutableSequence["RowFilter"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="RowFilter",
        )

    class Interleave(proto.Message):
        r"""A RowFilter which sends each row to each of several component
        RowFilters and interleaves the results.

        Attributes:
            filters (MutableSequence[google.cloud.bigtable_v2.types.RowFilter]):
                The elements of "filters" all process a copy of the input
                row, and the results are pooled, sorted, and combined into a
                single output row. If multiple cells are produced with the
                same column and timestamp, they will all appear in the
                output row in an unspecified mutual order. Consider the
                following example, with three filters:

                ::

                                                 input row
                                                     |
                           -----------------------------------------------------
                           |                         |                         |
                          f(0)                      f(1)                      f(2)
                           |                         |                         |
                    1: foo,bar,10,x             foo,bar,10,z              far,bar,7,a
                    2: foo,blah,11,z            far,blah,5,x              far,blah,5,x
                           |                         |                         |
                           -----------------------------------------------------
                                                     |
                    1:                      foo,bar,10,z   // could have switched with #2
                    2:                      foo,bar,10,x   // could have switched with #1
                    3:                      foo,blah,11,z
                    4:                      far,bar,7,a
                    5:                      far,blah,5,x   // identical to #6
                    6:                      far,blah,5,x   // identical to #5

                All interleaved filters are executed atomically.
        """

        filters: MutableSequence["RowFilter"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="RowFilter",
        )

    class Condition(proto.Message):
        r"""A RowFilter which evaluates one of two possible RowFilters,
        depending on whether or not a predicate RowFilter outputs any
        cells from the input row.

        IMPORTANT NOTE: The predicate filter does not execute atomically
        with the true and false filters, which may lead to inconsistent
        or unexpected results. Additionally, Condition filters have poor
        performance, especially when filters are set for the false
        condition.

        Attributes:
            predicate_filter (google.cloud.bigtable_v2.types.RowFilter):
                If ``predicate_filter`` outputs any cells, then
                ``true_filter`` will be evaluated on the input row.
                Otherwise, ``false_filter`` will be evaluated.
            true_filter (google.cloud.bigtable_v2.types.RowFilter):
                The filter to apply to the input row if ``predicate_filter``
                returns any results. If not provided, no results will be
                returned in the true case.
            false_filter (google.cloud.bigtable_v2.types.RowFilter):
                The filter to apply to the input row if ``predicate_filter``
                does not return any results. If not provided, no results
                will be returned in the false case.
        """

        predicate_filter: "RowFilter" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="RowFilter",
        )
        true_filter: "RowFilter" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="RowFilter",
        )
        false_filter: "RowFilter" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="RowFilter",
        )

    chain: Chain = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="filter",
        message=Chain,
    )
    interleave: Interleave = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="filter",
        message=Interleave,
    )
    condition: Condition = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="filter",
        message=Condition,
    )
    sink: bool = proto.Field(
        proto.BOOL,
        number=16,
        oneof="filter",
    )
    pass_all_filter: bool = proto.Field(
        proto.BOOL,
        number=17,
        oneof="filter",
    )
    block_all_filter: bool = proto.Field(
        proto.BOOL,
        number=18,
        oneof="filter",
    )
    row_key_regex_filter: bytes = proto.Field(
        proto.BYTES,
        number=4,
        oneof="filter",
    )
    row_sample_filter: float = proto.Field(
        proto.DOUBLE,
        number=14,
        oneof="filter",
    )
    family_name_regex_filter: str = proto.Field(
        proto.STRING,
        number=5,
        oneof="filter",
    )
    column_qualifier_regex_filter: bytes = proto.Field(
        proto.BYTES,
        number=6,
        oneof="filter",
    )
    column_range_filter: "ColumnRange" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="filter",
        message="ColumnRange",
    )
    timestamp_range_filter: "TimestampRange" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="filter",
        message="TimestampRange",
    )
    value_regex_filter: bytes = proto.Field(
        proto.BYTES,
        number=9,
        oneof="filter",
    )
    value_range_filter: "ValueRange" = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="filter",
        message="ValueRange",
    )
    cells_per_row_offset_filter: int = proto.Field(
        proto.INT32,
        number=10,
        oneof="filter",
    )
    cells_per_row_limit_filter: int = proto.Field(
        proto.INT32,
        number=11,
        oneof="filter",
    )
    cells_per_column_limit_filter: int = proto.Field(
        proto.INT32,
        number=12,
        oneof="filter",
    )
    strip_value_transformer: bool = proto.Field(
        proto.BOOL,
        number=13,
        oneof="filter",
    )
    apply_label_transformer: str = proto.Field(
        proto.STRING,
        number=19,
        oneof="filter",
    )


class Mutation(proto.Message):
    r"""Specifies a particular change to be made to the contents of a
    row.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        set_cell (google.cloud.bigtable_v2.types.Mutation.SetCell):
            Set a cell's value.

            This field is a member of `oneof`_ ``mutation``.
        add_to_cell (google.cloud.bigtable_v2.types.Mutation.AddToCell):
            Incrementally updates an ``Aggregate`` cell.

            This field is a member of `oneof`_ ``mutation``.
        merge_to_cell (google.cloud.bigtable_v2.types.Mutation.MergeToCell):
            Merges accumulated state to an ``Aggregate`` cell.

            This field is a member of `oneof`_ ``mutation``.
        delete_from_column (google.cloud.bigtable_v2.types.Mutation.DeleteFromColumn):
            Deletes cells from a column.

            This field is a member of `oneof`_ ``mutation``.
        delete_from_family (google.cloud.bigtable_v2.types.Mutation.DeleteFromFamily):
            Deletes cells from a column family.

            This field is a member of `oneof`_ ``mutation``.
        delete_from_row (google.cloud.bigtable_v2.types.Mutation.DeleteFromRow):
            Deletes cells from the entire row.

            This field is a member of `oneof`_ ``mutation``.
    """

    class SetCell(proto.Message):
        r"""A Mutation which sets the value of the specified cell.

        Attributes:
            family_name (str):
                The name of the family into which new data should be
                written. Must match ``[-_.a-zA-Z0-9]+``
            column_qualifier (bytes):
                The qualifier of the column into which new
                data should be written. Can be any byte string,
                including the empty string.
            timestamp_micros (int):
                The timestamp of the cell into which new data
                should be written. Use -1 for current Bigtable
                server time. Otherwise, the client should set
                this value itself, noting that the default value
                is a timestamp of zero if the field is left
                unspecified. Values must match the granularity
                of the table (e.g. micros, millis).
            value (bytes):
                The value to be written into the specified
                cell.
        """

        family_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        column_qualifier: bytes = proto.Field(
            proto.BYTES,
            number=2,
        )
        timestamp_micros: int = proto.Field(
            proto.INT64,
            number=3,
        )
        value: bytes = proto.Field(
            proto.BYTES,
            number=4,
        )

    class AddToCell(proto.Message):
        r"""A Mutation which incrementally updates a cell in an ``Aggregate``
        family.

        Attributes:
            family_name (str):
                The name of the ``Aggregate`` family into which new data
                should be added. This must be a family with a ``value_type``
                of ``Aggregate``. Format: ``[-_.a-zA-Z0-9]+``
            column_qualifier (google.cloud.bigtable_v2.types.Value):
                The qualifier of the column into which new data should be
                added. This must be a ``raw_value``.
            timestamp (google.cloud.bigtable_v2.types.Value):
                The timestamp of the cell to which new data should be added.
                This must be a ``raw_timestamp_micros`` that matches the
                table's ``granularity``.
            input (google.cloud.bigtable_v2.types.Value):
                The input value to be accumulated into the specified cell.
                This must be compatible with the family's
                ``value_type.input_type``.
        """

        family_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        column_qualifier: "Value" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="Value",
        )
        timestamp: "Value" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="Value",
        )
        input: "Value" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="Value",
        )

    class MergeToCell(proto.Message):
        r"""A Mutation which merges accumulated state into a cell in an
        ``Aggregate`` family.

        Attributes:
            family_name (str):
                The name of the ``Aggregate`` family into which new data
                should be added. This must be a family with a ``value_type``
                of ``Aggregate``. Format: ``[-_.a-zA-Z0-9]+``
            column_qualifier (google.cloud.bigtable_v2.types.Value):
                The qualifier of the column into which new data should be
                added. This must be a ``raw_value``.
            timestamp (google.cloud.bigtable_v2.types.Value):
                The timestamp of the cell to which new data should be added.
                This must be a ``raw_timestamp_micros`` that matches the
                table's ``granularity``.
            input (google.cloud.bigtable_v2.types.Value):
                The input value to be merged into the specified cell. This
                must be compatible with the family's
                ``value_type.state_type``. Merging ``NULL`` is allowed, but
                has no effect.
        """

        family_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        column_qualifier: "Value" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="Value",
        )
        timestamp: "Value" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="Value",
        )
        input: "Value" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="Value",
        )

    class DeleteFromColumn(proto.Message):
        r"""A Mutation which deletes cells from the specified column,
        optionally restricting the deletions to a given timestamp range.

        Attributes:
            family_name (str):
                The name of the family from which cells should be deleted.
                Must match ``[-_.a-zA-Z0-9]+``
            column_qualifier (bytes):
                The qualifier of the column from which cells
                should be deleted. Can be any byte string,
                including the empty string.
            time_range (google.cloud.bigtable_v2.types.TimestampRange):
                The range of timestamps within which cells
                should be deleted.
        """

        family_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        column_qualifier: bytes = proto.Field(
            proto.BYTES,
            number=2,
        )
        time_range: "TimestampRange" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="TimestampRange",
        )

    class DeleteFromFamily(proto.Message):
        r"""A Mutation which deletes all cells from the specified column
        family.

        Attributes:
            family_name (str):
                The name of the family from which cells should be deleted.
                Must match ``[-_.a-zA-Z0-9]+``
        """

        family_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class DeleteFromRow(proto.Message):
        r"""A Mutation which deletes all cells from the containing row."""

    set_cell: SetCell = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="mutation",
        message=SetCell,
    )
    add_to_cell: AddToCell = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="mutation",
        message=AddToCell,
    )
    merge_to_cell: MergeToCell = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="mutation",
        message=MergeToCell,
    )
    delete_from_column: DeleteFromColumn = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="mutation",
        message=DeleteFromColumn,
    )
    delete_from_family: DeleteFromFamily = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="mutation",
        message=DeleteFromFamily,
    )
    delete_from_row: DeleteFromRow = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="mutation",
        message=DeleteFromRow,
    )


class ReadModifyWriteRule(proto.Message):
    r"""Specifies an atomic read/modify/write operation on the latest
    value of the specified column.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        family_name (str):
            The name of the family to which the read/modify/write should
            be applied. Must match ``[-_.a-zA-Z0-9]+``
        column_qualifier (bytes):
            The qualifier of the column to which the
            read/modify/write should be applied.
            Can be any byte string, including the empty
            string.
        append_value (bytes):
            Rule specifying that ``append_value`` be appended to the
            existing value. If the targeted cell is unset, it will be
            treated as containing the empty string.

            This field is a member of `oneof`_ ``rule``.
        increment_amount (int):
            Rule specifying that ``increment_amount`` be added to the
            existing value. If the targeted cell is unset, it will be
            treated as containing a zero. Otherwise, the targeted cell
            must contain an 8-byte value (interpreted as a 64-bit
            big-endian signed integer), or the entire request will fail.

            This field is a member of `oneof`_ ``rule``.
    """

    family_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    column_qualifier: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )
    append_value: bytes = proto.Field(
        proto.BYTES,
        number=3,
        oneof="rule",
    )
    increment_amount: int = proto.Field(
        proto.INT64,
        number=4,
        oneof="rule",
    )


class StreamPartition(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam
    BigtableIO. A partition of a change stream.

    Attributes:
        row_range (google.cloud.bigtable_v2.types.RowRange):
            The row range covered by this partition and is specified by
            [``start_key_closed``, ``end_key_open``).
    """

    row_range: "RowRange" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RowRange",
    )


class StreamContinuationTokens(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam BigtableIO. The
    information required to continue reading the data from multiple
    ``StreamPartitions`` from where a previous read left off.

    Attributes:
        tokens (MutableSequence[google.cloud.bigtable_v2.types.StreamContinuationToken]):
            List of continuation tokens.
    """

    tokens: MutableSequence["StreamContinuationToken"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="StreamContinuationToken",
    )


class StreamContinuationToken(proto.Message):
    r"""NOTE: This API is intended to be used by Apache Beam BigtableIO. The
    information required to continue reading the data from a
    ``StreamPartition`` from where a previous read left off.

    Attributes:
        partition (google.cloud.bigtable_v2.types.StreamPartition):
            The partition that this token applies to.
        token (str):
            An encoded position in the stream to restart
            reading from.
    """

    partition: "StreamPartition" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="StreamPartition",
    )
    token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ProtoFormat(proto.Message):
    r"""Protocol buffers format descriptor, as described by Messages
    ProtoSchema and ProtoRows

    """


class ColumnMetadata(proto.Message):
    r"""Describes a column in a Bigtable Query Language result set.

    Attributes:
        name (str):
            The name of the column.
        type_ (google.cloud.bigtable_v2.types.Type):
            The type of the column.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    type_: types.Type = proto.Field(
        proto.MESSAGE,
        number=2,
        message=types.Type,
    )


class ProtoSchema(proto.Message):
    r"""ResultSet schema in proto format

    Attributes:
        columns (MutableSequence[google.cloud.bigtable_v2.types.ColumnMetadata]):
            The columns in the result set.
    """

    columns: MutableSequence["ColumnMetadata"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ColumnMetadata",
    )


class ResultSetMetadata(proto.Message):
    r"""Describes the structure of a Bigtable result set.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        proto_schema (google.cloud.bigtable_v2.types.ProtoSchema):
            Schema in proto format

            This field is a member of `oneof`_ ``schema``.
    """

    proto_schema: "ProtoSchema" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="schema",
        message="ProtoSchema",
    )


class ProtoRows(proto.Message):
    r"""Rows represented in proto format.

    This should be constructed by concatenating the ``batch_data`` from
    each of the relevant ``ProtoRowsBatch`` messages and parsing the
    result as a ``ProtoRows`` message.

    Attributes:
        values (MutableSequence[google.cloud.bigtable_v2.types.Value]):
            A proto rows message consists of a list of values. Every N
            complete values defines a row, where N is equal to the
            number of entries in the ``metadata.proto_schema.columns``
            value received in the first response.
    """

    values: MutableSequence["Value"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Value",
    )


class ProtoRowsBatch(proto.Message):
    r"""A part of a serialized ``ProtoRows`` message.

    Attributes:
        batch_data (bytes):
            Part of a serialized ``ProtoRows`` message. A complete,
            parseable ProtoRows message is constructed by concatenating
            ``batch_data`` from multiple ``ProtoRowsBatch`` messages.
            The ``PartialResultSet`` that contains the last part has
            ``complete_batch`` set to ``true``.
    """

    batch_data: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )


class PartialResultSet(proto.Message):
    r"""A partial result set from the streaming query API. Cloud Bigtable
    clients buffer partial results received in this message until a
    ``resume_token`` is received.

    The pseudocode below describes how to buffer and parse a stream of
    ``PartialResultSet`` messages.

    Having:

    -  queue of row results waiting to be returned ``queue``
    -  extensible buffer of bytes ``buffer``
    -  a place to keep track of the most recent ``resume_token`` for
       each PartialResultSet ``p`` received { if p.reset { ensure
       ``queue`` is empty ensure ``buffer`` is empty } if
       p.estimated_batch_size != 0 { (optional) ensure ``buffer`` is
       sized to at least ``p.estimated_batch_size`` } if
       ``p.proto_rows_batch`` is set { append
       ``p.proto_rows_batch.bytes`` to ``buffer`` } if p.batch_checksum
       is set and ``buffer`` is not empty { validate the checksum
       matches the contents of ``buffer`` (see comments on
       ``batch_checksum``) parse ``buffer`` as ``ProtoRows`` message,
       clearing ``buffer`` add parsed rows to end of ``queue`` } if
       p.resume_token is set { release results in ``queue`` save
       ``p.resume_token`` in ``resume_token`` } }


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        proto_rows_batch (google.cloud.bigtable_v2.types.ProtoRowsBatch):
            Partial rows in serialized ProtoRows format.

            This field is a member of `oneof`_ ``partial_rows``.
        batch_checksum (int):
            CRC32C checksum of concatenated ``partial_rows`` data for
            the current batch.

            When present, the buffered data from ``partial_rows`` forms
            a complete parseable message of the appropriate type.

            The client should mark the end of a parseable message and
            prepare to receive a new one starting from the next
            ``PartialResultSet`` message. Clients must verify the
            checksum of the serialized batch before yielding it to the
            caller.

            This does NOT mean the values can be yielded to the callers
            since a ``resume_token`` is required to safely do so.

            If ``resume_token`` is non-empty and any data has been
            received since the last one, this field is guaranteed to be
            non-empty. In other words, clients may assume that a batch
            will never cross a ``resume_token`` boundary.

            This field is a member of `oneof`_ ``_batch_checksum``.
        resume_token (bytes):
            An opaque token sent by the server to allow query resumption
            and signal that the buffered values constructed from
            received ``partial_rows`` can be yielded to the caller.
            Clients can provide this token in a subsequent request to
            resume the result stream from the current point.

            When ``resume_token`` is non-empty, the buffered values
            received from ``partial_rows`` since the last non-empty
            ``resume_token`` can be yielded to the callers, provided
            that the client keeps the value of ``resume_token`` and uses
            it on subsequent retries.

            A ``resume_token`` may be sent without information in
            ``partial_rows`` to checkpoint the progress of a sparse
            query. Any previous ``partial_rows`` data should still be
            yielded in this case, and the new ``resume_token`` should be
            saved for future retries as normal.

            A ``resume_token`` will only be sent on a boundary where
            there is either no ongoing result batch, or
            ``batch_checksum`` is also populated.

            The server will also send a sentinel ``resume_token`` when
            last batch of ``partial_rows`` is sent. If the client
            retries the ExecuteQueryRequest with the sentinel
            ``resume_token``, the server will emit it again without any
            data in ``partial_rows``, then return OK.
        reset (bool):
            If ``true``, any data buffered since the last non-empty
            ``resume_token`` must be discarded before the other parts of
            this message, if any, are handled.
        estimated_batch_size (int):
            Estimated size of the buffer required to hold the next batch
            of results.

            This value will be sent with the first ``partial_rows`` of a
            batch. That is, on the first ``partial_rows`` received in a
            stream, on the first message after a ``batch_checksum``
            message, and any time ``reset`` is true.

            The client can use this estimate to allocate a buffer for
            the next batch of results. This helps minimize the number of
            allocations required, though the buffer size may still need
            to be increased if the estimate is too low.
    """

    proto_rows_batch: "ProtoRowsBatch" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="partial_rows",
        message="ProtoRowsBatch",
    )
    batch_checksum: int = proto.Field(
        proto.UINT32,
        number=6,
        optional=True,
    )
    resume_token: bytes = proto.Field(
        proto.BYTES,
        number=5,
    )
    reset: bool = proto.Field(
        proto.BOOL,
        number=7,
    )
    estimated_batch_size: int = proto.Field(
        proto.INT32,
        number=4,
    )


class Idempotency(proto.Message):
    r"""Parameters on mutations where clients want to ensure
    idempotency (i.e. at-most-once semantics). This is currently
    only needed for certain aggregate types.

    Attributes:
        token (bytes):
            Unique token used to identify replays of this
            mutation. Must be at least 8 bytes long.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Client-assigned timestamp when the mutation's
            first attempt was sent. Used to reject mutations
            that arrive after idempotency protection may
            have expired. May cause spurious rejections if
            clock skew is too high.

            Leave unset or zero to always accept the
            mutation, at the risk of double counting if the
            protection for previous attempts has expired.
    """

    token: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

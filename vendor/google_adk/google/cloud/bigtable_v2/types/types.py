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


__protobuf__ = proto.module(
    package="google.bigtable.v2",
    manifest={
        "Type",
    },
)


class Type(proto.Message):
    r"""``Type`` represents the type of data that is written to, read from,
    or stored in Bigtable. It is heavily based on the GoogleSQL standard
    to help maintain familiarity and consistency across products and
    features.

    For compatibility with Bigtable's existing untyped APIs, each
    ``Type`` includes an ``Encoding`` which describes how to convert
    to/from the underlying data.

    Each encoding also defines the following properties:

    -  Order-preserving: Does the encoded value sort consistently with
       the original typed value? Note that Bigtable will always sort
       data based on the raw encoded value, *not* the decoded type.

       -  Example: BYTES values sort in the same order as their raw
          encodings.
       -  Counterexample: Encoding INT64 as a fixed-width decimal string
          does *not* preserve sort order when dealing with negative
          numbers. ``INT64(1) > INT64(-1)``, but
          ``STRING("-00001") > STRING("00001)``.

    -  Self-delimiting: If we concatenate two encoded values, can we
       always tell where the first one ends and the second one begins?

       -  Example: If we encode INT64s to fixed-width STRINGs, the first
          value will always contain exactly N digits, possibly preceded
          by a sign.
       -  Counterexample: If we concatenate two UTF-8 encoded STRINGs,
          we have no way to tell where the first one ends.

    -  Compatibility: Which other systems have matching encoding
       schemes? For example, does this encoding have a GoogleSQL
       equivalent? HBase? Java?

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bytes_type (google.cloud.bigtable_v2.types.Type.Bytes):
            Bytes

            This field is a member of `oneof`_ ``kind``.
        string_type (google.cloud.bigtable_v2.types.Type.String):
            String

            This field is a member of `oneof`_ ``kind``.
        int64_type (google.cloud.bigtable_v2.types.Type.Int64):
            Int64

            This field is a member of `oneof`_ ``kind``.
        float32_type (google.cloud.bigtable_v2.types.Type.Float32):
            Float32

            This field is a member of `oneof`_ ``kind``.
        float64_type (google.cloud.bigtable_v2.types.Type.Float64):
            Float64

            This field is a member of `oneof`_ ``kind``.
        bool_type (google.cloud.bigtable_v2.types.Type.Bool):
            Bool

            This field is a member of `oneof`_ ``kind``.
        timestamp_type (google.cloud.bigtable_v2.types.Type.Timestamp):
            Timestamp

            This field is a member of `oneof`_ ``kind``.
        date_type (google.cloud.bigtable_v2.types.Type.Date):
            Date

            This field is a member of `oneof`_ ``kind``.
        aggregate_type (google.cloud.bigtable_v2.types.Type.Aggregate):
            Aggregate

            This field is a member of `oneof`_ ``kind``.
        struct_type (google.cloud.bigtable_v2.types.Type.Struct):
            Struct

            This field is a member of `oneof`_ ``kind``.
        array_type (google.cloud.bigtable_v2.types.Type.Array):
            Array

            This field is a member of `oneof`_ ``kind``.
        map_type (google.cloud.bigtable_v2.types.Type.Map):
            Map

            This field is a member of `oneof`_ ``kind``.
        proto_type (google.cloud.bigtable_v2.types.Type.Proto):
            Proto

            This field is a member of `oneof`_ ``kind``.
        enum_type (google.cloud.bigtable_v2.types.Type.Enum):
            Enum

            This field is a member of `oneof`_ ``kind``.
    """

    class Bytes(proto.Message):
        r"""Bytes Values of type ``Bytes`` are stored in ``Value.bytes_value``.

        Attributes:
            encoding (google.cloud.bigtable_v2.types.Type.Bytes.Encoding):
                The encoding to use when converting to/from
                lower level types.
        """

        class Encoding(proto.Message):
            r"""Rules used to convert to/from lower level types.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                raw (google.cloud.bigtable_v2.types.Type.Bytes.Encoding.Raw):
                    Use ``Raw`` encoding.

                    This field is a member of `oneof`_ ``encoding``.
            """

            class Raw(proto.Message):
                r"""Leaves the value "as-is"

                -  Order-preserving? Yes
                -  Self-delimiting? No
                -  Compatibility? N/A

                """

            raw: "Type.Bytes.Encoding.Raw" = proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="encoding",
                message="Type.Bytes.Encoding.Raw",
            )

        encoding: "Type.Bytes.Encoding" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type.Bytes.Encoding",
        )

    class String(proto.Message):
        r"""String Values of type ``String`` are stored in
        ``Value.string_value``.

        Attributes:
            encoding (google.cloud.bigtable_v2.types.Type.String.Encoding):
                The encoding to use when converting to/from
                lower level types.
        """

        class Encoding(proto.Message):
            r"""Rules used to convert to/from lower level types.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                utf8_raw (google.cloud.bigtable_v2.types.Type.String.Encoding.Utf8Raw):
                    Deprecated: if set, converts to an empty ``utf8_bytes``.

                    This field is a member of `oneof`_ ``encoding``.
                utf8_bytes (google.cloud.bigtable_v2.types.Type.String.Encoding.Utf8Bytes):
                    Use ``Utf8Bytes`` encoding.

                    This field is a member of `oneof`_ ``encoding``.
            """

            class Utf8Raw(proto.Message):
                r"""Deprecated: prefer the equivalent ``Utf8Bytes``."""

            class Utf8Bytes(proto.Message):
                r"""UTF-8 encoding

                -  Order-preserving? Yes (code point order)
                -  Self-delimiting? No
                -  Compatibility?

                   -  BigQuery Federation ``TEXT`` encoding
                   -  HBase ``Bytes.toBytes``
                   -  Java ``String#getBytes(StandardCharsets.UTF_8)``

                """

            utf8_raw: "Type.String.Encoding.Utf8Raw" = proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="encoding",
                message="Type.String.Encoding.Utf8Raw",
            )
            utf8_bytes: "Type.String.Encoding.Utf8Bytes" = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="encoding",
                message="Type.String.Encoding.Utf8Bytes",
            )

        encoding: "Type.String.Encoding" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type.String.Encoding",
        )

    class Int64(proto.Message):
        r"""Int64 Values of type ``Int64`` are stored in ``Value.int_value``.

        Attributes:
            encoding (google.cloud.bigtable_v2.types.Type.Int64.Encoding):
                The encoding to use when converting to/from
                lower level types.
        """

        class Encoding(proto.Message):
            r"""Rules used to convert to/from lower level types.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                big_endian_bytes (google.cloud.bigtable_v2.types.Type.Int64.Encoding.BigEndianBytes):
                    Use ``BigEndianBytes`` encoding.

                    This field is a member of `oneof`_ ``encoding``.
            """

            class BigEndianBytes(proto.Message):
                r"""Encodes the value as an 8-byte big endian twos complement ``Bytes``
                value.

                -  Order-preserving? No (positive values only)
                -  Self-delimiting? Yes
                -  Compatibility?

                   -  BigQuery Federation ``BINARY`` encoding
                   -  HBase ``Bytes.toBytes``
                   -  Java ``ByteBuffer.putLong()`` with ``ByteOrder.BIG_ENDIAN``

                Attributes:
                    bytes_type (google.cloud.bigtable_v2.types.Type.Bytes):
                        Deprecated: ignored if set.
                """

                bytes_type: "Type.Bytes" = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    message="Type.Bytes",
                )

            big_endian_bytes: "Type.Int64.Encoding.BigEndianBytes" = proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="encoding",
                message="Type.Int64.Encoding.BigEndianBytes",
            )

        encoding: "Type.Int64.Encoding" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type.Int64.Encoding",
        )

    class Bool(proto.Message):
        r"""bool Values of type ``Bool`` are stored in ``Value.bool_value``."""

    class Float32(proto.Message):
        r"""Float32 Values of type ``Float32`` are stored in
        ``Value.float_value``.

        """

    class Float64(proto.Message):
        r"""Float64 Values of type ``Float64`` are stored in
        ``Value.float_value``.

        """

    class Timestamp(proto.Message):
        r"""Timestamp Values of type ``Timestamp`` are stored in
        ``Value.timestamp_value``.

        """

    class Date(proto.Message):
        r"""Date Values of type ``Date`` are stored in ``Value.date_value``."""

    class Struct(proto.Message):
        r"""A structured data value, consisting of fields which map to
        dynamically typed values. Values of type ``Struct`` are stored in
        ``Value.array_value`` where entries are in the same order and number
        as ``field_types``.

        Attributes:
            fields (MutableSequence[google.cloud.bigtable_v2.types.Type.Struct.Field]):
                The names and types of the fields in this
                struct.
        """

        class Field(proto.Message):
            r"""A struct field and its type.

            Attributes:
                field_name (str):
                    The field name (optional). Fields without a ``field_name``
                    are considered anonymous and cannot be referenced by name.
                type_ (google.cloud.bigtable_v2.types.Type):
                    The type of values in this field.
            """

            field_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            type_: "Type" = proto.Field(
                proto.MESSAGE,
                number=2,
                message="Type",
            )

        fields: MutableSequence["Type.Struct.Field"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="Type.Struct.Field",
        )

    class Proto(proto.Message):
        r"""A protobuf message type. Values of type ``Proto`` are stored in
        ``Value.bytes_value``.

        Attributes:
            schema_bundle_id (str):
                The ID of the schema bundle that this proto
                is defined in.
            message_name (str):
                The fully qualified name of the protobuf
                message, including package. In the format of
                "foo.bar.Message".
        """

        schema_bundle_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        message_name: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class Enum(proto.Message):
        r"""A protobuf enum type. Values of type ``Enum`` are stored in
        ``Value.int_value``.

        Attributes:
            schema_bundle_id (str):
                The ID of the schema bundle that this enum is
                defined in.
            enum_name (str):
                The fully qualified name of the protobuf enum
                message, including package. In the format of
                "foo.bar.EnumMessage".
        """

        schema_bundle_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        enum_name: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class Array(proto.Message):
        r"""An ordered list of elements of a given type. Values of type
        ``Array`` are stored in ``Value.array_value``.

        Attributes:
            element_type (google.cloud.bigtable_v2.types.Type):
                The type of the elements in the array. This must not be
                ``Array``.
        """

        element_type: "Type" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type",
        )

    class Map(proto.Message):
        r"""A mapping of keys to values of a given type. Values of type ``Map``
        are stored in a ``Value.array_value`` where each entry is another
        ``Value.array_value`` with two elements (the key and the value, in
        that order). Normally encoded Map values won't have repeated keys,
        however, clients are expected to handle the case in which they do.
        If the same key appears multiple times, the *last* value takes
        precedence.

        Attributes:
            key_type (google.cloud.bigtable_v2.types.Type):
                The type of a map key. Only ``Bytes``, ``String``, and
                ``Int64`` are allowed as key types.
            value_type (google.cloud.bigtable_v2.types.Type):
                The type of the values in a map.
        """

        key_type: "Type" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type",
        )
        value_type: "Type" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="Type",
        )

    class Aggregate(proto.Message):
        r"""A value that combines incremental updates into a summarized value.

        Data is never directly written or read using type ``Aggregate``.
        Writes will provide either the ``input_type`` or ``state_type``, and
        reads will always return the ``state_type`` .

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            input_type (google.cloud.bigtable_v2.types.Type):
                Type of the inputs that are accumulated by this
                ``Aggregate``, which must specify a full encoding. Use
                ``AddInput`` mutations to accumulate new inputs.
            state_type (google.cloud.bigtable_v2.types.Type):
                Output only. Type that holds the internal accumulator state
                for the ``Aggregate``. This is a function of the
                ``input_type`` and ``aggregator`` chosen, and will always
                specify a full encoding.
            sum (google.cloud.bigtable_v2.types.Type.Aggregate.Sum):
                Sum aggregator.

                This field is a member of `oneof`_ ``aggregator``.
            hllpp_unique_count (google.cloud.bigtable_v2.types.Type.Aggregate.HyperLogLogPlusPlusUniqueCount):
                HyperLogLogPlusPlusUniqueCount aggregator.

                This field is a member of `oneof`_ ``aggregator``.
            max_ (google.cloud.bigtable_v2.types.Type.Aggregate.Max):
                Max aggregator.

                This field is a member of `oneof`_ ``aggregator``.
            min_ (google.cloud.bigtable_v2.types.Type.Aggregate.Min):
                Min aggregator.

                This field is a member of `oneof`_ ``aggregator``.
        """

        class Sum(proto.Message):
            r"""Computes the sum of the input values. Allowed input: ``Int64``
            State: same as input

            """

        class Max(proto.Message):
            r"""Computes the max of the input values. Allowed input: ``Int64``
            State: same as input

            """

        class Min(proto.Message):
            r"""Computes the min of the input values. Allowed input: ``Int64``
            State: same as input

            """

        class HyperLogLogPlusPlusUniqueCount(proto.Message):
            r"""Computes an approximate unique count over the input values. When
            using raw data as input, be careful to use a consistent encoding.
            Otherwise the same value encoded differently could count more than
            once, or two distinct values could count as identical. Input: Any,
            or omit for Raw State: TBD Special state conversions: ``Int64`` (the
            unique count estimate)

            """

        input_type: "Type" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Type",
        )
        state_type: "Type" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="Type",
        )
        sum: "Type.Aggregate.Sum" = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof="aggregator",
            message="Type.Aggregate.Sum",
        )
        hllpp_unique_count: "Type.Aggregate.HyperLogLogPlusPlusUniqueCount" = (
            proto.Field(
                proto.MESSAGE,
                number=5,
                oneof="aggregator",
                message="Type.Aggregate.HyperLogLogPlusPlusUniqueCount",
            )
        )
        max_: "Type.Aggregate.Max" = proto.Field(
            proto.MESSAGE,
            number=6,
            oneof="aggregator",
            message="Type.Aggregate.Max",
        )
        min_: "Type.Aggregate.Min" = proto.Field(
            proto.MESSAGE,
            number=7,
            oneof="aggregator",
            message="Type.Aggregate.Min",
        )

    bytes_type: Bytes = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="kind",
        message=Bytes,
    )
    string_type: String = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="kind",
        message=String,
    )
    int64_type: Int64 = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="kind",
        message=Int64,
    )
    float32_type: Float32 = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="kind",
        message=Float32,
    )
    float64_type: Float64 = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="kind",
        message=Float64,
    )
    bool_type: Bool = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="kind",
        message=Bool,
    )
    timestamp_type: Timestamp = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="kind",
        message=Timestamp,
    )
    date_type: Date = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="kind",
        message=Date,
    )
    aggregate_type: Aggregate = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="kind",
        message=Aggregate,
    )
    struct_type: Struct = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="kind",
        message=Struct,
    )
    array_type: Array = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="kind",
        message=Array,
    )
    map_type: Map = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="kind",
        message=Map,
    )
    proto_type: Proto = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="kind",
        message=Proto,
    )
    enum_type: Enum = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="kind",
        message=Enum,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

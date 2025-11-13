# Copyright 2024 Google LLC
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
from __future__ import annotations

from typing import (
    List,
    TypeVar,
    Generic,
    Iterable,
    Optional,
    Sequence,
)
from abc import ABC, abstractmethod
from google.protobuf.message import Message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from google.cloud.bigtable_v2 import ProtoRows, Value as PBValue

from google.cloud.bigtable.data.execute_query._query_result_parsing_utils import (
    _parse_pb_value_to_python_value,
)

from google.cloud.bigtable.helpers import batched

from google.cloud.bigtable.data.execute_query.values import QueryResultRow
from google.cloud.bigtable.data.execute_query.metadata import Metadata


T = TypeVar("T")


class _Reader(ABC, Generic[T]):
    """
    An interface for classes that consume and parse bytes returned by ``_ByteCursor``.
    Parsed bytes should be gathered into bundles (rows or columns) of expected size
    and converted to an appropriate type ``T`` that will be returned as a semantically
    meaningful result to the library user by
    :meth:`google.cloud.bigtable.instance.Instance.execute_query` or
    :meth:`google.cloud.bigtable.data._async.client.BigtableDataClientAsync.execute_query`
    methods.

    This class consumes data obtained externally to be usable in both sync and async clients.

    See :class:`google.cloud.bigtable.byte_cursor._ByteCursor` for more context.
    """

    @abstractmethod
    def consume(
        self,
        batches_to_consume: List[bytes],
        metadata: Metadata,
        column_info: dict[str, Message | EnumTypeWrapper] | None = None,
    ) -> Optional[Iterable[T]]:
        """This method receives a list of batches of bytes to be parsed as ProtoRows messages.
        It then uses the metadata to group the values in the parsed messages into rows. Returns
        None if batches_to_consume is empty
        Args:
            bytes_to_consume (bytes): chunk of parsable byte batches received from
                :meth:`google.cloud.bigtable.byte_cursor._ByteCursor.consume`
                method.
            metadata: metadata used to transform values to rows
            column_info: (Optional) dict with mappings between column names and additional column information
                for protobuf deserialization.

        Returns:
            Iterable[T] or None: Iterable if gathered values can form one or more instances of T,
                or None if there is not enough data to construct at least one instance of T with
                appropriate number of entries.
        """
        raise NotImplementedError


class _QueryResultRowReader(_Reader[QueryResultRow]):
    """
    A :class:`._Reader` consuming bytes representing
    :class:`google.cloud.bigtable_v2.types.Type`
    and producing :class:`google.cloud.bigtable.execute_query.QueryResultRow`.

    Number of entries in each row is determined by number of columns in
    :class:`google.cloud.bigtable.execute_query.Metadata` obtained from
    :class:`google.cloud.bigtable.byte_cursor._ByteCursor` passed in the constructor.
    """

    def _parse_proto_rows(self, bytes_to_parse: bytes) -> Iterable[PBValue]:
        proto_rows = ProtoRows.pb().FromString(bytes_to_parse)
        return proto_rows.values

    def _construct_query_result_row(
        self,
        values: Sequence[PBValue],
        metadata: Metadata,
        column_info: dict[str, Message | EnumTypeWrapper] | None = None,
    ) -> QueryResultRow:
        result = QueryResultRow()
        columns = metadata.columns

        assert len(values) == len(
            columns
        ), "This function should be called only when count of values matches count of columns."

        for column, value in zip(columns, values):
            parsed_value = _parse_pb_value_to_python_value(
                value, column.column_type, column.column_name, column_info
            )
            result.add_field(column.column_name, parsed_value)
        return result

    def consume(
        self,
        batches_to_consume: List[bytes],
        metadata: Metadata,
        column_info: dict[str, Message | EnumTypeWrapper] | None = None,
    ) -> Optional[Iterable[QueryResultRow]]:
        num_columns = len(metadata.columns)
        rows = []
        for batch_bytes in batches_to_consume:
            values = self._parse_proto_rows(batch_bytes)
            for row_data in batched(values, n=num_columns):
                if len(row_data) == num_columns:
                    rows.append(
                        self._construct_query_result_row(
                            row_data, metadata, column_info
                        )
                    )
                else:
                    raise ValueError(
                        "Unexpected error, recieved bad number of values. "
                        f"Expected {num_columns} got {len(row_data)}."
                    )

        return rows

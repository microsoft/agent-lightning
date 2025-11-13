# Copyright 2023 Google LLC
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
from typing import Any
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sys import getsizeof

import google.cloud.bigtable_v2.types.bigtable as types_pb
import google.cloud.bigtable_v2.types.data as data_pb

from google.cloud.bigtable.data.read_modify_write_rules import _MAX_INCREMENT_VALUE


# special value for SetCell mutation timestamps. If set, server will assign a timestamp
_SERVER_SIDE_TIMESTAMP = -1

# mutation entries above this should be rejected
_MUTATE_ROWS_REQUEST_MUTATION_LIMIT = 100_000


class Mutation(ABC):
    """
    Abstract base class for mutations.

    This class defines the interface for different types of mutations that can be
    applied to Bigtable rows.
    """

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """
        Convert the mutation to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the mutation.
        """
        raise NotImplementedError

    def _to_pb(self) -> data_pb.Mutation:
        """
        Convert the mutation to a protobuf representation.

        Returns:
            Mutation: A protobuf representation of the mutation.
        """
        return data_pb.Mutation(**self._to_dict())

    def is_idempotent(self) -> bool:
        """
        Check if the mutation is idempotent

        Idempotent mutations can be safely retried on failure.

        Returns:
            bool: True if the mutation is idempotent, False otherwise.
        """
        return True

    def __str__(self) -> str:
        """
        Return a string representation of the mutation.

        Returns:
            str: A string representation of the mutation.
        """
        return str(self._to_dict())

    def size(self) -> int:
        """
        Get the size of the mutation in bytes

        Returns:
            int: The size of the mutation in bytes.
        """
        return getsizeof(self._to_dict())

    @classmethod
    def _from_dict(cls, input_dict: dict[str, Any]) -> Mutation:
        """
        Create a `Mutation` instance from a dictionary representation.

        Args:
            input_dict: A dictionary representation of the mutation.
        Returns:
            Mutation: A Mutation instance created from the dictionary.
        Raises:
            ValueError: If the input dictionary is invalid or does not represent a valid mutation type.
        """
        instance: Mutation | None = None
        try:
            if "set_cell" in input_dict:
                details = input_dict["set_cell"]
                instance = SetCell(
                    details["family_name"],
                    details["column_qualifier"],
                    details["value"],
                    details["timestamp_micros"],
                )
            elif "delete_from_column" in input_dict:
                details = input_dict["delete_from_column"]
                time_range = details.get("time_range", {})
                start = time_range.get("start_timestamp_micros", None)
                end = time_range.get("end_timestamp_micros", None)
                instance = DeleteRangeFromColumn(
                    details["family_name"], details["column_qualifier"], start, end
                )
            elif "delete_from_family" in input_dict:
                details = input_dict["delete_from_family"]
                instance = DeleteAllFromFamily(details["family_name"])
            elif "delete_from_row" in input_dict:
                instance = DeleteAllFromRow()
            elif "add_to_cell" in input_dict:
                details = input_dict["add_to_cell"]
                instance = AddToCell(
                    details["family_name"],
                    details["column_qualifier"]["raw_value"],
                    details["input"]["int_value"],
                    details["timestamp"]["raw_timestamp_micros"],
                )
        except KeyError as e:
            raise ValueError("Invalid mutation dictionary") from e
        if instance is None:
            raise ValueError("No valid mutation found")
        if not issubclass(instance.__class__, cls):
            raise ValueError("Mutation type mismatch")
        return instance


class SetCell(Mutation):
    """
    Mutation to set the value of a cell.

    Args:
        family: The name of the column family to which the new cell belongs.
        qualifier: The column qualifier of the new cell.
        new_value: The value of the new cell.
        timestamp_micros: The timestamp of the new cell. If `None`,
            the current timestamp will be used. Timestamps will be sent with
            millisecond precision. Extra precision will be truncated. If -1, the
            server will assign a timestamp. Note that `SetCell` mutations with
            server-side timestamps are non-idempotent operations and will not be retried.

    Raises:
        TypeError: If `qualifier` is not `bytes` or `str`.
        TypeError: If `new_value` is not `bytes`, `str`, or `int`.
        ValueError: If `timestamp_micros` is less than `_SERVER_SIDE_TIMESTAMP`.
    """

    def __init__(
        self,
        family: str,
        qualifier: bytes | str,
        new_value: bytes | str | int,
        timestamp_micros: int | None = None,
    ):
        qualifier = qualifier.encode() if isinstance(qualifier, str) else qualifier
        if not isinstance(qualifier, bytes):
            raise TypeError("qualifier must be bytes or str")
        if isinstance(new_value, str):
            new_value = new_value.encode()
        elif isinstance(new_value, int):
            if abs(new_value) > _MAX_INCREMENT_VALUE:
                raise ValueError(
                    "int values must be between -2**63 and 2**63 (64-bit signed int)"
                )
            new_value = new_value.to_bytes(8, "big", signed=True)
        if not isinstance(new_value, bytes):
            raise TypeError("new_value must be bytes, str, or int")
        if timestamp_micros is None:
            # use current timestamp, with milisecond precision
            timestamp_micros = time.time_ns() // 1000
            timestamp_micros = timestamp_micros - (timestamp_micros % 1000)
        if timestamp_micros < _SERVER_SIDE_TIMESTAMP:
            raise ValueError(
                f"timestamp_micros must be positive (or {_SERVER_SIDE_TIMESTAMP} for server-side timestamp)"
            )
        self.family = family
        self.qualifier = qualifier
        self.new_value = new_value
        self.timestamp_micros = timestamp_micros

    def _to_dict(self) -> dict[str, Any]:
        return {
            "set_cell": {
                "family_name": self.family,
                "column_qualifier": self.qualifier,
                "timestamp_micros": self.timestamp_micros,
                "value": self.new_value,
            }
        }

    def is_idempotent(self) -> bool:
        return self.timestamp_micros != _SERVER_SIDE_TIMESTAMP


@dataclass
class DeleteRangeFromColumn(Mutation):
    """
    Mutation to delete a range of cells from a column.

    Args:
        family: The name of the column family.
         qualifier: The column qualifier.
        start_timestamp_micros: The start timestamp of the range to
            delete. `None` represents 0. Defaults to `None`.
        end_timestamp_micros: The end timestamp of the range to
            delete. `None` represents infinity. Defaults to `None`.
    Raises:
        ValueError: If `start_timestamp_micros` is greater than `end_timestamp_micros`.
    """

    family: str
    qualifier: bytes
    # None represents 0
    start_timestamp_micros: int | None = None
    # None represents infinity
    end_timestamp_micros: int | None = None

    def __post_init__(self):
        if (
            self.start_timestamp_micros is not None
            and self.end_timestamp_micros is not None
            and self.start_timestamp_micros > self.end_timestamp_micros
        ):
            raise ValueError("start_timestamp_micros must be <= end_timestamp_micros")

    def _to_dict(self) -> dict[str, Any]:
        timestamp_range = {}
        if self.start_timestamp_micros is not None:
            timestamp_range["start_timestamp_micros"] = self.start_timestamp_micros
        if self.end_timestamp_micros is not None:
            timestamp_range["end_timestamp_micros"] = self.end_timestamp_micros
        return {
            "delete_from_column": {
                "family_name": self.family,
                "column_qualifier": self.qualifier,
                "time_range": timestamp_range,
            }
        }


@dataclass
class DeleteAllFromFamily(Mutation):
    """
    Mutation to delete all cells from a column family.

    Args:
        family_to_delete: The name of the column family to delete.
    """

    family_to_delete: str

    def _to_dict(self) -> dict[str, Any]:
        return {
            "delete_from_family": {
                "family_name": self.family_to_delete,
            }
        }


@dataclass
class DeleteAllFromRow(Mutation):
    """
    Mutation to delete all cells from a row.
    """

    def _to_dict(self) -> dict[str, Any]:
        return {
            "delete_from_row": {},
        }


@dataclass
class AddToCell(Mutation):
    """
    Adds an int64 value to an aggregate cell. The column family must be an
    aggregate family and have an "int64" input type or this mutation will be
    rejected.

    Note: The timestamp values are in microseconds but must match the
    granularity of the table (defaults to `MILLIS`). Therefore, the given value
    must be a multiple of 1000 (millisecond granularity). For example:
    `1571902339435000`.

    Args:
        family: The name of the column family to which the cell belongs.
        qualifier: The column qualifier of the cell.
        value: The value to be accumulated into the cell.
        timestamp_micros: The timestamp of the cell. Must be provided for
          cell aggregation to work correctly.


    Raises:
        TypeError: If `qualifier` is not `bytes` or `str`.
        TypeError: If `value` is not `int`.
        TypeError: If `timestamp_micros` is not `int`.
        ValueError: If `value` is out of bounds for a 64-bit signed int.
        ValueError: If `timestamp_micros` is less than 0.
    """

    def __init__(
        self,
        family: str,
        qualifier: bytes | str,
        value: int,
        timestamp_micros: int,
    ):
        qualifier = qualifier.encode() if isinstance(qualifier, str) else qualifier
        if not isinstance(qualifier, bytes):
            raise TypeError("qualifier must be bytes or str")
        if not isinstance(value, int):
            raise TypeError("value must be int")
        if not isinstance(timestamp_micros, int):
            raise TypeError("timestamp_micros must be int")
        if abs(value) > _MAX_INCREMENT_VALUE:
            raise ValueError(
                "int values must be between -2**63 and 2**63 (64-bit signed int)"
            )

        if timestamp_micros < 0:
            raise ValueError("timestamp must be non-negative")

        self.family = family
        self.qualifier = qualifier
        self.value = value
        self.timestamp = timestamp_micros

    def _to_dict(self) -> dict[str, Any]:
        return {
            "add_to_cell": {
                "family_name": self.family,
                "column_qualifier": {"raw_value": self.qualifier},
                "timestamp": {"raw_timestamp_micros": self.timestamp},
                "input": {"int_value": self.value},
            }
        }

    def is_idempotent(self) -> bool:
        return False


class RowMutationEntry:
    """
    A single entry in a `MutateRows` request.

    This class represents a set of mutations to apply to a specific row in a
    Bigtable table.

    Args:
        row_key: The key of the row to mutate.
        mutations: The mutation or list of mutations to apply
            to the row.

    Raises:
        ValueError: If `mutations` is empty or contains more than
            `_MUTATE_ROWS_REQUEST_MUTATION_LIMIT` mutations.
    """

    def __init__(self, row_key: bytes | str, mutations: Mutation | list[Mutation]):
        if isinstance(row_key, str):
            row_key = row_key.encode("utf-8")
        if isinstance(mutations, Mutation):
            mutations = [mutations]
        if len(mutations) == 0:
            raise ValueError("mutations must not be empty")
        elif len(mutations) > _MUTATE_ROWS_REQUEST_MUTATION_LIMIT:
            raise ValueError(
                f"entries must have <= {_MUTATE_ROWS_REQUEST_MUTATION_LIMIT} mutations"
            )
        self.row_key = row_key
        self.mutations = tuple(mutations)

    def _to_dict(self) -> dict[str, Any]:
        """
        Convert the mutation entry to a dictionary representation.

        Returns:
            dict[str, Any]: A dictionary representation of the mutation entry
        """
        return {
            "row_key": self.row_key,
            "mutations": [mutation._to_dict() for mutation in self.mutations],
        }

    def _to_pb(self) -> types_pb.MutateRowsRequest.Entry:
        """
        Convert the mutation entry to a protobuf representation.

        Returns:
            MutateRowsRequest.Entry: A protobuf representation of the mutation entry.
        """
        return types_pb.MutateRowsRequest.Entry(
            row_key=self.row_key,
            mutations=[mutation._to_pb() for mutation in self.mutations],
        )

    def is_idempotent(self) -> bool:
        """
        Check if all mutations in the entry are idempotent.

        Returns:
            bool: True if all mutations in the entry are idempotent, False otherwise.
        """
        return all(mutation.is_idempotent() for mutation in self.mutations)

    def size(self) -> int:
        """
        Get the size of the mutation entry in bytes.

        Returns:
            int: The size of the mutation entry in bytes.
        """
        return getsizeof(self._to_dict())

    @classmethod
    def _from_dict(cls, input_dict: dict[str, Any]) -> RowMutationEntry:
        """
        Create a `RowMutationEntry` instance from a dictionary representation.

        Args:
            input_dict: A dictionary representation of the mutation entry.

        Returns:
            RowMutationEntry: A RowMutationEntry instance created from the dictionary.
        """
        return RowMutationEntry(
            row_key=input_dict["row_key"],
            mutations=[
                Mutation._from_dict(mutation) for mutation in input_dict["mutations"]
            ],
        )


@dataclass
class _EntryWithProto:
    """
    A dataclass to hold a RowMutationEntry and its corresponding proto representation.

    Used in _MutateRowsOperation to avoid repeated conversion of RowMutationEntry to proto.
    """

    entry: RowMutationEntry
    proto: types_pb.MutateRowsRequest.Entry

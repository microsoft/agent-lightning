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

from collections import OrderedDict
from typing import Generator, overload, Any
from functools import total_ordering

from google.cloud.bigtable_v2.types import Row as RowPB

# Type aliases used internally for readability.
_family_type = str
_qualifier_type = bytes


class Row:
    """
    Model class for row data returned from server

    Does not represent all data contained in the row, only data returned by a
    query.
    Expected to be read-only to users, and written by backend

    Can be indexed by family and qualifier to get cells in the row::

        cells = row["family", "qualifier"]

    Args:
        key: Row key
        cells: List of cells in the row
    """

    __slots__ = ("row_key", "cells", "_index_data")

    def __init__(
        self,
        key: bytes,
        cells: list[Cell],
    ):
        """
        Row objects are not intended to be created by users.
        They are returned by the Bigtable backend.
        """
        self.row_key = key
        self.cells: list[Cell] = cells
        # index is lazily created when needed
        self._index_data: OrderedDict[
            _family_type, OrderedDict[_qualifier_type, list[Cell]]
        ] | None = None

    @property
    def _index(
        self,
    ) -> OrderedDict[_family_type, OrderedDict[_qualifier_type, list[Cell]]]:
        """
        Returns an index of cells associated with each family and qualifier.

        The index is lazily created when needed

        Returns:
            OrderedDict: Index of cells
        """
        if self._index_data is None:
            self._index_data = OrderedDict()
            for cell in self.cells:
                self._index_data.setdefault(cell.family, OrderedDict()).setdefault(
                    cell.qualifier, []
                ).append(cell)
        return self._index_data

    @classmethod
    def _from_pb(cls, row_pb: RowPB) -> Row:
        """
        Creates a row from a protobuf representation

        Row objects are not intended to be created by users.
        They are returned by the Bigtable backend.

        Args:
            row_pb (RowPB): Protobuf representation of the row
        Returns:
            Row: Row object created from the protobuf representation
        """
        row_key: bytes = row_pb.key
        cell_list: list[Cell] = []
        for family in row_pb.families:
            for column in family.columns:
                for cell in column.cells:
                    new_cell = Cell(
                        value=cell.value,
                        row_key=row_key,
                        family=family.name,
                        qualifier=column.qualifier,
                        timestamp_micros=cell.timestamp_micros,
                        labels=list(cell.labels) if cell.labels else None,
                    )
                    cell_list.append(new_cell)
        return cls(row_key, cells=cell_list)

    def get_cells(
        self, family: str | None = None, qualifier: str | bytes | None = None
    ) -> list[Cell]:
        """
        Returns cells sorted in Bigtable native order:
            - Family lexicographically ascending
            - Qualifier ascending
            - Timestamp in reverse chronological order

        If family or qualifier not passed, will include all

        Can also be accessed through indexing::
            cells = row["family", "qualifier"]
            cells = row["family"]

        Args:
            family: family to filter cells by
            qualifier: qualifier to filter cells by
        Returns:
            list[Cell]: List of cells in the row matching the filter
        Raises:
            ValueError: If family or qualifier is not found in the row
        """
        if family is None:
            if qualifier is not None:
                # get_cells(None, "qualifier") is not allowed
                raise ValueError("Qualifier passed without family")
            else:
                # return all cells on get_cells()
                return self.cells
        if qualifier is None:
            # return all cells in family on get_cells(family)
            return list(self._get_all_from_family(family))
        if isinstance(qualifier, str):
            qualifier = qualifier.encode("utf-8")
        # return cells in family and qualifier on get_cells(family, qualifier)
        if family not in self._index:
            raise ValueError(f"Family '{family}' not found in row '{self.row_key!r}'")
        if qualifier not in self._index[family]:
            raise ValueError(
                f"Qualifier '{qualifier!r}' not found in family '{family}' in row '{self.row_key!r}'"
            )
        return self._index[family][qualifier]

    def _get_all_from_family(self, family: str) -> Generator[Cell, None, None]:
        """
        Returns all cells in the row for the family_id

        Args:
            family: family to filter cells by
        Yields:
            Cell: cells in the row for the family_id
        Raises:
            ValueError: If family is not found in the row
        """
        if family not in self._index:
            raise ValueError(f"Family '{family}' not found in row '{self.row_key!r}'")
        for qualifier in self._index[family]:
            yield from self._index[family][qualifier]

    def __str__(self) -> str:
        """
        Human-readable string representation::

            {
              (family='fam', qualifier=b'col'): [b'value', (+1 more),],
              (family='fam', qualifier=b'col2'): [b'other'],
            }

        Returns:
            str: Human-readable string representation of the row
        """
        output = ["{"]
        for family, qualifier in self._get_column_components():
            cell_list = self[family, qualifier]
            line = [f"  (family={family!r}, qualifier={qualifier!r}): "]
            if len(cell_list) == 0:
                line.append("[],")
            elif len(cell_list) == 1:
                line.append(f"[{cell_list[0]}],")
            else:
                line.append(f"[{cell_list[0]}, (+{len(cell_list) - 1} more)],")
            output.append("".join(line))
        output.append("}")
        return "\n".join(output)

    def __repr__(self):
        cell_str_buffer = ["{"]
        for family, qualifier in self._get_column_components():
            cell_list = self[family, qualifier]
            repr_list = [cell._to_dict() for cell in cell_list]
            cell_str_buffer.append(f"  ('{family}', {qualifier!r}): {repr_list},")
        cell_str_buffer.append("}")
        cell_str = "\n".join(cell_str_buffer)
        output = f"Row(key={self.row_key!r}, cells={cell_str})"
        return output

    def _to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the cell in the Bigtable Row
        proto format

        https://cloud.google.com/bigtable/docs/reference/data/rpc/google.bigtable.v2#row
        """
        family_list = []
        for family_name, qualifier_dict in self._index.items():
            qualifier_list = []
            for qualifier_name, cell_list in qualifier_dict.items():
                cell_dicts = [cell._to_dict() for cell in cell_list]
                qualifier_list.append(
                    {"qualifier": qualifier_name, "cells": cell_dicts}
                )
            family_list.append({"name": family_name, "columns": qualifier_list})
        return {"key": self.row_key, "families": family_list}

    # Sequence and Mapping methods
    def __iter__(self):
        """
        Allow iterating over all cells in the row

        Returns:
            Iterator: Iterator over the cells in the row
        """
        return iter(self.cells)

    def __contains__(self, item):
        """
        Implements `in` operator

        Works for both cells in the internal list, and `family` or
        `(family, qualifier)` pairs associated with the cells

        Args:
            item: item to check for in the row
        Returns:
            bool: True if item is in the row, False otherwise
        """
        if isinstance(item, _family_type):
            return item in self._index
        elif (
            isinstance(item, tuple)
            and isinstance(item[0], _family_type)
            and isinstance(item[1], (bytes, str))
        ):
            q = item[1] if isinstance(item[1], bytes) else item[1].encode("utf-8")
            return item[0] in self._index and q in self._index[item[0]]
        # check if Cell is in Row
        return item in self.cells

    @overload
    def __getitem__(
        self,
        index: str | tuple[str, bytes | str],
    ) -> list[Cell]:
        # overload signature for type checking
        pass

    @overload
    def __getitem__(self, index: int) -> Cell:
        # overload signature for type checking
        pass

    @overload
    def __getitem__(self, index: slice) -> list[Cell]:
        # overload signature for type checking
        pass

    def __getitem__(self, index):
        """
        Implements [] indexing

        Supports indexing by family, (family, qualifier) pair,
        numerical index, and index slicing
        """
        if isinstance(index, _family_type):
            return self.get_cells(family=index)
        elif (
            isinstance(index, tuple)
            and isinstance(index[0], _family_type)
            and isinstance(index[1], (bytes, str))
        ):
            return self.get_cells(family=index[0], qualifier=index[1])
        elif isinstance(index, int) or isinstance(index, slice):
            # index is int or slice
            return self.cells[index]
        else:
            raise TypeError(
                "Index must be family_id, (family_id, qualifier), int, or slice"
            )

    def __len__(self):
        """
        Returns the number of cells in the row

        Returns:
            int: Number of cells in the row
        """
        return len(self.cells)

    def _get_column_components(self) -> list[tuple[str, bytes]]:
        """
        Returns a list of (family, qualifier) pairs associated with the cells

        Pairs can be used for indexing

        Returns:
            list[tuple[str, bytes]]: List of (family, qualifier) pairs
        """
        return [(f, q) for f in self._index for q in self._index[f]]

    def __eq__(self, other):
        """
        Implements `==` operator

        Returns:
            bool: True if rows are equal, False otherwise
        """
        # for performance reasons, check row metadata
        # before checking individual cells
        if not isinstance(other, Row):
            return False
        if self.row_key != other.row_key:
            return False
        if len(self.cells) != len(other.cells):
            return False
        components = self._get_column_components()
        other_components = other._get_column_components()
        if len(components) != len(other_components):
            return False
        if components != other_components:
            return False
        for family, qualifier in components:
            if len(self[family, qualifier]) != len(other[family, qualifier]):
                return False
        # compare individual cell lists
        if self.cells != other.cells:
            return False
        return True

    def __ne__(self, other) -> bool:
        """
        Implements `!=` operator

        Returns:
            bool: True if rows are not equal, False otherwise
        """
        return not self == other


@total_ordering
class Cell:
    """
    Model class for cell data

    Does not represent all data contained in the cell, only data returned by a
    query.
    Expected to be read-only to users, and written by backend

    Args:
        value: the byte string value of the cell
        row_key: the row key of the cell
        family: the family associated with the cell
        qualifier: the column qualifier associated with the cell
        timestamp_micros: the timestamp of the cell in microseconds
        labels: the list of labels associated with the cell
    """

    __slots__ = (
        "value",
        "row_key",
        "family",
        "qualifier",
        "timestamp_micros",
        "labels",
    )

    def __init__(
        self,
        value: bytes,
        row_key: bytes,
        family: str,
        qualifier: bytes | str,
        timestamp_micros: int,
        labels: list[str] | None = None,
    ):
        # Cell objects are not intended to be constructed by users.
        # They are returned by the Bigtable backend.
        self.value = value
        self.row_key = row_key
        self.family = family
        if isinstance(qualifier, str):
            qualifier = qualifier.encode()
        self.qualifier = qualifier
        self.timestamp_micros = timestamp_micros
        self.labels = labels if labels is not None else []

    def __int__(self) -> int:
        """
        Allows casting cell to int
        Interprets value as a 64-bit big-endian signed integer, as expected by
        ReadModifyWrite increment rule

        Returns:
            int: Value of the cell as a 64-bit big-endian signed integer
        """
        return int.from_bytes(self.value, byteorder="big", signed=True)

    def _to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the cell in the Bigtable Cell
        proto format

        https://cloud.google.com/bigtable/docs/reference/data/rpc/google.bigtable.v2#cell

        Returns:
            dict: Dictionary representation of the cell
        """
        cell_dict: dict[str, Any] = {
            "value": self.value,
        }
        cell_dict["timestamp_micros"] = self.timestamp_micros
        if self.labels:
            cell_dict["labels"] = self.labels
        return cell_dict

    def __str__(self) -> str:
        """
        Allows casting cell to str
        Prints encoded byte string, same as printing value directly.

        Returns:
            str: Encoded byte string of the value
        """
        return str(self.value)

    def __repr__(self):
        """
        Returns a string representation of the cell

        Returns:
            str: String representation of the cell
        """
        return f"Cell(value={self.value!r}, row_key={self.row_key!r}, family='{self.family}', qualifier={self.qualifier!r}, timestamp_micros={self.timestamp_micros}, labels={self.labels})"

    """For Bigtable native ordering"""

    def __lt__(self, other) -> bool:
        """
        Implements `<` operator

        Args:
            other: Cell to compare with
        Returns:
            bool: True if this cell is less than the other cell, False otherwise
        Raises:
            NotImplementedError: If other is not a Cell
        """
        if not isinstance(other, Cell):
            raise NotImplementedError
        this_ordering = (
            self.family,
            self.qualifier,
            -self.timestamp_micros,
            self.value,
            self.labels,
        )
        other_ordering = (
            other.family,
            other.qualifier,
            -other.timestamp_micros,
            other.value,
            other.labels,
        )
        return this_ordering < other_ordering

    def __eq__(self, other) -> bool:
        """
        Implements `==` operator

        Args:
            other: Cell to compare with
        Returns:
            bool: True if cells are equal, False otherwise
        """
        if not isinstance(other, Cell):
            return False
        return (
            self.row_key == other.row_key
            and self.family == other.family
            and self.qualifier == other.qualifier
            and self.value == other.value
            and self.timestamp_micros == other.timestamp_micros
            and len(self.labels) == len(other.labels)
            and all([label in other.labels for label in self.labels])
        )

    def __ne__(self, other) -> bool:
        """
        Implements `!=` operator

        Args:
            other: Cell to compare with
        Returns:
            bool: True if cells are not equal, False otherwise
        """
        return not self == other

    def __hash__(self):
        """
        Implements `hash()` function to fingerprint cell

        Returns:
            int: hash value of the cell
        """
        return hash(
            (
                self.row_key,
                self.family,
                self.qualifier,
                self.value,
                self.timestamp_micros,
                tuple(self.labels),
            )
        )

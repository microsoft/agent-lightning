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

from collections import defaultdict
from typing import (
    Optional,
    List,
    Dict,
    Set,
    Union,
    TypeVar,
    Generic,
    Tuple,
    Mapping,
)
from google.type import date_pb2  # type: ignore
from google.api_core.datetime_helpers import DatetimeWithNanoseconds

T = TypeVar("T")


class _NamedList(Generic[T]):
    """
    A class designed to store a list of elements, which can be accessed by
    name or index.
    This class is different from namedtuple, because namedtuple has some
    restrictions on names of fields and we do not want to have them.
    """

    _str_cls_name = "_NamedList"

    def __init__(self, fields: Optional[List[Tuple[Optional[str], T]]] = None):
        self._fields: List[Tuple[Optional[str], T]] = []
        self._field_indexes: Dict[str, List[int]] = defaultdict(list)
        self._duplicate_names: Set[str] = set()

        if fields:
            for field_name, field_type in fields:
                self.add_field(field_name, field_type)

    def add_field(self, name: Optional[str], value: T):
        if name:
            if name in self._field_indexes:
                self._duplicate_names.add(name)
            self._field_indexes[name].append(len(self._fields))
        self._fields.append((name, value))

    @property
    def fields(self):
        return self._fields

    def __getitem__(self, index_or_name: Union[str, int]):
        if isinstance(index_or_name, str):
            if index_or_name in self._duplicate_names:
                raise KeyError(
                    f"Ambigious field name: '{index_or_name}', use index instead."
                    f" Field present on indexes {', '.join(map(str, self._field_indexes[index_or_name]))}."
                )
            if index_or_name not in self._field_indexes:
                raise KeyError(f"No such field: {index_or_name}")
            index = self._field_indexes[index_or_name][0]
        else:
            index = index_or_name
        return self._fields[index][1]

    def __len__(self):
        return len(self._fields)

    def __eq__(self, other):
        if not isinstance(other, _NamedList):
            return False

        return (
            self._fields == other._fields
            and self._field_indexes == other._field_indexes
        )

    def __str__(self) -> str:
        fields_str = ", ".join([str(field) for field in self._fields])
        return f"{self.__class__.__name__}([{fields_str}])"

    def __repr__(self) -> str:
        return self.__str__()


ExecuteQueryValueType = Union[
    int,
    float,
    bool,
    bytes,
    str,
    # Note that Bigtable SQL does not currently support nanosecond precision,
    # only microseconds. We use this for compatibility with potential future
    # support
    DatetimeWithNanoseconds,
    date_pb2.Date,
    "Struct",
    List["ExecuteQueryValueType"],
    Mapping[Union[str, int, bytes], "ExecuteQueryValueType"],
]


class QueryResultRow(_NamedList[ExecuteQueryValueType]):
    """
    Represents a single row of the result
    """


class Struct(_NamedList[ExecuteQueryValueType]):
    """
    Represents a struct value in the result
    """

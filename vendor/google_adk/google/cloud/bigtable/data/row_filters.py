# Copyright 2016 Google LLC
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

"""Filters for Google Cloud Bigtable Row classes."""
from __future__ import annotations

import struct

from typing import Any, Sequence, TYPE_CHECKING, overload
from abc import ABC, abstractmethod

from google.cloud._helpers import _microseconds_from_datetime  # type: ignore
from google.cloud._helpers import _to_bytes  # type: ignore
from google.cloud.bigtable_v2.types import data as data_v2_pb2

if TYPE_CHECKING:
    # import dependencies when type checking
    from datetime import datetime

_PACK_I64 = struct.Struct(">q").pack


class RowFilter(ABC):
    """Basic filter to apply to cells in a row.

    These values can be combined via :class:`RowFilterChain`,
    :class:`RowFilterUnion` and :class:`ConditionalRowFilter`.

    .. note::

        This class is a do-nothing base class for all row filters.
    """

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        Returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(**self._to_dict())

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class _BoolFilter(RowFilter, ABC):
    """Row filter that uses a boolean flag.

    :type flag: bool
    :param flag: An indicator if a setting is turned on or off.
    """

    def __init__(self, flag: bool):
        self.flag = flag

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.flag == self.flag

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flag={self.flag})"


class SinkFilter(_BoolFilter):
    """Advanced row filter to skip parent filters.

    :type flag: bool
    :param flag: ADVANCED USE ONLY. Hook for introspection into the row filter.
                 Outputs all cells directly to the output of the read rather
                 than to any parent filter. Cannot be used within the
                 ``predicate_filter``, ``true_filter``, or ``false_filter``
                 of a :class:`ConditionalRowFilter`.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"sink": self.flag}


class PassAllFilter(_BoolFilter):
    """Row filter equivalent to not filtering at all.

    :type flag: bool
    :param flag: Matches all cells, regardless of input. Functionally
                 equivalent to leaving ``filter`` unset, but included for
                 completeness.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"pass_all_filter": self.flag}


class BlockAllFilter(_BoolFilter):
    """Row filter that doesn't match any cells.

    :type flag: bool
    :param flag: Does not match any cells, regardless of input. Useful for
                 temporarily disabling just part of a filter.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"block_all_filter": self.flag}


class _RegexFilter(RowFilter, ABC):
    """Row filter that uses a regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    :type regex: bytes or str
    :param regex:
        A regular expression (RE2) for some row filter.  String values
        will be encoded as ASCII.
    """

    def __init__(self, regex: str | bytes):
        self.regex: bytes = _to_bytes(regex)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.regex == self.regex

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regex={self.regex!r})"


class RowKeyRegexFilter(_RegexFilter):
    """Row filter for a row key regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    .. note::

        Special care need be used with the expression used. Since
        each of these properties can contain arbitrary bytes, the ``\\C``
        escape sequence must be used if a true wildcard is desired. The ``.``
        character will not match the new line character ``\\n``, which may be
        present in a binary value.

    :type regex: bytes
    :param regex: A regular expression (RE2) to match cells from rows with row
                  keys that satisfy this regex. For a
                  ``CheckAndMutateRowRequest``, this filter is unnecessary
                  since the row key is already specified.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"row_key_regex_filter": self.regex}


class RowSampleFilter(RowFilter):
    """Matches all cells from a row with probability p.

    :type sample: float
    :param sample: The probability of matching a cell (must be in the
                   interval ``(0, 1)``  The end points are excluded).
    """

    def __init__(self, sample: float):
        self.sample: float = sample

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.sample == self.sample

    def __ne__(self, other):
        return not self == other

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"row_sample_filter": self.sample}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sample={self.sample})"


class FamilyNameRegexFilter(_RegexFilter):
    """Row filter for a family name regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    :type regex: str
    :param regex: A regular expression (RE2) to match cells from columns in a
                  given column family. For technical reasons, the regex must
                  not contain the ``':'`` character, even if it is not being
                  used as a literal.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"family_name_regex_filter": self.regex}


class ColumnQualifierRegexFilter(_RegexFilter):
    """Row filter for a column qualifier regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    .. note::

        Special care need be used with the expression used. Since
        each of these properties can contain arbitrary bytes, the ``\\C``
        escape sequence must be used if a true wildcard is desired. The ``.``
        character will not match the new line character ``\\n``, which may be
        present in a binary value.

    :type regex: bytes
    :param regex: A regular expression (RE2) to match cells from column that
                  match this regex (irrespective of column family).
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"column_qualifier_regex_filter": self.regex}


class TimestampRange(object):
    """Range of time with inclusive lower and exclusive upper bounds.

    :type start: :class:`datetime.datetime`
    :param start: (Optional) The (inclusive) lower bound of the timestamp
                  range. If omitted, defaults to Unix epoch.

    :type end: :class:`datetime.datetime`
    :param end: (Optional) The (exclusive) upper bound of the timestamp
                range. If omitted, no upper bound is used.
    """

    def __init__(self, start: "datetime" | None = None, end: "datetime" | None = None):
        self.start: "datetime" | None = start
        self.end: "datetime" | None = end

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.start == self.start and other.end == self.end

    def __ne__(self, other):
        return not self == other

    def _to_pb(self) -> data_v2_pb2.TimestampRange:
        """Converts the :class:`TimestampRange` to a protobuf.

        Returns: The converted current object.
        """
        return data_v2_pb2.TimestampRange(**self._to_dict())

    def _to_dict(self) -> dict[str, int]:
        """Converts the timestamp range to a dict representation."""
        timestamp_range_kwargs = {}
        if self.start is not None:
            start_time = _microseconds_from_datetime(self.start) // 1000 * 1000
            timestamp_range_kwargs["start_timestamp_micros"] = start_time
        if self.end is not None:
            end_time = _microseconds_from_datetime(self.end)
            if end_time % 1000 != 0:
                # if not a whole milisecond value, round up
                end_time = end_time // 1000 * 1000 + 1000
            timestamp_range_kwargs["end_timestamp_micros"] = end_time
        return timestamp_range_kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start={self.start}, end={self.end})"


class TimestampRangeFilter(RowFilter):
    """Row filter that limits cells to a range of time.

    :type range_: :class:`TimestampRange`
    :param range_: Range of time that cells should match against.
    """

    def __init__(self, start: "datetime" | None = None, end: "datetime" | None = None):
        self.range_: TimestampRange = TimestampRange(start, end)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.range_ == self.range_

    def __ne__(self, other):
        return not self == other

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        First converts the ``range_`` on the current object to a protobuf and
        then uses it in the ``timestamp_range_filter`` field.

        Returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(timestamp_range_filter=self.range_._to_pb())

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"timestamp_range_filter": self.range_._to_dict()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start={self.range_.start!r}, end={self.range_.end!r})"


class ColumnRangeFilter(RowFilter):
    """A row filter to restrict to a range of columns.

    Both the start and end column can be included or excluded in the range.
    By default, we include them both, but this can be changed with optional
    flags.

    :type family_id: str
    :param family_id: The column family that contains the columns. Must
                             be of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

    :type start_qualifier: bytes
    :param start_qualifier: The start of the range of columns. If no value is
                         used, the backend applies no upper bound to the
                         values.

    :type end_qualifier: bytes
    :param end_qualifier: The end of the range of columns. If no value is used,
                       the backend applies no upper bound to the values.

    :type inclusive_start: bool
    :param inclusive_start: Boolean indicating if the start column should be
                            included in the range (or excluded). Defaults
                            to :data:`True` if ``start_qualifier`` is passed and
                            no ``inclusive_start`` was given.

    :type inclusive_end: bool
    :param inclusive_end: Boolean indicating if the end column should be
                          included in the range (or excluded). Defaults
                          to :data:`True` if ``end_qualifier`` is passed and
                          no ``inclusive_end`` was given.

    :raises: :class:`ValueError <exceptions.ValueError>` if ``inclusive_start``
             is set but no ``start_qualifier`` is given or if ``inclusive_end``
             is set but no ``end_qualifier`` is given
    """

    def __init__(
        self,
        family_id: str,
        start_qualifier: bytes | None = None,
        end_qualifier: bytes | None = None,
        inclusive_start: bool | None = None,
        inclusive_end: bool | None = None,
    ):
        if inclusive_start is None:
            inclusive_start = True
        elif start_qualifier is None:
            raise ValueError(
                "inclusive_start was specified but no start_qualifier was given."
            )
        if inclusive_end is None:
            inclusive_end = True
        elif end_qualifier is None:
            raise ValueError(
                "inclusive_end was specified but no end_qualifier was given."
            )

        self.family_id = family_id

        self.start_qualifier = start_qualifier
        self.inclusive_start = inclusive_start

        self.end_qualifier = end_qualifier
        self.inclusive_end = inclusive_end

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.family_id == self.family_id
            and other.start_qualifier == self.start_qualifier
            and other.end_qualifier == self.end_qualifier
            and other.inclusive_start == self.inclusive_start
            and other.inclusive_end == self.inclusive_end
        )

    def __ne__(self, other):
        return not self == other

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        First converts to a :class:`.data_v2_pb2.ColumnRange` and then uses it
        in the ``column_range_filter`` field.

        Returns: The converted current object.
        """
        column_range = data_v2_pb2.ColumnRange(**self._range_to_dict())
        return data_v2_pb2.RowFilter(column_range_filter=column_range)

    def _range_to_dict(self) -> dict[str, str | bytes]:
        """Converts the column range range to a dict representation."""
        column_range_kwargs: dict[str, str | bytes] = {}
        column_range_kwargs["family_name"] = self.family_id
        if self.start_qualifier is not None:
            if self.inclusive_start:
                key = "start_qualifier_closed"
            else:
                key = "start_qualifier_open"
            column_range_kwargs[key] = _to_bytes(self.start_qualifier)
        if self.end_qualifier is not None:
            if self.inclusive_end:
                key = "end_qualifier_closed"
            else:
                key = "end_qualifier_open"
            column_range_kwargs[key] = _to_bytes(self.end_qualifier)
        return column_range_kwargs

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"column_range_filter": self._range_to_dict()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(family_id='{self.family_id}', start_qualifier={self.start_qualifier!r}, end_qualifier={self.end_qualifier!r}, inclusive_start={self.inclusive_start}, inclusive_end={self.inclusive_end})"


class ValueRegexFilter(_RegexFilter):
    """Row filter for a value regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    .. note::

        Special care need be used with the expression used. Since
        each of these properties can contain arbitrary bytes, the ``\\C``
        escape sequence must be used if a true wildcard is desired. The ``.``
        character will not match the new line character ``\\n``, which may be
        present in a binary value.

    :type regex: bytes or str
    :param regex: A regular expression (RE2) to match cells with values that
                  match this regex.  String values will be encoded as ASCII.
    """

    def _to_dict(self) -> dict[str, bytes]:
        """Converts the row filter to a dict representation."""
        return {"value_regex_filter": self.regex}


class LiteralValueFilter(ValueRegexFilter):
    """Row filter for an exact value.


    :type value: bytes or str or int
    :param value:
        a literal string, integer, or the equivalent bytes.
        Integer values will be packed into signed 8-bytes.
    """

    def __init__(self, value: bytes | str | int):
        if isinstance(value, int):
            value = _PACK_I64(value)
        elif isinstance(value, str):
            value = value.encode("utf-8")
        value = self._write_literal_regex(value)
        super(LiteralValueFilter, self).__init__(value)

    @staticmethod
    def _write_literal_regex(input_bytes: bytes) -> bytes:
        """
        Escape re2 special characters from literal bytes.

        Extracted from: re2 QuoteMeta:
        https://github.com/google/re2/blob/70f66454c255080a54a8da806c52d1f618707f8a/re2/re2.cc#L456
        """
        result = bytearray()
        for byte in input_bytes:
            # If this is the part of a UTF8 or Latin1 character, we need \
            # to copy this byte without escaping.  Experimentally this is \
            # what works correctly with the regexp library. \
            utf8_latin1_check = (byte & 128) == 0
            if (
                (byte < ord("a") or byte > ord("z"))
                and (byte < ord("A") or byte > ord("Z"))
                and (byte < ord("0") or byte > ord("9"))
                and byte != ord("_")
                and utf8_latin1_check
            ):
                if byte == 0:
                    # Special handling for null chars.
                    # Note that this special handling is not strictly required for RE2,
                    # but this quoting is required for other regexp libraries such as
                    # PCRE.
                    # Can't use "\\0" since the next character might be a digit.
                    result.extend([ord("\\"), ord("x"), ord("0"), ord("0")])
                    continue
                result.append(ord(b"\\"))
            result.append(byte)
        return bytes(result)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.regex!r})"


class ValueRangeFilter(RowFilter):
    """A range of values to restrict to in a row filter.

    Will only match cells that have values in this range.

    Both the start and end value can be included or excluded in the range.
    By default, we include them both, but this can be changed with optional
    flags.

    :type start_value: bytes
    :param start_value: The start of the range of values. If no value is used,
                        the backend applies no lower bound to the values.

    :type end_value: bytes
    :param end_value: The end of the range of values. If no value is used,
                      the backend applies no upper bound to the values.

    :type inclusive_start: bool
    :param inclusive_start: Boolean indicating if the start value should be
                            included in the range (or excluded). Defaults
                            to :data:`True` if ``start_value`` is passed and
                            no ``inclusive_start`` was given.

    :type inclusive_end: bool
    :param inclusive_end: Boolean indicating if the end value should be
                          included in the range (or excluded). Defaults
                          to :data:`True` if ``end_value`` is passed and
                          no ``inclusive_end`` was given.

    :raises: :class:`ValueError <exceptions.ValueError>` if ``inclusive_start``
             is set but no ``start_value`` is given or if ``inclusive_end``
             is set but no ``end_value`` is given
    """

    def __init__(
        self,
        start_value: bytes | int | None = None,
        end_value: bytes | int | None = None,
        inclusive_start: bool | None = None,
        inclusive_end: bool | None = None,
    ):
        if inclusive_start is None:
            inclusive_start = True
        elif start_value is None:
            raise ValueError(
                "inclusive_start was specified but no start_value was given."
            )
        if inclusive_end is None:
            inclusive_end = True
        elif end_value is None:
            raise ValueError(
                "inclusive_end was specified but no end_qualifier was given."
            )
        if isinstance(start_value, int):
            start_value = _PACK_I64(start_value)
        self.start_value = start_value
        self.inclusive_start = inclusive_start

        if isinstance(end_value, int):
            end_value = _PACK_I64(end_value)
        self.end_value = end_value
        self.inclusive_end = inclusive_end

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.start_value == self.start_value
            and other.end_value == self.end_value
            and other.inclusive_start == self.inclusive_start
            and other.inclusive_end == self.inclusive_end
        )

    def __ne__(self, other):
        return not self == other

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        First converts to a :class:`.data_v2_pb2.ValueRange` and then uses
        it to create a row filter protobuf.

        Returns: The converted current object.
        """
        value_range = data_v2_pb2.ValueRange(**self._range_to_dict())
        return data_v2_pb2.RowFilter(value_range_filter=value_range)

    def _range_to_dict(self) -> dict[str, bytes]:
        """Converts the value range range to a dict representation."""
        value_range_kwargs = {}
        if self.start_value is not None:
            if self.inclusive_start:
                key = "start_value_closed"
            else:
                key = "start_value_open"
            value_range_kwargs[key] = _to_bytes(self.start_value)
        if self.end_value is not None:
            if self.inclusive_end:
                key = "end_value_closed"
            else:
                key = "end_value_open"
            value_range_kwargs[key] = _to_bytes(self.end_value)
        return value_range_kwargs

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"value_range_filter": self._range_to_dict()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start_value={self.start_value!r}, end_value={self.end_value!r}, inclusive_start={self.inclusive_start}, inclusive_end={self.inclusive_end})"


class _CellCountFilter(RowFilter, ABC):
    """Row filter that uses an integer count of cells.

    The cell count is used as an offset or a limit for the number
    of results returned.

    :type num_cells: int
    :param num_cells: An integer count / offset / limit.
    """

    def __init__(self, num_cells: int):
        self.num_cells = num_cells

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.num_cells == self.num_cells

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cells={self.num_cells})"


class CellsRowOffsetFilter(_CellCountFilter):
    """Row filter to skip cells in a row.

    :type num_cells: int
    :param num_cells: Skips the first N cells of the row.
    """

    def _to_dict(self) -> dict[str, int]:
        """Converts the row filter to a dict representation."""
        return {"cells_per_row_offset_filter": self.num_cells}


class CellsRowLimitFilter(_CellCountFilter):
    """Row filter to limit cells in a row.

    :type num_cells: int
    :param num_cells: Matches only the first N cells of the row.
    """

    def _to_dict(self) -> dict[str, int]:
        """Converts the row filter to a dict representation."""
        return {"cells_per_row_limit_filter": self.num_cells}


class CellsColumnLimitFilter(_CellCountFilter):
    """Row filter to limit cells in a column.

    :type num_cells: int
    :param num_cells: Matches only the most recent N cells within each column.
                      This filters a (family name, column) pair, based on
                      timestamps of each cell.
    """

    def _to_dict(self) -> dict[str, int]:
        """Converts the row filter to a dict representation."""
        return {"cells_per_column_limit_filter": self.num_cells}


class StripValueTransformerFilter(_BoolFilter):
    """Row filter that transforms cells into empty string (0 bytes).

    :type flag: bool
    :param flag: If :data:`True`, replaces each cell's value with the empty
                 string. As the name indicates, this is more useful as a
                 transformer than a generic query / filter.
    """

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"strip_value_transformer": self.flag}


class ApplyLabelFilter(RowFilter):
    """Filter to apply labels to cells.

    Intended to be used as an intermediate filter on a pre-existing filtered
    result set. This way if two sets are combined, the label can tell where
    the cell(s) originated.This allows the client to determine which results
    were produced from which part of the filter.

    .. note::

        Due to a technical limitation of the backend, it is not currently
        possible to apply multiple labels to a cell.

    :type label: str
    :param label: Label to apply to cells in the output row. Values must be
                  at most 15 characters long, and match the pattern
                  ``[a-z0-9\\-]+``.
    """

    def __init__(self, label: str):
        self.label = label

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.label == self.label

    def __ne__(self, other):
        return not self == other

    def _to_dict(self) -> dict[str, str]:
        """Converts the row filter to a dict representation."""
        return {"apply_label_transformer": self.label}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label})"


class _FilterCombination(RowFilter, Sequence[RowFilter], ABC):
    """Chain of row filters.

    Sends rows through several filters in sequence. The filters are "chained"
    together to process a row. After the first filter is applied, the second
    is applied to the filtered output and so on for subsequent filters.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def __init__(self, filters: list[RowFilter] | None = None):
        if filters is None:
            filters = []
        self.filters: list[RowFilter] = filters

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.filters == self.filters

    def __ne__(self, other):
        return not self == other

    def __len__(self) -> int:
        return len(self.filters)

    @overload
    def __getitem__(self, index: int) -> RowFilter:
        # overload signature for type checking
        pass

    @overload
    def __getitem__(self, index: slice) -> list[RowFilter]:
        # overload signature for type checking
        pass

    def __getitem__(self, index):
        return self.filters[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filters={self.filters})"

    def __str__(self) -> str:
        """
        Returns a string representation of the filter chain.

        Adds line breaks between each sub-filter for readability.
        """
        output = [f"{self.__class__.__name__}(["]
        for filter_ in self.filters:
            filter_lines = f"{filter_},".splitlines()
            output.extend([f"    {line}" for line in filter_lines])
        output.append("])")
        return "\n".join(output)


class RowFilterChain(_FilterCombination):
    """Chain of row filters.

    Sends rows through several filters in sequence. The filters are "chained"
    together to process a row. After the first filter is applied, the second
    is applied to the filtered output and so on for subsequent filters.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        Returns: The converted current object.
        """
        chain = data_v2_pb2.RowFilter.Chain(
            filters=[row_filter._to_pb() for row_filter in self.filters]
        )
        return data_v2_pb2.RowFilter(chain=chain)

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"chain": {"filters": [f._to_dict() for f in self.filters]}}


class RowFilterUnion(_FilterCombination):
    """Union of row filters.

    Sends rows through several filters simultaneously, then
    merges / interleaves all the filtered results together.

    If multiple cells are produced with the same column and timestamp,
    they will all appear in the output row in an unspecified mutual order.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        Returns: The converted current object.
        """
        interleave = data_v2_pb2.RowFilter.Interleave(
            filters=[row_filter._to_pb() for row_filter in self.filters]
        )
        return data_v2_pb2.RowFilter(interleave=interleave)

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"interleave": {"filters": [f._to_dict() for f in self.filters]}}


class ConditionalRowFilter(RowFilter):
    """Conditional row filter which exhibits ternary behavior.

    Executes one of two filters based on another filter. If the ``predicate_filter``
    returns any cells in the row, then ``true_filter`` is executed. If not,
    then ``false_filter`` is executed.

    .. note::

        The ``predicate_filter`` does not execute atomically with the true and false
        filters, which may lead to inconsistent or unexpected results.

        Additionally, executing a :class:`ConditionalRowFilter` has poor
        performance on the server, especially when ``false_filter`` is set.

    :type predicate_filter: :class:`RowFilter`
    :param predicate_filter: The filter to condition on before executing the
                        true/false filters.

    :type true_filter: :class:`RowFilter`
    :param true_filter: (Optional) The filter to execute if there are any cells
                        matching ``predicate_filter``. If not provided, no results
                        will be returned in the true case.

    :type false_filter: :class:`RowFilter`
    :param false_filter: (Optional) The filter to execute if there are no cells
                         matching ``predicate_filter``. If not provided, no results
                         will be returned in the false case.
    """

    def __init__(
        self,
        predicate_filter: RowFilter,
        true_filter: RowFilter | None = None,
        false_filter: RowFilter | None = None,
    ):
        self.predicate_filter = predicate_filter
        self.true_filter = true_filter
        self.false_filter = false_filter

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.predicate_filter == self.predicate_filter
            and other.true_filter == self.true_filter
            and other.false_filter == self.false_filter
        )

    def __ne__(self, other):
        return not self == other

    def _to_pb(self) -> data_v2_pb2.RowFilter:
        """Converts the row filter to a protobuf.

        Returns: The converted current object.
        """
        condition_kwargs = {"predicate_filter": self.predicate_filter._to_pb()}
        if self.true_filter is not None:
            condition_kwargs["true_filter"] = self.true_filter._to_pb()
        if self.false_filter is not None:
            condition_kwargs["false_filter"] = self.false_filter._to_pb()
        condition = data_v2_pb2.RowFilter.Condition(**condition_kwargs)
        return data_v2_pb2.RowFilter(condition=condition)

    def _condition_to_dict(self) -> dict[str, Any]:
        """Converts the condition to a dict representation."""
        condition_kwargs = {"predicate_filter": self.predicate_filter._to_dict()}
        if self.true_filter is not None:
            condition_kwargs["true_filter"] = self.true_filter._to_dict()
        if self.false_filter is not None:
            condition_kwargs["false_filter"] = self.false_filter._to_dict()
        return condition_kwargs

    def _to_dict(self) -> dict[str, Any]:
        """Converts the row filter to a dict representation."""
        return {"condition": self._condition_to_dict()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(predicate_filter={self.predicate_filter!r}, true_filter={self.true_filter!r}, false_filter={self.false_filter!r})"

    def __str__(self) -> str:
        output = [f"{self.__class__.__name__}("]
        for filter_type in ("predicate_filter", "true_filter", "false_filter"):
            filter_ = getattr(self, filter_type)
            if filter_ is None:
                continue
            # add the new filter set, adding indentations for readability
            filter_lines = f"{filter_type}={filter_},".splitlines()
            output.extend(f"    {line}" for line in filter_lines)
        output.append(")")
        return "\n".join(output)

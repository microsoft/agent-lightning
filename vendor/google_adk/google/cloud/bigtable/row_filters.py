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

import struct


from google.cloud._helpers import _microseconds_from_datetime  # type: ignore
from google.cloud._helpers import _to_bytes  # type: ignore
from google.cloud.bigtable_v2.types import data as data_v2_pb2

_PACK_I64 = struct.Struct(">q").pack


class RowFilter(object):
    """Basic filter to apply to cells in a row.

    These values can be combined via :class:`RowFilterChain`,
    :class:`RowFilterUnion` and :class:`ConditionalRowFilter`.

    .. note::

        This class is a do-nothing base class for all row filters.
    """


class _BoolFilter(RowFilter):
    """Row filter that uses a boolean flag.

    :type flag: bool
    :param flag: An indicator if a setting is turned on or off.
    """

    def __init__(self, flag):
        self.flag = flag

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.flag == self.flag

    def __ne__(self, other):
        return not self == other


class SinkFilter(_BoolFilter):
    """Advanced row filter to skip parent filters.

    :type flag: bool
    :param flag: ADVANCED USE ONLY. Hook for introspection into the row filter.
                 Outputs all cells directly to the output of the read rather
                 than to any parent filter. Cannot be used within the
                 ``predicate_filter``, ``true_filter``, or ``false_filter``
                 of a :class:`ConditionalRowFilter`.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(sink=self.flag)


class PassAllFilter(_BoolFilter):
    """Row filter equivalent to not filtering at all.

    :type flag: bool
    :param flag: Matches all cells, regardless of input. Functionally
                 equivalent to leaving ``filter`` unset, but included for
                 completeness.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(pass_all_filter=self.flag)


class BlockAllFilter(_BoolFilter):
    """Row filter that doesn't match any cells.

    :type flag: bool
    :param flag: Does not match any cells, regardless of input. Useful for
                 temporarily disabling just part of a filter.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(block_all_filter=self.flag)


class _RegexFilter(RowFilter):
    """Row filter that uses a regular expression.

    The ``regex`` must be valid RE2 patterns. See Google's
    `RE2 reference`_ for the accepted syntax.

    .. _RE2 reference: https://github.com/google/re2/wiki/Syntax

    :type regex: bytes or str
    :param regex:
        A regular expression (RE2) for some row filter.  String values
        will be encoded as ASCII.
    """

    def __init__(self, regex):
        self.regex = _to_bytes(regex)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.regex == self.regex

    def __ne__(self, other):
        return not self == other


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

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(row_key_regex_filter=self.regex)


class RowSampleFilter(RowFilter):
    """Matches all cells from a row with probability p.

    :type sample: float
    :param sample: The probability of matching a cell (must be in the
                   interval ``(0, 1)``  The end points are excluded).
    """

    def __init__(self, sample):
        self.sample = sample

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.sample == self.sample

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(row_sample_filter=self.sample)


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

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(family_name_regex_filter=self.regex)


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

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(column_qualifier_regex_filter=self.regex)


class TimestampRange(object):
    """Range of time with inclusive lower and exclusive upper bounds.

    :type start: :class:`datetime.datetime`
    :param start: (Optional) The (inclusive) lower bound of the timestamp
                  range. If omitted, defaults to Unix epoch.

    :type end: :class:`datetime.datetime`
    :param end: (Optional) The (exclusive) upper bound of the timestamp
                range. If omitted, no upper bound is used.
    """

    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.start == self.start and other.end == self.end

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the :class:`TimestampRange` to a protobuf.

        :rtype: :class:`.data_v2_pb2.TimestampRange`
        :returns: The converted current object.
        """
        timestamp_range_kwargs = {}
        if self.start is not None:
            timestamp_range_kwargs["start_timestamp_micros"] = (
                _microseconds_from_datetime(self.start) // 1000 * 1000
            )
        if self.end is not None:
            end_time = _microseconds_from_datetime(self.end)
            if end_time % 1000 != 0:
                end_time = end_time // 1000 * 1000 + 1000
            timestamp_range_kwargs["end_timestamp_micros"] = end_time
        return data_v2_pb2.TimestampRange(**timestamp_range_kwargs)


class TimestampRangeFilter(RowFilter):
    """Row filter that limits cells to a range of time.

    :type range_: :class:`TimestampRange`
    :param range_: Range of time that cells should match against.
    """

    def __init__(self, range_):
        self.range_ = range_

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.range_ == self.range_

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the row filter to a protobuf.

        First converts the ``range_`` on the current object to a protobuf and
        then uses it in the ``timestamp_range_filter`` field.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(timestamp_range_filter=self.range_.to_pb())


class ColumnRangeFilter(RowFilter):
    """A row filter to restrict to a range of columns.

    Both the start and end column can be included or excluded in the range.
    By default, we include them both, but this can be changed with optional
    flags.

    :type column_family_id: str
    :param column_family_id: The column family that contains the columns. Must
                             be of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

    :type start_column: bytes
    :param start_column: The start of the range of columns. If no value is
                         used, the backend applies no upper bound to the
                         values.

    :type end_column: bytes
    :param end_column: The end of the range of columns. If no value is used,
                       the backend applies no upper bound to the values.

    :type inclusive_start: bool
    :param inclusive_start: Boolean indicating if the start column should be
                            included in the range (or excluded). Defaults
                            to :data:`True` if ``start_column`` is passed and
                            no ``inclusive_start`` was given.

    :type inclusive_end: bool
    :param inclusive_end: Boolean indicating if the end column should be
                          included in the range (or excluded). Defaults
                          to :data:`True` if ``end_column`` is passed and
                          no ``inclusive_end`` was given.

    :raises: :class:`ValueError <exceptions.ValueError>` if ``inclusive_start``
             is set but no ``start_column`` is given or if ``inclusive_end``
             is set but no ``end_column`` is given
    """

    def __init__(
        self,
        column_family_id,
        start_column=None,
        end_column=None,
        inclusive_start=None,
        inclusive_end=None,
    ):
        self.column_family_id = column_family_id

        if inclusive_start is None:
            inclusive_start = True
        elif start_column is None:
            raise ValueError(
                "Inclusive start was specified but no " "start column was given."
            )
        self.start_column = start_column
        self.inclusive_start = inclusive_start

        if inclusive_end is None:
            inclusive_end = True
        elif end_column is None:
            raise ValueError(
                "Inclusive end was specified but no " "end column was given."
            )
        self.end_column = end_column
        self.inclusive_end = inclusive_end

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.column_family_id == self.column_family_id
            and other.start_column == self.start_column
            and other.end_column == self.end_column
            and other.inclusive_start == self.inclusive_start
            and other.inclusive_end == self.inclusive_end
        )

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the row filter to a protobuf.

        First converts to a :class:`.data_v2_pb2.ColumnRange` and then uses it
        in the ``column_range_filter`` field.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        column_range_kwargs = {"family_name": self.column_family_id}
        if self.start_column is not None:
            if self.inclusive_start:
                key = "start_qualifier_closed"
            else:
                key = "start_qualifier_open"
            column_range_kwargs[key] = _to_bytes(self.start_column)
        if self.end_column is not None:
            if self.inclusive_end:
                key = "end_qualifier_closed"
            else:
                key = "end_qualifier_open"
            column_range_kwargs[key] = _to_bytes(self.end_column)

        column_range = data_v2_pb2.ColumnRange(**column_range_kwargs)
        return data_v2_pb2.RowFilter(column_range_filter=column_range)


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

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(value_regex_filter=self.regex)


class ExactValueFilter(ValueRegexFilter):
    """Row filter for an exact value.


    :type value: bytes or str or int
    :param value:
        a literal string encodable as ASCII, or the
        equivalent bytes, or an integer (which will be packed into 8-bytes).
    """

    def __init__(self, value):
        if isinstance(value, int):
            value = _PACK_I64(value)
        super(ExactValueFilter, self).__init__(value)


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
        self, start_value=None, end_value=None, inclusive_start=None, inclusive_end=None
    ):
        if inclusive_start is None:
            inclusive_start = True
        elif start_value is None:
            raise ValueError(
                "Inclusive start was specified but no " "start value was given."
            )
        if isinstance(start_value, int):
            start_value = _PACK_I64(start_value)
        self.start_value = start_value
        self.inclusive_start = inclusive_start

        if inclusive_end is None:
            inclusive_end = True
        elif end_value is None:
            raise ValueError(
                "Inclusive end was specified but no " "end value was given."
            )
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

    def to_pb(self):
        """Converts the row filter to a protobuf.

        First converts to a :class:`.data_v2_pb2.ValueRange` and then uses
        it to create a row filter protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
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

        value_range = data_v2_pb2.ValueRange(**value_range_kwargs)
        return data_v2_pb2.RowFilter(value_range_filter=value_range)


class _CellCountFilter(RowFilter):
    """Row filter that uses an integer count of cells.

    The cell count is used as an offset or a limit for the number
    of results returned.

    :type num_cells: int
    :param num_cells: An integer count / offset / limit.
    """

    def __init__(self, num_cells):
        self.num_cells = num_cells

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.num_cells == self.num_cells

    def __ne__(self, other):
        return not self == other


class CellsRowOffsetFilter(_CellCountFilter):
    """Row filter to skip cells in a row.

    :type num_cells: int
    :param num_cells: Skips the first N cells of the row.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(cells_per_row_offset_filter=self.num_cells)


class CellsRowLimitFilter(_CellCountFilter):
    """Row filter to limit cells in a row.

    :type num_cells: int
    :param num_cells: Matches only the first N cells of the row.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(cells_per_row_limit_filter=self.num_cells)


class CellsColumnLimitFilter(_CellCountFilter):
    """Row filter to limit cells in a column.

    :type num_cells: int
    :param num_cells: Matches only the most recent N cells within each column.
                      This filters a (family name, column) pair, based on
                      timestamps of each cell.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(cells_per_column_limit_filter=self.num_cells)


class StripValueTransformerFilter(_BoolFilter):
    """Row filter that transforms cells into empty string (0 bytes).

    :type flag: bool
    :param flag: If :data:`True`, replaces each cell's value with the empty
                 string. As the name indicates, this is more useful as a
                 transformer than a generic query / filter.
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(strip_value_transformer=self.flag)


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

    def __init__(self, label):
        self.label = label

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.label == self.label

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        return data_v2_pb2.RowFilter(apply_label_transformer=self.label)


class _FilterCombination(RowFilter):
    """Chain of row filters.

    Sends rows through several filters in sequence. The filters are "chained"
    together to process a row. After the first filter is applied, the second
    is applied to the filtered output and so on for subsequent filters.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def __init__(self, filters=None):
        if filters is None:
            filters = []
        self.filters = filters

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.filters == self.filters

    def __ne__(self, other):
        return not self == other


class RowFilterChain(_FilterCombination):
    """Chain of row filters.

    Sends rows through several filters in sequence. The filters are "chained"
    together to process a row. After the first filter is applied, the second
    is applied to the filtered output and so on for subsequent filters.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        chain = data_v2_pb2.RowFilter.Chain(
            filters=[row_filter.to_pb() for row_filter in self.filters]
        )
        return data_v2_pb2.RowFilter(chain=chain)


class RowFilterUnion(_FilterCombination):
    """Union of row filters.

    Sends rows through several filters simultaneously, then
    merges / interleaves all the filtered results together.

    If multiple cells are produced with the same column and timestamp,
    they will all appear in the output row in an unspecified mutual order.

    :type filters: list
    :param filters: List of :class:`RowFilter`
    """

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        interleave = data_v2_pb2.RowFilter.Interleave(
            filters=[row_filter.to_pb() for row_filter in self.filters]
        )
        return data_v2_pb2.RowFilter(interleave=interleave)


class ConditionalRowFilter(RowFilter):
    """Conditional row filter which exhibits ternary behavior.

    Executes one of two filters based on another filter. If the ``base_filter``
    returns any cells in the row, then ``true_filter`` is executed. If not,
    then ``false_filter`` is executed.

    .. note::

        The ``base_filter`` does not execute atomically with the true and false
        filters, which may lead to inconsistent or unexpected results.

        Additionally, executing a :class:`ConditionalRowFilter` has poor
        performance on the server, especially when ``false_filter`` is set.

    :type base_filter: :class:`RowFilter`
    :param base_filter: The filter to condition on before executing the
                        true/false filters.

    :type true_filter: :class:`RowFilter`
    :param true_filter: (Optional) The filter to execute if there are any cells
                        matching ``base_filter``. If not provided, no results
                        will be returned in the true case.

    :type false_filter: :class:`RowFilter`
    :param false_filter: (Optional) The filter to execute if there are no cells
                         matching ``base_filter``. If not provided, no results
                         will be returned in the false case.
    """

    def __init__(self, base_filter, true_filter=None, false_filter=None):
        self.base_filter = base_filter
        self.true_filter = true_filter
        self.false_filter = false_filter

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.base_filter == self.base_filter
            and other.true_filter == self.true_filter
            and other.false_filter == self.false_filter
        )

    def __ne__(self, other):
        return not self == other

    def to_pb(self):
        """Converts the row filter to a protobuf.

        :rtype: :class:`.data_v2_pb2.RowFilter`
        :returns: The converted current object.
        """
        condition_kwargs = {"predicate_filter": self.base_filter.to_pb()}
        if self.true_filter is not None:
            condition_kwargs["true_filter"] = self.true_filter.to_pb()
        if self.false_filter is not None:
            condition_kwargs["false_filter"] = self.false_filter.to_pb()
        condition = data_v2_pb2.RowFilter.Condition(**condition_kwargs)
        return data_v2_pb2.RowFilter(condition=condition)

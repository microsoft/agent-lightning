# Copyright 2018 Google LLC
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

"""User-friendly container for Google Cloud Bigtable RowSet """


from google.cloud._helpers import _to_bytes  # type: ignore


class RowSet(object):
    """Convenience wrapper of google.bigtable.v2.RowSet

    Useful for creating a set of row keys and row ranges, which can
    be passed to read_rows method of class:`.Table.read_rows`.
    """

    def __init__(self):
        self.row_keys = []
        self.row_ranges = []

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if len(other.row_keys) != len(self.row_keys):
            return False

        if len(other.row_ranges) != len(self.row_ranges):
            return False

        if not set(other.row_keys) == set(self.row_keys):
            return False

        if not set(other.row_ranges) == set(self.row_ranges):
            return False

        return True

    def __ne__(self, other):
        return not self == other

    def add_row_key(self, row_key):
        """Add row key to row_keys list.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_add_row_key]
            :end-before: [END bigtable_api_add_row_key]
            :dedent: 4

        :type row_key: bytes
        :param row_key: The key of a row to read
        """
        self.row_keys.append(row_key)

    def add_row_range(self, row_range):
        """Add row_range to row_ranges list.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_add_row_range]
            :end-before: [END bigtable_api_add_row_range]
            :dedent: 4

        :type row_range: class:`RowRange`
        :param row_range: The row range object having start and end key
        """
        self.row_ranges.append(row_range)

    def add_row_range_from_keys(
        self, start_key=None, end_key=None, start_inclusive=True, end_inclusive=False
    ):
        """Add row range to row_ranges list from the row keys

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_range_from_keys]
            :end-before: [END bigtable_api_row_range_from_keys]
            :dedent: 4

        :type start_key: bytes
        :param start_key: (Optional) Start key of the row range. If left empty,
                          will be interpreted as the empty string.

        :type end_key: bytes
        :param end_key: (Optional) End key of the row range. If left empty,
                        will be interpreted as the empty string and range will
                        be unbounded on the high end.

        :type start_inclusive: bool
        :param start_inclusive: (Optional) Whether the ``start_key`` should be
                        considered inclusive. The default is True (inclusive).

        :type end_inclusive: bool
        :param end_inclusive: (Optional) Whether the ``end_key`` should be
                  considered inclusive. The default is False (exclusive).
        """
        row_range = RowRange(start_key, end_key, start_inclusive, end_inclusive)
        self.row_ranges.append(row_range)

    def add_row_range_with_prefix(self, row_key_prefix):
        """Add row range to row_ranges list that start with the row_key_prefix from the row keys

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_add_row_range_with_prefix]
            :end-before: [END bigtable_api_add_row_range_with_prefix]

        :type row_key_prefix: str
        :param row_key_prefix: To retrieve  all rows that start with this row key prefix.
                            Prefix cannot be zero length."""

        end_key = row_key_prefix[:-1] + chr(ord(row_key_prefix[-1]) + 1)
        self.add_row_range_from_keys(
            row_key_prefix.encode("utf-8"), end_key.encode("utf-8")
        )

    def _update_message_request(self, message):
        """Add row keys and row range to given request message

        :type message: class:`data_messages_v2_pb2.ReadRowsRequest`
        :param message: The ``ReadRowsRequest`` protobuf
        """
        for each in self.row_keys:
            message.rows.row_keys._pb.append(_to_bytes(each))

        for each in self.row_ranges:
            r_kwrags = each.get_range_kwargs()
            message.rows.row_ranges.append(r_kwrags)


class RowRange(object):
    """Convenience wrapper of google.bigtable.v2.RowRange

    :type start_key: bytes
    :param start_key: (Optional) Start key of the row range. If left empty,
                      will be interpreted as the empty string.

    :type end_key: bytes
    :param end_key: (Optional) End key of the row range. If left empty,
                    will be interpreted as the empty string and range will
                    be unbounded on the high end.

    :type start_inclusive: bool
    :param start_inclusive: (Optional) Whether the ``start_key`` should be
                  considered inclusive. The default is True (inclusive).

    :type end_inclusive: bool
    :param end_inclusive: (Optional) Whether the ``end_key`` should be
                  considered inclusive. The default is False (exclusive).
    """

    def __init__(
        self, start_key=None, end_key=None, start_inclusive=True, end_inclusive=False
    ):
        self.start_key = start_key
        self.start_inclusive = start_inclusive
        self.end_key = end_key
        self.end_inclusive = end_inclusive

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple[str]: The contents of this :class:`.RowRange`.
        """
        return (self.start_key, self.start_inclusive, self.end_key, self.end_inclusive)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def get_range_kwargs(self):
        """Convert row range object to dict which can be passed to
        google.bigtable.v2.RowRange add method.
        """
        range_kwargs = {}
        if self.start_key is not None:
            start_key_key = "start_key_open"
            if self.start_inclusive:
                start_key_key = "start_key_closed"
            range_kwargs[start_key_key] = _to_bytes(self.start_key)

        if self.end_key is not None:
            end_key_key = "end_key_open"
            if self.end_inclusive:
                end_key_key = "end_key_closed"
            range_kwargs[end_key_key] = _to_bytes(self.end_key)
        return range_kwargs

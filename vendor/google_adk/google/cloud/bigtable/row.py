# Copyright 2015 Google LLC
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

"""User-friendly container for Google Cloud Bigtable Row."""


import struct

from google.cloud._helpers import _datetime_from_microseconds  # type: ignore
from google.cloud._helpers import _microseconds_from_datetime  # type: ignore
from google.cloud._helpers import _to_bytes  # type: ignore
from google.cloud.bigtable_v2.types import data as data_v2_pb2


_PACK_I64 = struct.Struct(">q").pack

MAX_MUTATIONS = 100000
"""The maximum number of mutations that a row can accumulate."""

_MISSING_COLUMN_FAMILY = "Column family {} is not among the cells stored in this row."
_MISSING_COLUMN = (
    "Column {} is not among the cells stored in this row in the column family {}."
)
_MISSING_INDEX = (
    "Index {!r} is not valid for the cells stored in this row for column {} "
    "in the column family {}. There are {} such cells."
)


class Row(object):
    """Base representation of a Google Cloud Bigtable Row.

    This class has three subclasses corresponding to the three
    RPC methods for sending row mutations:

    * :class:`DirectRow` for ``MutateRow``
    * :class:`ConditionalRow` for ``CheckAndMutateRow``
    * :class:`AppendRow` for ``ReadModifyWriteRow``

    :type row_key: bytes
    :param row_key: The key for the current row.

    :type table: :class:`Table <google.cloud.bigtable.table.Table>`
    :param table: (Optional) The table that owns the row.
    """

    def __init__(self, row_key, table=None):
        self._row_key = _to_bytes(row_key)
        self._table = table

    @property
    def row_key(self):
        """Row key.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_row_key]
            :end-before: [END bigtable_api_row_row_key]
            :dedent: 4

        :rtype: bytes
        :returns: The key for the current row.
        """
        return self._row_key

    @property
    def table(self):
        """Row table.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_table]
            :end-before: [END bigtable_api_row_table]
            :dedent: 4

        :rtype: table: :class:`Table <google.cloud.bigtable.table.Table>`
        :returns: table: The table that owns the row.
        """
        return self._table


class _SetDeleteRow(Row):
    """Row helper for setting or deleting cell values.

    Implements helper methods to add mutations to set or delete cell contents:

    * :meth:`set_cell`
    * :meth:`delete`
    * :meth:`delete_cell`
    * :meth:`delete_cells`

    :type row_key: bytes
    :param row_key: The key for the current row.

    :type table: :class:`Table <google.cloud.bigtable.table.Table>`
    :param table: The table that owns the row.
    """

    ALL_COLUMNS = object()
    """Sentinel value used to indicate all columns in a column family."""

    def _get_mutations(self, state=None):
        """Gets the list of mutations for a given state.

        This method intended to be implemented by subclasses.

        ``state`` may not need to be used by all subclasses.

        :type state: bool
        :param state: The state that the mutation should be
                      applied in.

        :raises: :class:`NotImplementedError <exceptions.NotImplementedError>`
                 always.
        """
        raise NotImplementedError

    def _set_cell(self, column_family_id, column, value, timestamp=None, state=None):
        """Helper for :meth:`set_cell`

        Adds a mutation to set the value in a specific cell.

        ``state`` is unused by :class:`DirectRow` but is used by
        subclasses.

        :type column_family_id: str
        :param column_family_id: The column family that contains the column.
                                 Must be of the form
                                 ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family where the cell
                       is located.

        :type value: bytes or :class:`int`
        :param value: The value to set in the cell. If an integer is used,
                      will be interpreted as a 64-bit big-endian signed
                      integer (8 bytes).

        :type timestamp: :class:`datetime.datetime`
        :param timestamp: (Optional) The timestamp of the operation.

        :type state: bool
        :param state: (Optional) The state that is passed along to
                      :meth:`_get_mutations`.
        """
        column = _to_bytes(column)
        if isinstance(value, int):
            value = _PACK_I64(value)
        value = _to_bytes(value)
        if timestamp is None:
            # Use -1 for current Bigtable server time.
            timestamp_micros = -1
        else:
            timestamp_micros = _microseconds_from_datetime(timestamp)
            # Truncate to millisecond granularity.
            timestamp_micros -= timestamp_micros % 1000

        mutation_val = data_v2_pb2.Mutation.SetCell(
            family_name=column_family_id,
            column_qualifier=column,
            timestamp_micros=timestamp_micros,
            value=value,
        )
        mutation_pb = data_v2_pb2.Mutation(set_cell=mutation_val)
        self._get_mutations(state).append(mutation_pb)

    def _delete(self, state=None):
        """Helper for :meth:`delete`

        Adds a delete mutation (for the entire row) to the accumulated
        mutations.

        ``state`` is unused by :class:`DirectRow` but is used by
        subclasses.

        :type state: bool
        :param state: (Optional) The state that is passed along to
                      :meth:`_get_mutations`.
        """
        mutation_val = data_v2_pb2.Mutation.DeleteFromRow()
        mutation_pb = data_v2_pb2.Mutation(delete_from_row=mutation_val)
        self._get_mutations(state).append(mutation_pb)

    def _delete_cells(self, column_family_id, columns, time_range=None, state=None):
        """Helper for :meth:`delete_cell` and :meth:`delete_cells`.

        ``state`` is unused by :class:`DirectRow` but is used by
        subclasses.

        :type column_family_id: str
        :param column_family_id: The column family that contains the column
                                 or columns with cells being deleted. Must be
                                 of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type columns: :class:`list` of :class:`str` /
                       :func:`unicode <unicode>`, or :class:`object`
        :param columns: The columns within the column family that will have
                        cells deleted. If :attr:`ALL_COLUMNS` is used then
                        the entire column family will be deleted from the row.

        :type time_range: :class:`TimestampRange`
        :param time_range: (Optional) The range of time within which cells
                           should be deleted.

        :type state: bool
        :param state: (Optional) The state that is passed along to
                      :meth:`_get_mutations`.
        """
        mutations_list = self._get_mutations(state)
        if columns is self.ALL_COLUMNS:
            mutation_val = data_v2_pb2.Mutation.DeleteFromFamily(
                family_name=column_family_id
            )
            mutation_pb = data_v2_pb2.Mutation(delete_from_family=mutation_val)
            mutations_list.append(mutation_pb)
        else:
            delete_kwargs = {}
            if time_range is not None:
                delete_kwargs["time_range"] = time_range.to_pb()

            to_append = []
            for column in columns:
                column = _to_bytes(column)
                # time_range will never change if present, but the rest of
                # delete_kwargs will
                delete_kwargs.update(
                    family_name=column_family_id, column_qualifier=column
                )
                mutation_val = data_v2_pb2.Mutation.DeleteFromColumn(**delete_kwargs)
                mutation_pb = data_v2_pb2.Mutation(delete_from_column=mutation_val)
                to_append.append(mutation_pb)

            # We don't add the mutations until all columns have been
            # processed without error.
            mutations_list.extend(to_append)


class DirectRow(_SetDeleteRow):
    """Google Cloud Bigtable Row for sending "direct" mutations.

    These mutations directly set or delete cell contents:

    * :meth:`set_cell`
    * :meth:`delete`
    * :meth:`delete_cell`
    * :meth:`delete_cells`

    These methods can be used directly::

       >>> row = table.row(b'row-key1')
       >>> row.set_cell(u'fam', b'col1', b'cell-val')
       >>> row.delete_cell(u'fam', b'col2')

    .. note::

        A :class:`DirectRow` accumulates mutations locally via the
        :meth:`set_cell`, :meth:`delete`, :meth:`delete_cell` and
        :meth:`delete_cells` methods. To actually send these mutations to the
        Google Cloud Bigtable API, you must call :meth:`commit`.

    :type row_key: bytes
    :param row_key: The key for the current row.

    :type table: :class:`Table <google.cloud.bigtable.table.Table>`
    :param table: (Optional) The table that owns the row. This is
                  used for the :meth: `commit` only.  Alternatively,
                  DirectRows can be persisted via
                  :meth:`~google.cloud.bigtable.table.Table.mutate_rows`.
    """

    def __init__(self, row_key, table=None):
        super(DirectRow, self).__init__(row_key, table)
        self._pb_mutations = []

    def _get_mutations(self, state=None):  # pylint: disable=unused-argument
        """Gets the list of mutations for a given state.

        ``state`` is unused by :class:`DirectRow` but is used by
        subclasses.

        :type state: bool
        :param state: The state that the mutation should be
                      applied in.

        :rtype: list
        :returns: The list to add new mutations to (for the current state).
        """
        return self._pb_mutations

    def get_mutations_size(self):
        """Gets the total mutations size for current row

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_get_mutations_size]
            :end-before: [END bigtable_api_row_get_mutations_size]
            :dedent: 4
        """

        mutation_size = 0
        for mutation in self._get_mutations():
            mutation_size += mutation._pb.ByteSize()

        return mutation_size

    def set_cell(self, column_family_id, column, value, timestamp=None):
        """Sets a value in this row.

        The cell is determined by the ``row_key`` of this :class:`DirectRow`
        and the ``column``. The ``column`` must be in an existing
        :class:`.ColumnFamily` (as determined by ``column_family_id``).

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_set_cell]
            :end-before: [END bigtable_api_row_set_cell]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column.
                                 Must be of the form
                                 ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family where the cell
                       is located.

        :type value: bytes or :class:`int`
        :param value: The value to set in the cell. If an integer is used,
                      will be interpreted as a 64-bit big-endian signed
                      integer (8 bytes).

        :type timestamp: :class:`datetime.datetime`
        :param timestamp: (Optional) The timestamp of the operation.
        """
        self._set_cell(column_family_id, column, value, timestamp=timestamp, state=None)

    def delete(self):
        """Deletes this row from the table.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete]
            :end-before: [END bigtable_api_row_delete]
            :dedent: 4
        """
        self._delete(state=None)

    def delete_cell(self, column_family_id, column, time_range=None):
        """Deletes cell in this row.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete_cell]
            :end-before: [END bigtable_api_row_delete_cell]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column
                                 or columns with cells being deleted. Must be
                                 of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family that will have a
                       cell deleted.

        :type time_range: :class:`TimestampRange`
        :param time_range: (Optional) The range of time within which cells
                           should be deleted.
        """
        self._delete_cells(
            column_family_id, [column], time_range=time_range, state=None
        )

    def delete_cells(self, column_family_id, columns, time_range=None):
        """Deletes cells in this row.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete_cells]
            :end-before: [END bigtable_api_row_delete_cells]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column
                                 or columns with cells being deleted. Must be
                                 of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type columns: :class:`list` of :class:`str` /
                       :func:`unicode <unicode>`, or :class:`object`
        :param columns: The columns within the column family that will have
                        cells deleted. If :attr:`ALL_COLUMNS` is used then
                        the entire column family will be deleted from the row.

        :type time_range: :class:`TimestampRange`
        :param time_range: (Optional) The range of time within which cells
                           should be deleted.
        """
        self._delete_cells(column_family_id, columns, time_range=time_range, state=None)

    def commit(self):
        """Makes a ``MutateRow`` API request.

        If no mutations have been created in the row, no request is made.

        Mutations are applied atomically and in order, meaning that earlier
        mutations can be masked / negated by later ones. Cells already present
        in the row are left unchanged unless explicitly changed by a mutation.

        After committing the accumulated mutations, resets the local
        mutations to an empty list.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_commit]
            :end-before: [END bigtable_api_row_commit]
            :dedent: 4

        :rtype: :class:`~google.rpc.status_pb2.Status`
        :returns: A response status (`google.rpc.status_pb2.Status`)
                  representing success or failure of the row committed.
        :raises: :exc:`~.table.TooManyMutationsError` if the number of
                 mutations is greater than 100,000.
        """
        response = self._table.mutate_rows([self])

        self.clear()

        return response[0]

    def clear(self):
        """Removes all currently accumulated mutations on the current row.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_clear]
            :end-before: [END bigtable_api_row_clear]
            :dedent: 4
        """
        del self._pb_mutations[:]


class ConditionalRow(_SetDeleteRow):
    """Google Cloud Bigtable Row for sending mutations conditionally.

    Each mutation has an associated state: :data:`True` or :data:`False`.
    When :meth:`commit`-ed, the mutations for the :data:`True`
    state will be applied if the filter matches any cells in
    the row, otherwise the :data:`False` state will be applied.

    A :class:`ConditionalRow` accumulates mutations in the same way a
    :class:`DirectRow` does:

    * :meth:`set_cell`
    * :meth:`delete`
    * :meth:`delete_cell`
    * :meth:`delete_cells`

    with the only change the extra ``state`` parameter::

       >>> row_cond = table.row(b'row-key2', filter_=row_filter)
       >>> row_cond.set_cell(u'fam', b'col', b'cell-val', state=True)
       >>> row_cond.delete_cell(u'fam', b'col', state=False)

    .. note::

        As with :class:`DirectRow`, to actually send these mutations to the
        Google Cloud Bigtable API, you must call :meth:`commit`.

    :type row_key: bytes
    :param row_key: The key for the current row.

    :type table: :class:`Table <google.cloud.bigtable.table.Table>`
    :param table: The table that owns the row.

    :type filter_: :class:`.RowFilter`
    :param filter_: Filter to be used for conditional mutations.
    """

    def __init__(self, row_key, table, filter_):
        super(ConditionalRow, self).__init__(row_key, table)
        self._filter = filter_
        self._true_pb_mutations = []
        self._false_pb_mutations = []

    def _get_mutations(self, state=None):
        """Gets the list of mutations for a given state.

        Over-ridden so that the state can be used in:

        * :meth:`set_cell`
        * :meth:`delete`
        * :meth:`delete_cell`
        * :meth:`delete_cells`

        :type state: bool
        :param state: The state that the mutation should be
                      applied in.

        :rtype: list
        :returns: The list to add new mutations to (for the current state).
        """
        if state:
            return self._true_pb_mutations
        else:
            return self._false_pb_mutations

    def commit(self):
        """Makes a ``CheckAndMutateRow`` API request.

        If no mutations have been created in the row, no request is made.

        The mutations will be applied conditionally, based on whether the
        filter matches any cells in the :class:`ConditionalRow` or not. (Each
        method which adds a mutation has a ``state`` parameter for this
        purpose.)

        Mutations are applied atomically and in order, meaning that earlier
        mutations can be masked / negated by later ones. Cells already present
        in the row are left unchanged unless explicitly changed by a mutation.

        After committing the accumulated mutations, resets the local
        mutations.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_commit]
            :end-before: [END bigtable_api_row_commit]
            :dedent: 4

        :rtype: bool
        :returns: Flag indicating if the filter was matched (which also
                  indicates which set of mutations were applied by the server).
        :raises: :class:`ValueError <exceptions.ValueError>` if the number of
                 mutations exceeds the :data:`MAX_MUTATIONS`.
        """
        true_mutations = self._get_mutations(state=True)
        false_mutations = self._get_mutations(state=False)
        num_true_mutations = len(true_mutations)
        num_false_mutations = len(false_mutations)
        if num_true_mutations == 0 and num_false_mutations == 0:
            return
        if num_true_mutations > MAX_MUTATIONS or num_false_mutations > MAX_MUTATIONS:
            raise ValueError(
                "Exceed the maximum allowable mutations (%d). Had %s true "
                "mutations and %d false mutations."
                % (MAX_MUTATIONS, num_true_mutations, num_false_mutations)
            )

        data_client = self._table._instance._client.table_data_client
        resp = data_client.check_and_mutate_row(
            table_name=self._table.name,
            row_key=self._row_key,
            predicate_filter=self._filter.to_pb(),
            app_profile_id=self._table._app_profile_id,
            true_mutations=true_mutations,
            false_mutations=false_mutations,
        )
        self.clear()
        return resp.predicate_matched

    # pylint: disable=arguments-differ
    def set_cell(self, column_family_id, column, value, timestamp=None, state=True):
        """Sets a value in this row.

        The cell is determined by the ``row_key`` of this
        :class:`ConditionalRow` and the ``column``. The ``column`` must be in
        an existing :class:`.ColumnFamily` (as determined by
        ``column_family_id``).

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_set_cell]
            :end-before: [END bigtable_api_row_set_cell]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column.
                                 Must be of the form
                                 ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family where the cell
                       is located.

        :type value: bytes or :class:`int`
        :param value: The value to set in the cell. If an integer is used,
                      will be interpreted as a 64-bit big-endian signed
                      integer (8 bytes).

        :type timestamp: :class:`datetime.datetime`
        :param timestamp: (Optional) The timestamp of the operation.

        :type state: bool
        :param state: (Optional) The state that the mutation should be
                      applied in. Defaults to :data:`True`.
        """
        self._set_cell(
            column_family_id, column, value, timestamp=timestamp, state=state
        )

    def delete(self, state=True):
        """Deletes this row from the table.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete]
            :end-before: [END bigtable_api_row_delete]
            :dedent: 4

        :type state: bool
        :param state: (Optional) The state that the mutation should be
                      applied in. Defaults to :data:`True`.
        """
        self._delete(state=state)

    def delete_cell(self, column_family_id, column, time_range=None, state=True):
        """Deletes cell in this row.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

         For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete_cell]
            :end-before: [END bigtable_api_row_delete_cell]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column
                                 or columns with cells being deleted. Must be
                                 of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family that will have a
                       cell deleted.

        :type time_range: :class:`TimestampRange`
        :param time_range: (Optional) The range of time within which cells
                           should be deleted.

        :type state: bool
        :param state: (Optional) The state that the mutation should be
                      applied in. Defaults to :data:`True`.
        """
        self._delete_cells(
            column_family_id, [column], time_range=time_range, state=state
        )

    def delete_cells(self, column_family_id, columns, time_range=None, state=True):
        """Deletes cells in this row.

        .. note::

            This method adds a mutation to the accumulated mutations on this
            row, but does not make an API request. To actually
            send an API request (with the mutations) to the Google Cloud
            Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_delete_cells]
            :end-before: [END bigtable_api_row_delete_cells]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column
                                 or columns with cells being deleted. Must be
                                 of the form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type columns: :class:`list` of :class:`str` /
                       :func:`unicode <unicode>`, or :class:`object`
        :param columns: The columns within the column family that will have
                        cells deleted. If :attr:`ALL_COLUMNS` is used then the
                        entire column family will be deleted from the row.

        :type time_range: :class:`TimestampRange`
        :param time_range: (Optional) The range of time within which cells
                           should be deleted.

        :type state: bool
        :param state: (Optional) The state that the mutation should be
                      applied in. Defaults to :data:`True`.
        """
        self._delete_cells(
            column_family_id, columns, time_range=time_range, state=state
        )

    # pylint: enable=arguments-differ

    def clear(self):
        """Removes all currently accumulated mutations on the current row.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_clear]
            :end-before: [END bigtable_api_row_clear]
            :dedent: 4
        """
        del self._true_pb_mutations[:]
        del self._false_pb_mutations[:]


class AppendRow(Row):
    """Google Cloud Bigtable Row for sending append mutations.

    These mutations are intended to augment the value of an existing cell
    and uses the methods:

    * :meth:`append_cell_value`
    * :meth:`increment_cell_value`

    The first works by appending bytes and the second by incrementing an
    integer (stored in the cell as 8 bytes). In either case, if the
    cell is empty, assumes the default empty value (empty string for
    bytes or 0 for integer).

    :type row_key: bytes
    :param row_key: The key for the current row.

    :type table: :class:`Table <google.cloud.bigtable.table.Table>`
    :param table: The table that owns the row.
    """

    def __init__(self, row_key, table):
        super(AppendRow, self).__init__(row_key, table)
        self._rule_pb_list = []

    def clear(self):
        """Removes all currently accumulated modifications on current row.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_clear]
            :end-before: [END bigtable_api_row_clear]
            :dedent: 4
        """
        del self._rule_pb_list[:]

    def append_cell_value(self, column_family_id, column, value):
        """Appends a value to an existing cell.

        .. note::

            This method adds a read-modify rule protobuf to the accumulated
            read-modify rules on this row, but does not make an API
            request. To actually send an API request (with the rules) to the
            Google Cloud Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_append_cell_value]
            :end-before: [END bigtable_api_row_append_cell_value]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column.
                                 Must be of the form
                                 ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family where the cell
                       is located.

        :type value: bytes
        :param value: The value to append to the existing value in the cell. If
                      the targeted cell is unset, it will be treated as
                      containing the empty string.
        """
        column = _to_bytes(column)
        value = _to_bytes(value)
        rule_pb = data_v2_pb2.ReadModifyWriteRule(
            family_name=column_family_id, column_qualifier=column, append_value=value
        )
        self._rule_pb_list.append(rule_pb)

    def increment_cell_value(self, column_family_id, column, int_value):
        """Increments a value in an existing cell.

        Assumes the value in the cell is stored as a 64 bit integer
        serialized to bytes.

        .. note::

            This method adds a read-modify rule protobuf to the accumulated
            read-modify rules on this row, but does not make an API
            request. To actually send an API request (with the rules) to the
            Google Cloud Bigtable API, call :meth:`commit`.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_increment_cell_value]
            :end-before: [END bigtable_api_row_increment_cell_value]
            :dedent: 4

        :type column_family_id: str
        :param column_family_id: The column family that contains the column.
                                 Must be of the form
                                 ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

        :type column: bytes
        :param column: The column within the column family where the cell
                       is located.

        :type int_value: int
        :param int_value: The value to increment the existing value in the cell
                          by. If the targeted cell is unset, it will be treated
                          as containing a zero. Otherwise, the targeted cell
                          must contain an 8-byte value (interpreted as a 64-bit
                          big-endian signed integer), or the entire request
                          will fail.
        """
        column = _to_bytes(column)
        rule_pb = data_v2_pb2.ReadModifyWriteRule(
            family_name=column_family_id,
            column_qualifier=column,
            increment_amount=int_value,
        )
        self._rule_pb_list.append(rule_pb)

    def commit(self):
        """Makes a ``ReadModifyWriteRow`` API request.

        This commits modifications made by :meth:`append_cell_value` and
        :meth:`increment_cell_value`. If no modifications were made, makes
        no API request and just returns ``{}``.

        Modifies a row atomically, reading the latest existing
        timestamp / value from the specified columns and writing a new value by
        appending / incrementing. The new cell created uses either the current
        server time or the highest timestamp of a cell in that column (if it
        exceeds the server time).

        After committing the accumulated mutations, resets the local mutations.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_commit]
            :end-before: [END bigtable_api_row_commit]
            :dedent: 4

        :rtype: dict
        :returns: The new contents of all modified cells. Returned as a
                  dictionary of column families, each of which holds a
                  dictionary of columns. Each column contains a list of cells
                  modified. Each cell is represented with a two-tuple with the
                  value (in bytes) and the timestamp for the cell.
        :raises: :class:`ValueError <exceptions.ValueError>` if the number of
                 mutations exceeds the :data:`MAX_MUTATIONS`.
        """
        num_mutations = len(self._rule_pb_list)
        if num_mutations == 0:
            return {}
        if num_mutations > MAX_MUTATIONS:
            raise ValueError(
                "%d total append mutations exceed the maximum "
                "allowable %d." % (num_mutations, MAX_MUTATIONS)
            )

        data_client = self._table._instance._client.table_data_client
        row_response = data_client.read_modify_write_row(
            table_name=self._table.name,
            row_key=self._row_key,
            rules=self._rule_pb_list,
            app_profile_id=self._table._app_profile_id,
        )

        # Reset modifications after commit-ing request.
        self.clear()

        # NOTE: We expect row_response.key == self._row_key but don't check.
        return _parse_rmw_row_response(row_response)


def _parse_rmw_row_response(row_response):
    """Parses the response to a ``ReadModifyWriteRow`` request.

    :type row_response: :class:`.data_v2_pb2.Row`
    :param row_response: The response row (with only modified cells) from a
                         ``ReadModifyWriteRow`` request.

    :rtype: dict
    :returns: The new contents of all modified cells. Returned as a
              dictionary of column families, each of which holds a
              dictionary of columns. Each column contains a list of cells
              modified. Each cell is represented with a two-tuple with the
              value (in bytes) and the timestamp for the cell. For example:

              .. code:: python

                  {
                      u'col-fam-id': {
                          b'col-name1': [
                              (b'cell-val', datetime.datetime(...)),
                              (b'cell-val-newer', datetime.datetime(...)),
                          ],
                          b'col-name2': [
                              (b'altcol-cell-val', datetime.datetime(...)),
                          ],
                      },
                      u'col-fam-id2': {
                          b'col-name3-but-other-fam': [
                              (b'foo', datetime.datetime(...)),
                          ],
                      },
                  }
    """
    result = {}
    for column_family in row_response.row.families:
        column_family_id, curr_family = _parse_family_pb(column_family)
        result[column_family_id] = curr_family
    return result


def _parse_family_pb(family_pb):
    """Parses a Family protobuf into a dictionary.

    :type family_pb: :class:`._generated.data_pb2.Family`
    :param family_pb: A protobuf

    :rtype: tuple
    :returns: A string and dictionary. The string is the name of the
              column family and the dictionary has column names (within the
              family) as keys and cell lists as values. Each cell is
              represented with a two-tuple with the value (in bytes) and the
              timestamp for the cell. For example:

              .. code:: python

                  {
                      b'col-name1': [
                          (b'cell-val', datetime.datetime(...)),
                          (b'cell-val-newer', datetime.datetime(...)),
                      ],
                      b'col-name2': [
                          (b'altcol-cell-val', datetime.datetime(...)),
                      ],
                  }
    """
    result = {}
    for column in family_pb.columns:
        result[column.qualifier] = cells = []
        for cell in column.cells:
            val_pair = (cell.value, _datetime_from_microseconds(cell.timestamp_micros))
            cells.append(val_pair)

    return family_pb.name, result


class PartialRowData(object):
    """Representation of partial row in a Google Cloud Bigtable Table.

    These are expected to be updated directly from a
    :class:`._generated.bigtable_service_messages_pb2.ReadRowsResponse`

    :type row_key: bytes
    :param row_key: The key for the row holding the (partial) data.
    """

    def __init__(self, row_key):
        self._row_key = row_key
        self._cells = {}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other._row_key == self._row_key and other._cells == self._cells

    def __ne__(self, other):
        return not self == other

    def to_dict(self):
        """Convert the cells to a dictionary.

        This is intended to be used with HappyBase, so the column family and
        column qualiers are combined (with ``:``).

        :rtype: dict
        :returns: Dictionary containing all the data in the cells of this row.
        """
        result = {}
        for column_family_id, columns in self._cells.items():
            for column_qual, cells in columns.items():
                key = _to_bytes(column_family_id) + b":" + _to_bytes(column_qual)
                result[key] = cells
        return result

    @property
    def cells(self):
        """Property returning all the cells accumulated on this partial row.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_data_cells]
            :end-before: [END bigtable_api_row_data_cells]
            :dedent: 4

        :rtype: dict
        :returns: Dictionary of the :class:`Cell` objects accumulated. This
                  dictionary has two-levels of keys (first for column families
                  and second for column names/qualifiers within a family). For
                  a given column, a list of :class:`Cell` objects is stored.
        """
        return self._cells

    @property
    def row_key(self):
        """Getter for the current (partial) row's key.

        :rtype: bytes
        :returns: The current (partial) row's key.
        """
        return self._row_key

    def find_cells(self, column_family_id, column):
        """Get a time series of cells stored on this instance.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_find_cells]
            :end-before: [END bigtable_api_row_find_cells]
            :dedent: 4

        Args:
            column_family_id (str): The ID of the column family. Must be of the
                form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
            column (bytes): The column within the column family where the cells
                are located.

        Returns:
            List[~google.cloud.bigtable.row_data.Cell]: The cells stored in the
            specified column.

        Raises:
            KeyError: If ``column_family_id`` is not among the cells stored
                in this row.
            KeyError: If ``column`` is not among the cells stored in this row
                for the given ``column_family_id``.
        """
        try:
            column_family = self._cells[column_family_id]
        except KeyError:
            raise KeyError(_MISSING_COLUMN_FAMILY.format(column_family_id))

        try:
            cells = column_family[column]
        except KeyError:
            raise KeyError(_MISSING_COLUMN.format(column, column_family_id))

        return cells

    def cell_value(self, column_family_id, column, index=0):
        """Get a single cell value stored on this instance.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_cell_value]
            :end-before: [END bigtable_api_row_cell_value]
            :dedent: 4

        Args:
            column_family_id (str): The ID of the column family. Must be of the
                form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
            column (bytes): The column within the column family where the cell
                is located.
            index (Optional[int]): The offset within the series of values. If
                not specified, will return the first cell.

        Returns:
            ~google.cloud.bigtable.row_data.Cell value: The cell value stored
            in the specified column and specified index.

        Raises:
            KeyError: If ``column_family_id`` is not among the cells stored
                in this row.
            KeyError: If ``column`` is not among the cells stored in this row
                for the given ``column_family_id``.
            IndexError: If ``index`` cannot be found within the cells stored
                in this row for the given ``column_family_id``, ``column``
                pair.
        """
        cells = self.find_cells(column_family_id, column)

        try:
            cell = cells[index]
        except (TypeError, IndexError):
            num_cells = len(cells)
            msg = _MISSING_INDEX.format(index, column, column_family_id, num_cells)
            raise IndexError(msg)

        return cell.value

    def cell_values(self, column_family_id, column, max_count=None):
        """Get a time series of cells stored on this instance.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_row_cell_values]
            :end-before: [END bigtable_api_row_cell_values]
            :dedent: 4

        Args:
            column_family_id (str): The ID of the column family. Must be of the
                form ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.
            column (bytes): The column within the column family where the cells
                are located.
            max_count (int): The maximum number of cells to use.

        Returns:
            A generator which provides: cell.value, cell.timestamp_micros
                for each cell in the list of cells

        Raises:
            KeyError: If ``column_family_id`` is not among the cells stored
                in this row.
            KeyError: If ``column`` is not among the cells stored in this row
                for the given ``column_family_id``.
        """
        cells = self.find_cells(column_family_id, column)
        if max_count is None:
            max_count = len(cells)

        for index, cell in enumerate(cells):
            if index == max_count:
                break

            yield cell.value, cell.timestamp_micros


class Cell(object):
    """Representation of a Google Cloud Bigtable Cell.

    :type value: bytes
    :param value: The value stored in the cell.

    :type timestamp_micros: int
    :param timestamp_micros: The timestamp_micros when the cell was stored.

    :type labels: list
    :param labels: (Optional) List of strings. Labels applied to the cell.
    """

    def __init__(self, value, timestamp_micros, labels=None):
        self.value = value
        self.timestamp_micros = timestamp_micros
        self.labels = list(labels) if labels is not None else []

    @classmethod
    def from_pb(cls, cell_pb):
        """Create a new cell from a Cell protobuf.

        :type cell_pb: :class:`._generated.data_pb2.Cell`
        :param cell_pb: The protobuf to convert.

        :rtype: :class:`Cell`
        :returns: The cell corresponding to the protobuf.
        """
        if cell_pb.labels:
            return cls(cell_pb.value, cell_pb.timestamp_micros, labels=cell_pb.labels)
        else:
            return cls(cell_pb.value, cell_pb.timestamp_micros)

    @property
    def timestamp(self):
        return _datetime_from_microseconds(self.timestamp_micros)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.value == self.value
            and other.timestamp_micros == self.timestamp_micros
            and other.labels == self.labels
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "<{name} value={value!r} timestamp={timestamp}>".format(
            name=self.__class__.__name__, value=self.value, timestamp=self.timestamp
        )


class InvalidChunk(RuntimeError):
    """Exception raised to invalid chunk data from back-end."""

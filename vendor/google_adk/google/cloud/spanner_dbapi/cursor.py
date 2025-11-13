# Copyright 2020 Google LLC All rights reserved.
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

"""Database cursor for Google Cloud Spanner DB API."""
from collections import namedtuple

import sqlparse

from google.api_core.exceptions import Aborted
from google.api_core.exceptions import AlreadyExists
from google.api_core.exceptions import FailedPrecondition
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import InvalidArgument
from google.api_core.exceptions import OutOfRange

from google.cloud import spanner_v1 as spanner
from google.cloud.spanner_dbapi.batch_dml_executor import BatchMode
from google.cloud.spanner_dbapi.exceptions import IntegrityError
from google.cloud.spanner_dbapi.exceptions import InterfaceError
from google.cloud.spanner_dbapi.exceptions import OperationalError
from google.cloud.spanner_dbapi.exceptions import ProgrammingError

from google.cloud.spanner_dbapi import (
    _helpers,
    client_side_statement_executor,
    batch_dml_executor,
)
from google.cloud.spanner_dbapi._helpers import ColumnInfo
from google.cloud.spanner_dbapi._helpers import CODE_TO_DISPLAY_SIZE

from google.cloud.spanner_dbapi import parse_utils
from google.cloud.spanner_dbapi.parse_utils import get_param_types
from google.cloud.spanner_dbapi.parsed_statement import (
    StatementType,
    Statement,
    ParsedStatement,
    AutocommitDmlMode,
)
from google.cloud.spanner_dbapi.transaction_helper import CursorStatementType
from google.cloud.spanner_dbapi.utils import PeekIterator
from google.cloud.spanner_dbapi.utils import StreamedManyResultSets
from google.cloud.spanner_v1 import RequestOptions
from google.cloud.spanner_v1.merged_result_set import MergedResultSet

ColumnDetails = namedtuple("column_details", ["null_ok", "spanner_type"])


def check_not_closed(function):
    """`Cursor` class methods decorator.

    Raise an exception if the cursor is closed, or not bound to a
    connection, or the parent connection is closed.

    :raises: :class:`InterfaceError` if this cursor is closed.
    :raises: :class:`ProgrammingError` if this cursor is not bound to a connection.
    """

    def wrapper(cursor, *args, **kwargs):
        if not cursor.connection:
            raise ProgrammingError("Cursor is not connected to the database")

        if cursor.is_closed:
            raise InterfaceError("Cursor and/or connection is already closed.")

        return function(cursor, *args, **kwargs)

    return wrapper


class Cursor(object):
    """Database cursor to manage the context of a fetch operation.

    :type connection: :class:`~google.cloud.spanner_dbapi.connection.Connection`
    :param connection: A DB-API connection to Google Cloud Spanner.
    """

    def __init__(self, connection):
        self._itr = None
        self._result_set = None
        self._row_count = None
        self.lastrowid = None
        self.connection = connection
        self.transaction_helper = self.connection._transaction_helper
        self._is_closed = False
        # the number of rows to fetch at a time with fetchmany()
        self.arraysize = 1
        self._parsed_statement: ParsedStatement = None
        self._in_retry_mode = False
        self._batch_dml_rows_count = None
        self._request_tag = None

    @property
    def request_tag(self):
        """The request tag that will be applied to the next statement on this
        cursor. This property is automatically cleared when a statement is
        executed.

        Returns:
            str: The request tag that will be applied to the next statement on
                 this cursor.
        """
        return self._request_tag

    @request_tag.setter
    def request_tag(self, value):
        """Sets the request tag for the next statement on this cursor. This
        property is automatically cleared when a statement is executed.

        Args:
            value (str): The request tag for the statement.
        """
        self._request_tag = value

    @property
    def request_options(self):
        options = self.connection.request_options
        if self._request_tag:
            if not options:
                options = RequestOptions()
            options.request_tag = self._request_tag
            self._request_tag = None
        return options

    @property
    def is_closed(self):
        """The cursor close indicator.

        :rtype: bool
        :returns: True if the cursor or the parent connection is closed,
                  otherwise False.
        """
        return self._is_closed or self.connection.is_closed

    @property
    def description(self):
        """
        Read-only attribute containing the result columns description
        of a form:

        -   ``name``
        -   ``type_code``
        -   ``display_size``
        -   ``internal_size``
        -   ``precision``
        -   ``scale``
        -   ``null_ok``

        :rtype: tuple
        :returns: The result columns' description.
        """
        if (
            self._result_set is None
            or not getattr(self._result_set, "metadata", None)
            or self._result_set.metadata.row_type is None
            or self._result_set.metadata.row_type.fields is None
            or len(self._result_set.metadata.row_type.fields) == 0
        ):
            return

        columns = []
        for field in self._result_set.metadata.row_type.fields:
            columns.append(
                ColumnInfo(
                    name=field.name,
                    type_code=field.type_.code,
                    # Size of the SQL type of the column.
                    display_size=CODE_TO_DISPLAY_SIZE.get(field.type_.code),
                    # Client perceived size of the column.
                    internal_size=field._pb.ByteSize(),
                )
            )
        return tuple(columns)

    @property
    def rowcount(self):
        """The number of rows updated by the last INSERT, UPDATE, DELETE request's `execute()` call.
        For SELECT requests the rowcount returns -1.

        :rtype: int
        :returns: The number of rows updated by the last INSERT, UPDATE, DELETE request's .execute*() call.
        """

        if self._row_count is not None or self._result_set is None:
            return self._row_count

        stats = getattr(self._result_set, "stats", None)
        if stats is not None and "row_count_exact" in stats:
            return stats.row_count_exact

        return -1

    @check_not_closed
    def callproc(self, procname, args=None):
        """A no-op, raising an error if the cursor or connection is closed."""
        pass

    @check_not_closed
    def nextset(self):
        """A no-op, raising an error if the cursor or connection is closed."""
        pass

    @check_not_closed
    def setinputsizes(self, sizes):
        """A no-op, raising an error if the cursor or connection is closed."""
        pass

    @check_not_closed
    def setoutputsize(self, size, column=None):
        """A no-op, raising an error if the cursor or connection is closed."""
        pass

    def close(self):
        """Closes this cursor."""
        self._is_closed = True

    def _do_execute_update_in_autocommit(self, transaction, sql, params):
        """This function should only be used in autocommit mode."""
        self.connection._transaction = transaction
        self.connection._snapshot = None
        self._result_set = transaction.execute_sql(
            sql,
            params=params,
            param_types=get_param_types(params),
            last_statement=True,
        )
        self._itr = PeekIterator(self._result_set)
        self._row_count = None

    def _batch_DDLs(self, sql):
        """
        Check that the given operation contains only DDL
        statements and batch them into an internal list.

        :type sql: str
        :param sql: A SQL query statement.

        :raises: :class:`ValueError` in case not a DDL statement
                 present in the operation.
        """
        statements = []
        for ddl in sqlparse.split(sql):
            if ddl:
                ddl = ddl.rstrip(";")
                if (
                    parse_utils.classify_statement(ddl).statement_type
                    != StatementType.DDL
                ):
                    raise ValueError("Only DDL statements may be batched.")

                statements.append(ddl)

        # Only queue DDL statements if they are all correctly classified.
        self.connection._ddl_statements.extend(statements)

    def _reset(self):
        if self.connection.database is None:
            raise ValueError("Database needs to be passed for this operation")
        self._itr = None
        self._result_set = None
        self._row_count = None
        self._batch_dml_rows_count = None

    @check_not_closed
    def execute(self, sql, args=None):
        self._execute(sql, args, False)

    def _execute(self, sql, args=None, call_from_execute_many=False):
        """Prepares and executes a Spanner database operation.

        :type sql: str
        :param sql: A SQL query statement.

        :type args: list
        :param args: Additional parameters to supplement the SQL query.
        """
        self._reset()
        exception = None
        try:
            self._parsed_statement = parse_utils.classify_statement(sql, args)
            if self._parsed_statement is None:
                raise ProgrammingError("Invalid Statement.")

            if self._parsed_statement.statement_type == StatementType.CLIENT_SIDE:
                self._result_set = client_side_statement_executor.execute(
                    self, self._parsed_statement
                )
                if self._result_set is not None:
                    if isinstance(
                        self._result_set, StreamedManyResultSets
                    ) or isinstance(self._result_set, MergedResultSet):
                        self._itr = self._result_set
                    else:
                        self._itr = PeekIterator(self._result_set)
            elif self.connection._batch_mode == BatchMode.DML:
                self.connection.execute_batch_dml_statement(self._parsed_statement)
            elif self.connection.read_only or (
                not self.connection._client_transaction_started
                and self._parsed_statement.statement_type == StatementType.QUERY
            ):
                self._handle_DQL(sql, args or None)
            elif self._parsed_statement.statement_type == StatementType.DDL:
                self._batch_DDLs(sql)
                if not self.connection._client_transaction_started:
                    self.connection.run_prior_DDL_statements()
            elif (
                self.connection.autocommit_dml_mode
                is AutocommitDmlMode.PARTITIONED_NON_ATOMIC
            ):
                self._row_count = self.connection.database.execute_partitioned_dml(
                    sql,
                    params=args,
                    param_types=self._parsed_statement.statement.param_types,
                    request_options=self.request_options,
                )
                self._result_set = None
            else:
                self._execute_in_rw_transaction()

        except (AlreadyExists, FailedPrecondition, OutOfRange) as e:
            exception = e
            raise IntegrityError(getattr(e, "details", e)) from e
        except InvalidArgument as e:
            exception = e
            raise ProgrammingError(getattr(e, "details", e)) from e
        except InternalServerError as e:
            exception = e
            raise OperationalError(getattr(e, "details", e)) from e
        except Exception as e:
            exception = e
            raise
        finally:
            if not self._in_retry_mode and not call_from_execute_many:
                self.transaction_helper.add_execute_statement_for_retry(
                    self, sql, args, exception, False
                )
            if self.connection._client_transaction_started is False:
                self.connection._spanner_transaction_started = False

    def _execute_in_rw_transaction(self):
        # For every other operation, we've got to ensure that
        # any prior DDL statements were run.
        self.connection.run_prior_DDL_statements()
        statement = self._parsed_statement.statement
        if self.connection._client_transaction_started:
            while True:
                try:
                    self._result_set = self.connection.run_statement(
                        statement, self.request_options
                    )
                    self._itr = PeekIterator(self._result_set)
                    return
                except Aborted:
                    # We are raising it so it could be handled in transaction_helper.py and is retried
                    if self._in_retry_mode:
                        raise
                    else:
                        self.transaction_helper.retry_transaction()
        else:
            self.connection.database.run_in_transaction(
                self._do_execute_update_in_autocommit,
                statement.sql,
                statement.params or None,
            )

    @check_not_closed
    def executemany(self, operation, seq_of_params):
        """Execute the given SQL with every parameters set
        from the given sequence of parameters.

        :type operation: str
        :param operation: SQL code to execute.

        :type seq_of_params: list
        :param seq_of_params: Sequence of additional parameters to run
                              the query with.
        """
        self._reset()
        exception = None
        try:
            self._parsed_statement = parse_utils.classify_statement(operation)
            if self._parsed_statement.statement_type == StatementType.DDL:
                raise ProgrammingError(
                    "Executing DDL statements with executemany() method is not allowed."
                )

            if self._parsed_statement.statement_type == StatementType.CLIENT_SIDE:
                raise ProgrammingError(
                    "Executing the following operation: "
                    + operation
                    + ", with executemany() method is not allowed."
                )

            # For every operation, we've got to ensure that any prior DDL
            # statements were run.
            self.connection.run_prior_DDL_statements()
            # Treat UNKNOWN statements as if they are DML and let the server
            # determine what is wrong with it.
            if self._parsed_statement.statement_type in (
                StatementType.INSERT,
                StatementType.UPDATE,
                StatementType.UNKNOWN,
            ):
                statements = []
                for params in seq_of_params:
                    sql, params = parse_utils.sql_pyformat_args_to_spanner(
                        operation, params
                    )
                    statements.append(Statement(sql, params, get_param_types(params)))
                many_result_set = batch_dml_executor.run_batch_dml(self, statements)
            else:
                many_result_set = StreamedManyResultSets()
                for params in seq_of_params:
                    self._execute(operation, params, True)
                    many_result_set.add_iter(self._itr)

            self._result_set = many_result_set
            self._itr = many_result_set
        except Exception as e:
            exception = e
            raise
        finally:
            if not self._in_retry_mode:
                self.transaction_helper.add_execute_statement_for_retry(
                    self,
                    operation,
                    seq_of_params,
                    exception,
                    True,
                )
            if self.connection._client_transaction_started is False:
                self.connection._spanner_transaction_started = False

    @check_not_closed
    def fetchone(self):
        """Fetch the next row of a query result set, returning a single
        sequence, or None when no more data is available."""
        rows = self._fetch(CursorStatementType.FETCH_ONE)
        if not rows:
            return
        return rows[0]

    @check_not_closed
    def fetchall(self):
        """Fetch all (remaining) rows of a query result, returning them as
        a sequence of sequences.
        """
        return self._fetch(CursorStatementType.FETCH_ALL)

    @check_not_closed
    def fetchmany(self, size=None):
        """Fetch the next set of rows of a query result, returning a sequence
        of sequences. An empty sequence is returned when no more rows are available.

        :type size: int
        :param size: (Optional) The maximum number of results to fetch.

        :raises InterfaceError:
            if the previous call to .execute*() did not produce any result set
            or if no call was issued yet.
        """
        if size is None:
            size = self.arraysize
        return self._fetch(CursorStatementType.FETCH_MANY, size)

    def _fetch(self, cursor_statement_type, size=None):
        exception = None
        rows = []
        is_fetch_all = False
        try:
            while True:
                rows = []
                try:
                    if cursor_statement_type == CursorStatementType.FETCH_ALL:
                        is_fetch_all = True
                        for row in self:
                            rows.append(row)
                    elif cursor_statement_type == CursorStatementType.FETCH_MANY:
                        for _ in range(size):
                            try:
                                row = next(self)
                                rows.append(row)
                            except StopIteration:
                                break
                    elif cursor_statement_type == CursorStatementType.FETCH_ONE:
                        try:
                            row = next(self)
                            rows.append(row)
                        except StopIteration:
                            return
                    break
                except Aborted:
                    if not self.connection.read_only:
                        if self._in_retry_mode:
                            raise
                        else:
                            self.transaction_helper.retry_transaction()
        except Exception as e:
            exception = e
            raise
        finally:
            if not self._in_retry_mode:
                self.transaction_helper.add_fetch_statement_for_retry(
                    self, rows, exception, is_fetch_all
                )
            return rows

    def _handle_DQL_with_snapshot(self, snapshot, sql, params):
        self._result_set = snapshot.execute_sql(
            sql,
            params,
            get_param_types(params),
            request_options=self.request_options,
        )
        # Read the first element so that the StreamedResultSet can
        # return the metadata after a DQL statement.
        self._itr = PeekIterator(self._result_set)
        # Unfortunately, Spanner doesn't seem to send back
        # information about the number of rows available.
        self._row_count = None
        if self._result_set.metadata.transaction.read_timestamp is not None:
            snapshot._transaction_read_timestamp = (
                self._result_set.metadata.transaction.read_timestamp
            )

    def _handle_DQL(self, sql, params):
        if self.connection.database is None:
            raise ValueError("Database needs to be passed for this operation")
        sql, params = parse_utils.sql_pyformat_args_to_spanner(sql, params)
        if self.connection.read_only and self.connection._client_transaction_started:
            # initiate or use the existing multi-use snapshot
            self._handle_DQL_with_snapshot(
                self.connection.snapshot_checkout(), sql, params
            )
        else:
            # execute with single-use snapshot
            with self.connection.database.snapshot(
                **self.connection.staleness
            ) as snapshot:
                self.connection._snapshot = snapshot
                self.connection._transaction = None
                self._handle_DQL_with_snapshot(snapshot, sql, params)

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def __next__(self):
        if self._itr is None:
            raise ProgrammingError("no results to return")
        return next(self._itr)

    def __iter__(self):
        if self._itr is None:
            raise ProgrammingError("no results to return")
        return self._itr

    def list_tables(self, schema_name="", include_views=True):
        """List the tables of the linked Database.

        :rtype: list
        :returns: The list of tables within the Database.
        """
        return self.run_sql_in_snapshot(
            sql=_helpers.SQL_LIST_TABLES_AND_VIEWS
            if include_views
            else _helpers.SQL_LIST_TABLES,
            params={"table_schema": schema_name},
            param_types={"table_schema": spanner.param_types.STRING},
        )

    def run_sql_in_snapshot(self, sql, params=None, param_types=None):
        # Some SQL e.g. for INFORMATION_SCHEMA cannot be run in read-write transactions
        # hence this method exists to circumvent that limit.
        if self.connection.database is None:
            raise ValueError("Database needs to be passed for this operation")
        self.connection.run_prior_DDL_statements()

        with self.connection.database.snapshot() as snapshot:
            return list(snapshot.execute_sql(sql, params, param_types))

    def get_table_column_schema(self, table_name, schema_name=""):
        rows = self.run_sql_in_snapshot(
            sql=_helpers.SQL_GET_TABLE_COLUMN_SCHEMA,
            params={"schema_name": schema_name, "table_name": table_name},
            param_types={
                "schema_name": spanner.param_types.STRING,
                "table_name": spanner.param_types.STRING,
            },
        )

        column_details = {}
        for column_name, is_nullable, spanner_type in rows:
            column_details[column_name] = ColumnDetails(
                null_ok=is_nullable == "YES", spanner_type=spanner_type
            )
        return column_details

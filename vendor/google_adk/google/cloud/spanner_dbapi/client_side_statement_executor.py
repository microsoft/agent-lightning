# Copyright 2023 Google LLC All rights reserved.
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
from typing import TYPE_CHECKING, Union
from google.cloud.spanner_v1 import TransactionOptions

if TYPE_CHECKING:
    from google.cloud.spanner_dbapi.cursor import Cursor
    from google.cloud.spanner_dbapi import ProgrammingError

from google.cloud.spanner_dbapi.parsed_statement import (
    ParsedStatement,
    ClientSideStatementType,
)
from google.cloud.spanner_v1 import (
    Type,
    StructType,
    TypeCode,
    ResultSetMetadata,
    PartialResultSet,
)

from google.cloud.spanner_v1._helpers import _make_value_pb
from google.cloud.spanner_v1.streamed import StreamedResultSet

CONNECTION_CLOSED_ERROR = "This connection is closed"
TRANSACTION_NOT_STARTED_WARNING = (
    "This method is non-operational as a transaction has not been started."
)


def execute(cursor: "Cursor", parsed_statement: ParsedStatement):
    """Executes the client side statements by calling the relevant method.

    It is an internal method that can make backwards-incompatible changes.

    :type cursor: Cursor
    :param cursor: Cursor object of the dbApi

    :type parsed_statement: ParsedStatement
    :param parsed_statement: parsed_statement based on the sql query
    """
    connection = cursor.connection
    column_values = []
    if connection.is_closed:
        raise ProgrammingError(CONNECTION_CLOSED_ERROR)
    statement_type = parsed_statement.client_side_statement_type
    if statement_type == ClientSideStatementType.COMMIT:
        connection.commit()
        return None
    if statement_type == ClientSideStatementType.BEGIN:
        connection.begin(isolation_level=_get_isolation_level(parsed_statement))
        return None
    if statement_type == ClientSideStatementType.ROLLBACK:
        connection.rollback()
        return None
    if statement_type == ClientSideStatementType.SHOW_COMMIT_TIMESTAMP:
        if (
            connection._transaction is not None
            and connection._transaction.committed is not None
        ):
            column_values.append(connection._transaction.committed)
        return _get_streamed_result_set(
            ClientSideStatementType.SHOW_COMMIT_TIMESTAMP.name,
            TypeCode.TIMESTAMP,
            column_values,
        )
    if statement_type == ClientSideStatementType.SHOW_READ_TIMESTAMP:
        if (
            connection._snapshot is not None
            and connection._snapshot._transaction_read_timestamp is not None
        ):
            column_values.append(connection._snapshot._transaction_read_timestamp)
        return _get_streamed_result_set(
            ClientSideStatementType.SHOW_READ_TIMESTAMP.name,
            TypeCode.TIMESTAMP,
            column_values,
        )
    if statement_type == ClientSideStatementType.START_BATCH_DML:
        connection.start_batch_dml(cursor)
        return None
    if statement_type == ClientSideStatementType.RUN_BATCH:
        return connection.run_batch()
    if statement_type == ClientSideStatementType.ABORT_BATCH:
        return connection.abort_batch()
    if statement_type == ClientSideStatementType.PARTITION_QUERY:
        partition_ids = connection.partition_query(parsed_statement)
        return _get_streamed_result_set(
            "PARTITION",
            TypeCode.STRING,
            partition_ids,
        )
    if statement_type == ClientSideStatementType.RUN_PARTITION:
        return connection.run_partition(
            parsed_statement.client_side_statement_params[0]
        )
    if statement_type == ClientSideStatementType.RUN_PARTITIONED_QUERY:
        return connection.run_partitioned_query(parsed_statement)
    if statement_type == ClientSideStatementType.SET_AUTOCOMMIT_DML_MODE:
        return connection._set_autocommit_dml_mode(parsed_statement)


def _get_streamed_result_set(column_name, type_code, column_values):
    struct_type_pb = StructType(
        fields=[StructType.Field(name=column_name, type_=Type(code=type_code))]
    )

    result_set = PartialResultSet(metadata=ResultSetMetadata(row_type=struct_type_pb))
    if len(column_values) > 0:
        column_values_pb = []
        for column_value in column_values:
            column_values_pb.append(_make_value_pb(column_value))
        result_set.values.extend(column_values_pb)
    return StreamedResultSet(iter([result_set]))


def _get_isolation_level(
    statement: ParsedStatement,
) -> Union[TransactionOptions.IsolationLevel, None]:
    if (
        statement.client_side_statement_params is None
        or len(statement.client_side_statement_params) == 0
    ):
        return None
    level = statement.client_side_statement_params[0]
    if not isinstance(level, str) or level == "":
        return None
    # Replace (duplicate) whitespaces in the string with an underscore.
    level = "_".join(level.split()).upper()
    return TransactionOptions.IsolationLevel[level]

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

import re

from google.cloud.spanner_dbapi.parsed_statement import (
    ParsedStatement,
    StatementType,
    ClientSideStatementType,
    Statement,
)

RE_BEGIN = re.compile(
    r"^\s*(?:BEGIN|START)(?:\s+TRANSACTION)?(?:\s+ISOLATION\s+LEVEL\s+(REPEATABLE\s+READ|SERIALIZABLE))?\s*$",
    re.IGNORECASE,
)
RE_COMMIT = re.compile(r"^\s*(COMMIT)(\s+TRANSACTION)?\s*$", re.IGNORECASE)
RE_ROLLBACK = re.compile(r"^\s*(ROLLBACK)(\s+TRANSACTION)?\s*$", re.IGNORECASE)
RE_SHOW_COMMIT_TIMESTAMP = re.compile(
    r"^\s*(SHOW)\s+(VARIABLE)\s+(COMMIT_TIMESTAMP)\s*$", re.IGNORECASE
)
RE_SHOW_READ_TIMESTAMP = re.compile(
    r"^\s*(SHOW)\s+(VARIABLE)\s+(READ_TIMESTAMP)\s*$", re.IGNORECASE
)
RE_START_BATCH_DML = re.compile(r"^\s*(START)\s+(BATCH)\s+(DML)\s*$", re.IGNORECASE)
RE_RUN_BATCH = re.compile(r"^\s*(RUN)\s+(BATCH)\s*$", re.IGNORECASE)
RE_ABORT_BATCH = re.compile(r"^\s*(ABORT)\s+(BATCH)\s*$", re.IGNORECASE)
RE_PARTITION_QUERY = re.compile(r"^\s*(PARTITION)\s+(.+)", re.IGNORECASE)
RE_RUN_PARTITION = re.compile(r"^\s*(RUN)\s+(PARTITION)\s+(.+)", re.IGNORECASE)
RE_RUN_PARTITIONED_QUERY = re.compile(
    r"^\s*(RUN)\s+(PARTITIONED)\s+(QUERY)\s+(.+)", re.IGNORECASE
)
RE_SET_AUTOCOMMIT_DML_MODE = re.compile(
    r"^\s*(SET)\s+(AUTOCOMMIT_DML_MODE)\s+(=)\s+(.+)", re.IGNORECASE
)


def parse_stmt(query):
    """Parses the sql query to check if it matches with any of the client side
        statement regex.

    It is an internal method that can make backwards-incompatible changes.

    :type query: str
    :param query: sql query

    :rtype: ParsedStatement
    :returns: ParsedStatement object.
    """
    client_side_statement_type = None
    client_side_statement_params = []
    if RE_COMMIT.match(query):
        client_side_statement_type = ClientSideStatementType.COMMIT
    elif RE_ROLLBACK.match(query):
        client_side_statement_type = ClientSideStatementType.ROLLBACK
    elif RE_SHOW_COMMIT_TIMESTAMP.match(query):
        client_side_statement_type = ClientSideStatementType.SHOW_COMMIT_TIMESTAMP
    elif RE_SHOW_READ_TIMESTAMP.match(query):
        client_side_statement_type = ClientSideStatementType.SHOW_READ_TIMESTAMP
    elif RE_START_BATCH_DML.match(query):
        client_side_statement_type = ClientSideStatementType.START_BATCH_DML
    elif RE_BEGIN.match(query):
        match = re.search(RE_BEGIN, query)
        isolation_level = match.group(1)
        if isolation_level is not None:
            client_side_statement_params.append(isolation_level)
        client_side_statement_type = ClientSideStatementType.BEGIN
    elif RE_RUN_BATCH.match(query):
        client_side_statement_type = ClientSideStatementType.RUN_BATCH
    elif RE_ABORT_BATCH.match(query):
        client_side_statement_type = ClientSideStatementType.ABORT_BATCH
    elif RE_RUN_PARTITIONED_QUERY.match(query):
        match = re.search(RE_RUN_PARTITIONED_QUERY, query)
        client_side_statement_params.append(match.group(4))
        client_side_statement_type = ClientSideStatementType.RUN_PARTITIONED_QUERY
    elif RE_PARTITION_QUERY.match(query):
        match = re.search(RE_PARTITION_QUERY, query)
        client_side_statement_params.append(match.group(2))
        client_side_statement_type = ClientSideStatementType.PARTITION_QUERY
    elif RE_RUN_PARTITION.match(query):
        match = re.search(RE_RUN_PARTITION, query)
        client_side_statement_params.append(match.group(3))
        client_side_statement_type = ClientSideStatementType.RUN_PARTITION
    elif RE_SET_AUTOCOMMIT_DML_MODE.match(query):
        match = re.search(RE_SET_AUTOCOMMIT_DML_MODE, query)
        client_side_statement_params.append(match.group(4))
        client_side_statement_type = ClientSideStatementType.SET_AUTOCOMMIT_DML_MODE
    if client_side_statement_type is not None:
        return ParsedStatement(
            StatementType.CLIENT_SIDE,
            Statement(query),
            client_side_statement_type,
            client_side_statement_params,
        )
    return None

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

"""DB-API Connection for the Google Cloud Spanner."""
import warnings

from google.api_core.exceptions import Aborted
from google.api_core.gapic_v1.client_info import ClientInfo
from google.auth.credentials import AnonymousCredentials

from google.cloud import spanner_v1 as spanner
from google.cloud.spanner_dbapi import partition_helper
from google.cloud.spanner_dbapi.batch_dml_executor import BatchMode, BatchDmlExecutor
from google.cloud.spanner_dbapi.parsed_statement import AutocommitDmlMode
from google.cloud.spanner_dbapi.partition_helper import PartitionId
from google.cloud.spanner_dbapi.parsed_statement import ParsedStatement, Statement
from google.cloud.spanner_dbapi.transaction_helper import TransactionRetryHelper
from google.cloud.spanner_dbapi.cursor import Cursor
from google.cloud.spanner_v1 import RequestOptions, TransactionOptions
from google.cloud.spanner_v1.database_sessions_manager import TransactionType
from google.cloud.spanner_v1.snapshot import Snapshot

from google.cloud.spanner_dbapi.exceptions import (
    InterfaceError,
    OperationalError,
    ProgrammingError,
)
from google.cloud.spanner_dbapi.version import DEFAULT_USER_AGENT
from google.cloud.spanner_dbapi.version import PY_VERSION


CLIENT_TRANSACTION_NOT_STARTED_WARNING = (
    "This method is non-operational as a transaction has not been started."
)


def check_not_closed(function):
    """`Connection` class methods decorator.

    Raise an exception if the connection is closed.

    :raises: :class:`InterfaceError` if the connection is closed.
    """

    def wrapper(connection, *args, **kwargs):
        if connection.is_closed:
            raise InterfaceError("Connection is already closed")

        return function(connection, *args, **kwargs)

    return wrapper


class Connection:
    """Representation of a DB-API connection to a Cloud Spanner database.

    You most likely don't need to instantiate `Connection` objects
    directly, use the `connect` module function instead.

    :type instance: :class:`~google.cloud.spanner_v1.instance.Instance`
    :param instance: Cloud Spanner instance to connect to.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: The database to which the connection is linked.

    :type read_only: bool
    :param read_only:
        Flag to indicate that the connection may only execute queries and no update or DDL statements.
        If True, the connection will use a single use read-only transaction with strong timestamp
        bound for each new statement, and will immediately see any changes that have been committed by
        any other transaction.
        If autocommit is false, the connection will automatically start a new multi use read-only transaction
        with strong timestamp bound when the first statement is executed. This read-only transaction will be
        used for all subsequent statements until either commit() or rollback() is called on the connection. The
        read-only transaction will read from a consistent snapshot of the database at the time that the
        transaction started. This means that the transaction will not see any changes that have been
        committed by other transactions since the start of the read-only transaction. Commit or rolling back
        the read-only transaction is semantically the same, and only indicates that the read-only transaction
        should end a that a new one should be started when the next statement is executed.

    **kwargs: Initial value for connection variables.
    """

    def __init__(self, instance, database=None, read_only=False, **kwargs):
        self._instance = instance
        self._database = database
        self._ddl_statements = []

        self._transaction = None
        self._session = None
        self._snapshot = None

        self.is_closed = False
        self._autocommit = False
        # indicator to know if the session pool used by
        # this connection should be cleared on the
        # connection close
        self._own_pool = True
        self._read_only = read_only
        self._staleness = None
        self.request_priority = None
        self._transaction_begin_marked = False
        self._transaction_isolation_level = None
        # whether transaction started at Spanner. This means that we had
        # made at least one call to Spanner.
        self._spanner_transaction_started = False
        self._batch_mode = BatchMode.NONE
        self._batch_dml_executor: BatchDmlExecutor = None
        self._transaction_helper = TransactionRetryHelper(self)
        self._autocommit_dml_mode: AutocommitDmlMode = AutocommitDmlMode.TRANSACTIONAL
        self._connection_variables = kwargs

    @property
    def spanner_client(self):
        """Client for interacting with Cloud Spanner API. This property exposes
        the spanner client so that underlying methods can be accessed.
        """
        return self._instance._client

    @property
    def current_schema(self):
        """schema name for this connection.

        :rtype: str
        :returns: the current default schema of this connection. Currently, this
         is always "" for GoogleSQL and "public" for PostgreSQL databases.
        """
        if self.database is None:
            raise ValueError("database property not set on the connection")
        return self.database.default_schema_name

    @property
    def autocommit(self):
        """Autocommit mode flag for this connection.

        :rtype: bool
        :returns: Autocommit mode flag value.
        """
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        """Change this connection autocommit mode. Setting this value to True
        while a transaction is active will commit the current transaction.

        :type value: bool
        :param value: New autocommit mode state.
        """
        if value and not self._autocommit and self._spanner_transaction_started:
            self.commit()

        self._autocommit = value

    @property
    def database(self):
        """Database to which this connection relates.

        :rtype: :class:`~google.cloud.spanner_v1.database.Database`
        :returns: The related database object.
        """
        return self._database

    @property
    def autocommit_dml_mode(self):
        """Modes for executing DML statements in autocommit mode for this connection.

        The DML autocommit modes are:
        1) TRANSACTIONAL - DML statements are executed as single read-write transaction.
        After successful execution, the DML statement is guaranteed to have been applied
        exactly once to the database.

        2) PARTITIONED_NON_ATOMIC - DML statements are executed as partitioned DML transactions.
        If an error occurs during the execution of the DML statement, it is possible that the
        statement has been applied to some but not all of the rows specified in the statement.

        :rtype: :class:`~google.cloud.spanner_dbapi.parsed_statement.AutocommitDmlMode`
        """
        return self._autocommit_dml_mode

    @property
    def inside_transaction(self):
        warnings.warn(
            "This method is deprecated. Use _spanner_transaction_started field",
            DeprecationWarning,
        )
        return (
            self._transaction
            and not self._transaction.committed
            and not self._transaction.rolled_back
        )

    @property
    def _client_transaction_started(self):
        """Flag: whether transaction started at client side.

        Returns:
            bool: True if transaction started, False otherwise.
        """
        return (not self._autocommit) or self._transaction_begin_marked

    @property
    def _ignore_transaction_warnings(self):
        return self._connection_variables.get("ignore_transaction_warnings", False)

    @property
    def instance(self):
        """Instance to which this connection relates.

        :rtype: :class:`~google.cloud.spanner_v1.instance.Instance`
        :returns: The related instance object.
        """
        return self._instance

    @property
    def read_only(self):
        """Flag: the connection can be used only for database reads.

        Returns:
            bool:
                True if the connection may only be used for database reads.
        """
        return self._read_only

    @read_only.setter
    def read_only(self, value):
        """`read_only` flag setter.

        Args:
            value (bool): True for ReadOnly mode, False for ReadWrite.
        """
        if self._read_only != value and self._spanner_transaction_started:
            raise ValueError(
                "Connection read/write mode can't be changed while a transaction is in progress. "
                "Commit or rollback the current transaction and try again."
            )
        self._read_only = value

    @property
    def request_options(self):
        """Options for the next SQL operations.

        Returns:
            google.cloud.spanner_v1.RequestOptions:
                Request options.
        """
        if self.request_priority is None:
            return

        req_opts = RequestOptions(priority=self.request_priority)
        self.request_priority = None
        return req_opts

    @property
    def transaction_tag(self):
        """The transaction tag that will be applied to the next read/write
        transaction on this `Connection`. This property is automatically cleared
        when a new transaction is started.

        Returns:
            str: The transaction tag that will be applied to the next read/write transaction.
        """
        return self._connection_variables.get("transaction_tag", None)

    @transaction_tag.setter
    def transaction_tag(self, value):
        """Sets the transaction tag for the next read/write transaction on this
        `Connection`. This property is automatically cleared when a new transaction
        is started.

        Args:
            value (str): The transaction tag for the next read/write transaction.
        """
        self._connection_variables["transaction_tag"] = value

    @property
    def isolation_level(self):
        """The default isolation level that is used for all read/write
        transactions on this `Connection`.

        Returns:
            google.cloud.spanner_v1.types.TransactionOptions.IsolationLevel:
            The isolation level that is used for read/write transactions on
            this `Connection`.
        """
        return self._connection_variables.get(
            "isolation_level",
            TransactionOptions.IsolationLevel.ISOLATION_LEVEL_UNSPECIFIED,
        )

    @isolation_level.setter
    def isolation_level(self, value: TransactionOptions.IsolationLevel):
        """Sets the isolation level that is used for all read/write
        transactions on this `Connection`.

        Args:
            value (google.cloud.spanner_v1.types.TransactionOptions.IsolationLevel):
            The isolation level for all read/write transactions on this
            `Connection`.
        """
        self._connection_variables["isolation_level"] = value

    @property
    def staleness(self):
        """Current read staleness option value of this `Connection`.

        Returns:
            dict: Staleness type and value.
        """
        return self._staleness or {}

    @staleness.setter
    def staleness(self, value):
        """Read staleness option setter.

        Args:
            value (dict): Staleness type and value.
        """
        if self._spanner_transaction_started and value != self._staleness:
            raise ValueError(
                "`staleness` option can't be changed while a transaction is in progress. "
                "Commit or rollback the current transaction and try again."
            )

        possible_opts = (
            "read_timestamp",
            "min_read_timestamp",
            "max_staleness",
            "exact_staleness",
        )
        if value is not None and sum([opt in value for opt in possible_opts]) != 1:
            raise ValueError(
                "Expected one of the following staleness options: "
                "read_timestamp, min_read_timestamp, max_staleness, exact_staleness."
            )

        self._staleness = value

    def _session_checkout(self):
        """Get a Cloud Spanner session from the pool.

        If there is already a session associated with
        this connection, it'll be used instead.

        :rtype: :class:`google.cloud.spanner_v1.session.Session`
        :returns: Cloud Spanner session object ready to use.
        """
        if self.database is None:
            raise ValueError("Database needs to be passed for this operation")

        if not self._session:
            transaction_type = (
                TransactionType.READ_ONLY
                if self.read_only
                else TransactionType.READ_WRITE
            )
            self._session = self.database._sessions_manager.get_session(
                transaction_type
            )

        return self._session

    def _release_session(self):
        """Release the currently used Spanner session.

        The session will be returned into the sessions pool.
        """
        if self._session is None:
            return

        if self.database is None:
            raise ValueError("Database needs to be passed for this operation")

        self.database._sessions_manager.put_session(self._session)
        self._session = None

    def transaction_checkout(self):
        """Get a Cloud Spanner transaction.

        Begin a new transaction, if there is no transaction in
        this connection yet. Return the started one otherwise.

        This method is a no-op if the connection is in autocommit mode and no
        explicit transaction has been started

        :rtype: :class:`google.cloud.spanner_v1.transaction.Transaction`
        :returns: A Cloud Spanner transaction object, ready to use.
        """
        if not self.read_only and self._client_transaction_started:
            if not self._spanner_transaction_started:
                self._transaction = self._session_checkout().transaction()
                self._transaction.transaction_tag = self.transaction_tag
                if self._transaction_isolation_level:
                    self._transaction.isolation_level = (
                        self._transaction_isolation_level
                    )
                else:
                    self._transaction.isolation_level = self.isolation_level
                self.transaction_tag = None
                self._snapshot = None
                self._spanner_transaction_started = True
                self._transaction.begin()

            return self._transaction

    def snapshot_checkout(self):
        """Get a Cloud Spanner snapshot.

        Initiate a new multi-use snapshot, if there is no snapshot in
        this connection yet. Return the existing one otherwise.

        :rtype: :class:`google.cloud.spanner_v1.snapshot.Snapshot`
        :returns: A Cloud Spanner snapshot object, ready to use.
        """
        if self.read_only and self._client_transaction_started:
            if not self._spanner_transaction_started:
                self._snapshot = Snapshot(
                    self._session_checkout(), multi_use=True, **self.staleness
                )
                self._transaction = None
                self._snapshot.begin()
                self._spanner_transaction_started = True

            return self._snapshot

    def close(self):
        """Closes this connection.

        The connection will be unusable from this point forward. If the
        connection has an active transaction, it will be rolled back.
        """
        if self._spanner_transaction_started and not self._read_only:
            self._transaction.rollback()

        if self._own_pool and self.database:
            self.database._sessions_manager._pool.clear()

        self.is_closed = True

    @check_not_closed
    def begin(self, isolation_level=None):
        """
        Marks the transaction as started.

        :raises: :class:`InterfaceError`: if this connection is closed.
        :raises: :class:`OperationalError`: if there is an existing transaction
        that has been started
        """
        if self._transaction_begin_marked:
            raise OperationalError("A transaction has already started")
        if self._spanner_transaction_started:
            raise OperationalError(
                "Beginning a new transaction is not allowed when a transaction "
                "is already running"
            )
        self._transaction_begin_marked = True
        self._transaction_isolation_level = isolation_level

    def commit(self):
        """Commits any pending transaction to the database.
        This is a no-op if there is no active client transaction.
        """
        if self.database is None:
            raise ValueError("Database needs to be passed for this operation")
        if not self._client_transaction_started:
            if not self._ignore_transaction_warnings:
                warnings.warn(
                    CLIENT_TRANSACTION_NOT_STARTED_WARNING, UserWarning, stacklevel=2
                )
            return

        self.run_prior_DDL_statements()
        try:
            if self._spanner_transaction_started and not self._read_only:
                self._transaction.commit()
        except Aborted:
            self._transaction_helper.retry_transaction()
            self.commit()
        finally:
            self._reset_post_commit_or_rollback()

    def rollback(self):
        """Rolls back any pending transaction.
        This is a no-op if there is no active client transaction.
        """
        if not self._client_transaction_started:
            if not self._ignore_transaction_warnings:
                warnings.warn(
                    CLIENT_TRANSACTION_NOT_STARTED_WARNING, UserWarning, stacklevel=2
                )
            return
        try:
            if self._spanner_transaction_started and not self._read_only:
                self._transaction.rollback()
        finally:
            self._reset_post_commit_or_rollback()

    def _reset_post_commit_or_rollback(self):
        self._release_session()
        self._transaction_helper.reset()
        self._transaction_begin_marked = False
        self._transaction_isolation_level = None
        self._spanner_transaction_started = False

    @check_not_closed
    def cursor(self):
        """Factory to create a DB API Cursor."""
        return Cursor(self)

    @check_not_closed
    def run_prior_DDL_statements(self):
        if self.database is None:
            raise ValueError("Database needs to be passed for this operation")
        if self._ddl_statements:
            ddl_statements = self._ddl_statements
            self._ddl_statements = []

            return self.database.update_ddl(ddl_statements).result()

    def run_statement(
        self, statement: Statement, request_options: RequestOptions = None
    ):
        """Run single SQL statement in begun transaction.

        This method is never used in autocommit mode. In
        !autocommit mode however it remembers every executed
        SQL statement with its parameters.

        :type statement: :class:`Statement`
        :param statement: SQL statement to execute.

        :type retried: bool
        :param retried: (Optional) Retry the SQL statement if statement
                        execution failed. Defaults to false.

        :type request_options: :class:`RequestOptions`
        :param request_options: Request options to use for this statement.

        :rtype: :class:`google.cloud.spanner_v1.streamed.StreamedResultSet`,
                :class:`google.cloud.spanner_dbapi.checksum.ResultsChecksum`
        :returns: Streamed result set of the statement and a
                  checksum of this statement results.
        """
        transaction = self.transaction_checkout()
        return transaction.execute_sql(
            statement.sql,
            statement.params,
            param_types=statement.param_types,
            request_options=request_options or self.request_options,
        )

    @check_not_closed
    def validate(self):
        """
        Execute a minimal request to check if the connection
        is valid and the related database is reachable.

        Raise an exception in case if the connection is closed,
        invalid, target database is not found, or the request result
        is incorrect.

        :raises: :class:`InterfaceError`: if this connection is closed.
        :raises: :class:`OperationalError`: if the request result is incorrect.
        :raises: :class:`google.cloud.exceptions.NotFound`: if the linked instance
                  or database doesn't exist.
        """
        if self.database is None:
            raise ValueError("Database needs to be passed for this operation")
        with self.database.snapshot() as snapshot:
            result = list(snapshot.execute_sql("SELECT 1"))
            if result != [[1]]:
                raise OperationalError(
                    "The checking query (SELECT 1) returned an unexpected result: %s. "
                    "Expected: [[1]]" % result
                )

    @check_not_closed
    def start_batch_dml(self, cursor):
        if self._batch_mode is not BatchMode.NONE:
            raise ProgrammingError(
                "Cannot start a DML batch when a batch is already active"
            )
        if self.read_only:
            raise ProgrammingError(
                "Cannot start a DML batch when the connection is in read-only mode"
            )
        self._batch_mode = BatchMode.DML
        self._batch_dml_executor = BatchDmlExecutor(cursor)

    @check_not_closed
    def execute_batch_dml_statement(self, parsed_statement: ParsedStatement):
        if self._batch_mode is not BatchMode.DML:
            raise ProgrammingError(
                "Cannot execute statement when the BatchMode is not DML"
            )
        self._batch_dml_executor.execute_statement(parsed_statement)

    @check_not_closed
    def run_batch(self):
        if self._batch_mode is BatchMode.NONE:
            raise ProgrammingError("Cannot run a batch when the BatchMode is not set")
        try:
            if self._batch_mode is BatchMode.DML:
                many_result_set = self._batch_dml_executor.run_batch_dml()
        finally:
            self._batch_mode = BatchMode.NONE
            self._batch_dml_executor = None
        return many_result_set

    @check_not_closed
    def abort_batch(self):
        if self._batch_mode is BatchMode.NONE:
            raise ProgrammingError("Cannot abort a batch when the BatchMode is not set")
        if self._batch_mode is BatchMode.DML:
            self._batch_dml_executor = None
        self._batch_mode = BatchMode.NONE

    @check_not_closed
    def partition_query(
        self,
        parsed_statement: ParsedStatement,
        query_options=None,
    ):
        statement = parsed_statement.statement
        partitioned_query = parsed_statement.client_side_statement_params[0]
        self._partitioned_query_validation(partitioned_query, statement)

        batch_snapshot = self._database.batch_snapshot()
        partition_ids = []
        partitions = list(
            batch_snapshot.generate_query_batches(
                partitioned_query,
                statement.params,
                statement.param_types,
                query_options=query_options,
            )
        )

        batch_transaction_id = batch_snapshot.get_batch_transaction_id()
        for partition in partitions:
            partition_ids.append(
                partition_helper.encode_to_string(batch_transaction_id, partition)
            )
        return partition_ids

    @check_not_closed
    def run_partition(self, encoded_partition_id):
        partition_id: PartitionId = partition_helper.decode_from_string(
            encoded_partition_id
        )
        batch_transaction_id = partition_id.batch_transaction_id
        batch_snapshot = self._database.batch_snapshot(
            read_timestamp=batch_transaction_id.read_timestamp,
            session_id=batch_transaction_id.session_id,
            transaction_id=batch_transaction_id.transaction_id,
        )
        return batch_snapshot.process(partition_id.partition_result)

    @check_not_closed
    def run_partitioned_query(
        self,
        parsed_statement: ParsedStatement,
    ):
        statement = parsed_statement.statement
        partitioned_query = parsed_statement.client_side_statement_params[0]
        self._partitioned_query_validation(partitioned_query, statement)
        batch_snapshot = self._database.batch_snapshot()
        return batch_snapshot.run_partitioned_query(
            partitioned_query, statement.params, statement.param_types
        )

    @check_not_closed
    def _set_autocommit_dml_mode(
        self,
        parsed_statement: ParsedStatement,
    ):
        autocommit_dml_mode_str = parsed_statement.client_side_statement_params[0]
        autocommit_dml_mode = AutocommitDmlMode[autocommit_dml_mode_str.upper()]
        self.set_autocommit_dml_mode(autocommit_dml_mode)

    def set_autocommit_dml_mode(
        self,
        autocommit_dml_mode,
    ):
        """
        Sets the mode for executing DML statements in autocommit mode for this connection.
        This mode is only used when the connection is in autocommit mode, and may only
        be set while the transaction is in autocommit mode and not in a temporary transaction.
        """

        if self._client_transaction_started is True:
            raise ProgrammingError(
                "Cannot set autocommit DML mode while not in autocommit mode or while a transaction is active."
            )
        if self.read_only is True:
            raise ProgrammingError(
                "Cannot set autocommit DML mode for a read-only connection."
            )
        if self._batch_mode is not BatchMode.NONE:
            raise ProgrammingError("Cannot set autocommit DML mode while in a batch.")
        self._autocommit_dml_mode = autocommit_dml_mode

    def _partitioned_query_validation(self, partitioned_query, statement):
        if self.read_only is not True and self._client_transaction_started is True:
            raise ProgrammingError(
                "Partitioned query is not supported, because the connection is in a read/write transaction."
            )

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.commit()
        self.close()


def connect(
    instance_id,
    database_id=None,
    project=None,
    credentials=None,
    pool=None,
    user_agent=None,
    client=None,
    route_to_leader_enabled=True,
    database_role=None,
    **kwargs,
):
    """Creates a connection to a Google Cloud Spanner database.

    :type instance_id: str
    :param instance_id: The ID of the instance to connect to.

    :type database_id: str
    :param database_id: (Optional) The ID of the database to connect to.

    :type project: str
    :param project: (Optional) The ID of the project which owns the
                    instances, tables and data. If not provided, will
                    attempt to determine from the environment.

    :type credentials: Union[:class:`~google.auth.credentials.Credentials`, str]
    :param credentials: (Optional) The authorization credentials to attach to
                        requests. These credentials identify this application
                        to the service. These credentials may be specified as
                        a file path indicating where to retrieve the service
                        account JSON for the credentials to connect to
                        Cloud Spanner. If none are specified, the client will
                        attempt to ascertain the credentials from the
                        environment.

    :type pool: Concrete subclass of
                :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`.
    :param pool: (Optional). Session pool to be used by database.

    :type user_agent: str
    :param user_agent: (Optional) User agent to be used with this connection's
                       requests.

    :type client: Concrete subclass of
                  :class:`~google.cloud.spanner_v1.Client`.
    :param client: (Optional) Custom user provided Client Object

    :type route_to_leader_enabled: boolean
    :param route_to_leader_enabled:
        (Optional) Default True. Set route_to_leader_enabled as False to
        disable leader aware routing. Disabling leader aware routing would
        route all requests in RW/PDML transactions to the closest region.

    :type database_role: str
    :param database_role: (Optional) The database role to connect as when using
        fine-grained access controls.

    **kwargs: Initial value for connection variables.


    :rtype: :class:`google.cloud.spanner_dbapi.connection.Connection`
    :returns: Connection object associated with the given Google Cloud Spanner
              resource.
    """
    if client is None:
        client_info = ClientInfo(
            user_agent=user_agent or DEFAULT_USER_AGENT,
            python_version=PY_VERSION,
            client_library_version=spanner.__version__,
        )
        if isinstance(credentials, str):
            client = spanner.Client.from_service_account_json(
                credentials,
                project=project,
                client_info=client_info,
                route_to_leader_enabled=route_to_leader_enabled,
            )
        else:
            client_options = None
            if isinstance(credentials, AnonymousCredentials):
                client_options = kwargs.get("client_options")
            client = spanner.Client(
                project=project,
                credentials=credentials,
                client_info=client_info,
                route_to_leader_enabled=route_to_leader_enabled,
                client_options=client_options,
            )
    else:
        if project is not None and client.project != project:
            raise ValueError("project in url does not match client object project")

    instance = client.instance(instance_id)
    database = None
    if database_id:
        logger = kwargs.get("logger")
        database = instance.database(
            database_id, pool=pool, database_role=database_role, logger=logger
        )
    conn = Connection(instance, database, **kwargs)
    if pool is not None:
        conn._own_pool = False

    return conn

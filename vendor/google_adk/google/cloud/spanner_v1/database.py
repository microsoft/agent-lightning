# Copyright 2016 Google LLC All rights reserved.
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

"""User-friendly container for Cloud Spanner Database."""

import copy
import functools
from typing import Optional

import grpc
import logging
import re
import threading

import google.auth.credentials
from google.api_core.retry import Retry
from google.api_core.retry import if_exception_type
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import Aborted
from google.api_core import gapic_v1
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.protobuf.field_mask_pb2 import FieldMask

from google.cloud.spanner_admin_database_v1 import CreateDatabaseRequest
from google.cloud.spanner_admin_database_v1 import Database as DatabasePB
from google.cloud.spanner_admin_database_v1 import ListDatabaseRolesRequest
from google.cloud.spanner_admin_database_v1 import EncryptionConfig
from google.cloud.spanner_admin_database_v1 import RestoreDatabaseEncryptionConfig
from google.cloud.spanner_admin_database_v1 import RestoreDatabaseRequest
from google.cloud.spanner_admin_database_v1 import UpdateDatabaseDdlRequest
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1.transaction import BatchTransactionId
from google.cloud.spanner_v1 import ExecuteSqlRequest
from google.cloud.spanner_v1 import Type
from google.cloud.spanner_v1 import TypeCode
from google.cloud.spanner_v1 import TransactionSelector
from google.cloud.spanner_v1 import TransactionOptions
from google.cloud.spanner_v1 import DefaultTransactionOptions
from google.cloud.spanner_v1 import RequestOptions
from google.cloud.spanner_v1 import SpannerClient
from google.cloud.spanner_v1._helpers import _merge_query_options
from google.cloud.spanner_v1._helpers import (
    _metadata_with_prefix,
    _metadata_with_leader_aware_routing,
    _metadata_with_request_id,
)
from google.cloud.spanner_v1.batch import Batch
from google.cloud.spanner_v1.batch import MutationGroups
from google.cloud.spanner_v1.keyset import KeySet
from google.cloud.spanner_v1.merged_result_set import MergedResultSet
from google.cloud.spanner_v1.pool import BurstyPool
from google.cloud.spanner_v1.session import Session
from google.cloud.spanner_v1.database_sessions_manager import (
    DatabaseSessionsManager,
    TransactionType,
)
from google.cloud.spanner_v1.snapshot import _restart_on_unavailable
from google.cloud.spanner_v1.snapshot import Snapshot
from google.cloud.spanner_v1.streamed import StreamedResultSet
from google.cloud.spanner_v1.services.spanner.transports.grpc import (
    SpannerGrpcTransport,
)
from google.cloud.spanner_v1.table import Table
from google.cloud.spanner_v1._opentelemetry_tracing import (
    add_span_event,
    get_current_span,
    trace_call,
)
from google.cloud.spanner_v1.metrics.metrics_capture import MetricsCapture


SPANNER_DATA_SCOPE = "https://www.googleapis.com/auth/spanner.data"


_DATABASE_NAME_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/"
    r"instances/(?P<instance_id>[a-z][-a-z0-9]*)/"
    r"databases/(?P<database_id>[a-z][a-z0-9_\-]*[a-z0-9])$"
)

_DATABASE_METADATA_FILTER = "name:{0}/operations/"

_LIST_TABLES_QUERY = """SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
{}
"""

DEFAULT_RETRY_BACKOFF = Retry(initial=0.02, maximum=32, multiplier=1.3)


class Database(object):
    """Representation of a Cloud Spanner Database.

    We can use a :class:`Database` to:

    * :meth:`create` the database
    * :meth:`reload` the database
    * :meth:`update` the database
    * :meth:`drop` the database

    :type database_id: str
    :param database_id: The ID of the database.

    :type instance: :class:`~google.cloud.spanner_v1.instance.Instance`
    :param instance: The instance that owns the database.

    :type ddl_statements: list of string
    :param ddl_statements: (Optional) DDL statements, excluding the
                           CREATE DATABASE statement.

    :type pool: concrete subclass of
                :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`.
    :param pool: (Optional) session pool to be used by database.  If not
                 passed, the database will construct an instance of
                 :class:`~google.cloud.spanner_v1.pool.BurstyPool`.

    :type logger: :class:`logging.Logger`
    :param logger: (Optional) a custom logger that is used if `log_commit_stats`
                   is `True` to log commit statistics. If not passed, a logger
                   will be created when needed that will log the commit statistics
                   to stdout.
    :type encryption_config:
        :class:`~google.cloud.spanner_admin_database_v1.types.EncryptionConfig`
        or :class:`~google.cloud.spanner_admin_database_v1.types.RestoreDatabaseEncryptionConfig`
        or :class:`dict`
    :param encryption_config:
        (Optional) Encryption configuration for the database.
        If a dict is provided, it must be of the same form as either of the protobuf
        messages :class:`~google.cloud.spanner_admin_database_v1.types.EncryptionConfig`
        or :class:`~google.cloud.spanner_admin_database_v1.types.RestoreDatabaseEncryptionConfig`
    :type database_dialect:
        :class:`~google.cloud.spanner_admin_database_v1.types.DatabaseDialect`
    :param database_dialect:
        (Optional) database dialect for the database
    :type database_role: str or None
    :param database_role: (Optional) user-assigned database_role for the session.
    :type enable_drop_protection: boolean
    :param enable_drop_protection: (Optional) Represents whether the database
        has drop protection enabled or not.
    :type proto_descriptors: bytes
    :param proto_descriptors: (Optional) Proto descriptors used by CREATE/ALTER PROTO BUNDLE
                              statements in 'ddl_statements' above.
    """

    _spanner_api: SpannerClient = None

    __transport_lock = threading.Lock()
    __transports_to_channel_id = dict()

    def __init__(
        self,
        database_id,
        instance,
        ddl_statements=(),
        pool=None,
        logger=None,
        encryption_config=None,
        database_dialect=DatabaseDialect.DATABASE_DIALECT_UNSPECIFIED,
        database_role=None,
        enable_drop_protection=False,
        proto_descriptors=None,
    ):
        self.database_id = database_id
        self._instance = instance
        self._ddl_statements = _check_ddl_statements(ddl_statements)
        self._local = threading.local()
        self._state = None
        self._create_time = None
        self._restore_info = None
        self._version_retention_period = None
        self._earliest_version_time = None
        self._encryption_info = None
        self._default_leader = None
        self.log_commit_stats = False
        self._logger = logger
        self._encryption_config = encryption_config
        self._database_dialect = database_dialect
        self._database_role = database_role
        self._route_to_leader_enabled = self._instance._client.route_to_leader_enabled
        self._enable_drop_protection = enable_drop_protection
        self._reconciling = False
        self._directed_read_options = self._instance._client.directed_read_options
        self.default_transaction_options: DefaultTransactionOptions = (
            self._instance._client.default_transaction_options
        )
        self._proto_descriptors = proto_descriptors
        self._channel_id = 0  # It'll be created when _spanner_api is created.

        if pool is None:
            pool = BurstyPool(database_role=database_role)

        self._pool = pool
        pool.bind(self)

        self._sessions_manager = DatabaseSessionsManager(self, pool)

    @classmethod
    def from_pb(cls, database_pb, instance, pool=None):
        """Creates an instance of this class from a protobuf.

        :type database_pb:
            :class:`~google.cloud.spanner_admin_instance_v1.types.Instance`
        :param database_pb: A instance protobuf object.

        :type instance: :class:`~google.cloud.spanner_v1.instance.Instance`
        :param instance: The instance that owns the database.

        :type pool: concrete subclass of
                    :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`.
        :param pool: (Optional) session pool to be used by database.

        :rtype: :class:`Database`
        :returns: The database parsed from the protobuf response.
        :raises ValueError:
            if the instance name does not match the expected format
            or if the parsed project ID does not match the project ID
            on the instance's client, or if the parsed instance ID does
            not match the instance's ID.
        """
        match = _DATABASE_NAME_RE.match(database_pb.name)
        if match is None:
            raise ValueError(
                "Database protobuf name was not in the " "expected format.",
                database_pb.name,
            )
        if match.group("project") != instance._client.project:
            raise ValueError(
                "Project ID on database does not match the "
                "project ID on the instance's client"
            )
        instance_id = match.group("instance_id")
        if instance_id != instance.instance_id:
            raise ValueError(
                "Instance ID on database does not match the "
                "Instance ID on the instance"
            )
        database_id = match.group("database_id")

        return cls(database_id, instance, pool=pool)

    @property
    def name(self):
        """Database name used in requests.

        .. note::

          This property will not change if ``database_id`` does not, but the
          return value is not cached.

        The database name is of the form

            ``"projects/../instances/../databases/{database_id}"``

        :rtype: str
        :returns: The database name.
        """
        return self._instance.name + "/databases/" + self.database_id

    @property
    def state(self):
        """State of this database.

        :rtype: :class:`~google.cloud.spanner_admin_database_v1.types.Database.State`
        :returns: an enum describing the state of the database
        """
        return self._state

    @property
    def create_time(self):
        """Create time of this database.

        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the create time of
            this database
        """
        return self._create_time

    @property
    def restore_info(self):
        """Restore info for this database.

        :rtype: :class:`~google.cloud.spanner_v1.types.RestoreInfo`
        :returns: an object representing the restore info for this database
        """
        return self._restore_info

    @property
    def version_retention_period(self):
        """The period in which Cloud Spanner retains all versions of data
        for the database.

        :rtype: str
        :returns: a string representing the duration of the version retention period
        """
        return self._version_retention_period

    @property
    def earliest_version_time(self):
        """The earliest time at which older versions of the data can be read.

        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the earliest version time
        """
        return self._earliest_version_time

    @property
    def encryption_config(self):
        """Encryption config for this database.
        :rtype: :class:`~google.cloud.spanner_admin_instance_v1.types.EncryptionConfig`
        :returns: an object representing the encryption config for this database
        """
        return self._encryption_config

    @property
    def encryption_info(self):
        """Encryption info for this database.
        :rtype: a list of :class:`~google.cloud.spanner_admin_instance_v1.types.EncryptionInfo`
        :returns: a list of objects representing encryption info for this database
        """
        return self._encryption_info

    @property
    def default_leader(self):
        """The read-write region which contains the database's leader replicas.

        :rtype: str
        :returns: a string representing the read-write region
        """
        return self._default_leader

    @property
    def ddl_statements(self):
        """DDL Statements used to define database schema.

        See
        cloud.google.com/spanner/docs/data-definition-language

        :rtype: sequence of string
        :returns: the statements
        """
        return self._ddl_statements

    @property
    def database_dialect(self):
        """DDL Statements used to define database schema.

        See
        cloud.google.com/spanner/docs/data-definition-language

        :rtype: :class:`google.cloud.spanner_admin_database_v1.types.DatabaseDialect`
        :returns: the dialect of the database
        """
        if self._database_dialect == DatabaseDialect.DATABASE_DIALECT_UNSPECIFIED:
            self.reload()
        return self._database_dialect

    @property
    def default_schema_name(self):
        """Default schema name for this database.

        :rtype: str
        :returns: "" for GoogleSQL and "public" for PostgreSQL
        """
        if self.database_dialect == DatabaseDialect.POSTGRESQL:
            return "public"
        return ""

    @property
    def database_role(self):
        """User-assigned database_role for sessions created by the pool.
        :rtype: str
        :returns: a str with the name of the database role.
        """
        return self._database_role

    @property
    def reconciling(self):
        """Whether the database is currently reconciling.

        :rtype: boolean
        :returns: a boolean representing whether the database is reconciling
        """
        return self._reconciling

    @property
    def enable_drop_protection(self):
        """Whether the database has drop protection enabled.

        :rtype: boolean
        :returns: a boolean representing whether the database has drop
            protection enabled
        """
        return self._enable_drop_protection

    @enable_drop_protection.setter
    def enable_drop_protection(self, value):
        self._enable_drop_protection = value

    @property
    def proto_descriptors(self):
        """Proto Descriptors for this database.
        :rtype: bytes
        :returns: bytes representing the proto descriptors for this database
        """
        return self._proto_descriptors

    @property
    def logger(self):
        """Logger used by the database.

        The default logger will log commit stats at the log level INFO using
        `sys.stderr`.

        :rtype: :class:`logging.Logger` or `None`
        :returns: the logger
        """
        if self._logger is None:
            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self._logger.addHandler(ch)
        return self._logger

    @property
    def spanner_api(self):
        """Helper for session-related API calls."""
        if self._spanner_api is None:
            client_info = self._instance._client._client_info
            client_options = self._instance._client._client_options
            if self._instance.emulator_host is not None:
                transport = SpannerGrpcTransport(
                    channel=grpc.insecure_channel(self._instance.emulator_host)
                )
                self._spanner_api = SpannerClient(
                    client_info=client_info, transport=transport
                )
                return self._spanner_api
            credentials = self._instance._client.credentials
            if isinstance(credentials, google.auth.credentials.Scoped):
                credentials = credentials.with_scopes((SPANNER_DATA_SCOPE,))
            self._spanner_api = SpannerClient(
                credentials=credentials,
                client_info=client_info,
                client_options=client_options,
            )

            with self.__transport_lock:
                transport = self._spanner_api._transport
                channel_id = self.__transports_to_channel_id.get(transport, None)
                if channel_id is None:
                    channel_id = len(self.__transports_to_channel_id) + 1
                    self.__transports_to_channel_id[transport] = channel_id
                self._channel_id = channel_id

        return self._spanner_api

    def metadata_with_request_id(
        self, nth_request, nth_attempt, prior_metadata=[], span=None
    ):
        if span is None:
            span = get_current_span()

        return _metadata_with_request_id(
            self._nth_client_id,
            self._channel_id,
            nth_request,
            nth_attempt,
            prior_metadata,
            span,
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            other.database_id == self.database_id and other._instance == self._instance
        )

    def __ne__(self, other):
        return not self == other

    def create(self):
        """Create this database within its instance

        Includes any configured schema assigned to :attr:`ddl_statements`.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.CreateDatabase

        :rtype: :class:`~google.api_core.operation.Operation`
        :returns: a future used to poll the status of the create request
        :raises Conflict: if the database already exists
        :raises NotFound: if the instance owning the database does not exist
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        db_name = self.database_id
        if "-" in db_name:
            if self._database_dialect == DatabaseDialect.POSTGRESQL:
                db_name = f'"{db_name}"'
            else:
                db_name = f"`{db_name}`"
        if type(self._encryption_config) is dict:
            self._encryption_config = EncryptionConfig(**self._encryption_config)

        request = CreateDatabaseRequest(
            parent=self._instance.name,
            create_statement="CREATE DATABASE %s" % (db_name,),
            extra_statements=list(self._ddl_statements),
            encryption_config=self._encryption_config,
            database_dialect=self._database_dialect,
            proto_descriptors=self._proto_descriptors,
        )
        future = api.create_database(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        return future

    def exists(self):
        """Test whether this database exists.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.GetDatabaseDDL

        :rtype: bool
        :returns: True if the database exists, else false.
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        try:
            api.get_database_ddl(
                database=self.name,
                metadata=self.metadata_with_request_id(
                    self._next_nth_request, 1, metadata
                ),
            )
        except NotFound:
            return False
        return True

    def reload(self):
        """Reload this database.

        Refresh any configured schema into :attr:`ddl_statements`.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.GetDatabaseDDL

        :raises NotFound: if the database does not exist
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        response = api.get_database_ddl(
            database=self.name,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        self._ddl_statements = tuple(response.statements)
        self._proto_descriptors = response.proto_descriptors
        response = api.get_database(
            name=self.name,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        self._state = DatabasePB.State(response.state)
        self._create_time = response.create_time
        self._restore_info = response.restore_info
        self._version_retention_period = response.version_retention_period
        self._earliest_version_time = response.earliest_version_time
        self._encryption_config = response.encryption_config
        self._encryption_info = response.encryption_info
        self._default_leader = response.default_leader
        # Only update if the data is specific to avoid losing specificity.
        if response.database_dialect != DatabaseDialect.DATABASE_DIALECT_UNSPECIFIED:
            self._database_dialect = response.database_dialect
        self._enable_drop_protection = response.enable_drop_protection
        self._reconciling = response.reconciling

    def update_ddl(self, ddl_statements, operation_id="", proto_descriptors=None):
        """Update DDL for this database.

        Apply any configured schema from :attr:`ddl_statements`.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.UpdateDatabaseDdl

        :type ddl_statements: Sequence[str]
        :param ddl_statements: a list of DDL statements to use on this database
        :type operation_id: str
        :param operation_id: (optional) a string ID for the long-running operation
        :type proto_descriptors: bytes
        :param proto_descriptors: (optional) Proto descriptors used by CREATE/ALTER PROTO BUNDLE statements

        :rtype: :class:`google.api_core.operation.Operation`
        :returns: an operation instance
        :raises NotFound: if the database does not exist
        """
        client = self._instance._client
        api = client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        request = UpdateDatabaseDdlRequest(
            database=self.name,
            statements=ddl_statements,
            operation_id=operation_id,
            proto_descriptors=proto_descriptors,
        )

        future = api.update_database_ddl(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        return future

    def update(self, fields):
        """Update this database.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.UpdateDatabase

        .. note::

            Updates the specified fields of a Cloud Spanner database. Currently,
            only the `enable_drop_protection` field supports updates. To change
            this value before updating, set it via

            .. code:: python

                database.enable_drop_protection = True

           before calling :meth:`update`.

        :type fields: Sequence[str]
        :param fields: a list of fields to update

        :rtype: :class:`google.api_core.operation.Operation`
        :returns: an operation instance
        :raises NotFound: if the database does not exist
        """
        api = self._instance._client.database_admin_api
        database_pb = DatabasePB(
            name=self.name, enable_drop_protection=self._enable_drop_protection
        )

        # Only support updating drop protection for now.
        field_mask = FieldMask(paths=fields)
        metadata = _metadata_with_prefix(self.name)

        future = api.update_database(
            database=database_pb,
            update_mask=field_mask,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )

        return future

    def drop(self):
        """Drop this database.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.DropDatabase
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        api.drop_database(
            database=self.name,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )

    def execute_partitioned_dml(
        self,
        dml,
        params=None,
        param_types=None,
        query_options=None,
        request_options=None,
        exclude_txn_from_change_streams=False,
    ):
        """Execute a partitionable DML statement.

        :type dml: str
        :param dml: DML statement

        :type params: dict, {str -> column value}
        :param params: values for parameter replacement.  Keys must match
                       the names used in ``dml``.

        :type param_types: dict[str -> Union[dict, .types.Type]]
        :param param_types:
            (Optional) maps explicit types for one or more param values;
            required if parameters are passed.

        :type query_options:
            :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
            or :class:`dict`
        :param query_options:
                (Optional) Query optimizer configuration to use for the given query.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.QueryOptions`

        :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
        :param request_options:
            (Optional) Common options for this request.
            If a dict is provided, it must be of the same form as the protobuf
            message :class:`~google.cloud.spanner_v1.types.RequestOptions`.
            Please note, the `transactionTag` setting will be ignored as it is
            not supported for partitioned DML.

        :type exclude_txn_from_change_streams: bool
        :param exclude_txn_from_change_streams:
          (Optional) If true, instructs the transaction to be excluded from being recorded in change streams
          with the DDL option `allow_txn_exclusion=true`. This does not exclude the transaction from
          being recorded in the change streams with the DDL option `allow_txn_exclusion` being false or
          unset.

        :rtype: int
        :returns: Count of rows affected by the DML statement.
        """
        query_options = _merge_query_options(
            self._instance._client._query_options, query_options
        )
        if request_options is None:
            request_options = RequestOptions()
        elif type(request_options) is dict:
            request_options = RequestOptions(request_options)
        request_options.transaction_tag = None

        if params is not None:
            from google.cloud.spanner_v1.transaction import Transaction

            params_pb = Transaction._make_params_pb(params, param_types)
        else:
            params_pb = {}

        api = self.spanner_api

        txn_options = TransactionOptions(
            partitioned_dml=TransactionOptions.PartitionedDml(),
            exclude_txn_from_change_streams=exclude_txn_from_change_streams,
        )

        metadata = _metadata_with_prefix(self.name)
        if self._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(self._route_to_leader_enabled)
            )

        def execute_pdml():
            with trace_call(
                "CloudSpanner.Database.execute_partitioned_pdml",
                observability_options=self.observability_options,
            ) as span, MetricsCapture():
                transaction_type = TransactionType.PARTITIONED
                session = self._sessions_manager.get_session(transaction_type)

                try:
                    add_span_event(span, "Starting BeginTransaction")
                    txn = api.begin_transaction(
                        session=session.name,
                        options=txn_options,
                        metadata=self.metadata_with_request_id(
                            self._next_nth_request,
                            1,
                            metadata,
                            span,
                        ),
                    )

                    txn_selector = TransactionSelector(id=txn.id)

                    request = ExecuteSqlRequest(
                        session=session.name,
                        sql=dml,
                        params=params_pb,
                        param_types=param_types,
                        query_options=query_options,
                        request_options=request_options,
                    )

                    method = functools.partial(
                        api.execute_streaming_sql,
                        metadata=metadata,
                    )

                    iterator = _restart_on_unavailable(
                        method=method,
                        request=request,
                        trace_name="CloudSpanner.ExecuteStreamingSql",
                        session=session,
                        metadata=metadata,
                        transaction_selector=txn_selector,
                        observability_options=self.observability_options,
                        request_id_manager=self,
                    )

                    result_set = StreamedResultSet(iterator)
                    list(result_set)  # consume all partials

                    return result_set.stats.row_count_lower_bound
                finally:
                    self._sessions_manager.put_session(session)

        return _retry_on_aborted(execute_pdml, DEFAULT_RETRY_BACKOFF)()

    @property
    def _next_nth_request(self):
        if self._instance and self._instance._client:
            return self._instance._client._next_nth_request
        return 1

    @property
    def _nth_client_id(self):
        if self._instance and self._instance._client:
            return self._instance._client._nth_client_id
        return 0

    def session(self, labels=None, database_role=None):
        """Factory to create a session for this database.

        Deprecated. Sessions should be checked out indirectly using context
        managers or :meth:`~google.cloud.spanner_v1.database.Database.run_in_transaction`,
        rather than built directly from the database.

        :type labels: dict (str -> str) or None
        :param labels: (Optional) user-assigned labels for the session.

        :type database_role: str
        :param database_role: (Optional) user-assigned database_role for the session.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: a session bound to this database.
        """
        # If role is specified in param, then that role is used
        # instead.
        role = database_role or self._database_role
        is_multiplexed = False
        if self.sessions_manager._use_multiplexed(
            transaction_type=TransactionType.READ_ONLY
        ):
            is_multiplexed = True
        return Session(
            self, labels=labels, database_role=role, is_multiplexed=is_multiplexed
        )

    def snapshot(self, **kw):
        """Return an object which wraps a snapshot.

        The wrapper *must* be used as a context manager, with the snapshot
        as the value returned by the wrapper.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.v1#google.spanner.v1.TransactionOptions.ReadOnly

        :type kw: dict
        :param kw:
            Passed through to
            :class:`~google.cloud.spanner_v1.snapshot.Snapshot` constructor.

        :rtype: :class:`~google.cloud.spanner_v1.database.SnapshotCheckout`
        :returns: new wrapper
        """
        return SnapshotCheckout(self, **kw)

    def batch(
        self,
        request_options=None,
        max_commit_delay=None,
        exclude_txn_from_change_streams=False,
        isolation_level=TransactionOptions.IsolationLevel.ISOLATION_LEVEL_UNSPECIFIED,
        read_lock_mode=TransactionOptions.ReadWrite.ReadLockMode.READ_LOCK_MODE_UNSPECIFIED,
        **kw,
    ):
        """Return an object which wraps a batch.

        The wrapper *must* be used as a context manager, with the batch
        as the value returned by the wrapper.

        :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
        :param request_options:
                (Optional) Common options for the commit request.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.RequestOptions`.

        :type max_commit_delay: :class:`datetime.timedelta`
        :param max_commit_delay:
                (Optional) The amount of latency this request is willing to incur
                in order to improve throughput. Value must be between 0ms and
                500ms.

        :type exclude_txn_from_change_streams: bool
        :param exclude_txn_from_change_streams:
          (Optional) If true, instructs the transaction to be excluded from being recorded in change streams
          with the DDL option `allow_txn_exclusion=true`. This does not exclude the transaction from
          being recorded in the change streams with the DDL option `allow_txn_exclusion` being false or
          unset.

        :type isolation_level:
            :class:`google.cloud.spanner_v1.types.TransactionOptions.IsolationLevel`
        :param isolation_level:
                (Optional) Sets the isolation level for this transaction. This overrides any default isolation level set for the client.

        :type read_lock_mode:
            :class:`google.cloud.spanner_v1.types.TransactionOptions.ReadWrite.ReadLockMode`
        :param read_lock_mode:
                (Optional) Sets the read lock mode for this transaction. This overrides any default read lock mode set for the client.

        :rtype: :class:`~google.cloud.spanner_v1.database.BatchCheckout`
        :returns: new wrapper
        """

        return BatchCheckout(
            self,
            request_options,
            max_commit_delay,
            exclude_txn_from_change_streams,
            isolation_level,
            read_lock_mode,
            **kw,
        )

    def mutation_groups(self):
        """Return an object which wraps a mutation_group.

        The wrapper *must* be used as a context manager, with the mutation group
        as the value returned by the wrapper.

        :rtype: :class:`~google.cloud.spanner_v1.database.MutationGroupsCheckout`
        :returns: new wrapper
        """
        return MutationGroupsCheckout(self)

    def batch_snapshot(
        self,
        read_timestamp=None,
        exact_staleness=None,
        session_id=None,
        transaction_id=None,
    ):
        """Return an object which wraps a batch read / query.

        :type read_timestamp: :class:`datetime.datetime`
        :param read_timestamp: Execute all reads at the given timestamp.

        :type exact_staleness: :class:`datetime.timedelta`
        :param exact_staleness: Execute all reads at a timestamp that is
                                ``exact_staleness`` old.

        :type session_id: str
        :param session_id: id of the session used in transaction

        :type transaction_id: str
        :param transaction_id: id of the transaction

        :rtype: :class:`~google.cloud.spanner_v1.database.BatchSnapshot`
        :returns: new wrapper
        """
        return BatchSnapshot(
            self,
            read_timestamp=read_timestamp,
            exact_staleness=exact_staleness,
            session_id=session_id,
            transaction_id=transaction_id,
        )

    def run_in_transaction(self, func, *args, **kw):
        """Perform a unit of work in a transaction, retrying on abort.

        :type func: callable
        :param func: takes a required positional argument, the transaction,
                     and additional positional / keyword arguments as supplied
                     by the caller.

        :type args: tuple
        :param args: additional positional arguments to be passed to ``func``.

        :type kw: dict
        :param kw: (Optional) keyword arguments to be passed to ``func``.
                   If passed,
                   "timeout_secs" will be removed and used to
                   override the default retry timeout which defines maximum timestamp
                   to continue retrying the transaction.
                   "max_commit_delay" will be removed and used to set the
                   max_commit_delay for the request. Value must be between
                   0ms and 500ms.
                   "exclude_txn_from_change_streams" if true, instructs the transaction to be excluded
                   from being recorded in change streams with the DDL option `allow_txn_exclusion=true`.
                   This does not exclude the transaction from being recorded in the change streams with
                   the DDL option `allow_txn_exclusion` being false or unset.
                   "isolation_level" sets the isolation level for the transaction.
                   "read_lock_mode" sets the read lock mode for the transaction.

        :rtype: Any
        :returns: The return value of ``func``.

        :raises Exception:
            reraises any non-ABORT exceptions raised by ``func``.
        """
        observability_options = getattr(self, "observability_options", None)
        with trace_call(
            "CloudSpanner.Database.run_in_transaction",
            observability_options=observability_options,
        ), MetricsCapture():
            # Sanity check: Is there a transaction already running?
            # If there is, then raise a red flag. Otherwise, mark that this one
            # is running.
            if getattr(self._local, "transaction_running", False):
                raise RuntimeError("Spanner does not support nested transactions.")

            self._local.transaction_running = True

            # Check out a session and run the function in a transaction; once
            # done, flip the sanity check bit back and return the session.
            transaction_type = TransactionType.READ_WRITE
            session = self._sessions_manager.get_session(transaction_type)

            try:
                return session.run_in_transaction(func, *args, **kw)

            finally:
                self._local.transaction_running = False
                self._sessions_manager.put_session(session)

    def restore(self, source):
        """Restore from a backup to this database.

        :type source: :class:`~google.cloud.spanner_v1.backup.Backup`
        :param source: the path of the source being restored from.

        :rtype: :class:`~google.api_core.operation.Operation`
        :returns: a future used to poll the status of the create request
        :raises Conflict: if the database already exists
        :raises NotFound:
            if the instance owning the database does not exist, or
            if the backup being restored from does not exist
        :raises ValueError: if backup is not set
        """
        if source is None:
            raise ValueError("Restore source not specified")
        if type(self._encryption_config) is dict:
            self._encryption_config = RestoreDatabaseEncryptionConfig(
                **self._encryption_config
            )
        if (
            self.encryption_config
            and self.encryption_config.kms_key_name
            and self.encryption_config.encryption_type
            != RestoreDatabaseEncryptionConfig.EncryptionType.CUSTOMER_MANAGED_ENCRYPTION
        ):
            raise ValueError("kms_key_name only used with CUSTOMER_MANAGED_ENCRYPTION")
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        request = RestoreDatabaseRequest(
            parent=self._instance.name,
            database_id=self.database_id,
            backup=source.name,
            encryption_config=self._encryption_config or None,
        )
        future = api.restore_database(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        return future

    def is_ready(self):
        """Test whether this database is ready for use.

        :rtype: bool
        :returns: True if the database state is READY_OPTIMIZING or READY, else False.
        """
        return (
            self.state == DatabasePB.State.READY_OPTIMIZING
            or self.state == DatabasePB.State.READY
        )

    def is_optimized(self):
        """Test whether this database has finished optimizing.

        :rtype: bool
        :returns: True if the database state is READY, else False.
        """
        return self.state == DatabasePB.State.READY

    def list_database_operations(self, filter_="", page_size=None):
        """List database operations for the database.

        :type filter_: str
        :param filter_:
            Optional. A string specifying a filter for which database operations to list.

        :type page_size: int
        :param page_size:
            Optional. The maximum number of operations in each page of results from this
            request. Non-positive values are ignored. Defaults to a sensible value set
            by the API.

        :type: :class:`~google.api_core.page_iterator.Iterator`
        :returns:
            Iterator of :class:`~google.api_core.operation.Operation`
            resources within the current instance.
        """
        database_filter = _DATABASE_METADATA_FILTER.format(self.name)
        if filter_:
            database_filter = "({0}) AND ({1})".format(filter_, database_filter)
        return self._instance.list_database_operations(
            filter_=database_filter, page_size=page_size
        )

    def list_database_roles(self, page_size=None):
        """Lists Cloud Spanner database roles.

        :type page_size: int
        :param page_size:
            Optional. The maximum number of database roles in each page of results
            from this request. Non-positive values are ignored. Defaults to a
            sensible value set by the API.

        :type: Iterable
        :returns:
            Iterable of :class:`~google.cloud.spanner_admin_database_v1.types.spanner_database_admin.DatabaseRole`
            resources within the current database.
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        request = ListDatabaseRolesRequest(
            parent=self.name,
            page_size=page_size,
        )
        return api.list_database_roles(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )

    def table(self, table_id):
        """Factory to create a table object within this database.

        Note: This method does not create a table in Cloud Spanner, but it can
        be used to check if a table exists.

        .. code-block:: python

           my_table = database.table("my_table")
           if my_table.exists():
               print("Table with ID 'my_table' exists.")
           else:
               print("Table with ID 'my_table' does not exist.")

        :type table_id: str
        :param table_id: The ID of the table.

        :rtype: :class:`~google.cloud.spanner_v1.table.Table`
        :returns: a table owned by this database.
        """
        return Table(table_id, self)

    def list_tables(self, schema="_default"):
        """List tables within the database.

        :type schema: str
        :param schema: The schema to search for tables, or None for all schemas. Use the special string "_default" to
                       search for tables in the default schema of the database.

        :type: Iterable
        :returns:
            Iterable of :class:`~google.cloud.spanner_v1.table.Table`
            resources within the current database.
        """
        if "_default" == schema:
            schema = self.default_schema_name

        with self.snapshot() as snapshot:
            if schema is None:
                results = snapshot.execute_sql(
                    sql=_LIST_TABLES_QUERY.format(""),
                )
            else:
                if self._database_dialect == DatabaseDialect.POSTGRESQL:
                    where_clause = "WHERE TABLE_SCHEMA = $1"
                    param_name = "p1"
                else:
                    where_clause = (
                        "WHERE TABLE_SCHEMA = @schema AND SPANNER_STATE = 'COMMITTED'"
                    )
                    param_name = "schema"
                results = snapshot.execute_sql(
                    sql=_LIST_TABLES_QUERY.format(where_clause),
                    params={param_name: schema},
                    param_types={param_name: Type(code=TypeCode.STRING)},
                )
            for row in results:
                yield self.table(row[0])

    def get_iam_policy(self, policy_version=None):
        """Gets the access control policy for a database resource.

        :type policy_version: int
        :param policy_version:
            (Optional) the maximum policy version that will be
            used to format the policy. Valid values are 0, 1 ,3.

        :rtype: :class:`~google.iam.v1.policy_pb2.Policy`
        :returns:
            returns an Identity and Access Management (IAM) policy. It is used to
            specify access control policies for Cloud Platform
            resources.
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        request = iam_policy_pb2.GetIamPolicyRequest(
            resource=self.name,
            options=options_pb2.GetPolicyOptions(
                requested_policy_version=policy_version
            ),
        )
        response = api.get_iam_policy(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        return response

    def set_iam_policy(self, policy):
        """Sets the access control policy on a database resource.
        Replaces any existing policy.

        :type policy: :class:`~google.iam.v1.policy_pb2.Policy`
        :param policy_version:
            the complete policy to be applied to the resource.

        :rtype: :class:`~google.iam.v1.policy_pb2.Policy`
        :returns:
            returns the new Identity and Access Management (IAM) policy.
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        request = iam_policy_pb2.SetIamPolicyRequest(
            resource=self.name,
            policy=policy,
        )
        response = api.set_iam_policy(
            request=request,
            metadata=self.metadata_with_request_id(self._next_nth_request, 1, metadata),
        )
        return response

    @property
    def observability_options(self):
        """
        Returns the observability options that you set when creating
        the SpannerClient.
        """
        if not (self._instance and self._instance._client):
            return None

        opts = getattr(self._instance._client, "observability_options", None)
        if not opts:
            opts = dict()

        opts["db_name"] = self.name
        return opts

    @property
    def sessions_manager(self) -> DatabaseSessionsManager:
        """Returns the database sessions manager.

        :rtype: :class:`~google.cloud.spanner_v1.database_sessions_manager.DatabaseSessionsManager`
        :returns: The sessions manager for this database.
        """
        return self._sessions_manager


class BatchCheckout(object):
    """Context manager for using a batch from a database.

    Inside the context manager, checks out a session from the database,
    creates a batch from it, making the batch available.

    Caller must *not* use the batch to perform API requests outside the scope
    of the context manager.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: database to use

    :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
    :param request_options:
            (Optional) Common options for the commit request.
            If a dict is provided, it must be of the same form as the protobuf
            message :class:`~google.cloud.spanner_v1.types.RequestOptions`.

    :type max_commit_delay: :class:`datetime.timedelta`
    :param max_commit_delay:
            (Optional) The amount of latency this request is willing to incur
            in order to improve throughput.
    """

    def __init__(
        self,
        database,
        request_options=None,
        max_commit_delay=None,
        exclude_txn_from_change_streams=False,
        isolation_level=TransactionOptions.IsolationLevel.ISOLATION_LEVEL_UNSPECIFIED,
        read_lock_mode=TransactionOptions.ReadWrite.ReadLockMode.READ_LOCK_MODE_UNSPECIFIED,
        **kw,
    ):
        self._database: Database = database
        self._session: Optional[Session] = None
        self._batch: Optional[Batch] = None

        if request_options is None:
            self._request_options = RequestOptions()
        elif type(request_options) is dict:
            self._request_options = RequestOptions(request_options)
        else:
            self._request_options = request_options
        self._max_commit_delay = max_commit_delay
        self._exclude_txn_from_change_streams = exclude_txn_from_change_streams
        self._isolation_level = isolation_level
        self._read_lock_mode = read_lock_mode
        self._kw = kw

    def __enter__(self):
        """Begin ``with`` block."""

        # Batch transactions are performed as blind writes,
        # which are treated as read-only transactions.
        transaction_type = TransactionType.READ_ONLY
        self._session = self._database.sessions_manager.get_session(transaction_type)

        add_span_event(
            span=get_current_span(),
            event_name="Using session",
            event_attributes={"id": self._session.session_id},
        )

        batch = self._batch = Batch(session=self._session)
        if self._request_options.transaction_tag:
            batch.transaction_tag = self._request_options.transaction_tag

        return batch

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End ``with`` block."""
        try:
            if exc_type is None:
                self._batch.commit(
                    return_commit_stats=self._database.log_commit_stats,
                    request_options=self._request_options,
                    max_commit_delay=self._max_commit_delay,
                    exclude_txn_from_change_streams=self._exclude_txn_from_change_streams,
                    isolation_level=self._isolation_level,
                    read_lock_mode=self._read_lock_mode,
                    **self._kw,
                )
        finally:
            if self._database.log_commit_stats and self._batch.commit_stats:
                self._database.logger.info(
                    "CommitStats: {}".format(self._batch.commit_stats),
                    extra={"commit_stats": self._batch.commit_stats},
                )
            self._database.sessions_manager.put_session(self._session)
            current_span = get_current_span()
            add_span_event(
                current_span,
                "Returned session to pool",
                {"id": self._session.session_id},
            )


class MutationGroupsCheckout(object):
    """Context manager for using mutation groups from a database.

    Inside the context manager, checks out a session from the database,
    creates mutation groups from it, making the groups available.

    Caller must *not* use the object to perform API requests outside the scope
    of the context manager.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: database to use
    """

    def __init__(self, database):
        self._database: Database = database
        self._session: Optional[Session] = None

    def __enter__(self):
        """Begin ``with`` block."""
        transaction_type = TransactionType.READ_WRITE
        self._session = self._database.sessions_manager.get_session(transaction_type)

        return MutationGroups(session=self._session)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End ``with`` block."""
        if isinstance(exc_val, NotFound):
            # If NotFound exception occurs inside the with block
            # then we validate if the session still exists.
            if not self._session.exists():
                self._session = self._database._pool._new_session()
                self._session.create()
        self._database.sessions_manager.put_session(self._session)


class SnapshotCheckout(object):
    """Context manager for using a snapshot from a database.

    Inside the context manager, checks out a session from the database,
    creates a snapshot from it, making the snapshot available.

    Caller must *not* use the snapshot to perform API requests outside the
    scope of the context manager.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: database to use

    :type kw: dict
    :param kw:
        Passed through to
        :class:`~google.cloud.spanner_v1.snapshot.Snapshot` constructor.
    """

    def __init__(self, database, **kw):
        self._database: Database = database
        self._session: Optional[Session] = None
        self._kw: dict = kw

    def __enter__(self):
        """Begin ``with`` block."""
        transaction_type = TransactionType.READ_ONLY
        self._session = self._database.sessions_manager.get_session(transaction_type)

        return Snapshot(session=self._session, **self._kw)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End ``with`` block."""
        if isinstance(exc_val, NotFound):
            # If NotFound exception occurs inside the with block
            # then we validate if the session still exists.
            if not self._session.exists():
                self._session = self._database._pool._new_session()
                self._session.create()
        self._database.sessions_manager.put_session(self._session)


class BatchSnapshot(object):
    """Wrapper for generating and processing read / query batches.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: database to use

    :type read_timestamp: :class:`datetime.datetime`
    :param read_timestamp: Execute all reads at the given timestamp.

    :type exact_staleness: :class:`datetime.timedelta`
    :param exact_staleness: Execute all reads at a timestamp that is
                            ``exact_staleness`` old.
    """

    def __init__(
        self,
        database,
        read_timestamp=None,
        exact_staleness=None,
        session_id=None,
        transaction_id=None,
    ):
        self._database: Database = database

        self._session_id: Optional[str] = session_id
        self._transaction_id: Optional[bytes] = transaction_id

        self._session: Optional[Session] = None
        self._snapshot: Optional[Snapshot] = None

        self._read_timestamp = read_timestamp
        self._exact_staleness = exact_staleness

    @classmethod
    def from_dict(cls, database, mapping):
        """Reconstruct an instance from a mapping.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database to use

        :type mapping: mapping
        :param mapping: serialized state of the instance

        :rtype: :class:`BatchSnapshot`
        """

        instance = cls(database)

        session = instance._session = Session(database=database)
        instance._session_id = session._session_id = mapping["session_id"]

        snapshot = instance._snapshot = session.snapshot()
        instance._transaction_id = snapshot._transaction_id = mapping["transaction_id"]

        return instance

    def to_dict(self):
        """Return state as a dictionary.

        Result can be used to serialize the instance and reconstitute
        it later using :meth:`from_dict`.

        :rtype: dict
        """
        session = self._get_session()
        snapshot = self._get_snapshot()
        return {
            "session_id": session._session_id,
            "transaction_id": snapshot._transaction_id,
        }

    def __enter__(self):
        """Begin ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End ``with`` block."""
        self.close()

    @property
    def observability_options(self):
        return getattr(self._database, "observability_options", {})

    def _get_session(self):
        """Create session as needed.

        .. note::

           Caller is responsible for cleaning up the session after
           all partitions have been processed.
        """
        if self._session is None:
            database = self._database

            # If the session ID is not specified, check out a new session for
            # partitioned transactions from  the database session manager; otherwise,
            # the session has already been checked out, so just create a session to
            # represent it.
            if self._session_id is None:
                transaction_type = TransactionType.PARTITIONED
                session = database.sessions_manager.get_session(transaction_type)
                self._session_id = session.session_id

            else:
                session = Session(database=database)
                session._session_id = self._session_id

            self._session = session

        return self._session

    def _get_snapshot(self):
        """Create snapshot if needed."""

        if self._snapshot is None:
            self._snapshot = self._get_session().snapshot(
                read_timestamp=self._read_timestamp,
                exact_staleness=self._exact_staleness,
                multi_use=True,
                transaction_id=self._transaction_id,
            )

            if self._transaction_id is None:
                self._snapshot.begin()

        return self._snapshot

    def get_batch_transaction_id(self):
        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("Read-only transaction not begun")
        return BatchTransactionId(
            snapshot._transaction_id,
            snapshot._session.session_id,
            snapshot._read_timestamp,
        )

    def read(self, *args, **kw):
        """Convenience method:  perform read operation via snapshot.

        See :meth:`~google.cloud.spanner_v1.snapshot.Snapshot.read`.
        """
        return self._get_snapshot().read(*args, **kw)

    def execute_sql(self, *args, **kw):
        """Convenience method:  perform query operation via snapshot.

        See :meth:`~google.cloud.spanner_v1.snapshot.Snapshot.execute_sql`.
        """
        return self._get_snapshot().execute_sql(*args, **kw)

    def generate_read_batches(
        self,
        table,
        columns,
        keyset,
        index="",
        partition_size_bytes=None,
        max_partitions=None,
        data_boost_enabled=False,
        directed_read_options=None,
        *,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
    ):
        """Start a partitioned batch read operation.

        Uses the ``PartitionRead`` API request to initiate the partitioned
        read.  Returns a list of batch information needed to perform the
        actual reads.

        :type table: str
        :param table: name of the table from which to fetch data

        :type columns: list of str
        :param columns: names of columns to be retrieved

        :type keyset: :class:`~google.cloud.spanner_v1.keyset.KeySet`
        :param keyset: keys / ranges identifying rows to be retrieved

        :type index: str
        :param index: (Optional) name of index to use, rather than the
                      table's primary key

        :type partition_size_bytes: int
        :param partition_size_bytes:
            (Optional) desired size for each partition generated.  The service
            uses this as a hint, the actual partition size may differ.

        :type max_partitions: int
        :param max_partitions:
            (Optional) desired maximum number of partitions generated. The
            service uses this as a hint, the actual number of partitions may
            differ.

        :type data_boost_enabled:
        :param data_boost_enabled:
                (Optional) If this is for a partitioned read and this field is
                set ``true``, the request will be executed via offline access.

        :type directed_read_options: :class:`~google.cloud.spanner_v1.DirectedReadOptions`
            or :class:`dict`
        :param directed_read_options: (Optional) Request level option used to set the directed_read_options
            for ReadRequests that indicates which replicas
            or regions should be used for non-transactional reads.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :rtype: iterable of dict
        :returns:
            mappings of information used perform actual partitioned reads via
            :meth:`process_read_batch`.
        """
        with trace_call(
            f"CloudSpanner.{type(self).__name__}.generate_read_batches",
            extra_attributes=dict(table=table, columns=columns),
            observability_options=self.observability_options,
        ), MetricsCapture():
            partitions = self._get_snapshot().partition_read(
                table=table,
                columns=columns,
                keyset=keyset,
                index=index,
                partition_size_bytes=partition_size_bytes,
                max_partitions=max_partitions,
                retry=retry,
                timeout=timeout,
            )

            read_info = {
                "table": table,
                "columns": columns,
                "keyset": keyset._to_dict(),
                "index": index,
                "data_boost_enabled": data_boost_enabled,
                "directed_read_options": directed_read_options,
            }
            for partition in partitions:
                yield {"partition": partition, "read": read_info.copy()}

    def process_read_batch(
        self,
        batch,
        *,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
        lazy_decode=False,
    ):
        """Process a single, partitioned read.

        :type batch: mapping
        :param batch:
            one of the mappings returned from an earlier call to
            :meth:`generate_read_batches`.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :type lazy_decode: bool
        :param lazy_decode:
            (Optional) If this argument is set to ``true``, the iterator
            returns the underlying protobuf values instead of decoded Python
            objects. This reduces the time that is needed to iterate through
            large result sets. The application is responsible for decoding
            the data that is needed.


        :rtype: :class:`~google.cloud.spanner_v1.streamed.StreamedResultSet`
        :returns: a result set instance which can be used to consume rows.
        """
        observability_options = self.observability_options
        with trace_call(
            f"CloudSpanner.{type(self).__name__}.process_read_batch",
            observability_options=observability_options,
        ), MetricsCapture():
            kwargs = copy.deepcopy(batch["read"])
            keyset_dict = kwargs.pop("keyset")
            kwargs["keyset"] = KeySet._from_dict(keyset_dict)
            return self._get_snapshot().read(
                partition=batch["partition"], **kwargs, retry=retry, timeout=timeout
            )

    def generate_query_batches(
        self,
        sql,
        params=None,
        param_types=None,
        partition_size_bytes=None,
        max_partitions=None,
        query_options=None,
        data_boost_enabled=False,
        directed_read_options=None,
        *,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
    ):
        """Start a partitioned query operation.

        Uses the ``PartitionQuery`` API request to start a partitioned
        query operation.  Returns a list of batch information needed to
        perform the actual queries.

        :type sql: str
        :param sql: SQL query statement

        :type params: dict, {str -> column value}
        :param params: values for parameter replacement.  Keys must match
                       the names used in ``sql``.

        :type param_types: dict[str -> Union[dict, .types.Type]]
        :param param_types:
            (Optional) maps explicit types for one or more param values;
            required if parameters are passed.

        :type partition_size_bytes: int
        :param partition_size_bytes:
            (Optional) desired size for each partition generated.  The service
            uses this as a hint, the actual partition size may differ.

        :type max_partitions: int
        :param max_partitions:
            (Optional) desired maximum number of partitions generated. The
            service uses this as a hint, the actual number of partitions may
            differ.

        :type query_options:
            :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
            or :class:`dict`
        :param query_options:
                (Optional) Query optimizer configuration to use for the given query.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.QueryOptions`

        :type data_boost_enabled:
        :param data_boost_enabled:
                (Optional) If this is for a partitioned query and this field is
                set ``true``, the request will be executed via offline access.

        :type directed_read_options: :class:`~google.cloud.spanner_v1.DirectedReadOptions`
            or :class:`dict`
        :param directed_read_options: (Optional) Request level option used to set the directed_read_options
            for ExecuteSqlRequests that indicates which replicas
            or regions should be used for non-transactional queries.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :rtype: iterable of dict
        :returns:
            mappings of information used perform actual partitioned reads via
            :meth:`process_read_batch`.
        """
        with trace_call(
            f"CloudSpanner.{type(self).__name__}.generate_query_batches",
            extra_attributes=dict(sql=sql),
            observability_options=self.observability_options,
        ), MetricsCapture():
            partitions = self._get_snapshot().partition_query(
                sql=sql,
                params=params,
                param_types=param_types,
                partition_size_bytes=partition_size_bytes,
                max_partitions=max_partitions,
                retry=retry,
                timeout=timeout,
            )

            query_info = {
                "sql": sql,
                "data_boost_enabled": data_boost_enabled,
                "directed_read_options": directed_read_options,
            }
            if params:
                query_info["params"] = params
                query_info["param_types"] = param_types

            # Query-level options have higher precedence than client-level and
            # environment-level options
            default_query_options = self._database._instance._client._query_options
            query_info["query_options"] = _merge_query_options(
                default_query_options, query_options
            )

            for partition in partitions:
                yield {"partition": partition, "query": query_info}

    def process_query_batch(
        self,
        batch,
        *,
        lazy_decode: bool = False,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
    ):
        """Process a single, partitioned query.

        :type batch: mapping
        :param batch:
            one of the mappings returned from an earlier call to
            :meth:`generate_query_batches`.

        :type lazy_decode: bool
        :param lazy_decode:
            (Optional) If this argument is set to ``true``, the iterator
            returns the underlying protobuf values instead of decoded Python
            objects. This reduces the time that is needed to iterate through
            large result sets.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :rtype: :class:`~google.cloud.spanner_v1.streamed.StreamedResultSet`
        :returns: a result set instance which can be used to consume rows.
        """
        with trace_call(
            f"CloudSpanner.{type(self).__name__}.process_query_batch",
            observability_options=self.observability_options,
        ), MetricsCapture():
            return self._get_snapshot().execute_sql(
                partition=batch["partition"],
                **batch["query"],
                lazy_decode=lazy_decode,
                retry=retry,
                timeout=timeout,
            )

    def run_partitioned_query(
        self,
        sql,
        params=None,
        param_types=None,
        partition_size_bytes=None,
        max_partitions=None,
        query_options=None,
        data_boost_enabled=False,
        lazy_decode=False,
    ):
        """Start a partitioned query operation to get list of partitions and
        then executes each partition on a separate thread

        :type sql: str
        :param sql: SQL query statement

        :type params: dict, {str -> column value}
        :param params: values for parameter replacement.  Keys must match
                       the names used in ``sql``.

        :type param_types: dict[str -> Union[dict, .types.Type]]
        :param param_types:
            (Optional) maps explicit types for one or more param values;
            required if parameters are passed.

        :type partition_size_bytes: int
        :param partition_size_bytes:
            (Optional) desired size for each partition generated.  The service
            uses this as a hint, the actual partition size may differ.

        :type max_partitions: int
        :param max_partitions:
            (Optional) desired maximum number of partitions generated. The
            service uses this as a hint, the actual number of partitions may
            differ.

        :type query_options:
            :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
            or :class:`dict`
        :param query_options:
                (Optional) Query optimizer configuration to use for the given query.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.QueryOptions`

        :type data_boost_enabled:
        :param data_boost_enabled:
                (Optional) If this is for a partitioned query and this field is
                set ``true``, the request will be executed using data boost.
                Please see https://cloud.google.com/spanner/docs/databoost/databoost-overview

        :rtype: :class:`~google.cloud.spanner_v1.merged_result_set.MergedResultSet`
        :returns: a result set instance which can be used to consume rows.
        """
        with trace_call(
            f"CloudSpanner.${type(self).__name__}.run_partitioned_query",
            extra_attributes=dict(sql=sql),
            observability_options=self.observability_options,
        ), MetricsCapture():
            partitions = list(
                self.generate_query_batches(
                    sql,
                    params,
                    param_types,
                    partition_size_bytes,
                    max_partitions,
                    query_options,
                    data_boost_enabled,
                )
            )
            return MergedResultSet(self, partitions, 0, lazy_decode=lazy_decode)

    def process(self, batch):
        """Process a single, partitioned query or read.

        :type batch: mapping
        :param batch:
            one of the mappings returned from an earlier call to
            :meth:`generate_query_batches`.

        :rtype: :class:`~google.cloud.spanner_v1.streamed.StreamedResultSet`
        :returns: a result set instance which can be used to consume rows.
        :raises ValueError: if batch does not contain either 'read' or 'query'
        """
        if "query" in batch:
            return self.process_query_batch(batch)
        if "read" in batch:
            return self.process_read_batch(batch)
        raise ValueError("Invalid batch")

    def close(self):
        """Clean up underlying session.

        .. note::

           If the transaction has been shared across multiple machines,
           calling this on any machine would invalidate the transaction
           everywhere. Ideally this would be called when data has been read
           from all the partitions.
        """
        if self._session is not None:
            if not self._session.is_multiplexed:
                self._session.delete()


def _check_ddl_statements(value):
    """Validate DDL Statements used to define database schema.

    See
    https://cloud.google.com/spanner/docs/data-definition-language

    :type value: list of string
    :param value: DDL statements, excluding the 'CREATE DATABASE' statement

    :rtype: tuple
    :returns: tuple of validated DDL statement strings.
    :raises ValueError:
        if elements in ``value`` are not strings, or if ``value`` contains
        a ``CREATE DATABASE`` statement.
    """
    if not all(isinstance(line, str) for line in value):
        raise ValueError("Pass a list of strings")

    if any("create database" in line.lower() for line in value):
        raise ValueError("Do not pass a 'CREATE DATABASE' statement")

    return tuple(value)


def _retry_on_aborted(func, retry_config):
    """Helper for :meth:`Database.execute_partitioned_dml`.

    Wrap function in a Retry that will retry on Aborted exceptions
    with the retry config specified.

    :type func: callable
    :param func: the function to be retried on Aborted exceptions

    :type retry_config: Retry
    :param retry_config: retry object with the settings to be used
    """
    retry = retry_config.with_predicate(if_exception_type(Aborted))
    return retry(func)

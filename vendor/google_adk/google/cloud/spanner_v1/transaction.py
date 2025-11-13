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

"""Spanner read-write transaction support."""
import functools
from google.protobuf.struct_pb2 import Struct
from typing import Optional

from google.cloud.spanner_v1._helpers import (
    _make_value_pb,
    _merge_query_options,
    _metadata_with_prefix,
    _metadata_with_leader_aware_routing,
    _retry,
    _check_rst_stream_error,
    _merge_Transaction_Options,
)
from google.cloud.spanner_v1 import (
    CommitRequest,
    CommitResponse,
    ResultSet,
    ExecuteBatchDmlResponse,
    Mutation,
)
from google.cloud.spanner_v1 import ExecuteBatchDmlRequest
from google.cloud.spanner_v1 import ExecuteSqlRequest
from google.cloud.spanner_v1 import TransactionOptions
from google.cloud.spanner_v1._helpers import AtomicCounter
from google.cloud.spanner_v1.snapshot import _SnapshotBase
from google.cloud.spanner_v1.batch import _BatchBase
from google.cloud.spanner_v1._opentelemetry_tracing import add_span_event, trace_call
from google.cloud.spanner_v1 import RequestOptions
from google.cloud.spanner_v1.metrics.metrics_capture import MetricsCapture
from google.api_core import gapic_v1
from google.api_core.exceptions import InternalServerError
from dataclasses import dataclass, field
from typing import Any


class Transaction(_SnapshotBase, _BatchBase):
    """Implement read-write transaction semantics for a session.

    :type session: :class:`~google.cloud.spanner_v1.session.Session`
    :param session: the session used to perform the commit

    :raises ValueError: if session has an existing transaction
    """

    exclude_txn_from_change_streams: bool = False
    isolation_level: TransactionOptions.IsolationLevel = (
        TransactionOptions.IsolationLevel.ISOLATION_LEVEL_UNSPECIFIED
    )
    read_lock_mode: TransactionOptions.ReadWrite.ReadLockMode = (
        TransactionOptions.ReadWrite.ReadLockMode.READ_LOCK_MODE_UNSPECIFIED
    )

    # Override defaults from _SnapshotBase.
    _multi_use: bool = True
    _read_only: bool = False

    def __init__(self, session):
        super(Transaction, self).__init__(session)
        self.rolled_back: bool = False

        # If this transaction is used to retry a previous aborted transaction with a
        # multiplexed session, the identifier for that transaction is used to increase
        # the lock order of the new transaction (see :meth:`_build_transaction_options_pb`).
        # This attribute should only be set by :meth:`~google.cloud.spanner_v1.session.Session.run_in_transaction`.
        self._multiplexed_session_previous_transaction_id: Optional[bytes] = None

    def _build_transaction_options_pb(self) -> TransactionOptions:
        """Builds and returns transaction options for this transaction.

        :rtype: :class:`~.transaction_pb2.TransactionOptions`
        :returns: transaction options for this transaction.
        """

        default_transaction_options = (
            self._session._database.default_transaction_options.default_read_write_transaction_options
        )

        merge_transaction_options = TransactionOptions(
            read_write=TransactionOptions.ReadWrite(
                multiplexed_session_previous_transaction_id=self._multiplexed_session_previous_transaction_id,
                read_lock_mode=self.read_lock_mode,
            ),
            exclude_txn_from_change_streams=self.exclude_txn_from_change_streams,
            isolation_level=self.isolation_level,
        )

        return _merge_Transaction_Options(
            defaultTransactionOptions=default_transaction_options,
            mergeTransactionOptions=merge_transaction_options,
        )

    def _execute_request(
        self,
        method,
        request,
        metadata,
        trace_name=None,
        attributes=None,
    ):
        """Helper method to execute request after fetching transaction selector.

        :type method: callable
        :param method: function returning iterator

        :type request: proto
        :param request: request proto to call the method with

        :raises: ValueError: if the transaction is not ready to update.
        """

        if self.committed is not None:
            raise ValueError("Transaction already committed.")
        if self.rolled_back:
            raise ValueError("Transaction already rolled back.")

        session = self._session
        transaction = self._build_transaction_selector_pb()
        request.transaction = transaction

        with trace_call(
            trace_name,
            session,
            attributes,
            observability_options=getattr(
                session._database, "observability_options", None
            ),
            metadata=metadata,
        ), MetricsCapture():
            method = functools.partial(method, request=request)
            response = _retry(
                method,
                allowed_exceptions={InternalServerError: _check_rst_stream_error},
            )

        return response

    def rollback(self) -> None:
        """Roll back a transaction on the database.

        :raises: ValueError: if the transaction is not ready to roll back.
        """

        if self.committed is not None:
            raise ValueError("Transaction already committed.")
        if self.rolled_back:
            raise ValueError("Transaction already rolled back.")

        if self._transaction_id is not None:
            session = self._session
            database = session._database
            api = database.spanner_api

            metadata = _metadata_with_prefix(database.name)
            if database._route_to_leader_enabled:
                metadata.append(
                    _metadata_with_leader_aware_routing(
                        database._route_to_leader_enabled
                    )
                )

            observability_options = getattr(database, "observability_options", None)
            with trace_call(
                f"CloudSpanner.{type(self).__name__}.rollback",
                session,
                observability_options=observability_options,
                metadata=metadata,
            ) as span, MetricsCapture():
                attempt = AtomicCounter(0)
                nth_request = database._next_nth_request

                def wrapped_method(*args, **kwargs):
                    attempt.increment()
                    rollback_method = functools.partial(
                        api.rollback,
                        session=session.name,
                        transaction_id=self._transaction_id,
                        metadata=database.metadata_with_request_id(
                            nth_request,
                            attempt.value,
                            metadata,
                            span,
                        ),
                    )
                    return rollback_method(*args, **kwargs)

                _retry(
                    wrapped_method,
                    allowed_exceptions={InternalServerError: _check_rst_stream_error},
                )

        self.rolled_back = True

    def commit(
        self, return_commit_stats=False, request_options=None, max_commit_delay=None
    ):
        """Commit mutations to the database.

        :type return_commit_stats: bool
        :param return_commit_stats:
          If true, the response will return commit stats which can be accessed though commit_stats.

        :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
        :param request_options:
                (Optional) Common options for this request.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.RequestOptions`.

        :type max_commit_delay: :class:`datetime.timedelta`
        :param max_commit_delay:
                (Optional) The amount of latency this request is willing to incur
                in order to improve throughput.
                :class:`~google.cloud.spanner_v1.types.MaxCommitDelay`.

        :rtype: datetime
        :returns: timestamp of the committed changes.

        :raises: ValueError: if the transaction is not ready to commit.
        """

        mutations = self._mutations
        num_mutations = len(mutations)

        session = self._session
        database = session._database
        api = database.spanner_api

        metadata = _metadata_with_prefix(database.name)
        if database._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(database._route_to_leader_enabled)
            )

        with trace_call(
            name=f"CloudSpanner.{type(self).__name__}.commit",
            session=session,
            extra_attributes={"num_mutations": num_mutations},
            observability_options=getattr(database, "observability_options", None),
            metadata=metadata,
        ) as span, MetricsCapture():
            if self.committed is not None:
                raise ValueError("Transaction already committed.")
            if self.rolled_back:
                raise ValueError("Transaction already rolled back.")

            if self._transaction_id is None:
                if num_mutations > 0:
                    self._begin_mutations_only_transaction()
                else:
                    raise ValueError("Transaction has not begun.")

            if request_options is None:
                request_options = RequestOptions()
            elif type(request_options) is dict:
                request_options = RequestOptions(request_options)
            if self.transaction_tag is not None:
                request_options.transaction_tag = self.transaction_tag

            # Request tags are not supported for commit requests.
            request_options.request_tag = None

            common_commit_request_args = {
                "session": session.name,
                "transaction_id": self._transaction_id,
                "return_commit_stats": return_commit_stats,
                "max_commit_delay": max_commit_delay,
                "request_options": request_options,
            }

            add_span_event(span, "Starting Commit")

            attempt = AtomicCounter(0)
            nth_request = database._next_nth_request

            def wrapped_method(*args, **kwargs):
                attempt.increment()
                commit_request_args = {
                    "mutations": mutations,
                    **common_commit_request_args,
                }
                # Check if session is multiplexed (safely handle mock sessions)
                is_multiplexed = getattr(self._session, "is_multiplexed", False)
                if is_multiplexed and self._precommit_token is not None:
                    commit_request_args["precommit_token"] = self._precommit_token

                commit_method = functools.partial(
                    api.commit,
                    request=CommitRequest(**commit_request_args),
                    metadata=database.metadata_with_request_id(
                        nth_request,
                        attempt.value,
                        metadata,
                        span,
                    ),
                )
                return commit_method(*args, **kwargs)

            commit_retry_event_name = "Transaction Commit Attempt Failed. Retrying"

            def before_next_retry(nth_retry, delay_in_seconds):
                add_span_event(
                    span=span,
                    event_name=commit_retry_event_name,
                    event_attributes={
                        "attempt": nth_retry,
                        "sleep_seconds": delay_in_seconds,
                    },
                )

            commit_response_pb: CommitResponse = _retry(
                wrapped_method,
                allowed_exceptions={InternalServerError: _check_rst_stream_error},
                before_next_retry=before_next_retry,
            )

            # If the response contains a precommit token, the transaction did not
            # successfully commit, and must be retried with the new precommit token.
            # The mutations should not be included in the new request, and no further
            # retries or exception handling should be performed.
            if commit_response_pb._pb.HasField("precommit_token"):
                add_span_event(span, commit_retry_event_name)
                nth_request = database._next_nth_request
                commit_response_pb = api.commit(
                    request=CommitRequest(
                        precommit_token=commit_response_pb.precommit_token,
                        **common_commit_request_args,
                    ),
                    metadata=database.metadata_with_request_id(
                        nth_request,
                        1,
                        metadata,
                        span,
                    ),
                )

            add_span_event(span, "Commit Done")

        self.committed = commit_response_pb.commit_timestamp
        if return_commit_stats:
            self.commit_stats = commit_response_pb.commit_stats

        return self.committed

    @staticmethod
    def _make_params_pb(params, param_types):
        """Helper for :meth:`execute_update`.

        :type params: dict, {str -> column value}
        :param params: values for parameter replacement.  Keys must match
                       the names used in ``dml``.

        :type param_types: dict[str -> Union[dict, .types.Type]]
        :param param_types:
            (Optional) maps explicit types for one or more param values;
            required if parameters are passed.

        :rtype: Union[None, :class:`Struct`]
        :returns: a struct message for the passed params, or None
        :raises ValueError:
            If ``param_types`` is None but ``params`` is not None.
        :raises ValueError:
            If ``params`` is None but ``param_types`` is not None.
        """
        if params:
            return Struct(
                fields={key: _make_value_pb(value) for key, value in params.items()}
            )

        return {}

    def execute_update(
        self,
        dml,
        params=None,
        param_types=None,
        query_mode=None,
        query_options=None,
        request_options=None,
        last_statement=False,
        *,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
    ):
        """Perform an ``ExecuteSql`` API request with DML.

        :type dml: str
        :param dml: SQL DML statement

        :type params: dict, {str -> column value}
        :param params: values for parameter replacement.  Keys must match
                       the names used in ``dml``.

        :type param_types: dict[str -> Union[dict, .types.Type]]
        :param param_types:
            (Optional) maps explicit types for one or more param values;
            required if parameters are passed.

        :type query_mode:
            :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryMode`
        :param query_mode: Mode governing return of results / query plan.
            See:
            `QueryMode <https://cloud.google.com/spanner/reference/rpc/google.spanner.v1#google.spanner.v1.ExecuteSqlRequest.QueryMode>`_.

        :type query_options:
            :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
            or :class:`dict`
        :param query_options: (Optional) Options that are provided for query plan stability.

        :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
        :param request_options:
                (Optional) Common options for this request.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.RequestOptions`.

        :type last_statement: bool
        :param last_statement:
                If set to true, this option marks the end of the transaction. The
                transaction should be committed or aborted after this statement
                executes, and attempts to execute any other requests against this
                transaction (including reads and queries) will be rejected. Mixing
                mutations with statements that are marked as the last statement is
                not allowed.
                For DML statements, setting this option may cause some error
                reporting to be deferred until commit time (e.g. validation of
                unique constraints). Given this, successful execution of a DML
                statement should not be assumed until the transaction commits.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :rtype: int
        :returns: Count of rows affected by the DML statement.
        """

        session = self._session
        database = session._database
        api = database.spanner_api

        params_pb = self._make_params_pb(params, param_types)

        metadata = _metadata_with_prefix(database.name)
        if database._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(database._route_to_leader_enabled)
            )

        seqno, self._execute_sql_request_count = (
            self._execute_sql_request_count,
            self._execute_sql_request_count + 1,
        )

        # Query-level options have higher precedence than client-level and
        # environment-level options
        default_query_options = database._instance._client._query_options
        query_options = _merge_query_options(default_query_options, query_options)

        if request_options is None:
            request_options = RequestOptions()
        elif type(request_options) is dict:
            request_options = RequestOptions(request_options)
        request_options.transaction_tag = self.transaction_tag

        trace_attributes = {"db.statement": dml}

        # If this request begins the transaction, we need to lock
        # the transaction until the transaction ID is updated.
        is_inline_begin = False

        if self._transaction_id is None:
            is_inline_begin = True
            self._lock.acquire()

        execute_sql_request = ExecuteSqlRequest(
            session=session.name,
            transaction=self._build_transaction_selector_pb(),
            sql=dml,
            params=params_pb,
            param_types=param_types,
            query_mode=query_mode,
            query_options=query_options,
            seqno=seqno,
            request_options=request_options,
            last_statement=last_statement,
        )

        nth_request = database._next_nth_request
        attempt = AtomicCounter(0)

        def wrapped_method(*args, **kwargs):
            attempt.increment()
            execute_sql_method = functools.partial(
                api.execute_sql,
                request=execute_sql_request,
                metadata=database.metadata_with_request_id(
                    nth_request, attempt.value, metadata
                ),
                retry=retry,
                timeout=timeout,
            )
            return execute_sql_method(*args, **kwargs)

        result_set_pb: ResultSet = self._execute_request(
            wrapped_method,
            execute_sql_request,
            metadata,
            f"CloudSpanner.{type(self).__name__}.execute_update",
            trace_attributes,
        )

        self._update_for_result_set_pb(result_set_pb)

        if is_inline_begin:
            self._lock.release()

        if result_set_pb._pb.HasField("precommit_token"):
            self._update_for_precommit_token_pb(result_set_pb.precommit_token)

        return result_set_pb.stats.row_count_exact

    def batch_update(
        self,
        statements,
        request_options=None,
        last_statement=False,
        *,
        retry=gapic_v1.method.DEFAULT,
        timeout=gapic_v1.method.DEFAULT,
    ):
        """Perform a batch of DML statements via an ``ExecuteBatchDml`` request.

        :type statements:
            Sequence[Union[ str, Tuple[str, Dict[str, Any], Dict[str, Union[dict, .types.Type]]]]]

        :param statements:
            List of DML statements, with optional params / param types.
            If passed, 'params' is a dict mapping names to the values
            for parameter replacement.  Keys must match the names used in the
            corresponding DML statement.  If 'params' is passed, 'param_types'
            must also be passed, as a dict mapping names to the type of
            value passed in 'params'.

        :type request_options:
            :class:`google.cloud.spanner_v1.types.RequestOptions`
        :param request_options:
                (Optional) Common options for this request.
                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.spanner_v1.types.RequestOptions`.

        :type last_statement: bool
        :param last_statement:
                If set to true, this option marks the end of the transaction. The
                transaction should be committed or aborted after this statement
                executes, and attempts to execute any other requests against this
                transaction (including reads and queries) will be rejected. Mixing
                mutations with statements that are marked as the last statement is
                not allowed.
                For DML statements, setting this option may cause some error
                reporting to be deferred until commit time (e.g. validation of
                unique constraints). Given this, successful execution of a DML
                statement should not be assumed until the transaction commits.

        :type retry: :class:`~google.api_core.retry.Retry`
        :param retry: (Optional) The retry settings for this request.

        :type timeout: float
        :param timeout: (Optional) The timeout for this request.

        :rtype:
            Tuple(status, Sequence[int])
        :returns:
            Status code, plus counts of rows affected by each completed DML
            statement.  Note that if the status code is not ``OK``, the
            statement triggering the error will not have an entry in the
            list, nor will any statements following that one.
        """

        session = self._session
        database = session._database
        api = database.spanner_api

        parsed = []
        for statement in statements:
            if isinstance(statement, str):
                parsed.append(ExecuteBatchDmlRequest.Statement(sql=statement))
            else:
                dml, params, param_types = statement
                params_pb = self._make_params_pb(params, param_types)
                parsed.append(
                    ExecuteBatchDmlRequest.Statement(
                        sql=dml, params=params_pb, param_types=param_types
                    )
                )

        metadata = _metadata_with_prefix(database.name)
        if database._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(database._route_to_leader_enabled)
            )

        seqno, self._execute_sql_request_count = (
            self._execute_sql_request_count,
            self._execute_sql_request_count + 1,
        )

        if request_options is None:
            request_options = RequestOptions()
        elif type(request_options) is dict:
            request_options = RequestOptions(request_options)
        request_options.transaction_tag = self.transaction_tag

        trace_attributes = {
            # Get just the queries from the DML statement batch
            "db.statement": ";".join([statement.sql for statement in parsed])
        }

        # If this request begins the transaction, we need to lock
        # the transaction until the transaction ID is updated.
        is_inline_begin = False

        if self._transaction_id is None:
            is_inline_begin = True
            self._lock.acquire()

        execute_batch_dml_request = ExecuteBatchDmlRequest(
            session=session.name,
            transaction=self._build_transaction_selector_pb(),
            statements=parsed,
            seqno=seqno,
            request_options=request_options,
            last_statements=last_statement,
        )

        nth_request = database._next_nth_request
        attempt = AtomicCounter(0)

        def wrapped_method(*args, **kwargs):
            attempt.increment()
            execute_batch_dml_method = functools.partial(
                api.execute_batch_dml,
                request=execute_batch_dml_request,
                metadata=database.metadata_with_request_id(
                    nth_request, attempt.value, metadata
                ),
                retry=retry,
                timeout=timeout,
            )
            return execute_batch_dml_method(*args, **kwargs)

        response_pb: ExecuteBatchDmlResponse = self._execute_request(
            wrapped_method,
            execute_batch_dml_request,
            metadata,
            "CloudSpanner.DMLTransaction",
            trace_attributes,
        )

        self._update_for_execute_batch_dml_response_pb(response_pb)

        if is_inline_begin:
            self._lock.release()

        if (
            len(response_pb.result_sets) > 0
            and response_pb.result_sets[0].precommit_token
        ):
            self._update_for_precommit_token_pb(
                response_pb.result_sets[0].precommit_token
            )

        row_counts = [
            result_set.stats.row_count_exact for result_set in response_pb.result_sets
        ]

        return response_pb.status, row_counts

    def _begin_transaction(self, mutation: Mutation = None) -> bytes:
        """Begins a transaction on the database.

        :type mutation: :class:`~google.cloud.spanner_v1.mutation.Mutation`
        :param mutation: (Optional) Mutation to include in the begin transaction
            request. Required for mutation-only transactions with multiplexed sessions.

        :rtype: bytes
        :returns: identifier for the transaction.

        :raises ValueError: if the transaction has already begun or is single-use.
        """

        if self.committed is not None:
            raise ValueError("Transaction is already committed")
        if self.rolled_back:
            raise ValueError("Transaction is already rolled back")

        return super(Transaction, self)._begin_transaction(mutation=mutation)

    def _begin_mutations_only_transaction(self) -> None:
        """Begins a mutations-only transaction on the database."""

        mutation = self._get_mutation_for_begin_mutations_only_transaction()
        self._begin_transaction(mutation=mutation)

    def _get_mutation_for_begin_mutations_only_transaction(self) -> Optional[Mutation]:
        """Returns a mutation to use for beginning a mutations-only transaction.
        Returns None if a mutation does not need to be included.

        :rtype: :class:`~google.cloud.spanner_v1.types.Mutation`
        :returns: A mutation to use for beginning a mutations-only transaction.
        """

        # A mutation only needs to be included
        # for transaction with multiplexed sessions.
        if not self._session.is_multiplexed:
            return None

        mutations: list[Mutation] = self._mutations

        # If there are multiple mutations, select the mutation as follows:
        #   1. Choose a delete, update, or replace mutation instead
        #      of an insert mutation (since inserts could involve an auto-
        #      generated column and the client doesn't have that information).
        #   2. If there are no delete, update, or replace mutations, choose
        #      the insert mutation that includes the largest number of values.

        insert_mutation: Mutation = None
        max_insert_values: int = -1

        for mut in mutations:
            if mut.insert:
                num_values = len(mut.insert.values)
                if num_values > max_insert_values:
                    insert_mutation = mut
                    max_insert_values = num_values
            else:
                return mut

        return insert_mutation

    def _update_for_execute_batch_dml_response_pb(
        self, response_pb: ExecuteBatchDmlResponse
    ) -> None:
        """Update the transaction for the given execute batch DML response.

        :type response_pb: :class:`~google.cloud.spanner_v1.types.ExecuteBatchDmlResponse`
        :param response_pb: The execute batch DML response to update the transaction with.
        """
        # Only the first result set contains the result set metadata.
        if len(response_pb.result_sets) > 0:
            self._update_for_result_set_pb(response_pb.result_sets[0])

    def __enter__(self):
        """Begin ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End ``with`` block."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


@dataclass
class BatchTransactionId:
    transaction_id: str
    session_id: str
    read_timestamp: Any


@dataclass
class DefaultTransactionOptions:
    isolation_level: str = TransactionOptions.IsolationLevel.ISOLATION_LEVEL_UNSPECIFIED
    read_lock_mode: str = (
        TransactionOptions.ReadWrite.ReadLockMode.READ_LOCK_MODE_UNSPECIFIED
    )
    _defaultReadWriteTransactionOptions: Optional[TransactionOptions] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        """Initialize _defaultReadWriteTransactionOptions automatically"""
        self._defaultReadWriteTransactionOptions = TransactionOptions(
            read_write=TransactionOptions.ReadWrite(
                read_lock_mode=self.read_lock_mode,
            ),
            isolation_level=self.isolation_level,
        )

    @property
    def default_read_write_transaction_options(self) -> TransactionOptions:
        """Public accessor for _defaultReadWriteTransactionOptions"""
        return self._defaultReadWriteTransactionOptions

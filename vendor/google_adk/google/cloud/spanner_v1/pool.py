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

"""Pools managing shared Session objects."""

import datetime
import queue
import time

from google.cloud.exceptions import NotFound
from google.cloud.spanner_v1 import BatchCreateSessionsRequest
from google.cloud.spanner_v1 import Session as SessionProto
from google.cloud.spanner_v1.session import Session
from google.cloud.spanner_v1._helpers import (
    _metadata_with_prefix,
    _metadata_with_leader_aware_routing,
)
from google.cloud.spanner_v1._opentelemetry_tracing import (
    add_span_event,
    get_current_span,
    trace_call,
)
from warnings import warn

from google.cloud.spanner_v1.metrics.metrics_capture import MetricsCapture

_NOW = datetime.datetime.utcnow  # unit tests may replace


class AbstractSessionPool(object):
    """Specifies required API for concrete session pool implementations.

    :type labels: dict (str -> str) or None
    :param labels: (Optional) user-assigned labels for sessions created
                    by the pool.

    :type database_role: str
    :param database_role: (Optional) user-assigned database_role for the session.
    """

    _database = None

    def __init__(self, labels=None, database_role=None):
        if labels is None:
            labels = {}
        self._labels = labels
        self._database_role = database_role

    @property
    def labels(self):
        """User-assigned labels for sessions created by the pool.

        :rtype: dict (str -> str)
        :returns: labels assigned by the user
        """
        return self._labels

    @property
    def database_role(self):
        """User-assigned database_role for sessions created by the pool.

        :rtype: str
        :returns: database_role assigned by the user
        """
        return self._database_role

    def bind(self, database):
        """Associate the pool with a database.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database used by the pool to create sessions
                         when needed.

        Concrete implementations of this method may pre-fill the pool
        using the database.

        :raises NotImplementedError: abstract method
        """
        raise NotImplementedError()

    def get(self):
        """Check a session out from the pool.

        Concrete implementations of this method are allowed to raise an
        error to signal that the pool is exhausted, or to block until a
        session is available.

        :raises NotImplementedError: abstract method
        """
        raise NotImplementedError()

    def put(self, session):
        """Return a session to the pool.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: the session being returned.

        Concrete implementations of this method are allowed to raise an
        error to signal that the pool is full, or to block until it is
        not full.

        :raises NotImplementedError: abstract method
        """
        raise NotImplementedError()

    def clear(self):
        """Delete all sessions in the pool.

        Concrete implementations of this method are allowed to raise an
        error to signal that the pool is full, or to block until it is
        not full.

        :raises NotImplementedError: abstract method
        """
        raise NotImplementedError()

    def _new_session(self):
        """Helper for concrete methods creating session instances.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: new session instance.
        """

        role = self.database_role or self._database.database_role
        return Session(database=self._database, labels=self.labels, database_role=role)

    def session(self, **kwargs):
        """Check out a session from the pool.

        Deprecated. Sessions should be checked out indirectly using context
        managers or :meth:`~google.cloud.spanner_v1.database.Database.run_in_transaction`,
        rather than checked out directly from the pool.

        :param kwargs: (optional) keyword arguments, passed through to
                       the returned checkout.

        :rtype: :class:`~google.cloud.spanner_v1.session.SessionCheckout`
        :returns: a checkout instance, to be used as a context manager for
                  accessing the session and returning it to the pool.
        """
        return SessionCheckout(self, **kwargs)


class FixedSizePool(AbstractSessionPool):
    """Concrete session pool implementation:

    - Pre-allocates / creates a fixed number of sessions.

    - "Pings" existing sessions via :meth:`session.exists` before returning
      sessions that have not been used for more than 55 minutes and replaces
      expired sessions.

    - Blocks, with a timeout, when :meth:`get` is called on an empty pool.
      Raises after timing out.

    - Raises when :meth:`put` is called on a full pool.  That error is
      never expected in normal practice, as users should be calling
      :meth:`get` followed by :meth:`put` whenever in need of a session.

    :type size: int
    :param size: fixed pool size

    :type default_timeout: int
    :param default_timeout: default timeout, in seconds, to wait for
                                 a returned session.

    :type labels: dict (str -> str) or None
    :param labels: (Optional) user-assigned labels for sessions created
                    by the pool.

    :type database_role: str
    :param database_role: (Optional) user-assigned database_role for the session.
    """

    DEFAULT_SIZE = 10
    DEFAULT_TIMEOUT = 10
    DEFAULT_MAX_AGE_MINUTES = 55

    def __init__(
        self,
        size=DEFAULT_SIZE,
        default_timeout=DEFAULT_TIMEOUT,
        labels=None,
        database_role=None,
        max_age_minutes=DEFAULT_MAX_AGE_MINUTES,
    ):
        super(FixedSizePool, self).__init__(labels=labels, database_role=database_role)
        self.size = size
        self.default_timeout = default_timeout
        self._sessions = queue.LifoQueue(size)
        self._max_age = datetime.timedelta(minutes=max_age_minutes)

    def bind(self, database):
        """Associate the pool with a database.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database used by the pool to used to create sessions
                         when needed.
        """
        self._database = database
        requested_session_count = self.size - self._sessions.qsize()
        span = get_current_span()
        span_event_attributes = {"kind": type(self).__name__}

        if requested_session_count <= 0:
            add_span_event(
                span,
                f"Invalid session pool size({requested_session_count}) <= 0",
                span_event_attributes,
            )
            return

        api = database.spanner_api
        metadata = _metadata_with_prefix(database.name)
        if database._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(database._route_to_leader_enabled)
            )
        self._database_role = self._database_role or self._database.database_role
        if requested_session_count > 0:
            add_span_event(
                span,
                f"Requesting {requested_session_count} sessions",
                span_event_attributes,
            )

        if self._sessions.full():
            add_span_event(span, "Session pool is already full", span_event_attributes)
            return

        request = BatchCreateSessionsRequest(
            database=database.name,
            session_count=requested_session_count,
            session_template=SessionProto(creator_role=self.database_role),
        )

        observability_options = getattr(self._database, "observability_options", None)
        with trace_call(
            "CloudSpanner.FixedPool.BatchCreateSessions",
            observability_options=observability_options,
            metadata=metadata,
        ) as span, MetricsCapture():
            returned_session_count = 0
            while not self._sessions.full():
                request.session_count = requested_session_count - self._sessions.qsize()
                add_span_event(
                    span,
                    f"Creating {request.session_count} sessions",
                    span_event_attributes,
                )
                resp = api.batch_create_sessions(
                    request=request,
                    metadata=database.metadata_with_request_id(
                        database._next_nth_request,
                        1,
                        metadata,
                        span,
                    ),
                )

                add_span_event(
                    span,
                    "Created sessions",
                    dict(count=len(resp.session)),
                )

                for session_pb in resp.session:
                    session = self._new_session()
                    session._session_id = session_pb.name.split("/")[-1]
                    self._sessions.put(session)
                    returned_session_count += 1

            add_span_event(
                span,
                f"Requested for {requested_session_count} sessions, returned {returned_session_count}",
                span_event_attributes,
            )

    def get(self, timeout=None):
        """Check a session out from the pool.

        :type timeout: int
        :param timeout: seconds to block waiting for an available session

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: an existing session from the pool, or a newly-created
                  session.
        :raises: :exc:`queue.Empty` if the queue is empty.
        """
        if timeout is None:
            timeout = self.default_timeout

        start_time = time.time()
        current_span = get_current_span()
        span_event_attributes = {"kind": type(self).__name__}
        add_span_event(current_span, "Acquiring session", span_event_attributes)

        session = None
        try:
            add_span_event(
                current_span,
                "Waiting for a session to become available",
                span_event_attributes,
            )

            session = self._sessions.get(block=True, timeout=timeout)
            age = _NOW() - session.last_use_time

            if age >= self._max_age and not session.exists():
                if not session.exists():
                    add_span_event(
                        current_span,
                        "Session is not valid, recreating it",
                        span_event_attributes,
                    )
                session = self._new_session()
                session.create()
                # Replacing with the updated session.id.
                span_event_attributes["session.id"] = session._session_id

            span_event_attributes["session.id"] = session._session_id
            span_event_attributes["time.elapsed"] = time.time() - start_time
            add_span_event(current_span, "Acquired session", span_event_attributes)

        except queue.Empty as e:
            add_span_event(
                current_span, "No sessions available in the pool", span_event_attributes
            )
            raise e

        return session

    def put(self, session):
        """Return a session to the pool.

        Never blocks:  if the pool is full, raises.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: the session being returned.

        :raises: :exc:`queue.Full` if the queue is full.
        """
        self._sessions.put_nowait(session)

    def clear(self):
        """Delete all sessions in the pool."""

        while True:
            try:
                session = self._sessions.get(block=False)
            except queue.Empty:
                break
            else:
                session.delete()


class BurstyPool(AbstractSessionPool):
    """Concrete session pool implementation:

    - "Pings" existing sessions via :meth:`session.exists` before returning
      them.

    - Creates a new session, rather than blocking, when :meth:`get` is called
      on an empty pool.

    - Discards the returned session, rather than blocking, when :meth:`put`
      is called on a full pool.

    :type target_size: int
    :param target_size: max pool size

    :type labels: dict (str -> str) or None
    :param labels: (Optional) user-assigned labels for sessions created
                    by the pool.

    :type database_role: str
    :param database_role: (Optional) user-assigned database_role for the session.
    """

    def __init__(self, target_size=10, labels=None, database_role=None):
        super(BurstyPool, self).__init__(labels=labels, database_role=database_role)
        self.target_size = target_size
        self._database = None
        self._sessions = queue.LifoQueue(target_size)

    def bind(self, database):
        """Associate the pool with a database.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database used by the pool to create sessions
                         when needed.
        """
        self._database = database
        self._database_role = self._database_role or self._database.database_role

    def get(self):
        """Check a session out from the pool.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: an existing session from the pool, or a newly-created
                  session.
        """
        current_span = get_current_span()
        span_event_attributes = {"kind": type(self).__name__}
        add_span_event(current_span, "Acquiring session", span_event_attributes)

        try:
            add_span_event(
                current_span,
                "Waiting for a session to become available",
                span_event_attributes,
            )
            session = self._sessions.get_nowait()
        except queue.Empty:
            add_span_event(
                current_span,
                "No sessions available in pool. Creating session",
                span_event_attributes,
            )
            session = self._new_session()
            session.create()
        else:
            if not session.exists():
                add_span_event(
                    current_span,
                    "Session is not valid, recreating it",
                    span_event_attributes,
                )
                session = self._new_session()
                session.create()
        return session

    def put(self, session):
        """Return a session to the pool.

        Never blocks:  if the pool is full, the returned session is
        discarded.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: the session being returned.
        """
        try:
            self._sessions.put_nowait(session)
        except queue.Full:
            try:
                # Sessions from pools are never multiplexed, so we can always delete them
                session.delete()
            except NotFound:
                pass

    def clear(self):
        """Delete all sessions in the pool."""

        while True:
            try:
                session = self._sessions.get(block=False)
            except queue.Empty:
                break
            else:
                session.delete()


class PingingPool(AbstractSessionPool):
    """Concrete session pool implementation:

    - Pre-allocates / creates a fixed number of sessions.

    - Sessions are used in "round-robin" order (LRU first).

    - "Pings" existing sessions in the background after a specified interval
      via an API call (``session.ping()``).

    - Blocks, with a timeout, when :meth:`get` is called on an empty pool.
      Raises after timing out.

    - Raises when :meth:`put` is called on a full pool.  That error is
      never expected in normal practice, as users should be calling
      :meth:`get` followed by :meth:`put` whenever in need of a session.

    The application is responsible for calling :meth:`ping` at appropriate
    times, e.g. from a background thread.

    :type size: int
    :param size: fixed pool size

    :type default_timeout: int
    :param default_timeout: default timeout, in seconds, to wait for
                            a returned session.

    :type ping_interval: int
    :param ping_interval: interval at which to ping sessions.

    :type labels: dict (str -> str) or None
    :param labels: (Optional) user-assigned labels for sessions created
                    by the pool.

    :type database_role: str
    :param database_role: (Optional) user-assigned database_role for the session.
    """

    def __init__(
        self,
        size=10,
        default_timeout=10,
        ping_interval=3000,
        labels=None,
        database_role=None,
    ):
        super(PingingPool, self).__init__(labels=labels, database_role=database_role)
        self.size = size
        self.default_timeout = default_timeout
        self._delta = datetime.timedelta(seconds=ping_interval)
        self._sessions = queue.PriorityQueue(size)

    def bind(self, database):
        """Associate the pool with a database.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database used by the pool to create sessions
                         when needed.
        """
        self._database = database
        api = database.spanner_api
        metadata = _metadata_with_prefix(database.name)
        if database._route_to_leader_enabled:
            metadata.append(
                _metadata_with_leader_aware_routing(database._route_to_leader_enabled)
            )
        self._database_role = self._database_role or self._database.database_role

        request = BatchCreateSessionsRequest(
            database=database.name,
            session_count=self.size,
            session_template=SessionProto(creator_role=self.database_role),
        )

        span_event_attributes = {"kind": type(self).__name__}
        current_span = get_current_span()
        requested_session_count = request.session_count
        if requested_session_count <= 0:
            add_span_event(
                current_span,
                f"Invalid session pool size({requested_session_count}) <= 0",
                span_event_attributes,
            )
            return

        add_span_event(
            current_span,
            f"Requesting {requested_session_count} sessions",
            span_event_attributes,
        )

        observability_options = getattr(self._database, "observability_options", None)
        with trace_call(
            "CloudSpanner.PingingPool.BatchCreateSessions",
            observability_options=observability_options,
            metadata=metadata,
        ) as span, MetricsCapture():
            returned_session_count = 0
            while returned_session_count < self.size:
                resp = api.batch_create_sessions(
                    request=request,
                    metadata=database.metadata_with_request_id(
                        database._next_nth_request,
                        1,
                        metadata,
                        span,
                    ),
                )

                add_span_event(
                    span,
                    f"Created {len(resp.session)} sessions",
                )

                for session_pb in resp.session:
                    session = self._new_session()
                    returned_session_count += 1
                    session._session_id = session_pb.name.split("/")[-1]
                    self.put(session)

            add_span_event(
                span,
                f"Requested for {requested_session_count} sessions, returned {returned_session_count}",
                span_event_attributes,
            )

    def get(self, timeout=None):
        """Check a session out from the pool.

        :type timeout: int
        :param timeout: seconds to block waiting for an available session

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: an existing session from the pool, or a newly-created
                  session.
        :raises: :exc:`queue.Empty` if the queue is empty.
        """
        if timeout is None:
            timeout = self.default_timeout

        start_time = time.time()
        span_event_attributes = {"kind": type(self).__name__}
        current_span = get_current_span()
        add_span_event(
            current_span,
            "Waiting for a session to become available",
            span_event_attributes,
        )

        ping_after = None
        session = None
        try:
            ping_after, session = self._sessions.get(block=True, timeout=timeout)
        except queue.Empty as e:
            add_span_event(
                current_span,
                "No sessions available in the pool within the specified timeout",
                span_event_attributes,
            )
            raise e

        if _NOW() > ping_after:
            # Using session.exists() guarantees the returned session exists.
            # session.ping() uses a cached result in the backend which could
            # result in a recently deleted session being returned.
            if not session.exists():
                session = self._new_session()
                session.create()

        span_event_attributes.update(
            {
                "time.elapsed": time.time() - start_time,
                "session.id": session._session_id,
                "kind": "pinging_pool",
            }
        )
        add_span_event(current_span, "Acquired session", span_event_attributes)
        return session

    def put(self, session):
        """Return a session to the pool.

        Never blocks:  if the pool is full, raises.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: the session being returned.

        :raises: :exc:`queue.Full` if the queue is full.
        """
        self._sessions.put_nowait((_NOW() + self._delta, session))

    def clear(self):
        """Delete all sessions in the pool."""
        while True:
            try:
                _, session = self._sessions.get(block=False)
            except queue.Empty:
                break
            else:
                session.delete()

    def ping(self):
        """Refresh maybe-expired sessions in the pool.

        This method is designed to be called from a background thread,
        or during the "idle" phase of an event loop.
        """
        while True:
            try:
                ping_after, session = self._sessions.get(block=False)
            except queue.Empty:  # all sessions in use
                break
            if ping_after > _NOW():  # oldest session is fresh
                # Re-add to queue with existing expiration
                self._sessions.put((ping_after, session))
                break
            try:
                session.ping()
            except NotFound:
                session = self._new_session()
                session.create()
            # Re-add to queue with new expiration
            self.put(session)


class TransactionPingingPool(PingingPool):
    """Concrete session pool implementation:

    Deprecated: TransactionPingingPool no longer begins a transaction for each of its sessions at startup.
    Hence the TransactionPingingPool is same as :class:`PingingPool` and maybe removed in the future.


    In addition to the features of :class:`PingingPool`, this class
    creates and begins a transaction for each of its sessions at startup.

    When a session is returned to the pool, if its transaction has been
    committed or rolled back, the pool creates a new transaction for the
    session and pushes the transaction onto a separate queue of "transactions
    to begin."  The application is responsible for flushing this queue
    as appropriate via the pool's :meth:`begin_pending_transactions` method.

    :type size: int
    :param size: fixed pool size

    :type default_timeout: int
    :param default_timeout: default timeout, in seconds, to wait for
                            a returned session.

    :type ping_interval: int
    :param ping_interval: interval at which to ping sessions.

    :type labels: dict (str -> str) or None
    :param labels: (Optional) user-assigned labels for sessions created
                    by the pool.

    :type database_role: str
    :param database_role: (Optional) user-assigned database_role for the session.
    """

    def __init__(
        self,
        size=10,
        default_timeout=10,
        ping_interval=3000,
        labels=None,
        database_role=None,
    ):
        """This throws a deprecation warning on initialization."""
        warn(
            f"{self.__class__.__name__} is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._pending_sessions = queue.Queue()

        super(TransactionPingingPool, self).__init__(
            size,
            default_timeout,
            ping_interval,
            labels=labels,
            database_role=database_role,
        )

        self.begin_pending_transactions()

    def bind(self, database):
        """Associate the pool with a database.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database: database used by the pool to create sessions
                         when needed.
        """
        super(TransactionPingingPool, self).bind(database)
        self._database_role = self._database_role or self._database.database_role
        self.begin_pending_transactions()

    def put(self, session):
        """Return a session to the pool.

        Never blocks:  if the pool is full, raises.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: the session being returned.

        :raises: :exc:`queue.Full` if the queue is full.
        """
        if self._sessions.full():
            raise queue.Full

        txn = session._transaction
        if txn is None or txn.committed or txn.rolled_back:
            session.transaction()
            self._pending_sessions.put(session)
        else:
            super(TransactionPingingPool, self).put(session)

    def begin_pending_transactions(self):
        """Begin all transactions for sessions added to the pool."""
        while not self._pending_sessions.empty():
            session = self._pending_sessions.get()
            super(TransactionPingingPool, self).put(session)


class SessionCheckout(object):
    """Context manager: hold session checked out from a pool.

    Deprecated. Sessions should be checked out indirectly using context
    managers or :meth:`~google.cloud.spanner_v1.database.Database.run_in_transaction`,
    rather than checked out directly from the pool.

    :type pool: concrete subclass of
        :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`
    :param pool: Pool from which to check out a session.

    :param kwargs: extra keyword arguments to be passed to :meth:`pool.get`.
    """

    _session = None

    def __init__(self, pool, **kwargs):
        self._pool = pool
        self._kwargs = kwargs.copy()

    def __enter__(self):
        self._session = self._pool.get(**self._kwargs)
        return self._session

    def __exit__(self, *ignored):
        self._pool.put(self._session)

# Copyright 2025 Google LLC All rights reserved.
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
from enum import Enum
from os import getenv
from datetime import timedelta
from threading import Event, Lock, Thread
from time import sleep, time
from typing import Optional
from weakref import ref

from google.cloud.spanner_v1.session import Session
from google.cloud.spanner_v1._opentelemetry_tracing import (
    get_current_span,
    add_span_event,
)


class TransactionType(Enum):
    """Transaction types for session options."""

    READ_ONLY = "read-only"
    PARTITIONED = "partitioned"
    READ_WRITE = "read/write"


class DatabaseSessionsManager(object):
    """Manages sessions for a Cloud Spanner database.

    Sessions can be checked out from the database session manager for a specific
    transaction type using :meth:`get_session`, and returned to the session manager
    using :meth:`put_session`.

    The sessions returned by the session manager depend on the configured environment variables
    and the provided session pool (see :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`).

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: The database to manage sessions for.

    :type pool: :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`
    :param pool: The pool to get non-multiplexed sessions from.
    """

    # Environment variables for multiplexed sessions
    _ENV_VAR_MULTIPLEXED = "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS"
    _ENV_VAR_MULTIPLEXED_PARTITIONED = (
        "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS_PARTITIONED_OPS"
    )
    _ENV_VAR_MULTIPLEXED_READ_WRITE = "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS_FOR_RW"

    # Intervals for the maintenance thread to check and refresh the multiplexed session.
    _MAINTENANCE_THREAD_POLLING_INTERVAL = timedelta(minutes=10)
    _MAINTENANCE_THREAD_REFRESH_INTERVAL = timedelta(days=7)

    def __init__(self, database, pool):
        self._database = database
        self._pool = pool

        # Declare multiplexed session attributes. When a multiplexed session for the
        # database session manager is created, a maintenance thread is initialized to
        # periodically delete and recreate the multiplexed session so that it remains
        # valid. Because of this concurrency, we need to use a lock whenever we access
        # the multiplexed session to avoid any race conditions.
        self._multiplexed_session: Optional[Session] = None
        self._multiplexed_session_thread: Optional[Thread] = None
        self._multiplexed_session_lock: Lock = Lock()

        # Event to terminate the maintenance thread.
        # Only used for testing purposes.
        self._multiplexed_session_terminate_event: Event = Event()

    def get_session(self, transaction_type: TransactionType) -> Session:
        """Returns a session for the given transaction type from the database session manager.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: a session for the given transaction type.
        """

        session = (
            self._get_multiplexed_session()
            if self._use_multiplexed(transaction_type)
            else self._pool.get()
        )

        add_span_event(
            get_current_span(),
            "Using session",
            {"id": session.session_id, "multiplexed": session.is_multiplexed},
        )

        return session

    def put_session(self, session: Session) -> None:
        """Returns the session to the database session manager.

        :type session: :class:`~google.cloud.spanner_v1.session.Session`
        :param session: The session to return to the database session manager.
        """

        add_span_event(
            get_current_span(),
            "Returning session",
            {"id": session.session_id, "multiplexed": session.is_multiplexed},
        )

        # No action is needed for multiplexed sessions: the session
        # pool is only used for managing non-multiplexed sessions,
        # since they can only process one transaction at a time.
        if not session.is_multiplexed:
            self._pool.put(session)

    def _get_multiplexed_session(self) -> Session:
        """Returns a multiplexed session from the database session manager.

        If the multiplexed session is not defined, creates a new multiplexed
        session and starts a maintenance thread to periodically delete and
        recreate it so that it remains valid. Otherwise, simply returns the
        current multiplexed session.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: a multiplexed session.
        """

        with self._multiplexed_session_lock:
            if self._multiplexed_session is None:
                self._multiplexed_session = self._build_multiplexed_session()

                self._multiplexed_session_thread = self._build_maintenance_thread()
                self._multiplexed_session_thread.start()

            return self._multiplexed_session

    def _build_multiplexed_session(self) -> Session:
        """Builds and returns a new multiplexed session for the database session manager.

        :rtype: :class:`~google.cloud.spanner_v1.session.Session`
        :returns: a new multiplexed session.
        """

        session = Session(
            database=self._database,
            database_role=self._database.database_role,
            is_multiplexed=True,
        )
        session.create()

        self._database.logger.info("Created multiplexed session.")

        return session

    def _build_maintenance_thread(self) -> Thread:
        """Builds and returns a multiplexed session maintenance thread for
        the database session manager. This thread will periodically delete
        and recreate the multiplexed session to ensure that it is always valid.

        :rtype: :class:`threading.Thread`
        :returns: a multiplexed session maintenance thread.
        """

        # Use a weak reference to the database session manager to avoid
        # creating a circular reference that would prevent the database
        # session manager from being garbage collected.
        session_manager_ref = ref(self)

        return Thread(
            target=self._maintain_multiplexed_session,
            name=f"maintenance-multiplexed-session-{self._multiplexed_session.name}",
            args=[session_manager_ref],
            daemon=True,
        )

    @staticmethod
    def _maintain_multiplexed_session(session_manager_ref) -> None:
        """Maintains the multiplexed session for the database session manager.

        This method will delete and recreate the referenced database session manager's
        multiplexed session to ensure that it is always valid. The method will run until
        the database session manager is deleted or the multiplexed session is deleted.

        :type session_manager_ref: :class:`_weakref.ReferenceType`
        :param session_manager_ref: A weak reference to the database session manager.
        """

        manager = session_manager_ref()
        if manager is None:
            return

        polling_interval_seconds = (
            manager._MAINTENANCE_THREAD_POLLING_INTERVAL.total_seconds()
        )
        refresh_interval_seconds = (
            manager._MAINTENANCE_THREAD_REFRESH_INTERVAL.total_seconds()
        )

        session_created_time = time()

        while True:
            # Terminate the thread is the database session manager has been deleted.
            manager = session_manager_ref()
            if manager is None:
                return

            # Terminate the thread if corresponding event is set.
            if manager._multiplexed_session_terminate_event.is_set():
                return

            # Wait for until the refresh interval has elapsed.
            if time() - session_created_time < refresh_interval_seconds:
                sleep(polling_interval_seconds)
                continue

            with manager._multiplexed_session_lock:
                manager._multiplexed_session.delete()
                manager._multiplexed_session = manager._build_multiplexed_session()

            session_created_time = time()

    @classmethod
    def _use_multiplexed(cls, transaction_type: TransactionType) -> bool:
        """Returns whether to use multiplexed sessions for the given transaction type.

        Multiplexed sessions are enabled for read-only transactions if:
            * _ENV_VAR_MULTIPLEXED != 'false'.

        Multiplexed sessions are enabled for partitioned transactions if:
            * _ENV_VAR_MULTIPLEXED_PARTITIONED != 'false'.

        Multiplexed sessions are enabled for read/write transactions if:
            * _ENV_VAR_MULTIPLEXED_READ_WRITE != 'false'.

        :type transaction_type: :class:`TransactionType`
        :param transaction_type: the type of transaction

        :rtype: bool
        :returns: True if multiplexed sessions should be used for the given transaction
            type, False otherwise.

        :raises ValueError: if the transaction type is not supported.
        """

        if transaction_type is TransactionType.READ_ONLY:
            return cls._getenv(cls._ENV_VAR_MULTIPLEXED)

        elif transaction_type is TransactionType.PARTITIONED:
            return cls._getenv(cls._ENV_VAR_MULTIPLEXED_PARTITIONED)

        elif transaction_type is TransactionType.READ_WRITE:
            return cls._getenv(cls._ENV_VAR_MULTIPLEXED_READ_WRITE)

        raise ValueError(f"Transaction type {transaction_type} is not supported.")

    @classmethod
    def _getenv(cls, env_var_name: str) -> bool:
        """Returns the value of the given environment variable as a boolean.

        True unless explicitly 'false' (case-insensitive).
        All other values (including unset) are considered true.

        :type env_var_name: str
        :param env_var_name: the name of the boolean environment variable

        :rtype: bool
        :returns: True unless the environment variable is set to 'false', False otherwise.
        """

        env_var_value = getenv(env_var_name, "true").lower().strip()
        return env_var_value != "false"

# Copyright 2024 Google LLC
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
#

from __future__ import annotations

from typing import (
    cast,
    Any,
    AsyncIterable,
    Optional,
    Set,
    Sequence,
    TYPE_CHECKING,
)

import abc
import time
import warnings
import random
import os
import concurrent.futures

from functools import partial
from grpc import Channel

from google.cloud.bigtable.data.execute_query.values import ExecuteQueryValueType
from google.cloud.bigtable.data.execute_query.metadata import (
    SqlType,
    _pb_metadata_to_metadata_types,
)
from google.cloud.bigtable.data.execute_query._parameters_formatting import (
    _format_execute_query_params,
    _to_param_types,
)
from google.cloud.bigtable_v2.services.bigtable.transports.base import (
    DEFAULT_CLIENT_INFO,
)
from google.cloud.bigtable_v2.types.bigtable import PingAndWarmRequest
from google.cloud.bigtable_v2.types.bigtable import SampleRowKeysRequest
from google.cloud.bigtable_v2.types.bigtable import MutateRowRequest
from google.cloud.bigtable_v2.types.bigtable import CheckAndMutateRowRequest
from google.cloud.bigtable_v2.types.bigtable import ReadModifyWriteRowRequest
from google.cloud.client import ClientWithProject
from google.cloud.environment_vars import BIGTABLE_EMULATOR  # type: ignore
from google.api_core import retry as retries
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.exceptions import Aborted
from google.protobuf.message import Message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

import google.auth.credentials
import google.auth._default
from google.api_core import client_options as client_options_lib
from google.cloud.bigtable.client import _DEFAULT_BIGTABLE_EMULATOR_CLIENT
from google.cloud.bigtable.data.row import Row
from google.cloud.bigtable.data.read_rows_query import ReadRowsQuery
from google.cloud.bigtable.data.exceptions import FailedQueryShardError
from google.cloud.bigtable.data.exceptions import ShardedReadRowsExceptionGroup

from google.cloud.bigtable.data._helpers import TABLE_DEFAULT, _align_timeouts
from google.cloud.bigtable.data._helpers import _WarmedInstanceKey
from google.cloud.bigtable.data._helpers import _CONCURRENCY_LIMIT
from google.cloud.bigtable.data._helpers import _retry_exception_factory
from google.cloud.bigtable.data._helpers import _validate_timeouts
from google.cloud.bigtable.data._helpers import _get_error_type
from google.cloud.bigtable.data._helpers import _get_retryable_errors
from google.cloud.bigtable.data._helpers import _get_timeouts
from google.cloud.bigtable.data._helpers import _attempt_timeout_generator
from google.cloud.bigtable.data.mutations import Mutation, RowMutationEntry

from google.cloud.bigtable.data.read_modify_write_rules import ReadModifyWriteRule
from google.cloud.bigtable.data.row_filters import RowFilter
from google.cloud.bigtable.data.row_filters import StripValueTransformerFilter
from google.cloud.bigtable.data.row_filters import CellsRowLimitFilter
from google.cloud.bigtable.data.row_filters import RowFilterChain

from google.cloud.bigtable.data._cross_sync import CrossSync

if CrossSync.is_async:
    from grpc.aio import insecure_channel
    from google.cloud.bigtable_v2.services.bigtable.transports import (
        BigtableGrpcAsyncIOTransport as TransportType,
    )
    from google.cloud.bigtable_v2.services.bigtable import (
        BigtableAsyncClient as GapicClient,
    )
    from google.cloud.bigtable.data._async.mutations_batcher import _MB_SIZE
    from google.cloud.bigtable.data._async._swappable_channel import (
        AsyncSwappableChannel,
    )
else:
    from typing import Iterable  # noqa: F401
    from grpc import insecure_channel
    from google.cloud.bigtable_v2.services.bigtable.transports import BigtableGrpcTransport as TransportType  # type: ignore
    from google.cloud.bigtable_v2.services.bigtable import BigtableClient as GapicClient  # type: ignore
    from google.cloud.bigtable.data._sync_autogen.mutations_batcher import _MB_SIZE
    from google.cloud.bigtable.data._sync_autogen._swappable_channel import (  # noqa: F401
        SwappableChannel,
    )


if TYPE_CHECKING:
    from google.cloud.bigtable.data._helpers import RowKeySamples
    from google.cloud.bigtable.data._helpers import ShardedQuery

    if CrossSync.is_async:
        from google.cloud.bigtable.data._async.mutations_batcher import (
            MutationsBatcherAsync,
        )
        from google.cloud.bigtable.data.execute_query._async.execute_query_iterator import (
            ExecuteQueryIteratorAsync,
        )
    else:
        from google.cloud.bigtable.data._sync_autogen.mutations_batcher import (  # noqa: F401
            MutationsBatcher,
        )
        from google.cloud.bigtable.data.execute_query._sync_autogen.execute_query_iterator import (  # noqa: F401
            ExecuteQueryIterator,
        )


__CROSS_SYNC_OUTPUT__ = "google.cloud.bigtable.data._sync_autogen.client"


@CrossSync.convert_class(
    sync_name="BigtableDataClient",
    add_mapping_for_name="DataClient",
)
class BigtableDataClientAsync(ClientWithProject):
    @CrossSync.convert(
        docstring_format_vars={
            "LOOP_MESSAGE": (
                "Client should be created within an async context (running event loop)",
                None,
            ),
            "RAISE_NO_LOOP": (
                "RuntimeError: if called outside of an async context (no running event loop)",
                None,
            ),
        }
    )
    def __init__(
        self,
        *,
        project: str | None = None,
        credentials: google.auth.credentials.Credentials | None = None,
        client_options: dict[str, Any]
        | "google.api_core.client_options.ClientOptions"
        | None = None,
        **kwargs,
    ):
        """
        Create a client instance for the Bigtable Data API

        {LOOP_MESSAGE}

        Args:
            project: the project which the client acts on behalf of.
                If not passed, falls back to the default inferred
                from the environment.
            credentials:
                Thehe OAuth2 Credentials to use for this
                client. If not passed (and if no ``_http`` object is
                passed), falls back to the default inferred from the
                environment.
            client_options:
                Client options used to set user options
                on the client. API Endpoint should be set through client_options.
        Raises:
            {RAISE_NO_LOOP}
        """
        if "pool_size" in kwargs:
            warnings.warn("pool_size no longer supported")
        # set up client info headers for veneer library
        self.client_info = DEFAULT_CLIENT_INFO
        self.client_info.client_library_version = self._client_version()
        # parse client options
        if type(client_options) is dict:
            client_options = client_options_lib.from_dict(client_options)
        client_options = cast(
            Optional[client_options_lib.ClientOptions], client_options
        )
        self._emulator_host = os.getenv(BIGTABLE_EMULATOR)
        if self._emulator_host is not None:
            warnings.warn(
                "Connecting to Bigtable emulator at {}".format(self._emulator_host),
                RuntimeWarning,
                stacklevel=2,
            )
            # use insecure channel if emulator is set
            if credentials is None:
                credentials = google.auth.credentials.AnonymousCredentials()
            if project is None:
                project = _DEFAULT_BIGTABLE_EMULATOR_CLIENT

        # initialize client
        ClientWithProject.__init__(
            self,
            credentials=credentials,
            project=project,
            client_options=client_options,
        )
        self._gapic_client = GapicClient(
            credentials=credentials,
            client_options=client_options,
            client_info=self.client_info,
            transport=lambda *args, **kwargs: TransportType(
                *args, **kwargs, channel=self._build_grpc_channel
            ),
        )
        if (
            credentials
            and credentials.universe_domain != self.universe_domain
            and self._emulator_host is None
        ):
            # validate that the universe domain of the credentials matches the
            # universe domain configured in client_options
            raise ValueError(
                f"The configured universe domain ({self.universe_domain}) does "
                "not match the universe domain found in the credentials "
                f"({self._credentials.universe_domain}). If you haven't "
                "configured the universe domain explicitly, `googleapis.com` "
                "is the default."
            )
        self._is_closed = CrossSync.Event()
        self.transport = cast(TransportType, self._gapic_client.transport)
        # keep track of active instances to for warmup on channel refresh
        self._active_instances: Set[_WarmedInstanceKey] = set()
        # keep track of _DataApiTarget objects associated with each instance
        # only remove instance from _active_instances when all associated targets are closed
        self._instance_owners: dict[_WarmedInstanceKey, Set[int]] = {}
        self._channel_init_time = time.monotonic()
        self._channel_refresh_task: CrossSync.Task[None] | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor() if not CrossSync.is_async else None
        )
        if self._emulator_host is None:
            # attempt to start background channel refresh tasks
            try:
                self._start_background_channel_refresh()
            except RuntimeError:
                warnings.warn(
                    f"{self.__class__.__name__} should be started in an "
                    "asyncio event loop. Channel refresh will not be started",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @CrossSync.convert(replace_symbols={"AsyncSwappableChannel": "SwappableChannel"})
    def _build_grpc_channel(self, *args, **kwargs) -> AsyncSwappableChannel:
        """
        This method is called by the gapic transport to create a grpc channel.

        The init arguments passed down are captured in a partial used by AsyncSwappableChannel
        to create new channel instances in the future, as part of the channel refresh logic

        Emulators always use an inseucre channel

        Args:
          - *args: positional arguments passed by the gapic layer to create a new channel with
          - **kwargs: keyword arguments passed by the gapic layer to create a new channel with
        Returns:
          a custom wrapped swappable channel
        """
        if self._emulator_host is not None:
            # emulators use insecure channel
            create_channel_fn = partial(insecure_channel, self._emulator_host)
        else:
            create_channel_fn = partial(TransportType.create_channel, *args, **kwargs)
        return AsyncSwappableChannel(create_channel_fn)

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used by the client instance.
        """
        return self._gapic_client.universe_domain

    @property
    def api_endpoint(self) -> str:
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._gapic_client.api_endpoint

    @staticmethod
    def _client_version() -> str:
        """
        Helper function to return the client version string for this client
        """
        version_str = f"{google.cloud.bigtable.__version__}-data"
        if CrossSync.is_async:
            version_str += "-async"
        return version_str

    @CrossSync.convert(
        docstring_format_vars={
            "RAISE_NO_LOOP": (
                "RuntimeError: if not called in an asyncio event loop",
                "None",
            )
        }
    )
    def _start_background_channel_refresh(self) -> None:
        """
        Starts a background task to ping and warm grpc channel

        Raises:
            {RAISE_NO_LOOP}
        """
        if (
            not self._channel_refresh_task
            and not self._emulator_host
            and not self._is_closed.is_set()
        ):
            # raise error if not in an event loop in async client
            CrossSync.verify_async_event_loop()
            self._channel_refresh_task = CrossSync.create_task(
                self._manage_channel,
                sync_executor=self._executor,
                task_name=f"{self.__class__.__name__} channel refresh",
            )

    @CrossSync.convert
    async def close(self, timeout: float | None = 2.0):
        """
        Cancel all background tasks
        """
        self._is_closed.set()
        if self._channel_refresh_task is not None:
            self._channel_refresh_task.cancel()
            await CrossSync.wait([self._channel_refresh_task], timeout=timeout)
        await self.transport.close()
        if self._executor:
            self._executor.shutdown(wait=False)
        self._channel_refresh_task = None

    @CrossSync.convert
    async def _ping_and_warm_instances(
        self,
        instance_key: _WarmedInstanceKey | None = None,
        channel: Channel | None = None,
    ) -> list[BaseException | None]:
        """
        Prepares the backend for requests on a channel

        Pings each Bigtable instance registered in `_active_instances` on the client

        Args:
            instance_key: if provided, only warm the instance associated with the key
            channel: grpc channel to warm. If none, warms `self.transport.grpc_channel`
        Returns:
            list[BaseException | None]: sequence of results or exceptions from the ping requests
        """
        channel = channel or self.transport.grpc_channel
        instance_list = (
            [instance_key] if instance_key is not None else self._active_instances
        )
        ping_rpc = channel.unary_unary(
            "/google.bigtable.v2.Bigtable/PingAndWarm",
            request_serializer=PingAndWarmRequest.serialize,
        )
        # prepare list of coroutines to run
        partial_list = [
            partial(
                ping_rpc,
                request={"name": instance_name, "app_profile_id": app_profile_id},
                metadata=[
                    (
                        "x-goog-request-params",
                        f"name={instance_name}&app_profile_id={app_profile_id}",
                    )
                ],
                wait_for_ready=True,
            )
            for (instance_name, app_profile_id) in instance_list
        ]
        result_list = await CrossSync.gather_partials(
            partial_list, return_exceptions=True, sync_executor=self._executor
        )
        return [r or None for r in result_list]

    def _invalidate_channel_stubs(self):
        """Helper to reset the cached stubs. Needed when changing out the grpc channel"""
        self.transport._stubs = {}
        self.transport._prep_wrapped_messages(self.client_info)

    @CrossSync.convert(replace_symbols={"AsyncSwappableChannel": "SwappableChannel"})
    async def _manage_channel(
        self,
        refresh_interval_min: float = 60 * 35,
        refresh_interval_max: float = 60 * 45,
        grace_period: float = 60 * 10,
    ) -> None:
        """
        Background task that periodically refreshes and warms a grpc channel

        The backend will automatically close channels after 60 minutes, so
        `refresh_interval` + `grace_period` should be < 60 minutes

        Runs continuously until the client is closed

        Args:
            refresh_interval_min: minimum interval before initiating refresh
                process in seconds. Actual interval will be a random value
                between `refresh_interval_min` and `refresh_interval_max`
            refresh_interval_max: maximum interval before initiating refresh
                process in seconds. Actual interval will be a random value
                between `refresh_interval_min` and `refresh_interval_max`
            grace_period: time to allow previous channel to serve existing
                requests before closing, in seconds
        """
        if not isinstance(self.transport.grpc_channel, AsyncSwappableChannel):
            warnings.warn("Channel does not support auto-refresh.")
            return
        super_channel: AsyncSwappableChannel = self.transport.grpc_channel
        first_refresh = self._channel_init_time + random.uniform(
            refresh_interval_min, refresh_interval_max
        )
        next_sleep = max(first_refresh - time.monotonic(), 0)
        if next_sleep > 0:
            # warm the current channel immediately
            await self._ping_and_warm_instances(channel=super_channel)
        # continuously refresh the channel every `refresh_interval` seconds
        while not self._is_closed.is_set():
            await CrossSync.event_wait(
                self._is_closed,
                next_sleep,
                async_break_early=False,  # no need to interrupt sleep. Task will be cancelled on close
            )
            if self._is_closed.is_set():
                # don't refresh if client is closed
                break
            start_timestamp = time.monotonic()
            # prepare new channel for use
            new_channel = super_channel.create_channel()
            await self._ping_and_warm_instances(channel=new_channel)
            # cycle channel out of use, with long grace window before closure
            old_channel = super_channel.swap_channel(new_channel)
            self._invalidate_channel_stubs()
            # give old_channel a chance to complete existing rpcs
            if CrossSync.is_async:
                await old_channel.close(grace_period)
            else:
                if grace_period:
                    self._is_closed.wait(grace_period)  # type: ignore
                old_channel.close()  # type: ignore
            # subtract the time spent waiting for the channel to be replaced
            next_refresh = random.uniform(refresh_interval_min, refresh_interval_max)
            next_sleep = max(next_refresh - (time.monotonic() - start_timestamp), 0)

    @CrossSync.convert(
        replace_symbols={
            "TableAsync": "Table",
            "ExecuteQueryIteratorAsync": "ExecuteQueryIterator",
            "_DataApiTargetAsync": "_DataApiTarget",
        }
    )
    async def _register_instance(
        self,
        instance_id: str,
        app_profile_id: Optional[str],
        owner_id: int,
    ) -> None:
        """
        Registers an instance with the client, and warms the channel for the instance
        The client will periodically refresh grpc channel used to make
        requests, and new channels will be warmed for each registered instance
        Channels will not be refreshed unless at least one instance is registered

        Args:
          instance_id: id of the instance to register.
          app_profile_id: id of the app profile calling the instance.
          owner_id: integer id of the object owning the instance. Owners will be tracked in
              _instance_owners, and instances will only be unregistered when all
              owners call _remove_instance_registration. Can be obtained by calling
              `id` identity funcion, using `id(owner)`
        """
        instance_name = self._gapic_client.instance_path(self.project, instance_id)
        instance_key = _WarmedInstanceKey(instance_name, app_profile_id)
        self._instance_owners.setdefault(instance_key, set()).add(owner_id)
        if instance_key not in self._active_instances:
            self._active_instances.add(instance_key)
            if self._channel_refresh_task:
                # refresh tasks already running
                # call ping and warm on all existing channels
                await self._ping_and_warm_instances(instance_key)
            else:
                # refresh tasks aren't active. start them as background tasks
                self._start_background_channel_refresh()

    @CrossSync.convert(
        replace_symbols={
            "TableAsync": "Table",
            "ExecuteQueryIteratorAsync": "ExecuteQueryIterator",
            "_DataApiTargetAsync": "_DataApiTarget",
        }
    )
    def _remove_instance_registration(
        self,
        instance_id: str,
        app_profile_id: Optional[str],
        owner_id: int,
    ) -> bool:
        """
        Removes an instance from the client's registered instances, to prevent
        warming new channels for the instance

        If instance_id is not registered, or is still in use by other tables, returns False

        Args:
            instance_id: id of the instance to remove
            app_profile_id: id of the app profile calling the instance.
            owner_id: integer id of the object owning the instance. Can be
                obtained by the `id` identity funcion, using `id(owner)`.
        Returns:
            bool: True if instance was removed, else False
        """
        instance_name = self._gapic_client.instance_path(self.project, instance_id)
        instance_key = _WarmedInstanceKey(instance_name, app_profile_id)
        owner_list = self._instance_owners.get(instance_key, set())
        try:
            owner_list.remove(owner_id)
            if len(owner_list) == 0:
                self._active_instances.remove(instance_key)
            return True
        except KeyError:
            return False

    @CrossSync.convert(
        replace_symbols={"TableAsync": "Table"},
        docstring_format_vars={
            "LOOP_MESSAGE": (
                "Must be created within an async context (running event loop)",
                "",
            ),
            "RAISE_NO_LOOP": (
                "RuntimeError: if called outside of an async context (no running event loop)",
                "None",
            ),
        },
    )
    def get_table(self, instance_id: str, table_id: str, *args, **kwargs) -> TableAsync:
        """
        Returns a table instance for making data API requests. All arguments are passed
        directly to the TableAsync constructor.

        {LOOP_MESSAGE}

        Args:
            instance_id: The Bigtable instance ID to associate with this client.
                instance_id is combined with the client's project to fully
                specify the instance
            table_id: The ID of the table. table_id is combined with the
                instance_id and the client's project to fully specify the table
            app_profile_id: The app profile to associate with requests.
                https://cloud.google.com/bigtable/docs/app-profiles
            default_read_rows_operation_timeout: The default timeout for read rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_read_rows_attempt_timeout: The default timeout for individual
                read rows rpc requests, in seconds. If not set, defaults to 20 seconds
            default_mutate_rows_operation_timeout: The default timeout for mutate rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_mutate_rows_attempt_timeout: The default timeout for individual
                mutate rows rpc requests, in seconds. If not set, defaults to 60 seconds
            default_operation_timeout: The default timeout for all other operations, in
                seconds. If not set, defaults to 60 seconds
            default_attempt_timeout: The default timeout for all other individual rpc
                requests, in seconds. If not set, defaults to 20 seconds
            default_read_rows_retryable_errors: a list of errors that will be retried
                if encountered during read_rows and related operations.
                Defaults to 4 (DeadlineExceeded), 14 (ServiceUnavailable), and 10 (Aborted)
            default_mutate_rows_retryable_errors: a list of errors that will be retried
                if encountered during mutate_rows and related operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
            default_retryable_errors: a list of errors that will be retried if
                encountered during all other operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
        Returns:
            TableAsync: a table instance for making data API requests
        Raises:
            {RAISE_NO_LOOP}
        """
        return TableAsync(self, instance_id, table_id, *args, **kwargs)

    @CrossSync.convert(
        replace_symbols={"AuthorizedViewAsync": "AuthorizedView"},
        docstring_format_vars={
            "LOOP_MESSAGE": (
                "Must be created within an async context (running event loop)",
                "",
            ),
            "RAISE_NO_LOOP": (
                "RuntimeError: if called outside of an async context (no running event loop)",
                "None",
            ),
        },
    )
    def get_authorized_view(
        self, instance_id: str, table_id: str, authorized_view_id: str, *args, **kwargs
    ) -> AuthorizedViewAsync:
        """
        Returns an authorized view instance for making data API requests. All arguments are passed
        directly to the AuthorizedViewAsync constructor.

        {LOOP_MESSAGE}

        Args:
            instance_id: The Bigtable instance ID to associate with this client.
                instance_id is combined with the client's project to fully
                specify the instance
            table_id: The ID of the table. table_id is combined with the
                instance_id and the client's project to fully specify the table
            authorized_view_id: The id for the authorized view to use for requests
            app_profile_id: The app profile to associate with requests.
                https://cloud.google.com/bigtable/docs/app-profiles
            default_read_rows_operation_timeout: The default timeout for read rows
                operations, in seconds. If not set, defaults to Table's value
            default_read_rows_attempt_timeout: The default timeout for individual
                read rows rpc requests, in seconds. If not set, defaults Table's value
            default_mutate_rows_operation_timeout: The default timeout for mutate rows
                operations, in seconds. If not set, defaults to Table's value
            default_mutate_rows_attempt_timeout: The default timeout for individual
                mutate rows rpc requests, in seconds. If not set, defaults Table's value
            default_operation_timeout: The default timeout for all other operations, in
                seconds. If not set, defaults to Table's value
            default_attempt_timeout: The default timeout for all other individual rpc
                requests, in seconds. If not set, defaults to Table's value
            default_read_rows_retryable_errors: a list of errors that will be retried
                if encountered during read_rows and related operations. If not set,
                defaults to Table's value
            default_mutate_rows_retryable_errors: a list of errors that will be retried
                if encountered during mutate_rows and related operations. If not set,
                defaults to Table's value
            default_retryable_errors: a list of errors that will be retried if
                encountered during all other operations. If not set, defaults to
                Table's value
        Returns:
            AuthorizedViewAsync: a table instance for making data API requests
        Raises:
            {RAISE_NO_LOOP}
        """
        return CrossSync.AuthorizedView(
            self,
            instance_id,
            table_id,
            authorized_view_id,
            *args,
            **kwargs,
        )

    @CrossSync.convert(
        replace_symbols={"ExecuteQueryIteratorAsync": "ExecuteQueryIterator"}
    )
    async def execute_query(
        self,
        query: str,
        instance_id: str,
        *,
        parameters: dict[str, ExecuteQueryValueType] | None = None,
        parameter_types: dict[str, SqlType.Type] | None = None,
        app_profile_id: str | None = None,
        operation_timeout: float = 600,
        attempt_timeout: float | None = 20,
        retryable_errors: Sequence[type[Exception]] = (
            DeadlineExceeded,
            ServiceUnavailable,
            Aborted,
        ),
        prepare_operation_timeout: float = 60,
        prepare_attempt_timeout: float | None = 20,
        prepare_retryable_errors: Sequence[type[Exception]] = (
            DeadlineExceeded,
            ServiceUnavailable,
        ),
        column_info: dict[str, Message | EnumTypeWrapper] | None = None,
    ) -> "ExecuteQueryIteratorAsync":
        """
        Executes an SQL query on an instance.
        Returns an iterator to asynchronously stream back columns from selected rows.

        Failed requests within operation_timeout will be retried based on the
        retryable_errors list until operation_timeout is reached.

        Note that this makes two requests, one to ``PrepareQuery`` and one to ``ExecuteQuery``.
        These have separate retry configurations. ``ExecuteQuery`` is where the bulk of the
        work happens.

        Args:
            query: Query to be run on Bigtable instance. The query can use ``@param``
                placeholders to use parameter interpolation on the server. Values for all
                parameters should be provided in ``parameters``. Types of parameters are
                inferred but should be provided in ``parameter_types`` if the inference is
                not possible (i.e. when value can be None, an empty list or an empty dict).
            instance_id: The Bigtable instance ID to perform the query on.
                instance_id is combined with the client's project to fully
                specify the instance.
            parameters: Dictionary with values for all parameters used in the ``query``.
            parameter_types: Dictionary with types of parameters used in the ``query``.
                Required to contain entries only for parameters whose type cannot be
                detected automatically (i.e. the value can be None, an empty list or
                an empty dict).
            app_profile_id: The app profile to associate with requests.
                https://cloud.google.com/bigtable/docs/app-profiles
            operation_timeout: the time budget for the entire executeQuery operation, in seconds.
                Failed requests will be retried within the budget.
                Defaults to 600 seconds.
            attempt_timeout: the time budget for an individual executeQuery network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the 20 seconds.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered during executeQuery.
                Defaults to 4 (DeadlineExceeded), 14 (ServiceUnavailable), and 10 (Aborted)
            prepare_operation_timeout: the time budget for the entire prepareQuery operation, in seconds.
                Failed requests will be retried within the budget.
                Defaults to 60 seconds.
            prepare_attempt_timeout: the time budget for an individual prepareQuery network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the 20 seconds.
                If None, defaults to prepare_operation_timeout.
            prepare_retryable_errors: a list of errors that will be retried if encountered during prepareQuery.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
            column_info: (Optional) A dictionary mapping column names to Protobuf message classes or EnumTypeWrapper objects.
                This dictionary provides the necessary type information for deserializing PROTO and
                ENUM column values from the query results. When an entry is provided
                for a PROTO or ENUM column, the client library will attempt to deserialize the raw data.

                    - For PROTO columns: The value in the dictionary should be the
                      Protobuf Message class (e.g., ``my_pb2.MyMessage``).
                    - For ENUM columns: The value should be the Protobuf EnumTypeWrapper
                      object (e.g., ``my_pb2.MyEnum``).

                Example::

                    import my_pb2

                    column_info = {
                        "my_proto_column": my_pb2.MyMessage,
                        "my_enum_column": my_pb2.MyEnum
                    }

                If ``column_info`` is not provided, or if a specific column name is not found
                in the dictionary:

                    - PROTO columns will be returned as raw bytes.
                    - ENUM columns will be returned as integers.

                Note for Nested PROTO or ENUM Fields:

                    To specify types for PROTO or ENUM fields within STRUCTs or MAPs, use a dot-separated
                    path from the top-level column name.

                        - For STRUCTs: ``struct_column_name.field_name``
                        - For MAPs: ``map_column_name.key`` or ``map_column_name.value`` to specify types
                          for the map keys or values, respectively.

                    Example::

                        import my_pb2

                        column_info = {
                            # Top-level column
                            "my_proto_column": my_pb2.MyMessage,
                            "my_enum_column": my_pb2.MyEnum,

                            # Nested field in a STRUCT column named 'my_struct'
                            "my_struct.nested_proto_field": my_pb2.OtherMessage,
                            "my_struct.nested_enum_field": my_pb2.AnotherEnum,

                            # Nested field in a MAP column named 'my_map'
                            "my_map.key": my_pb2.MapKeyEnum, # If map keys were enums
                            "my_map.value": my_pb2.MapValueMessage,

                            # PROTO field inside a STRUCT, where the STRUCT is the value in a MAP column
                            "struct_map.value.nested_proto_field": my_pb2.DeeplyNestedProto,
                            "struct_map.value.nested_enum_field": my_pb2.DeeplyNestedEnum
                        }

        Returns:
            ExecuteQueryIteratorAsync: an asynchronous iterator that yields rows returned by the query
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
            google.cloud.bigtable.data.exceptions.ParameterTypeInferenceFailed: Raised if
                a parameter is passed without an explicit type, and the type cannot be infered
            google.protobuf.message.DecodeError: raised if the deserialization of a PROTO/ENUM value fails.
        """
        instance_name = self._gapic_client.instance_path(self.project, instance_id)
        converted_param_types = _to_param_types(parameters, parameter_types)
        prepare_request = {
            "instance_name": instance_name,
            "query": query,
            "app_profile_id": app_profile_id,
            "param_types": converted_param_types,
            "proto_format": {},
        }
        prepare_predicate = retries.if_exception_type(
            *[_get_error_type(e) for e in prepare_retryable_errors]
        )
        prepare_operation_timeout, prepare_attempt_timeout = _align_timeouts(
            prepare_operation_timeout, prepare_attempt_timeout
        )
        prepare_sleep_generator = retries.exponential_sleep_generator(0.01, 2, 60)

        target = partial(
            self._gapic_client.prepare_query,
            request=prepare_request,
            timeout=prepare_attempt_timeout,
            retry=None,
        )
        prepare_result = await CrossSync.retry_target(
            target,
            prepare_predicate,
            prepare_sleep_generator,
            prepare_operation_timeout,
            exception_factory=_retry_exception_factory,
        )

        prepare_metadata = _pb_metadata_to_metadata_types(prepare_result.metadata)

        retryable_excs = [_get_error_type(e) for e in retryable_errors]

        pb_params = _format_execute_query_params(parameters, parameter_types)

        request_body = {
            "instance_name": instance_name,
            "app_profile_id": app_profile_id,
            "prepared_query": prepare_result.prepared_query,
            "params": pb_params,
        }
        operation_timeout, attempt_timeout = _align_timeouts(
            operation_timeout, attempt_timeout
        )

        return CrossSync.ExecuteQueryIterator(
            self,
            instance_id,
            app_profile_id,
            request_body,
            prepare_metadata,
            attempt_timeout,
            operation_timeout,
            retryable_excs=retryable_excs,
            column_info=column_info,
        )

    @CrossSync.convert(sync_name="__enter__")
    async def __aenter__(self):
        self._start_background_channel_refresh()
        return self

    @CrossSync.convert(sync_name="__exit__", replace_symbols={"__aexit__": "__exit__"})
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        await self._gapic_client.__aexit__(exc_type, exc_val, exc_tb)


@CrossSync.convert_class(sync_name="_DataApiTarget")
class _DataApiTargetAsync(abc.ABC):
    """
    Abstract class containing API surface for BigtableDataClient. Should not be created directly

    Can be instantiated as a Table or an AuthorizedView
    """

    @CrossSync.convert(
        replace_symbols={"BigtableDataClientAsync": "BigtableDataClient"},
        docstring_format_vars={
            "LOOP_MESSAGE": (
                "Must be created within an async context (running event loop)",
                "",
            ),
            "RAISE_NO_LOOP": (
                "RuntimeError: if called outside of an async context (no running event loop)",
                "None",
            ),
        },
    )
    def __init__(
        self,
        client: BigtableDataClientAsync,
        instance_id: str,
        table_id: str,
        app_profile_id: str | None = None,
        *,
        default_read_rows_operation_timeout: float = 600,
        default_read_rows_attempt_timeout: float | None = 20,
        default_mutate_rows_operation_timeout: float = 600,
        default_mutate_rows_attempt_timeout: float | None = 60,
        default_operation_timeout: float = 60,
        default_attempt_timeout: float | None = 20,
        default_read_rows_retryable_errors: Sequence[type[Exception]] = (
            DeadlineExceeded,
            ServiceUnavailable,
            Aborted,
        ),
        default_mutate_rows_retryable_errors: Sequence[type[Exception]] = (
            DeadlineExceeded,
            ServiceUnavailable,
        ),
        default_retryable_errors: Sequence[type[Exception]] = (
            DeadlineExceeded,
            ServiceUnavailable,
        ),
    ):
        """
        Initialize a Table instance

        {LOOP_MESSAGE}

        Args:
            instance_id: The Bigtable instance ID to associate with this client.
                instance_id is combined with the client's project to fully
                specify the instance
            table_id: The ID of the table. table_id is combined with the
                instance_id and the client's project to fully specify the table
            app_profile_id: The app profile to associate with requests.
                https://cloud.google.com/bigtable/docs/app-profiles
            default_read_rows_operation_timeout: The default timeout for read rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_read_rows_attempt_timeout: The default timeout for individual
                read rows rpc requests, in seconds. If not set, defaults to 20 seconds
            default_mutate_rows_operation_timeout: The default timeout for mutate rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_mutate_rows_attempt_timeout: The default timeout for individual
                mutate rows rpc requests, in seconds. If not set, defaults to 60 seconds
            default_operation_timeout: The default timeout for all other operations, in
                seconds. If not set, defaults to 60 seconds
            default_attempt_timeout: The default timeout for all other individual rpc
                requests, in seconds. If not set, defaults to 20 seconds
            default_read_rows_retryable_errors: a list of errors that will be retried
                if encountered during read_rows and related operations.
                Defaults to 4 (DeadlineExceeded), 14 (ServiceUnavailable), and 10 (Aborted)
            default_mutate_rows_retryable_errors: a list of errors that will be retried
                if encountered during mutate_rows and related operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
            default_retryable_errors: a list of errors that will be retried if
                encountered during all other operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
        Raises:
            {RAISE_NO_LOOP}
        """
        # NOTE: any changes to the signature of this method should also be reflected
        # in client.get_table()
        # validate timeouts
        _validate_timeouts(
            default_operation_timeout, default_attempt_timeout, allow_none=True
        )
        _validate_timeouts(
            default_read_rows_operation_timeout,
            default_read_rows_attempt_timeout,
            allow_none=True,
        )
        _validate_timeouts(
            default_mutate_rows_operation_timeout,
            default_mutate_rows_attempt_timeout,
            allow_none=True,
        )

        self.client = client
        self.instance_id = instance_id
        self.instance_name = self.client._gapic_client.instance_path(
            self.client.project, instance_id
        )
        self.table_id = table_id
        self.table_name = self.client._gapic_client.table_path(
            self.client.project, instance_id, table_id
        )
        self.app_profile_id: str | None = app_profile_id

        self.default_operation_timeout: float = default_operation_timeout
        self.default_attempt_timeout: float | None = default_attempt_timeout
        self.default_read_rows_operation_timeout: float = (
            default_read_rows_operation_timeout
        )
        self.default_read_rows_attempt_timeout: float | None = (
            default_read_rows_attempt_timeout
        )
        self.default_mutate_rows_operation_timeout: float = (
            default_mutate_rows_operation_timeout
        )
        self.default_mutate_rows_attempt_timeout: float | None = (
            default_mutate_rows_attempt_timeout
        )

        self.default_read_rows_retryable_errors: Sequence[type[Exception]] = (
            default_read_rows_retryable_errors or ()
        )
        self.default_mutate_rows_retryable_errors: Sequence[type[Exception]] = (
            default_mutate_rows_retryable_errors or ()
        )
        self.default_retryable_errors: Sequence[type[Exception]] = (
            default_retryable_errors or ()
        )

        try:
            self._register_instance_future = CrossSync.create_task(
                self.client._register_instance,
                self.instance_id,
                self.app_profile_id,
                id(self),
                sync_executor=self.client._executor,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{self.__class__.__name__} must be created within an async event loop context."
            ) from e

    @property
    @abc.abstractmethod
    def _request_path(self) -> dict[str, str]:
        """
        Used to populate table_name or authorized_view_name for rpc requests, depending on the subclass

        Unimplemented in base class
        """
        raise NotImplementedError

    def __str__(self):
        path_str = list(self._request_path.values())[0] if self._request_path else ""
        return f"{self.__class__.__name__}<{path_str!r}>"

    @CrossSync.convert(replace_symbols={"AsyncIterable": "Iterable"})
    async def read_rows_stream(
        self,
        query: ReadRowsQuery,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
    ) -> AsyncIterable[Row]:
        """
        Read a set of rows from the table, based on the specified query.
        Returns an iterator to asynchronously stream back row data.

        Failed requests within operation_timeout will be retried based on the
        retryable_errors list until operation_timeout is reached.

        Args:
            query: contains details about which rows to return
            operation_timeout: the time budget for the entire operation, in seconds.
                 Failed requests will be retried within the budget.
                 Defaults to the Table's default_read_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_read_rows_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_read_rows_retryable_errors
        Returns:
            AsyncIterable[Row]: an asynchronous iterator that yields rows returned by the query
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
        """
        operation_timeout, attempt_timeout = _get_timeouts(
            operation_timeout, attempt_timeout, self
        )
        retryable_excs = _get_retryable_errors(retryable_errors, self)

        row_merger = CrossSync._ReadRowsOperation(
            query,
            self,
            operation_timeout=operation_timeout,
            attempt_timeout=attempt_timeout,
            retryable_exceptions=retryable_excs,
        )
        return row_merger.start_operation()

    @CrossSync.convert
    async def read_rows(
        self,
        query: ReadRowsQuery,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
    ) -> list[Row]:
        """
        Read a set of rows from the table, based on the specified query.
        Retruns results as a list of Row objects when the request is complete.
        For streamed results, use read_rows_stream.

        Failed requests within operation_timeout will be retried based on the
        retryable_errors list until operation_timeout is reached.

        Args:
            query: contains details about which rows to return
            operation_timeout: the time budget for the entire operation, in seconds.
                 Failed requests will be retried within the budget.
                 Defaults to the Table's default_read_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_read_rows_attempt_timeout.
                If None, defaults to operation_timeout.
                If None, defaults to the Table's default_read_rows_attempt_timeout,
                or the operation_timeout if that is also None.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_read_rows_retryable_errors.
        Returns:
            list[Row]: a list of Rows returned by the query
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
        """
        row_generator = await self.read_rows_stream(
            query,
            operation_timeout=operation_timeout,
            attempt_timeout=attempt_timeout,
            retryable_errors=retryable_errors,
        )
        return [row async for row in row_generator]

    @CrossSync.convert
    async def read_row(
        self,
        row_key: str | bytes,
        *,
        row_filter: RowFilter | None = None,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
    ) -> Row | None:
        """
        Read a single row from the table, based on the specified key.

        Failed requests within operation_timeout will be retried based on the
        retryable_errors list until operation_timeout is reached.

        Args:
            query: contains details about which rows to return
            operation_timeout: the time budget for the entire operation, in seconds.
                 Failed requests will be retried within the budget.
                 Defaults to the Table's default_read_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_read_rows_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_read_rows_retryable_errors.
        Returns:
            Row | None: a Row object if the row exists, otherwise None
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
        """
        if row_key is None:
            raise ValueError("row_key must be string or bytes")
        query = ReadRowsQuery(row_keys=row_key, row_filter=row_filter, limit=1)
        results = await self.read_rows(
            query,
            operation_timeout=operation_timeout,
            attempt_timeout=attempt_timeout,
            retryable_errors=retryable_errors,
        )
        if len(results) == 0:
            return None
        return results[0]

    @CrossSync.convert
    async def read_rows_sharded(
        self,
        sharded_query: ShardedQuery,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
    ) -> list[Row]:
        """
        Runs a sharded query in parallel, then return the results in a single list.
        Results will be returned in the order of the input queries.

        This function is intended to be run on the results on a query.shard() call.
        For example::

            table_shard_keys = await table.sample_row_keys()
            query = ReadRowsQuery(...)
            shard_queries = query.shard(table_shard_keys)
            results = await table.read_rows_sharded(shard_queries)

        Args:
            sharded_query: a sharded query to execute
            operation_timeout: the time budget for the entire operation, in seconds.
                 Failed requests will be retried within the budget.
                 Defaults to the Table's default_read_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_read_rows_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_read_rows_retryable_errors.
        Returns:
            list[Row]: a list of Rows returned by the query
        Raises:
            ShardedReadRowsExceptionGroup: if any of the queries failed
            ValueError: if the query_list is empty
        """
        if not sharded_query:
            raise ValueError("empty sharded_query")
        operation_timeout, attempt_timeout = _get_timeouts(
            operation_timeout, attempt_timeout, self
        )
        # make sure each rpc stays within overall operation timeout
        rpc_timeout_generator = _attempt_timeout_generator(
            operation_timeout, operation_timeout
        )

        # limit the number of concurrent requests using a semaphore
        concurrency_sem = CrossSync.Semaphore(_CONCURRENCY_LIMIT)

        @CrossSync.convert
        async def read_rows_with_semaphore(query):
            async with concurrency_sem:
                # calculate new timeout based on time left in overall operation
                shard_timeout = next(rpc_timeout_generator)
                if shard_timeout <= 0:
                    raise DeadlineExceeded(
                        "Operation timeout exceeded before starting query"
                    )
                return await self.read_rows(
                    query,
                    operation_timeout=shard_timeout,
                    attempt_timeout=min(attempt_timeout, shard_timeout),
                    retryable_errors=retryable_errors,
                )

        routine_list = [
            partial(read_rows_with_semaphore, query) for query in sharded_query
        ]
        batch_result = await CrossSync.gather_partials(
            routine_list,
            return_exceptions=True,
            sync_executor=self.client._executor,
        )

        # collect results and errors
        error_dict = {}
        shard_idx = 0
        results_list = []
        for result in batch_result:
            if isinstance(result, Exception):
                error_dict[shard_idx] = result
            elif isinstance(result, BaseException):
                # BaseException not expected; raise immediately
                raise result
            else:
                results_list.extend(result)
            shard_idx += 1
        if error_dict:
            # if any sub-request failed, raise an exception instead of returning results
            raise ShardedReadRowsExceptionGroup(
                [
                    FailedQueryShardError(idx, sharded_query[idx], e)
                    for idx, e in error_dict.items()
                ],
                results_list,
                len(sharded_query),
            )
        return results_list

    @CrossSync.convert
    async def row_exists(
        self,
        row_key: str | bytes,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.READ_ROWS,
    ) -> bool:
        """
        Return a boolean indicating whether the specified row exists in the table.
        uses the filters: chain(limit cells per row = 1, strip value)

        Args:
            row_key: the key of the row to check
            operation_timeout: the time budget for the entire operation, in seconds.
                 Failed requests will be retried within the budget.
                 Defaults to the Table's default_read_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_read_rows_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_read_rows_retryable_errors.
        Returns:
            bool: a bool indicating whether the row exists
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
        """
        if row_key is None:
            raise ValueError("row_key must be string or bytes")

        strip_filter = StripValueTransformerFilter(flag=True)
        limit_filter = CellsRowLimitFilter(1)
        chain_filter = RowFilterChain(filters=[limit_filter, strip_filter])
        query = ReadRowsQuery(row_keys=row_key, limit=1, row_filter=chain_filter)
        results = await self.read_rows(
            query,
            operation_timeout=operation_timeout,
            attempt_timeout=attempt_timeout,
            retryable_errors=retryable_errors,
        )
        return len(results) > 0

    @CrossSync.convert
    async def sample_row_keys(
        self,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
    ) -> RowKeySamples:
        """
        Return a set of RowKeySamples that delimit contiguous sections of the table of
        approximately equal size

        RowKeySamples output can be used with ReadRowsQuery.shard() to create a sharded query that
        can be parallelized across multiple backend nodes read_rows and read_rows_stream
        requests will call sample_row_keys internally for this purpose when sharding is enabled

        RowKeySamples is simply a type alias for list[tuple[bytes, int]]; a list of
        row_keys, along with offset positions in the table

        Args:
            operation_timeout: the time budget for the entire operation, in seconds.
                Failed requests will be retried within the budget.i
                Defaults to the Table's default_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_retryable_errors.
        Returns:
            RowKeySamples: a set of RowKeySamples the delimit contiguous sections of the table
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing GoogleAPIError exceptions
                from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised if the request encounters an unrecoverable error
        """
        # prepare timeouts
        operation_timeout, attempt_timeout = _get_timeouts(
            operation_timeout, attempt_timeout, self
        )
        attempt_timeout_gen = _attempt_timeout_generator(
            attempt_timeout, operation_timeout
        )
        # prepare retryable
        retryable_excs = _get_retryable_errors(retryable_errors, self)
        predicate = retries.if_exception_type(*retryable_excs)

        sleep_generator = retries.exponential_sleep_generator(0.01, 2, 60)

        @CrossSync.convert
        async def execute_rpc():
            results = await self.client._gapic_client.sample_row_keys(
                request=SampleRowKeysRequest(
                    app_profile_id=self.app_profile_id, **self._request_path
                ),
                timeout=next(attempt_timeout_gen),
                retry=None,
            )
            return [(s.row_key, s.offset_bytes) async for s in results]

        return await CrossSync.retry_target(
            execute_rpc,
            predicate,
            sleep_generator,
            operation_timeout,
            exception_factory=_retry_exception_factory,
        )

    @CrossSync.convert(replace_symbols={"MutationsBatcherAsync": "MutationsBatcher"})
    def mutations_batcher(
        self,
        *,
        flush_interval: float | None = 5,
        flush_limit_mutation_count: int | None = 1000,
        flush_limit_bytes: int = 20 * _MB_SIZE,
        flow_control_max_mutation_count: int = 100_000,
        flow_control_max_bytes: int = 100 * _MB_SIZE,
        batch_operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
        batch_attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
        batch_retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
    ) -> "MutationsBatcherAsync":
        """
        Returns a new mutations batcher instance.

        Can be used to iteratively add mutations that are flushed as a group,
        to avoid excess network calls

        Args:
          flush_interval: Automatically flush every flush_interval seconds. If None,
              a table default will be used
          flush_limit_mutation_count: Flush immediately after flush_limit_mutation_count
              mutations are added across all entries. If None, this limit is ignored.
          flush_limit_bytes: Flush immediately after flush_limit_bytes bytes are added.
          flow_control_max_mutation_count: Maximum number of inflight mutations.
          flow_control_max_bytes: Maximum number of inflight bytes.
          batch_operation_timeout: timeout for each mutate_rows operation, in seconds.
              Defaults to the Table's default_mutate_rows_operation_timeout
          batch_attempt_timeout: timeout for each individual request, in seconds.
              Defaults to the Table's default_mutate_rows_attempt_timeout.
              If None, defaults to batch_operation_timeout.
          batch_retryable_errors: a list of errors that will be retried if encountered.
              Defaults to the Table's default_mutate_rows_retryable_errors.
        Returns:
            MutationsBatcherAsync: a MutationsBatcherAsync context manager that can batch requests
        """
        return CrossSync.MutationsBatcher(
            self,
            flush_interval=flush_interval,
            flush_limit_mutation_count=flush_limit_mutation_count,
            flush_limit_bytes=flush_limit_bytes,
            flow_control_max_mutation_count=flow_control_max_mutation_count,
            flow_control_max_bytes=flow_control_max_bytes,
            batch_operation_timeout=batch_operation_timeout,
            batch_attempt_timeout=batch_attempt_timeout,
            batch_retryable_errors=batch_retryable_errors,
        )

    @CrossSync.convert
    async def mutate_row(
        self,
        row_key: str | bytes,
        mutations: list[Mutation] | Mutation,
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
    ):
        """
        Mutates a row atomically.

        Cells already present in the row are left unchanged unless explicitly changed
        by ``mutation``.

        Idempotent operations (i.e, all mutations have an explicit timestamp) will be
        retried on server failure. Non-idempotent operations will not.

        Args:
            row_key: the row to apply mutations to
            mutations: the set of mutations to apply to the row
            operation_timeout: the time budget for the entire operation, in seconds.
                Failed requests will be retried within the budget.
                Defaults to the Table's default_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Only idempotent mutations will be retried. Defaults to the Table's
                default_retryable_errors.
        Raises:
            google.api_core.exceptions.DeadlineExceeded: raised after operation timeout
                will be chained with a RetryExceptionGroup containing all
                GoogleAPIError exceptions from any retries that failed
            google.api_core.exceptions.GoogleAPIError: raised on non-idempotent operations that cannot be
                safely retried.
            ValueError: if invalid arguments are provided
        """
        operation_timeout, attempt_timeout = _get_timeouts(
            operation_timeout, attempt_timeout, self
        )

        if not mutations:
            raise ValueError("No mutations provided")
        mutations_list = mutations if isinstance(mutations, list) else [mutations]

        if all(mutation.is_idempotent() for mutation in mutations_list):
            # mutations are all idempotent and safe to retry
            predicate = retries.if_exception_type(
                *_get_retryable_errors(retryable_errors, self)
            )
        else:
            # mutations should not be retried
            predicate = retries.if_exception_type()

        sleep_generator = retries.exponential_sleep_generator(0.01, 2, 60)

        target = partial(
            self.client._gapic_client.mutate_row,
            request=MutateRowRequest(
                row_key=row_key.encode("utf-8")
                if isinstance(row_key, str)
                else row_key,
                mutations=[mutation._to_pb() for mutation in mutations_list],
                app_profile_id=self.app_profile_id,
                **self._request_path,
            ),
            timeout=attempt_timeout,
            retry=None,
        )
        return await CrossSync.retry_target(
            target,
            predicate,
            sleep_generator,
            operation_timeout,
            exception_factory=_retry_exception_factory,
        )

    @CrossSync.convert
    async def bulk_mutate_rows(
        self,
        mutation_entries: list[RowMutationEntry],
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
        attempt_timeout: float | None | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
        retryable_errors: Sequence[type[Exception]]
        | TABLE_DEFAULT = TABLE_DEFAULT.MUTATE_ROWS,
    ):
        """
        Applies mutations for multiple rows in a single batched request.

        Each individual RowMutationEntry is applied atomically, but separate entries
        may be applied in arbitrary order (even for entries targetting the same row)
        In total, the row_mutations can contain at most 100000 individual mutations
        across all entries

        Idempotent entries (i.e., entries with mutations with explicit timestamps)
        will be retried on failure. Non-idempotent will not, and will reported in a
        raised exception group

        Args:
            mutation_entries: the batches of mutations to apply
                Each entry will be applied atomically, but entries will be applied
                in arbitrary order
            operation_timeout: the time budget for the entire operation, in seconds.
                Failed requests will be retried within the budget.
                Defaults to the Table's default_mutate_rows_operation_timeout
            attempt_timeout: the time budget for an individual network request, in seconds.
                If it takes longer than this time to complete, the request will be cancelled with
                a DeadlineExceeded exception, and a retry will be attempted.
                Defaults to the Table's default_mutate_rows_attempt_timeout.
                If None, defaults to operation_timeout.
            retryable_errors: a list of errors that will be retried if encountered.
                Defaults to the Table's default_mutate_rows_retryable_errors
        Raises:
            MutationsExceptionGroup: if one or more mutations fails
                Contains details about any failed entries in .exceptions
            ValueError: if invalid arguments are provided
        """
        operation_timeout, attempt_timeout = _get_timeouts(
            operation_timeout, attempt_timeout, self
        )
        retryable_excs = _get_retryable_errors(retryable_errors, self)

        operation = CrossSync._MutateRowsOperation(
            self.client._gapic_client,
            self,
            mutation_entries,
            operation_timeout,
            attempt_timeout,
            retryable_exceptions=retryable_excs,
        )
        await operation.start()

    @CrossSync.convert
    async def check_and_mutate_row(
        self,
        row_key: str | bytes,
        predicate: RowFilter | None,
        *,
        true_case_mutations: Mutation | list[Mutation] | None = None,
        false_case_mutations: Mutation | list[Mutation] | None = None,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
    ) -> bool:
        """
        Mutates a row atomically based on the output of a predicate filter

        Non-idempotent operation: will not be retried

        Args:
            row_key: the key of the row to mutate
            predicate: the filter to be applied to the contents of the specified row.
                Depending on whether or not any results  are yielded,
                either true_case_mutations or false_case_mutations will be executed.
                If None, checks that the row contains any values at all.
            true_case_mutations:
                Changes to be atomically applied to the specified row if
                predicate yields at least one cell when
                applied to row_key. Entries are applied in order,
                meaning that earlier mutations can be masked by later
                ones. Must contain at least one entry if
                false_case_mutations is empty, and at most 100000.
            false_case_mutations:
                Changes to be atomically applied to the specified row if
                predicate_filter does not yield any cells when
                applied to row_key. Entries are applied in order,
                meaning that earlier mutations can be masked by later
                ones. Must contain at least one entry if
                `true_case_mutations` is empty, and at most 100000.
            operation_timeout: the time budget for the entire operation, in seconds.
                Failed requests will not be retried. Defaults to the Table's default_operation_timeout
        Returns:
            bool indicating whether the predicate was true or false
        Raises:
            google.api_core.exceptions.GoogleAPIError: exceptions from grpc call
        """
        operation_timeout, _ = _get_timeouts(operation_timeout, None, self)
        if true_case_mutations is not None and not isinstance(
            true_case_mutations, list
        ):
            true_case_mutations = [true_case_mutations]
        true_case_list = [m._to_pb() for m in true_case_mutations or []]
        if false_case_mutations is not None and not isinstance(
            false_case_mutations, list
        ):
            false_case_mutations = [false_case_mutations]
        false_case_list = [m._to_pb() for m in false_case_mutations or []]
        result = await self.client._gapic_client.check_and_mutate_row(
            request=CheckAndMutateRowRequest(
                true_mutations=true_case_list,
                false_mutations=false_case_list,
                predicate_filter=predicate._to_pb() if predicate is not None else None,
                row_key=row_key.encode("utf-8")
                if isinstance(row_key, str)
                else row_key,
                app_profile_id=self.app_profile_id,
                **self._request_path,
            ),
            timeout=operation_timeout,
            retry=None,
        )
        return result.predicate_matched

    @CrossSync.convert
    async def read_modify_write_row(
        self,
        row_key: str | bytes,
        rules: ReadModifyWriteRule | list[ReadModifyWriteRule],
        *,
        operation_timeout: float | TABLE_DEFAULT = TABLE_DEFAULT.DEFAULT,
    ) -> Row:
        """
        Reads and modifies a row atomically according to input ReadModifyWriteRules,
        and returns the contents of all modified cells

        The new value for the timestamp is the greater of the existing timestamp or
        the current server time.

        Non-idempotent operation: will not be retried

        Args:
            row_key: the key of the row to apply read/modify/write rules to
            rules: A rule or set of rules to apply to the row.
                Rules are applied in order, meaning that earlier rules will affect the
                results of later ones.
            operation_timeout: the time budget for the entire operation, in seconds.
                Failed requests will not be retried.
                Defaults to the Table's default_operation_timeout.
        Returns:
            Row: a Row containing cell data that was modified as part of the operation
        Raises:
            google.api_core.exceptions.GoogleAPIError: exceptions from grpc call
            ValueError: if invalid arguments are provided
        """
        operation_timeout, _ = _get_timeouts(operation_timeout, None, self)
        if operation_timeout <= 0:
            raise ValueError("operation_timeout must be greater than 0")
        if rules is not None and not isinstance(rules, list):
            rules = [rules]
        if not rules:
            raise ValueError("rules must contain at least one item")
        result = await self.client._gapic_client.read_modify_write_row(
            request=ReadModifyWriteRowRequest(
                rules=[rule._to_pb() for rule in rules],
                row_key=row_key.encode("utf-8")
                if isinstance(row_key, str)
                else row_key,
                app_profile_id=self.app_profile_id,
                **self._request_path,
            ),
            timeout=operation_timeout,
            retry=None,
        )
        # construct Row from result
        return Row._from_pb(result.row)

    @CrossSync.convert
    async def close(self):
        """
        Called to close the Table instance and release any resources held by it.
        """
        if self._register_instance_future:
            self._register_instance_future.cancel()
        self.client._remove_instance_registration(
            self.instance_id, self.app_profile_id, id(self)
        )

    @CrossSync.convert(sync_name="__enter__")
    async def __aenter__(self):
        """
        Implement async context manager protocol

        Ensure registration task has time to run, so that
        grpc channels will be warmed for the specified instance
        """
        if self._register_instance_future:
            await self._register_instance_future
        return self

    @CrossSync.convert(sync_name="__exit__")
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Implement async context manager protocol

        Unregister this instance with the client, so that
        grpc channels will no longer be warmed
        """
        await self.close()


@CrossSync.convert_class(
    sync_name="Table",
    add_mapping_for_name="Table",
    replace_symbols={"_DataApiTargetAsync": "_DataApiTarget"},
)
class TableAsync(_DataApiTargetAsync):
    """
    Main Data API surface for interacting with a Bigtable table.

    Table object maintains table_id, and app_profile_id context, and passes them with
    each call
    """

    @property
    def _request_path(self) -> dict[str, str]:
        return {"table_name": self.table_name}


@CrossSync.convert_class(
    sync_name="AuthorizedView",
    add_mapping_for_name="AuthorizedView",
    replace_symbols={"_DataApiTargetAsync": "_DataApiTarget"},
)
class AuthorizedViewAsync(_DataApiTargetAsync):
    """
    Provides access to an authorized view of a table.

    An authorized view is a subset of a table that you configure to include specific table data.
    Then you grant access to the authorized view separately from access to the table.

    AuthorizedView object maintains table_id, app_profile_id, and authorized_view_id context,
    and passed them with each call
    """

    @CrossSync.convert(
        docstring_format_vars={
            "LOOP_MESSAGE": (
                "Must be created within an async context (running event loop)",
                "",
            ),
            "RAISE_NO_LOOP": (
                "RuntimeError: if called outside of an async context (no running event loop)",
                "None",
            ),
        }
    )
    def __init__(
        self,
        client,
        instance_id,
        table_id,
        authorized_view_id,
        app_profile_id: str | None = None,
        **kwargs,
    ):
        """
        Initialize an AuthorizedView instance

        {LOOP_MESSAGE}

        Args:
            instance_id: The Bigtable instance ID to associate with this client.
                instance_id is combined with the client's project to fully
                specify the instance
            table_id: The ID of the table. table_id is combined with the
                instance_id and the client's project to fully specify the table
            authorized_view_id: The id for the authorized view to use for requests
            app_profile_id: The app profile to associate with requests.
                https://cloud.google.com/bigtable/docs/app-profiles
            default_read_rows_operation_timeout: The default timeout for read rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_read_rows_attempt_timeout: The default timeout for individual
                read rows rpc requests, in seconds. If not set, defaults to 20 seconds
            default_mutate_rows_operation_timeout: The default timeout for mutate rows
                operations, in seconds. If not set, defaults to 600 seconds (10 minutes)
            default_mutate_rows_attempt_timeout: The default timeout for individual
                mutate rows rpc requests, in seconds. If not set, defaults to 60 seconds
            default_operation_timeout: The default timeout for all other operations, in
                seconds. If not set, defaults to 60 seconds
            default_attempt_timeout: The default timeout for all other individual rpc
                requests, in seconds. If not set, defaults to 20 seconds
            default_read_rows_retryable_errors: a list of errors that will be retried
                if encountered during read_rows and related operations.
                Defaults to 4 (DeadlineExceeded), 14 (ServiceUnavailable), and 10 (Aborted)
            default_mutate_rows_retryable_errors: a list of errors that will be retried
                if encountered during mutate_rows and related operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
            default_retryable_errors: a list of errors that will be retried if
                encountered during all other operations.
                Defaults to 4 (DeadlineExceeded) and 14 (ServiceUnavailable)
        Raises:
            {RAISE_NO_LOOP}
        """
        super().__init__(client, instance_id, table_id, app_profile_id, **kwargs)
        self.authorized_view_id = authorized_view_id
        self.authorized_view_name: str = self.client._gapic_client.authorized_view_path(
            self.client.project, instance_id, table_id, authorized_view_id
        )

    @property
    def _request_path(self) -> dict[str, str]:
        return {"authorized_view_name": self.authorized_view_name}

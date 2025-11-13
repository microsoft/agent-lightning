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

from typing import Sequence, TYPE_CHECKING

from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
import google.cloud.bigtable_v2.types.bigtable as types_pb
import google.cloud.bigtable.data.exceptions as bt_exceptions
from google.cloud.bigtable.data._helpers import _attempt_timeout_generator
from google.cloud.bigtable.data._helpers import _retry_exception_factory

# mutate_rows requests are limited to this number of mutations
from google.cloud.bigtable.data.mutations import _MUTATE_ROWS_REQUEST_MUTATION_LIMIT
from google.cloud.bigtable.data.mutations import _EntryWithProto

from google.cloud.bigtable.data._cross_sync import CrossSync

if TYPE_CHECKING:
    from google.cloud.bigtable.data.mutations import RowMutationEntry

    if CrossSync.is_async:
        from google.cloud.bigtable_v2.services.bigtable.async_client import (
            BigtableAsyncClient as GapicClientType,
        )
        from google.cloud.bigtable.data._async.client import (  # type: ignore
            _DataApiTargetAsync as TargetType,
        )
    else:
        from google.cloud.bigtable_v2.services.bigtable.client import (  # type: ignore
            BigtableClient as GapicClientType,
        )
        from google.cloud.bigtable.data._sync_autogen.client import (  # type: ignore
            _DataApiTarget as TargetType,
        )

__CROSS_SYNC_OUTPUT__ = "google.cloud.bigtable.data._sync_autogen._mutate_rows"


@CrossSync.convert_class("_MutateRowsOperation")
class _MutateRowsOperationAsync:
    """
    MutateRowsOperation manages the logic of sending a set of row mutations,
    and retrying on failed entries. It manages this using the _run_attempt
    function, which attempts to mutate all outstanding entries, and raises
    _MutateRowsIncomplete if any retryable errors are encountered.

    Errors are exposed as a MutationsExceptionGroup, which contains a list of
    exceptions organized by the related failed mutation entries.

    Args:
        gapic_client: the client to use for the mutate_rows call
        target: the table or view associated with the request
        mutation_entries: a list of RowMutationEntry objects to send to the server
        operation_timeout: the timeout to use for the entire operation, in seconds.
        attempt_timeout: the timeout to use for each mutate_rows attempt, in seconds.
            If not specified, the request will run until operation_timeout is reached.
    """

    @CrossSync.convert
    def __init__(
        self,
        gapic_client: GapicClientType,
        target: TargetType,
        mutation_entries: list["RowMutationEntry"],
        operation_timeout: float,
        attempt_timeout: float | None,
        retryable_exceptions: Sequence[type[Exception]] = (),
    ):
        # check that mutations are within limits
        total_mutations = sum(len(entry.mutations) for entry in mutation_entries)
        if total_mutations > _MUTATE_ROWS_REQUEST_MUTATION_LIMIT:
            raise ValueError(
                "mutate_rows requests can contain at most "
                f"{_MUTATE_ROWS_REQUEST_MUTATION_LIMIT} mutations across "
                f"all entries. Found {total_mutations}."
            )
        self._target = target
        self._gapic_fn = gapic_client.mutate_rows
        # create predicate for determining which errors are retryable
        self.is_retryable = retries.if_exception_type(
            # RPC level errors
            *retryable_exceptions,
            # Entry level errors
            bt_exceptions._MutateRowsIncomplete,
        )
        sleep_generator = retries.exponential_sleep_generator(0.01, 2, 60)
        self._operation = lambda: CrossSync.retry_target(
            self._run_attempt,
            self.is_retryable,
            sleep_generator,
            operation_timeout,
            exception_factory=_retry_exception_factory,
        )
        # initialize state
        self.timeout_generator = _attempt_timeout_generator(
            attempt_timeout, operation_timeout
        )
        self.mutations = [_EntryWithProto(m, m._to_pb()) for m in mutation_entries]
        self.remaining_indices = list(range(len(self.mutations)))
        self.errors: dict[int, list[Exception]] = {}

    @CrossSync.convert
    async def start(self):
        """
        Start the operation, and run until completion

        Raises:
            MutationsExceptionGroup: if any mutations failed
        """
        try:
            # trigger mutate_rows
            await self._operation()
        except Exception as exc:
            # exceptions raised by retryable are added to the list of exceptions for all unfinalized mutations
            incomplete_indices = self.remaining_indices.copy()
            for idx in incomplete_indices:
                self._handle_entry_error(idx, exc)
        finally:
            # raise exception detailing incomplete mutations
            all_errors: list[Exception] = []
            for idx, exc_list in self.errors.items():
                if len(exc_list) == 0:
                    raise core_exceptions.ClientError(
                        f"Mutation {idx} failed with no associated errors"
                    )
                elif len(exc_list) == 1:
                    cause_exc = exc_list[0]
                else:
                    cause_exc = bt_exceptions.RetryExceptionGroup(exc_list)
                entry = self.mutations[idx].entry
                all_errors.append(
                    bt_exceptions.FailedMutationEntryError(idx, entry, cause_exc)
                )
            if all_errors:
                raise bt_exceptions.MutationsExceptionGroup(
                    all_errors, len(self.mutations)
                )

    @CrossSync.convert
    async def _run_attempt(self):
        """
        Run a single attempt of the mutate_rows rpc.

        Raises:
            _MutateRowsIncomplete: if there are failed mutations eligible for
                retry after the attempt is complete
            GoogleAPICallError: if the gapic rpc fails
        """
        request_entries = [self.mutations[idx].proto for idx in self.remaining_indices]
        # track mutations in this request that have not been finalized yet
        active_request_indices = {
            req_idx: orig_idx for req_idx, orig_idx in enumerate(self.remaining_indices)
        }
        self.remaining_indices = []
        if not request_entries:
            # no more mutations. return early
            return
        # make gapic request
        try:
            result_generator = await self._gapic_fn(
                request=types_pb.MutateRowsRequest(
                    entries=request_entries,
                    app_profile_id=self._target.app_profile_id,
                    **self._target._request_path,
                ),
                timeout=next(self.timeout_generator),
                retry=None,
            )
            async for result_list in result_generator:
                for result in result_list.entries:
                    # convert sub-request index to global index
                    orig_idx = active_request_indices[result.index]
                    entry_error = core_exceptions.from_grpc_status(
                        result.status.code,
                        result.status.message,
                        details=result.status.details,
                    )
                    if result.status.code != 0:
                        # mutation failed; update error list (and remaining_indices if retryable)
                        self._handle_entry_error(orig_idx, entry_error)
                    elif orig_idx in self.errors:
                        # mutation succeeded; remove from error list
                        del self.errors[orig_idx]
                    # remove processed entry from active list
                    del active_request_indices[result.index]
        except Exception as exc:
            # add this exception to list for each mutation that wasn't
            # already handled, and update remaining_indices if mutation is retryable
            for idx in active_request_indices.values():
                self._handle_entry_error(idx, exc)
            # bubble up exception to be handled by retry wrapper
            raise
        # check if attempt succeeded, or needs to be retried
        if self.remaining_indices:
            # unfinished work; raise exception to trigger retry
            raise bt_exceptions._MutateRowsIncomplete

    def _handle_entry_error(self, idx: int, exc: Exception):
        """
        Add an exception to the list of exceptions for a given mutation index,
        and add the index to the list of remaining indices if the exception is
        retryable.

        Args:
            idx: the index of the mutation that failed
            exc: the exception to add to the list
        """
        entry = self.mutations[idx].entry
        self.errors.setdefault(idx, []).append(exc)
        if (
            entry.is_idempotent()
            and self.is_retryable(exc)
            and idx not in self.remaining_indices
        ):
            self.remaining_indices.append(idx)

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
"""
Helper functions used in various places in the library.
"""
from __future__ import annotations

from typing import Sequence, List, Tuple, TYPE_CHECKING, Union
import time
import enum
from collections import namedtuple
from google.cloud.bigtable.data.read_rows_query import ReadRowsQuery

from google.api_core import exceptions as core_exceptions
from google.api_core.retry import RetryFailureReason
from google.cloud.bigtable.data.exceptions import RetryExceptionGroup

if TYPE_CHECKING:
    import grpc
    from google.cloud.bigtable.data._async.client import _DataApiTargetAsync
    from google.cloud.bigtable.data._sync_autogen.client import _DataApiTarget

"""
Helper functions used in various places in the library.
"""

# Type alias for the output of sample_keys
RowKeySamples = List[Tuple[bytes, int]]

# type alias for the output of query.shard()
ShardedQuery = List[ReadRowsQuery]

# used by read_rows_sharded to limit how many requests are attempted in parallel
_CONCURRENCY_LIMIT = 10

# used to identify an active bigtable resource that needs to be warmed through PingAndWarm
# each instance/app_profile_id pair needs to be individually tracked
_WarmedInstanceKey = namedtuple(
    "_WarmedInstanceKey", ["instance_name", "app_profile_id"]
)


# enum used on method calls when table defaults should be used
class TABLE_DEFAULT(enum.Enum):
    # default for mutate_row, sample_row_keys, check_and_mutate_row, and read_modify_write_row
    DEFAULT = "DEFAULT"
    # default for read_rows, read_rows_stream, read_rows_sharded, row_exists, and read_row
    READ_ROWS = "READ_ROWS_DEFAULT"
    # default for bulk_mutate_rows and mutations_batcher
    MUTATE_ROWS = "MUTATE_ROWS_DEFAULT"


def _attempt_timeout_generator(
    per_request_timeout: float | None, operation_timeout: float
):
    """
    Generator that yields the timeout value for each attempt of a retry loop.

    Will return per_request_timeout until the operation_timeout is approached,
    at which point it will return the remaining time in the operation_timeout.

    Args:
        per_request_timeout: The timeout value to use for each request, in seconds.
            If None, the operation_timeout will be used for each request.
        operation_timeout: The timeout value to use for the entire operationm in seconds.
    Yields:
        float: The timeout value to use for the next request, in seonds
    """
    per_request_timeout = (
        per_request_timeout if per_request_timeout is not None else operation_timeout
    )
    deadline = operation_timeout + time.monotonic()
    while True:
        yield max(0, min(per_request_timeout, deadline - time.monotonic()))


def _retry_exception_factory(
    exc_list: list[Exception],
    reason: RetryFailureReason,
    timeout_val: float | None,
) -> tuple[Exception, Exception | None]:
    """
    Build retry error based on exceptions encountered during operation

    Args:
        exc_list: list of exceptions encountered during operation
        is_timeout: whether the operation failed due to timeout
        timeout_val: the operation timeout value in seconds, for constructing
            the error message
    Returns:
        tuple[Exception, Exception|None]:
            tuple of the exception to raise, and a cause exception if applicable
    """
    if reason == RetryFailureReason.TIMEOUT:
        timeout_val_str = f"of {timeout_val:0.1f}s " if timeout_val is not None else ""
        # if failed due to timeout, raise deadline exceeded as primary exception
        source_exc: Exception = core_exceptions.DeadlineExceeded(
            f"operation_timeout{timeout_val_str} exceeded"
        )
    elif exc_list:
        # otherwise, raise non-retryable error as primary exception
        source_exc = exc_list.pop()
    else:
        source_exc = RuntimeError("failed with unspecified exception")
    # use the retry exception group as the cause of the exception
    cause_exc: Exception | None = RetryExceptionGroup(exc_list) if exc_list else None
    source_exc.__cause__ = cause_exc
    return source_exc, cause_exc


def _get_timeouts(
    operation: float | TABLE_DEFAULT,
    attempt: float | None | TABLE_DEFAULT,
    table: "_DataApiTargetAsync" | "_DataApiTarget",
) -> tuple[float, float]:
    """
    Convert passed in timeout values to floats, using table defaults if necessary.

    attempt will use operation value if None, or if larger than operation.

    Will call _validate_timeouts on the outputs, and raise ValueError if the
    resulting timeouts are invalid.

    Args:
        operation: The timeout value to use for the entire operation, in seconds.
        attempt: The timeout value to use for each attempt, in seconds.
        table: The table to use for default values.
    Returns:
        tuple[float, float]: A tuple of (operation_timeout, attempt_timeout)
    """
    # load table defaults if necessary
    if operation == TABLE_DEFAULT.DEFAULT:
        final_operation = table.default_operation_timeout
    elif operation == TABLE_DEFAULT.READ_ROWS:
        final_operation = table.default_read_rows_operation_timeout
    elif operation == TABLE_DEFAULT.MUTATE_ROWS:
        final_operation = table.default_mutate_rows_operation_timeout
    else:
        final_operation = operation
    if attempt == TABLE_DEFAULT.DEFAULT:
        attempt = table.default_attempt_timeout
    elif attempt == TABLE_DEFAULT.READ_ROWS:
        attempt = table.default_read_rows_attempt_timeout
    elif attempt == TABLE_DEFAULT.MUTATE_ROWS:
        attempt = table.default_mutate_rows_attempt_timeout

    return _align_timeouts(final_operation, attempt)


def _align_timeouts(operation: float, attempt: float | None) -> tuple[float, float]:
    """
    Convert passed in timeout values to floats.

    attempt will use operation value if None, or if larger than operation.

    Will call _validate_timeouts on the outputs, and raise ValueError if the
    resulting timeouts are invalid.

    Args:
        operation: The timeout value to use for the entire operation, in seconds.
        attempt: The timeout value to use for each attempt, in seconds.
    Returns:
        tuple[float, float]: A tuple of (operation_timeout, attempt_timeout)
    """
    if attempt is None:
        # no timeout specified, use operation timeout for both
        final_attempt = operation
    else:
        # cap attempt timeout at operation timeout
        final_attempt = min(attempt, operation) if operation else attempt

    _validate_timeouts(operation, final_attempt, allow_none=False)
    return operation, final_attempt


def _validate_timeouts(
    operation_timeout: float, attempt_timeout: float | None, allow_none: bool = False
):
    """
    Helper function that will verify that timeout values are valid, and raise
    an exception if they are not.

    Args:
        operation_timeout: The timeout value to use for the entire operation, in seconds.
        attempt_timeout: The timeout value to use for each attempt, in seconds.
        allow_none: If True, attempt_timeout can be None. If False, None values will raise an exception.
    Raises:
        ValueError: if operation_timeout or attempt_timeout are invalid.
    """
    if operation_timeout is None:
        raise ValueError("operation_timeout cannot be None")
    if operation_timeout <= 0:
        raise ValueError("operation_timeout must be greater than 0")
    if not allow_none and attempt_timeout is None:
        raise ValueError("attempt_timeout must not be None")
    elif attempt_timeout is not None:
        if attempt_timeout <= 0:
            raise ValueError("attempt_timeout must be greater than 0")


def _get_error_type(
    call_code: Union["grpc.StatusCode", int, type[Exception]]
) -> type[Exception]:
    """Helper function for ensuring the object is an exception type.
    If it is not, the proper GoogleAPICallError type is infered from the status
    code.

    Args:
      - call_code: Exception type or gRPC status code.
    """
    if isinstance(call_code, type):
        return call_code
    else:
        return type(core_exceptions.from_grpc_status(call_code, ""))


def _get_retryable_errors(
    call_codes: Sequence["grpc.StatusCode" | int | type[Exception]] | TABLE_DEFAULT,
    table: "_DataApiTargetAsync" | "_DataApiTarget",
) -> list[type[Exception]]:
    """
    Convert passed in retryable error codes to a list of exception types.

    Args:
        call_codes: The error codes to convert. Can be a list of grpc.StatusCode values,
            int values, or Exception types, or a TABLE_DEFAULT value.
        table: The table to use for default values.
    Returns:
        list[type[Exception]]: A list of exception types to retry on.
    """
    # load table defaults if necessary
    if call_codes == TABLE_DEFAULT.DEFAULT:
        call_codes = table.default_retryable_errors
    elif call_codes == TABLE_DEFAULT.READ_ROWS:
        call_codes = table.default_read_rows_retryable_errors
    elif call_codes == TABLE_DEFAULT.MUTATE_ROWS:
        call_codes = table.default_mutate_rows_retryable_errors

    return [_get_error_type(e) for e in call_codes]

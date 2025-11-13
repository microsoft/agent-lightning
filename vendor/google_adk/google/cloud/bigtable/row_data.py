# Copyright 2016 Google LLC
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

"""Container for Google Cloud Bigtable Cells and Streaming Row Contents."""


import copy

import grpc  # type: ignore
import warnings
from google.api_core import exceptions
from google.api_core import retry
from google.cloud._helpers import _to_bytes  # type: ignore

from google.cloud.bigtable.row_merger import _RowMerger, _State
from google.cloud.bigtable_v2.types import bigtable as data_messages_v2_pb2
from google.cloud.bigtable_v2.types import data as data_v2_pb2
from google.cloud.bigtable.row import Cell, InvalidChunk, PartialRowData


# Some classes need to be re-exported here to keep backwards
# compatibility. Those classes were moved to row_merger, but we dont want to
# break enduser's imports. This hack, ensures they don't get marked as unused.
_ = (Cell, InvalidChunk, PartialRowData)


class PartialCellData(object):  # pragma: NO COVER
    """This class is no longer used and will be removed in the future"""

    def __init__(
        self, row_key, family_name, qualifier, timestamp_micros, labels=(), value=b""
    ):
        self.row_key = row_key
        self.family_name = family_name
        self.qualifier = qualifier
        self.timestamp_micros = timestamp_micros
        self.labels = labels
        self.value = value

    def append_value(self, value):
        self.value += value


class InvalidReadRowsResponse(RuntimeError):
    """Exception raised to invalid response data from back-end."""


class InvalidRetryRequest(RuntimeError):
    """Exception raised when retry request is invalid."""


RETRYABLE_INTERNAL_ERROR_MESSAGES = (
    "rst_stream",
    "rst stream",
    "received unexpected eos on data frame from server",
)
"""Internal error messages that can be retried during read row and mutation."""


def _retriable_internal_server_error(exc):
    """
    Return True if the internal server error is retriable.
    """
    return isinstance(exc, exceptions.InternalServerError) and any(
        retryable_message in exc.message.lower()
        for retryable_message in RETRYABLE_INTERNAL_ERROR_MESSAGES
    )


def _retry_read_rows_exception(exc):
    """Return True if the exception is retriable for read row requests."""
    if isinstance(exc, grpc.RpcError):
        exc = exceptions.from_grpc_error(exc)

    return _retriable_internal_server_error(exc) or isinstance(
        exc, (exceptions.ServiceUnavailable, exceptions.DeadlineExceeded)
    )


DEFAULT_RETRY_READ_ROWS = retry.Retry(
    predicate=_retry_read_rows_exception,
    initial=1.0,
    maximum=15.0,
    multiplier=2.0,
    deadline=60.0,  # 60 seconds
)
"""The default retry strategy to be used on retry-able errors.

Used by
:meth:`~google.cloud.bigtable.row_data.PartialRowsData._read_next_response`.
"""


class PartialRowsData(object):
    """Convenience wrapper for consuming a ``ReadRows`` streaming response.

    :type read_method: :class:`client._table_data_client.read_rows`
    :param read_method: ``ReadRows`` method.

    :type request: :class:`data_messages_v2_pb2.ReadRowsRequest`
    :param request: The ``ReadRowsRequest`` message used to create a
                    ReadRowsResponse iterator. If the iterator fails, a new
                    iterator is created, allowing the scan to continue from
                    the point just beyond the last successfully read row,
                    identified by self.last_scanned_row_key. The retry happens
                    inside of the Retry class, using a predicate for the
                    expected exceptions during iteration.

    :type retry: :class:`~google.api_core.retry.Retry`
    :param retry: (Optional) Retry delay and deadline arguments. To override,
                  the default value :attr:`DEFAULT_RETRY_READ_ROWS` can be
                  used and modified with the
                  :meth:`~google.api_core.retry.Retry.with_delay` method
                  or the
                  :meth:`~google.api_core.retry.Retry.with_deadline` method.
    """

    NEW_ROW = "New row"  # No cells yet complete for row
    ROW_IN_PROGRESS = "Row in progress"  # Some cells complete for row
    CELL_IN_PROGRESS = "Cell in progress"  # Incomplete cell for row

    STATE_NEW_ROW = 1
    STATE_ROW_IN_PROGRESS = 2
    STATE_CELL_IN_PROGRESS = 3

    read_states = {
        STATE_NEW_ROW: NEW_ROW,
        STATE_ROW_IN_PROGRESS: ROW_IN_PROGRESS,
        STATE_CELL_IN_PROGRESS: CELL_IN_PROGRESS,
    }

    def __init__(self, read_method, request, retry=DEFAULT_RETRY_READ_ROWS):
        # Counter for rows returned to the user
        self._counter = 0
        self._row_merger = _RowMerger()

        # May be cached from previous response
        self.last_scanned_row_key = None
        self.read_method = read_method
        self.request = request
        self.retry = retry

        # The `timeout` parameter must be somewhat greater than the value
        # contained in `self.retry`, in order to avoid race-like condition and
        # allow registering the first deadline error before invoking the retry.
        # Otherwise there is a risk of entering an infinite loop that resets
        # the timeout counter just before it being triggered. The increment
        # by 1 second here is customary but should not be much less than that.
        self.response_iterator = read_method(
            request, timeout=self.retry._deadline + 1, retry=self.retry
        )

        self.rows = {}

        # Flag to stop iteration, for any reason not related to self.retry()
        self._cancelled = False

    @property
    def state(self):  # pragma: NO COVER
        """
        DEPRECATED: this property is deprecated and will be removed in the
        future.
        """
        warnings.warn(
            "`PartialRowsData#state()` is deprecated and will be removed in the future",
            DeprecationWarning,
            stacklevel=2,
        )

        # Best effort: try to map internal RowMerger states to old strings for
        # backwards compatibility
        internal_state = self._row_merger.state
        if internal_state == _State.ROW_START:
            return self.NEW_ROW
        # note: _State.CELL_START, _State.CELL_COMPLETE are transient states
        # and will not be visible in between chunks
        elif internal_state == _State.CELL_IN_PROGRESS:
            return self.CELL_IN_PROGRESS
        elif internal_state == _State.ROW_COMPLETE:
            return self.NEW_ROW
        else:
            raise RuntimeError("unexpected internal state: " + self._)

    def cancel(self):
        """Cancels the iterator, closing the stream."""
        self._cancelled = True
        self.response_iterator.cancel()

    def consume_all(self, max_loops=None):
        """Consume the streamed responses until there are no more.

        .. warning::
           This method will be removed in future releases.  Please use this
           class as a generator instead.

        :type max_loops: int
        :param max_loops: (Optional) Maximum number of times to try to consume
                          an additional ``ReadRowsResponse``. You can use this
                          to avoid long wait times.
        """
        for row in self:
            self.rows[row.row_key] = row

    def _create_retry_request(self):
        """Helper for :meth:`__iter__`."""
        req_manager = _ReadRowsRequestManager(
            self.request, self.last_scanned_row_key, self._counter
        )
        return req_manager.build_updated_request()

    def _on_error(self, exc):
        """Helper for :meth:`__iter__`."""
        # restart the read scan from AFTER the last successfully read row
        retry_request = self.request
        if self.last_scanned_row_key:
            retry_request = self._create_retry_request()

        self._row_merger = _RowMerger(self._row_merger.last_seen_row_key)
        self.response_iterator = self.read_method(retry_request)

    def _read_next(self):
        """Helper for :meth:`__iter__`."""
        return next(self.response_iterator)

    def _read_next_response(self):
        """Helper for :meth:`__iter__`."""
        resp_protoplus = self.retry(self._read_next, on_error=self._on_error)()
        # unwrap the underlying protobuf, there is a significant amount of
        # overhead that protoplus imposes for very little gain. The protos
        # are not user visible, so we just use the raw protos for merging.
        return data_messages_v2_pb2.ReadRowsResponse.pb(resp_protoplus)

    def __iter__(self):
        """Consume the ``ReadRowsResponse`` s from the stream.
        Read the rows and yield each to the reader

        Parse the response and its chunks into a new/existing row in
        :attr:`_rows`. Rows are returned in order by row key.
        """
        while not self._cancelled:
            try:
                response = self._read_next_response()
            except StopIteration:
                self._row_merger.finalize()
                break
            except InvalidRetryRequest:
                self._cancelled = True
                break

            for row in self._row_merger.process_chunks(response):
                self.last_scanned_row_key = self._row_merger.last_seen_row_key
                self._counter += 1

                yield row

                if self._cancelled:
                    break
            # The last response might not have generated any rows, but it
            # could've updated last_scanned_row_key
            self.last_scanned_row_key = self._row_merger.last_seen_row_key


class _ReadRowsRequestManager(object):
    """Update the ReadRowsRequest message in case of failures by
        filtering the already read keys.

    :type message: class:`data_messages_v2_pb2.ReadRowsRequest`
    :param message: Original ReadRowsRequest containing all of the parameters
                    of API call

    :type last_scanned_key: bytes
    :param last_scanned_key: last successfully scanned key

    :type rows_read_so_far: int
    :param rows_read_so_far: total no of rows successfully read so far.
                            this will be used for updating rows_limit

    """

    def __init__(self, message, last_scanned_key, rows_read_so_far):
        self.message = message
        self.last_scanned_key = last_scanned_key
        self.rows_read_so_far = rows_read_so_far

    def build_updated_request(self):
        """Updates the given message request as per last scanned key"""

        resume_request = data_messages_v2_pb2.ReadRowsRequest()
        data_messages_v2_pb2.ReadRowsRequest.copy_from(resume_request, self.message)

        if self.message.rows_limit != 0:
            row_limit_remaining = self.message.rows_limit - self.rows_read_so_far
            if row_limit_remaining > 0:
                resume_request.rows_limit = row_limit_remaining
            else:
                raise InvalidRetryRequest

        # if neither RowSet.row_keys nor RowSet.row_ranges currently exist,
        # add row_range that starts with last_scanned_key as start_key_open
        # to request only rows that have not been returned yet
        if "rows" not in self.message:
            row_range = data_v2_pb2.RowRange(start_key_open=self.last_scanned_key)
            resume_request.rows = data_v2_pb2.RowSet(row_ranges=[row_range])
        else:
            row_keys = self._filter_rows_keys()
            row_ranges = self._filter_row_ranges()

            if len(row_keys) == 0 and len(row_ranges) == 0:
                # Avoid sending empty row_keys and row_ranges
                # if that was not the intention
                raise InvalidRetryRequest

            resume_request.rows = data_v2_pb2.RowSet(
                row_keys=row_keys, row_ranges=row_ranges
            )
        return resume_request

    def _filter_rows_keys(self):
        """Helper for :meth:`build_updated_request`"""
        return [
            row_key
            for row_key in self.message.rows.row_keys
            if row_key > self.last_scanned_key
        ]

    def _filter_row_ranges(self):
        """Helper for :meth:`build_updated_request`"""
        new_row_ranges = []

        for row_range in self.message.rows.row_ranges:
            # if current end_key (open or closed) is set, return its value,
            # if not, set to empty string ('').
            # NOTE: Empty string in end_key means "end of table"
            end_key = self._end_key_set(row_range)
            # if end_key is already read, skip to the next row_range
            if end_key and self._key_already_read(end_key):
                continue

            # if current start_key (open or closed) is set, return its value,
            # if not, then set to empty string ('')
            # NOTE: Empty string in start_key means "beginning of table"
            start_key = self._start_key_set(row_range)

            # if start_key was already read or doesn't exist,
            # create a row_range with last_scanned_key as start_key_open
            # to be passed to retry request
            retry_row_range = row_range
            if self._key_already_read(start_key):
                retry_row_range = copy.deepcopy(row_range)
                retry_row_range.start_key_closed = _to_bytes("")
                retry_row_range.start_key_open = self.last_scanned_key

            new_row_ranges.append(retry_row_range)

        return new_row_ranges

    def _key_already_read(self, key):
        """Helper for :meth:`_filter_row_ranges`"""
        return key <= self.last_scanned_key

    @staticmethod
    def _start_key_set(row_range):
        """Helper for :meth:`_filter_row_ranges`"""
        return row_range.start_key_open or row_range.start_key_closed

    @staticmethod
    def _end_key_set(row_range):
        """Helper for :meth:`_filter_row_ranges`"""
        return row_range.end_key_open or row_range.end_key_closed

# Copyright 2018 Google LLC
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

"""User friendly container for Google Cloud Bigtable MutationBatcher."""
import threading
import queue
import concurrent.futures
import atexit


from google.api_core.exceptions import from_grpc_status
from dataclasses import dataclass


FLUSH_COUNT = 100  # after this many elements, send out the batch

MAX_MUTATION_SIZE = 20 * 1024 * 1024  # 20MB # after this many bytes, send out the batch

MAX_OUTSTANDING_BYTES = 100 * 1024 * 1024  # 100MB # max inflight byte size.

MAX_OUTSTANDING_ELEMENTS = 100000  # max inflight mutations.


class MutationsBatchError(Exception):
    """Error in the batch request"""

    def __init__(self, message, exc):
        self.exc = exc
        self.message = message
        super().__init__(self.message)


class _MutationsBatchQueue(object):
    """Private Threadsafe Queue to hold rows for batching."""

    def __init__(self, max_mutation_bytes=MAX_MUTATION_SIZE, flush_count=FLUSH_COUNT):
        """Specify the queue constraints"""
        self._queue = queue.Queue()
        self.total_mutation_count = 0
        self.total_size = 0
        self.max_mutation_bytes = max_mutation_bytes
        self.flush_count = flush_count

    def get(self):
        """
        Retrieve an item from the queue. Recalculate queue size.

        If the queue is empty, return None.
        """
        try:
            row = self._queue.get_nowait()
            mutation_size = row.get_mutations_size()
            self.total_mutation_count -= len(row._get_mutations())
            self.total_size -= mutation_size
            return row
        except queue.Empty:
            return None

    def put(self, item):
        """Insert an item to the queue. Recalculate queue size."""

        mutation_count = len(item._get_mutations())

        self._queue.put(item)

        self.total_size += item.get_mutations_size()
        self.total_mutation_count += mutation_count

    def full(self):
        """Check if the queue is full."""
        if (
            self.total_mutation_count >= self.flush_count
            or self.total_size >= self.max_mutation_bytes
        ):
            return True
        return False


@dataclass
class _BatchInfo:
    """Keeping track of size of a batch"""

    mutations_count: int = 0
    rows_count: int = 0
    mutations_size: int = 0


class _FlowControl(object):
    def __init__(
        self,
        max_mutations=MAX_OUTSTANDING_ELEMENTS,
        max_mutation_bytes=MAX_OUTSTANDING_BYTES,
    ):
        """Control the inflight requests. Keep track of the mutations, row bytes and row counts.
        As requests to backend are being made, adjust the number of mutations being processed.

        If threshold is reached, block the flow.
        Reopen the flow as requests are finished.
        """
        self.max_mutations = max_mutations
        self.max_mutation_bytes = max_mutation_bytes
        self.inflight_mutations = 0
        self.inflight_size = 0
        self.event = threading.Event()
        self.event.set()
        self._lock = threading.Lock()

    def is_blocked(self):
        """Returns True if:

        - inflight mutations >= max_mutations, or
        - inflight bytes size >= max_mutation_bytes, or
        """

        return (
            self.inflight_mutations >= self.max_mutations
            or self.inflight_size >= self.max_mutation_bytes
        )

    def control_flow(self, batch_info):
        """
        Calculate the resources used by this batch
        """

        with self._lock:
            self.inflight_mutations += batch_info.mutations_count
            self.inflight_size += batch_info.mutations_size
        self.set_flow_control_status()

    def wait(self):
        """
        Wait until flow control pushback has been released.
        It awakens as soon as `event` is set.
        """
        self.event.wait()

    def set_flow_control_status(self):
        """Check the inflight mutations and size.

        If values exceed the allowed threshold, block the event.
        """
        if self.is_blocked():
            self.event.clear()  # sleep
        else:
            self.event.set()  # awaken the threads

    def release(self, batch_info):
        """
        Release the resources.
        Decrement the row size to allow enqueued mutations to be run.
        """
        with self._lock:
            self.inflight_mutations -= batch_info.mutations_count
            self.inflight_size -= batch_info.mutations_size
        self.set_flow_control_status()


class MutationsBatcher(object):
    """A MutationsBatcher is used in batch cases where the number of mutations
    is large or unknown. It will store :class:`DirectRow` in memory until one of the
    size limits is reached, or an explicit call to :func:`flush()` is performed. When
    a flush event occurs, the :class:`DirectRow` in memory will be sent to Cloud
    Bigtable. Batching mutations is more efficient than sending individual
    request.

    This class is not suited for usage in systems where each mutation
    must be guaranteed to be sent, since calling mutate may only result in an
    in-memory change. In a case of a system crash, any :class:`DirectRow` remaining in
    memory will not necessarily be sent to the service, even after the
    completion of the :func:`mutate()` method.

    Note on thread safety: The same :class:`MutationBatcher` cannot be shared by multiple end-user threads.

    :type table: class
    :param table: class:`~google.cloud.bigtable.table.Table`.

    :type flush_count: int
    :param flush_count: (Optional) Max number of rows to flush. If it
        reaches the max number of rows it calls finish_batch() to mutate the
        current row batch. Default is FLUSH_COUNT (1000 rows).

    :type max_row_bytes: int
    :param max_row_bytes: (Optional) Max number of row mutations size to
        flush. If it reaches the max number of row mutations size it calls
        finish_batch() to mutate the current row batch. Default is MAX_ROW_BYTES
        (5 MB).

    :type flush_interval: float
    :param flush_interval: (Optional) The interval (in seconds) between asynchronous flush.
        Default is 1 second.

    :type batch_completed_callback: Callable[list:[`~google.rpc.status_pb2.Status`]] = None
    :param batch_completed_callback: (Optional) A callable for handling responses
        after the current batch is sent. The callable function expect a list of grpc
        Status.
    """

    def __init__(
        self,
        table,
        flush_count=FLUSH_COUNT,
        max_row_bytes=MAX_MUTATION_SIZE,
        flush_interval=1,
        batch_completed_callback=None,
    ):
        self._rows = _MutationsBatchQueue(
            max_mutation_bytes=max_row_bytes, flush_count=flush_count
        )
        self.table = table
        self._executor = concurrent.futures.ThreadPoolExecutor()
        atexit.register(self.close)
        self._timer = threading.Timer(flush_interval, self.flush)
        self._timer.start()
        self.flow_control = _FlowControl(
            max_mutations=MAX_OUTSTANDING_ELEMENTS,
            max_mutation_bytes=MAX_OUTSTANDING_BYTES,
        )
        self.futures_mapping = {}
        self.exceptions = queue.Queue()
        self._user_batch_completed_callback = batch_completed_callback

    @property
    def flush_count(self):
        return self._rows.flush_count

    @property
    def max_row_bytes(self):
        return self._rows.max_mutation_bytes

    def __enter__(self):
        """Starting the MutationsBatcher as a context manager"""
        return self

    def mutate(self, row):
        """Add a row to the batch. If the current batch meets one of the size
        limits, the batch is sent asynchronously.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_batcher_mutate]
            :end-before: [END bigtable_api_batcher_mutate]
            :dedent: 4

        :type row: class
        :param row: :class:`~google.cloud.bigtable.row.DirectRow`.

        :raises: One of the following:
            * :exc:`~.table._BigtableRetryableError` if any row returned a transient error.
            * :exc:`RuntimeError` if the number of responses doesn't match the number of rows that were retried
        """
        self._rows.put(row)

        if self._rows.full():
            self._flush_async()

    def mutate_rows(self, rows):
        """Add multiple rows to the batch. If the current batch meets one of the size
        limits, the batch is sent asynchronously.

        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_batcher_mutate_rows]
            :end-before: [END bigtable_api_batcher_mutate_rows]
            :dedent: 4

        :type rows: list:[`~google.cloud.bigtable.row.DirectRow`]
        :param rows: list:[`~google.cloud.bigtable.row.DirectRow`].

        :raises: One of the following:
            * :exc:`~.table._BigtableRetryableError` if any row returned a transient error.
            * :exc:`RuntimeError` if the number of responses doesn't match the number of rows that were retried
        """
        for row in rows:
            self.mutate(row)

    def flush(self):
        """Sends the current batch to Cloud Bigtable synchronously.
        For example:

        .. literalinclude:: snippets_table.py
            :start-after: [START bigtable_api_batcher_flush]
            :end-before: [END bigtable_api_batcher_flush]
            :dedent: 4

        :raises:
            * :exc:`.batcherMutationsBatchError` if there's any error in the mutations.
        """
        rows_to_flush = []
        row = self._rows.get()
        while row is not None:
            rows_to_flush.append(row)
            row = self._rows.get()
        response = self._flush_rows(rows_to_flush)
        return response

    def _flush_async(self):
        """Sends the current batch to Cloud Bigtable asynchronously.

        :raises:
            * :exc:`.batcherMutationsBatchError` if there's any error in the mutations.
        """
        next_row = self._rows.get()
        while next_row is not None:
            # start a new batch
            rows_to_flush = [next_row]
            batch_info = _BatchInfo(
                mutations_count=len(next_row._get_mutations()),
                rows_count=1,
                mutations_size=next_row.get_mutations_size(),
            )
            # fill up batch with rows
            next_row = self._rows.get()
            while next_row is not None and self._row_fits_in_batch(
                next_row, batch_info
            ):
                rows_to_flush.append(next_row)
                batch_info.mutations_count += len(next_row._get_mutations())
                batch_info.rows_count += 1
                batch_info.mutations_size += next_row.get_mutations_size()
                next_row = self._rows.get()
            # send batch over network
            # wait for resources to become available
            self.flow_control.wait()
            # once unblocked, submit the batch
            # event flag will be set by control_flow to block subsequent thread, but not blocking this one
            self.flow_control.control_flow(batch_info)
            future = self._executor.submit(self._flush_rows, rows_to_flush)
            # schedule release of resources from flow control
            self.futures_mapping[future] = batch_info
            future.add_done_callback(self._batch_completed_callback)

    def _batch_completed_callback(self, future):
        """Callback for when the mutation has finished to clean up the current batch
        and release items from the flow controller.
        Raise exceptions if there's any.
        Release the resources locked by the flow control and allow enqueued tasks to be run.
        """
        processed_rows = self.futures_mapping[future]
        self.flow_control.release(processed_rows)
        del self.futures_mapping[future]

    def _row_fits_in_batch(self, row, batch_info):
        """Checks if a row can fit in the current batch.

        :type row: class
        :param row: :class:`~google.cloud.bigtable.row.DirectRow`.

        :type batch_info: :class:`_BatchInfo`
        :param batch_info: Information about the current batch.

        :rtype: bool
        :returns: True if the row can fit in the current batch.
        """
        new_rows_count = batch_info.rows_count + 1
        new_mutations_count = batch_info.mutations_count + len(row._get_mutations())
        new_mutations_size = batch_info.mutations_size + row.get_mutations_size()
        return (
            new_rows_count <= self.flush_count
            and new_mutations_size <= self.max_row_bytes
            and new_mutations_count <= self.flow_control.max_mutations
            and new_mutations_size <= self.flow_control.max_mutation_bytes
        )

    def _flush_rows(self, rows_to_flush):
        """Mutate the specified rows.

        :raises:
            * :exc:`.batcherMutationsBatchError` if there's any error in the mutations.
        """
        responses = []
        if len(rows_to_flush) > 0:
            response = self.table.mutate_rows(rows_to_flush)

            if self._user_batch_completed_callback:
                self._user_batch_completed_callback(response)

            for result in response:
                if result.code != 0:
                    exc = from_grpc_status(result.code, result.message)
                    self.exceptions.put(exc)
                responses.append(result)

        return responses

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Clean up resources. Flush and shutdown the ThreadPoolExecutor."""
        self.close()

    def close(self):
        """Clean up resources. Flush and shutdown the ThreadPoolExecutor.
        Any errors will be raised.

        :raises:
            * :exc:`.batcherMutationsBatchError` if there's any error in the mutations.
        """
        self.flush()
        self._executor.shutdown(wait=True)
        atexit.unregister(self.close)
        if self.exceptions.qsize() > 0:
            exc = list(self.exceptions.queue)
            raise MutationsBatchError("Errors in batch mutations.", exc=exc)

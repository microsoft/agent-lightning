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

"""Transport for Python logging handler

Uses a background worker to log to Cloud Logging asynchronously.
"""

from __future__ import print_function

import atexit
import datetime
import logging
import queue
import sys
import threading
import time

from google.cloud.logging_v2 import _helpers
from google.cloud.logging_v2.handlers.transports.base import Transport
from google.cloud.logging_v2.logger import _GLOBAL_RESOURCE

_DEFAULT_GRACE_PERIOD = 5.0  # Seconds
_DEFAULT_MAX_BATCH_SIZE = 10
_DEFAULT_MAX_LATENCY = 0  # Seconds
_WORKER_THREAD_NAME = "google.cloud.logging.Worker"
_WORKER_TERMINATOR = object()
_LOGGER = logging.getLogger(__name__)

_CLOSE_THREAD_SHUTDOWN_ERROR_MSG = (
    "CloudLoggingHandler shutting down, cannot send logs entries to Cloud Logging due to "
    "inconsistent threading behavior at shutdown. To avoid this issue, flush the logging handler "
    "manually or switch to StructuredLogHandler. You can also close the CloudLoggingHandler manually "
    "via handler.close or client.close."
)


def _get_many(queue_, *, max_items=None, max_latency=0):
    """Get multiple items from a Queue.

    Gets at least one (blocking) and at most ``max_items`` items
    (non-blocking) from a given Queue. Does not mark the items as done.

    Args:
        queue_ (queue.Queue): The Queue to get items from.
        max_items (Optional[int]): The maximum number of items to get.
            If ``None``, then all available items in the queue are returned.
        max_latency (Optional[float]): The maximum number of seconds to wait
            for more than one item from a queue. This number includes
            the time required to retrieve the first item.

    Returns:
        list: items retrieved from the queue
    """
    start = time.time()
    # Always return at least one item.
    items = [queue_.get()]
    while max_items is None or len(items) < max_items:
        try:
            elapsed = time.time() - start
            timeout = max(0, max_latency - elapsed)
            items.append(queue_.get(timeout=timeout))
        except queue.Empty:
            break
    return items


class _Worker(object):
    """A background thread that writes batches of log entries."""

    def __init__(
        self,
        cloud_logger,
        *,
        grace_period=_DEFAULT_GRACE_PERIOD,
        max_batch_size=_DEFAULT_MAX_BATCH_SIZE,
        max_latency=_DEFAULT_MAX_LATENCY,
    ):
        """
        Args:
            cloud_logger (logging_v2.logger.Logger):
                The logger to send entries to.
            grace_period (Optional[float]): The amount of time to wait for pending logs to
                be submitted when the process is shutting down.
            max_batch (Optional[int]): The maximum number of items to send at a time
                in the background thread.
            max_latency (Optional[float]): The amount of time to wait for new logs before
                sending a new batch. It is strongly recommended to keep this smaller
                than the grace_period. This means this is effectively the longest
                amount of time the background thread will hold onto log entries
                before sending them to the server.
        """
        self._cloud_logger = cloud_logger
        self._grace_period = grace_period
        self._max_batch_size = max_batch_size
        self._max_latency = max_latency
        self._queue = queue.Queue(0)
        self._operational_lock = threading.Lock()
        self._thread = None

    @property
    def is_alive(self):
        """Returns True is the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _safely_commit_batch(self, batch):
        total_logs = len(batch.entries)

        try:
            if total_logs > 0:
                batch.commit()
                _LOGGER.debug("Submitted %d logs", total_logs)
        except Exception:
            _LOGGER.error("Failed to submit %d logs.", total_logs, exc_info=True)

    def _thread_main(self):
        """The entry point for the worker thread.

        Pulls pending log entries off the queue and writes them in batches to
        the Cloud Logger.
        """
        _LOGGER.debug("Background thread started.")

        done = False
        while not done:
            batch = self._cloud_logger.batch()
            items = _get_many(
                self._queue,
                max_items=self._max_batch_size,
                max_latency=self._max_latency,
            )

            for item in items:
                if item is _WORKER_TERMINATOR:
                    done = True  # Continue processing items.
                else:
                    batch.log(**item)

            # We cannot commit logs upstream if the main thread is shutting down
            if threading.main_thread().is_alive():
                self._safely_commit_batch(batch)

            for it in items:
                self._queue.task_done()

        _LOGGER.debug("Background thread exited gracefully.")

    def start(self):
        """Starts the background thread.

        Additionally, this registers a handler for process exit to attempt
        to send any pending log entries before shutdown.
        """
        with self._operational_lock:
            if self.is_alive:
                return

            self._thread = threading.Thread(
                target=self._thread_main, name=_WORKER_THREAD_NAME
            )
            self._thread.daemon = True
            self._thread.start()
            atexit.register(self._handle_exit)

    def stop(self, *, grace_period=None):
        """Signals the background thread to stop.

        This does not terminate the background thread. It simply queues the
        stop signal. If the main process exits before the background thread
        processes the stop signal, it will be terminated without finishing
        work. The ``grace_period`` parameter will give the background
        thread some time to finish processing before this function returns.

        Args:
            grace_period (Optional[float]): If specified, this method will
                block up to this many seconds to allow the background thread
                to finish work before returning.

        Returns:
            bool: True if the thread terminated. False if the thread is still
            running.
        """
        if not self.is_alive:
            return True

        with self._operational_lock:
            self._queue.put_nowait(_WORKER_TERMINATOR)

            if grace_period is not None:
                print("Waiting up to %d seconds." % (grace_period,), file=sys.stderr)

            self._thread.join(timeout=grace_period)

            # Check this before disowning the thread, because after we disown
            # the thread is_alive will be False regardless of if the thread
            # exited or not.
            success = not self.is_alive

            self._thread = None

            return success

    def _close(self, close_msg):
        """Callback that attempts to send pending logs before termination if the main thread is alive."""
        if not self.is_alive:
            return

        if not self._queue.empty():
            print(close_msg, file=sys.stderr)

        if threading.main_thread().is_alive() and self.stop(
            grace_period=self._grace_period
        ):
            print("Sent all pending logs.", file=sys.stderr)
        elif not self._queue.empty():
            print(
                "Failed to send %d pending logs." % (self._queue.qsize(),),
                file=sys.stderr,
            )

        self._thread = None

    def enqueue(self, record, message, **kwargs):
        """Queues a log entry to be written by the background thread.

        Args:
            record (logging.LogRecord): Python log record that the handler was called with.
            message (str or dict): The message from the ``LogRecord`` after being
                        formatted by the associated log formatters.
            kwargs: Additional optional arguments for the logger
        """
        # set python logger name as label if missing
        labels = kwargs.pop("labels", {})
        if record.name:
            labels["python_logger"] = labels.get("python_logger", record.name)
        kwargs["labels"] = labels
        # enqueue new entry
        queue_entry = {
            "message": message,
            "severity": _helpers._normalize_severity(record.levelno),
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, datetime.timezone.utc
            ),
        }
        queue_entry.update(kwargs)
        self._queue.put_nowait(queue_entry)

    def flush(self):
        """Submit any pending log records."""
        self._queue.join()

    def close(self):
        """Signals the worker thread to stop, then closes the transport thread.

        This call will attempt to send pending logs before termination, and
        should be followed up by disowning the transport object.
        """
        atexit.unregister(self._handle_exit)
        self._close(
            "Background thread shutting down, attempting to send %d queued log "
            "entries to Cloud Logging..." % (self._queue.qsize(),)
        )

    def _handle_exit(self):
        """Handle system exit.

        Since we cannot send pending logs during system shutdown due to thread errors,
        log an error message to stderr to notify the user.
        """
        self._close(_CLOSE_THREAD_SHUTDOWN_ERROR_MSG)


class BackgroundThreadTransport(Transport):
    """Asynchronous transport that uses a background thread."""

    def __init__(
        self,
        client,
        name,
        *,
        grace_period=_DEFAULT_GRACE_PERIOD,
        batch_size=_DEFAULT_MAX_BATCH_SIZE,
        max_latency=_DEFAULT_MAX_LATENCY,
        resource=_GLOBAL_RESOURCE,
        **kwargs,
    ):
        """
        Args:
            client (~logging_v2.client.Client):
                The Logging client.
            name (str): The name of the lgoger.
            grace_period (Optional[float]): The amount of time to wait for pending logs to
                be submitted when the process is shutting down.
            batch_size (Optional[int]): The maximum number of items to send at a time in the
                background thread.
            max_latency (Optional[float]): The amount of time to wait for new logs before
                sending a new batch. It is strongly recommended to keep this smaller
                than the grace_period. This means this is effectively the longest
                amount of time the background thread will hold onto log entries
                before sending them to the server.
            resource (Optional[Resource|dict]): The default monitored resource to associate
                with logs when not specified
        """
        self.client = client
        logger = self.client.logger(name, resource=resource)
        self.grace_period = grace_period
        self.worker = _Worker(
            logger,
            grace_period=grace_period,
            max_batch_size=batch_size,
            max_latency=max_latency,
        )
        self.worker.start()

    def send(self, record, message, **kwargs):
        """Overrides Transport.send().

        Args:
            record (logging.LogRecord): Python log record that the handler was called with.
            message (str or dict): The message from the ``LogRecord`` after being
                formatted by the associated log formatters.
            kwargs: Additional optional arguments for the logger
        """
        self.worker.enqueue(record, message, **kwargs)

    def flush(self):
        """Submit any pending log records."""
        self.worker.flush()

    def close(self):
        """Closes the worker thread."""
        self.worker.close()

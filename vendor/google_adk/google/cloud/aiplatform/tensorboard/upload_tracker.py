# -*- coding: utf-8 -*-
# Copyright 2019-2024 The TensorFlow Authors. All Rights Reserved.
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Progress tracker for uploader."""


import contextlib
from datetime import datetime
import sys
import time


def readable_time_string():
    """Get a human-readable time string for the present."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def readable_bytes_string(bytes):
    """Get a human-readable string for number of bytes."""
    if bytes >= 2**20:
        return "%.1f MB" % (float(bytes) / 2**20)
    elif bytes >= 2**10:
        return "%.1f kB" % (float(bytes) / 2**10)
    else:
        return "%d B" % bytes


class UploadStats:
    """Statistics of uploading."""

    def __init__(self):
        self._last_summarized_timestamp = time.time()
        self._last_data_added_timestamp = 0
        self._num_scalars = 0
        self._num_tensors = 0
        self._num_tensors_skipped = 0
        self._tensor_bytes = 0
        self._tensor_bytes_skipped = 0
        self._num_blobs = 0
        self._num_blobs_skipped = 0
        self._blob_bytes = 0
        self._blob_bytes_skipped = 0
        self._plugin_names = set()

    def add_scalars(self, num_scalars):
        """Add a batch of scalars.

        Args:
          num_scalars: Number of scalars uploaded in this batch.
        """
        self._refresh_last_data_added_timestamp()
        self._num_scalars += num_scalars

    def add_tensors(
        self,
        num_tensors,
        num_tensors_skipped,
        tensor_bytes,
        tensor_bytes_skipped,
    ):
        """Add a batch of tensors.

        Args:
          num_tensors: Number of tensors encountered in this batch, including
            the ones skipped due to reasons such as large exceeding limit.
          num_tensors: Number of tensors skipped. This describes a subset of
            `num_tensors` and hence must be `<= num_tensors`.
          tensor_bytes: Total byte size of tensors encountered in this batch,
            including the skipped ones.
          tensor_bytes_skipped: Total byte size of the tensors skipped due to
            reasons such as size exceeding limit.
        """
        assert num_tensors_skipped <= num_tensors
        assert tensor_bytes_skipped <= tensor_bytes
        self._refresh_last_data_added_timestamp()
        self._num_tensors += num_tensors
        self._num_tensors_skipped += num_tensors_skipped
        self._tensor_bytes += tensor_bytes
        self._tensor_bytes_skipped = tensor_bytes_skipped

    def add_blob(self, blob_bytes, is_skipped):
        """Add a blob.

        Args:
          blob_bytes: Byte size of the blob.
          is_skipped: Whether the uploading of the blob is skipped due to
            reasons such as size exceeding limit.
        """
        self._refresh_last_data_added_timestamp()
        self._num_blobs += 1
        self._blob_bytes += blob_bytes
        if is_skipped:
            self._num_blobs_skipped += 1
            self._blob_bytes_skipped += blob_bytes

    def add_plugin(self, plugin_name):
        """Add a plugin.

        Args:
          plugin_name: Name of the plugin.
        """
        self._refresh_last_data_added_timestamp()
        self._plugin_names.add(plugin_name)

    @property
    def num_scalars(self):
        return self._num_scalars

    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def num_tensors_skipped(self):
        return self._num_tensors_skipped

    @property
    def tensor_bytes(self):
        return self._tensor_bytes

    @property
    def tensor_bytes_skipped(self):
        return self._tensor_bytes_skipped

    @property
    def num_blobs(self):
        return self._num_blobs

    @property
    def num_blobs_skipped(self):
        return self._num_blobs_skipped

    @property
    def blob_bytes(self):
        return self._blob_bytes

    @property
    def blob_bytes_skipped(self):
        return self._blob_bytes_skipped

    @property
    def plugin_names(self):
        return self._plugin_names

    def has_data(self):
        """Has any data been tracked by this instance.

        This counts the tensor and blob data that have been scanned
        but skipped.

        Returns:
          Whether this stats tracking object has tracked any data.
        """
        return self._num_scalars > 0 or self._num_tensors > 0 or self._num_blobs > 0

    def summarize(self):
        """Get a summary string for actually-uploaded and skipped data.

        Calling this property also marks the "last_summarized" timestamp, so that
        the has_new_data_since_last_summarize() will be able to report the correct value
        later.

        Returns:
          A tuple with two items:
          - A string summarizing all data uploaded so far.
          - If any data was skipped, a string for all skipped data. Else, `None`.
        """
        self._last_summarized_timestamp = time.time()
        string_pieces = []
        string_pieces.append("%d scalars" % self._num_scalars)
        uploaded_tensor_count = self._num_tensors - self._num_tensors_skipped
        uploaded_tensor_bytes = self._tensor_bytes - self._tensor_bytes_skipped
        string_pieces.append(
            "0 tensors"
            if not uploaded_tensor_count
            else (
                "%d tensors (%s)"
                % (
                    uploaded_tensor_count,
                    readable_bytes_string(uploaded_tensor_bytes),
                )
            )
        )
        uploaded_blob_count = self._num_blobs - self._num_blobs_skipped
        uploaded_blob_bytes = self._blob_bytes - self._blob_bytes_skipped
        string_pieces.append(
            "0 binary objects"
            if not uploaded_blob_count
            else (
                "%d binary objects (%s)"
                % (
                    uploaded_blob_count,
                    readable_bytes_string(uploaded_blob_bytes),
                )
            )
        )
        skipped_string = self._skipped_summary() if self._skipped_any() else None
        return ", ".join(string_pieces), skipped_string

    def _skipped_any(self):
        """Whether any data was skipped."""
        return self._num_tensors_skipped or self._num_blobs_skipped

    def has_new_data_since_last_summarize(self):
        return self._last_data_added_timestamp > self._last_summarized_timestamp

    def _skipped_summary(self):
        """Get a summary string for skipped data."""
        string_pieces = []
        if self._num_tensors_skipped:
            string_pieces.append(
                "%d tensors (%s)"
                % (
                    self._num_tensors_skipped,
                    readable_bytes_string(self._tensor_bytes_skipped),
                )
            )
        if self._num_blobs_skipped:
            string_pieces.append(
                "%d binary objects (%s)"
                % (
                    self._num_blobs_skipped,
                    readable_bytes_string(self._blob_bytes_skipped),
                )
            )
        return ", ".join(string_pieces)

    def _refresh_last_data_added_timestamp(self):
        self._last_data_added_timestamp = time.time()


_STYLE_RESET = "\033[0m"
_STYLE_BOLD = "\033[1m"
_STYLE_GREEN = "\033[32m"
_STYLE_YELLOW = "\033[33m"
_STYLE_DARKGRAY = "\033[90m"
_STYLE_ERASE_LINE = "\033[2K"


class UploadTracker:
    """Tracker for uploader progress and status."""

    _SUPPORTED_VERBISITY_VALUES = (0, 1)

    def __init__(self, verbosity, one_shot=False):
        if verbosity not in self._SUPPORTED_VERBISITY_VALUES:
            raise ValueError(
                "Unsupported verbosity value %s (supported values: %s)"
                % (verbosity, self._SUPPORTED_VERBISITY_VALUES)
            )
        self._verbosity = verbosity
        self._stats = UploadStats()
        self._send_count = 0
        self._one_shot = one_shot

    def _dummy_generator(self):
        while True:
            # Yield an arbitrary value 0: The progress bar is indefinite.
            yield 0

    def _overwrite_line_message(self, message, color_code=_STYLE_GREEN):
        """Overwrite the current line with a stylized message."""
        if not self._verbosity:
            return
        message += "." * 3
        sys.stdout.write(_STYLE_ERASE_LINE + color_code + message + _STYLE_RESET + "\r")
        sys.stdout.flush()

    def _single_line_message(self, message):
        """Write a timestamped single line, with newline, to stdout."""
        if not self._verbosity:
            return
        start_message = "%s[%s]%s %s\n" % (
            _STYLE_BOLD,
            readable_time_string(),
            _STYLE_RESET,
            message,
        )
        sys.stdout.write(start_message)
        sys.stdout.flush()

    def has_data(self):
        """Determine if any data has been uploaded under the tracker's watch."""
        return self._stats.has_data()

    def _update_cumulative_status(self):
        """Write an update summarizing the data uploaded since the start."""
        if not self._verbosity:
            return
        if not self._stats.has_new_data_since_last_summarize():
            return
        uploaded_str, skipped_str = self._stats.summarize()
        uploaded_message = "%s[%s]%s Total uploaded: %s\n" % (
            _STYLE_BOLD,
            readable_time_string(),
            _STYLE_RESET,
            uploaded_str,
        )
        sys.stdout.write(uploaded_message)
        if skipped_str:
            sys.stdout.write(
                "%sTotal skipped: %s\n%s" % (_STYLE_DARKGRAY, skipped_str, _STYLE_RESET)
            )
        sys.stdout.flush()
        # TODO(cais): Add summary of what plugins have been involved, once it's
        # clear how to get canonical plugin names.

    def add_plugin_name(self, plugin_name):
        self._stats.add_plugin(plugin_name)

    @contextlib.contextmanager
    def send_tracker(self):
        """Create a context manager for a round of data sending."""
        self._send_count += 1
        if self._send_count == 1:
            self._single_line_message("Started scanning logdir.")
        try:
            # self._reset_bars()
            self._overwrite_line_message("Data upload starting")
            yield
        finally:
            self._update_cumulative_status()
            if self._one_shot:
                self._single_line_message("Done scanning logdir.")
            else:
                self._overwrite_line_message(
                    "Listening for new data in logdir",
                    color_code=_STYLE_YELLOW,
                )

    @contextlib.contextmanager
    def scalars_tracker(self, num_scalars):
        """Create a context manager for tracking a scalar batch upload.

        Args:
          num_scalars: Number of scalars in the batch.
        """
        self._overwrite_line_message("Uploading %d scalars" % num_scalars)
        try:
            yield
        finally:
            self._stats.add_scalars(num_scalars)

    @contextlib.contextmanager
    def tensors_tracker(
        self,
        num_tensors,
        num_tensors_skipped,
        tensor_bytes,
        tensor_bytes_skipped,
    ):
        """Create a context manager for tracking a tensor batch upload.

        Args:
          num_tensors: Total number of tensors in the batch.
          num_tensors_skipped: Number of tensors skipped (a subset of
            `num_tensors`). Hence this must be `<= num_tensors`.
          tensor_bytes: Total byte size of the tensors in the batch.
          tensor_bytes_skipped: Byte size of skipped tensors in the batch (a
            subset of `tensor_bytes`). Must be `<= tensor_bytes`.
        """
        if num_tensors_skipped:
            message = "Uploading %d tensors (%s) (Skipping %d tensors, %s)" % (
                num_tensors - num_tensors_skipped,
                readable_bytes_string(tensor_bytes - tensor_bytes_skipped),
                num_tensors_skipped,
                readable_bytes_string(tensor_bytes_skipped),
            )
        else:
            message = "Uploading %d tensors (%s)" % (
                num_tensors,
                readable_bytes_string(tensor_bytes),
            )
        self._overwrite_line_message(message)
        try:
            yield
        finally:
            self._stats.add_tensors(
                num_tensors,
                num_tensors_skipped,
                tensor_bytes,
                tensor_bytes_skipped,
            )

    @contextlib.contextmanager
    def blob_tracker(self, blob_bytes):
        """Creates context manager tracker for uploading a blob.

        Args:
          blob_bytes: Total byte size of the blob being uploaded.
        """
        self._overwrite_line_message(
            "Uploading binary object (%s)" % readable_bytes_string(blob_bytes)
        )
        try:
            yield _BlobTracker(self._stats, blob_bytes)
        finally:
            pass


class _BlobTracker:
    def __init__(self, upload_stats, blob_bytes):
        self._upload_stats = upload_stats
        self._blob_bytes = blob_bytes

    def mark_uploaded(self, is_uploaded):
        self._upload_stats.add_blob(self._blob_bytes, is_skipped=(not is_uploaded))

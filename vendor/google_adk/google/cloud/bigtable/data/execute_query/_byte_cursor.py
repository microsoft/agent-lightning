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

from typing import List, Optional

from google.cloud.bigtable.data.execute_query._checksum import _CRC32C
from google.cloud.bigtable_v2 import ExecuteQueryResponse


class _ByteCursor:
    """
    Buffers bytes from `ExecuteQuery` responses until resume_token is received or end-of-stream
    is reached. :class:`google.cloud.bigtable_v2.types.bigtable.ExecuteQueryResponse` obtained from
    the server should be passed to the ``consume`` method and its non-None results should be passed
    to appropriate :class:`google.cloud.bigtable.execute_query_reader._Reader` for parsing gathered
    bytes.

    This class consumes data obtained externally to be usable in both sync and async clients.

    See :class:`google.cloud.bigtable.execute_query_reader._Reader` for more context.
    """

    def __init__(self):
        self._batch_buffer = bytearray()
        self._batches: List[bytes] = []
        self._resume_token = None

    def reset(self):
        self._batch_buffer = bytearray()
        self._batches = []

    def prepare_for_new_request(self):
        """
        Prepares this ``_ByteCursor`` for retrying an ``ExecuteQuery`` request.

        Clears internal buffers of this ``_ByteCursor`` and returns last received
        ``resume_token`` to be used in retried request.

        This is the only method that returns ``resume_token`` to the user.
        Returning the token to the user is tightly coupled with clearing internal
        buffers to prevent accidental retry without clearing the state, what would
        cause invalid results. ``resume_token`` are not needed in other cases,
        thus they is no separate getter for it.

        Returns:
            bytes: Last received resume_token.
        """
        # The first response of any retried stream will always contain reset, so
        # this isn't actually necessary, but we do it for safety
        self.reset()
        return self._resume_token

    def empty(self) -> bool:
        return not self._batch_buffer and not self._batches

    def consume(self, response: ExecuteQueryResponse) -> Optional[List[bytes]]:
        """
        Reads results bytes from an ``ExecuteQuery`` response and adds them to a buffer.

        If the response contains a ``resume_token``:
        - the ``resume_token`` is saved in this ``_ByteCursor``, and
        - internal buffers are flushed and returned to the caller.

        ``resume_token`` is not available directly, but can be retrieved by calling
        :meth:`._ByteCursor.prepare_for_new_request` when preparing to retry a request.

        Args:
            response (google.cloud.bigtable_v2.types.bigtable.ExecuteQueryResponse):
                Response obtained from the stream.

        Returns:
            bytes or None: List of bytes if buffers were flushed or None otherwise.
            Each element in the list represents the bytes of a `ProtoRows` message.

        Raises:
            ValueError: If provided ``ExecuteQueryResponse`` is not valid
                or contains bytes representing response of a different kind than previously
                processed responses.
        """
        response_pb = response._pb  # proto-plus attribute retrieval is slow.

        if response_pb.HasField("results"):
            results = response_pb.results
            if results.reset:
                self.reset()
            if results.HasField("proto_rows_batch"):
                self._batch_buffer.extend(results.proto_rows_batch.batch_data)
                # Note that 0 is a valid checksum so we must check for field presence
                if results.HasField("batch_checksum"):
                    expected_checksum = results.batch_checksum
                    checksum = _CRC32C.checksum(self._batch_buffer)
                    if expected_checksum != checksum:
                        raise ValueError(
                            f"Unexpected checksum mismatch. Expected: {expected_checksum}, got: {checksum}"
                        )
                    # We have a complete batch so we move it to batches and reset the
                    # batch_buffer
                    self._batches.append(memoryview(self._batch_buffer))
                    self._batch_buffer = bytearray()

            if results.resume_token:
                self._resume_token = results.resume_token

                if self._batches:
                    if self._batch_buffer:
                        raise ValueError("Unexpected resume_token without checksum")
                    return_value = self._batches
                    self._batches = []
                    return return_value
        else:
            raise ValueError(f"Unexpected ExecuteQueryResponse: {response}")
        return None

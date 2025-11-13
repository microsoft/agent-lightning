# Copyright 2025 Google LLC
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
"""
NOTE:
This is _experimental module for upcoming support for Rapid Storage.
(https://cloud.google.com/blog/products/storage-data-transfer/high-performance-storage-innovations-for-ai-hpc#:~:text=your%20AI%20workloads%3A-,Rapid%20Storage,-%3A%20A%20new)

APIs may not work as intended and are not stable yet. Feature is not
GA(Generally Available) yet, please contact your TAM(Technical Account Manager)
if you want to use these APIs.

"""

from typing import Optional
from google.cloud import _storage_v2
from google.cloud.storage._experimental.asyncio.async_grpc_client import AsyncGrpcClient
from google.cloud.storage._experimental.asyncio.async_abstract_object_stream import (
    _AsyncAbstractObjectStream,
)

from google.api_core.bidi_async import AsyncBidiRpc


class _AsyncReadObjectStream(_AsyncAbstractObjectStream):
    """Class representing a gRPC bidi-stream for reading data from a GCS ``Object``.

    This class provides a unix socket-like interface to a GCS ``Object``, with
    methods like ``open``, ``close``, ``send``, and ``recv``.

    :type client: :class:`~google.cloud.storage._experimental.asyncio.async_grpc_client.AsyncGrpcClient.grpc_client`
    :param client: async grpc client to use for making API requests.

    :type bucket_name: str
    :param bucket_name: The name of the GCS ``bucket`` containing the object.

    :type object_name: str
    :param object_name: The name of the GCS ``object`` to be read.

    :type generation_number: int
    :param generation_number: (Optional) If present, selects a specific revision of
                              this object.

    :type read_handle: bytes
    :param read_handle: (Optional) An existing handle for reading the object.
                        If provided, opening the bidi-gRPC connection will be faster.
    """

    def __init__(
        self,
        client: AsyncGrpcClient.grpc_client,
        bucket_name: str,
        object_name: str,
        generation_number: Optional[int] = None,
        read_handle: Optional[bytes] = None,
    ) -> None:
        if client is None:
            raise ValueError("client must be provided")
        if bucket_name is None:
            raise ValueError("bucket_name must be provided")
        if object_name is None:
            raise ValueError("object_name must be provided")

        super().__init__(
            bucket_name=bucket_name,
            object_name=object_name,
            generation_number=generation_number,
        )
        self.client: AsyncGrpcClient.grpc_client = client
        self.read_handle: Optional[bytes] = read_handle

        self._full_bucket_name = f"projects/_/buckets/{self.bucket_name}"

        self.rpc = self.client._client._transport._wrapped_methods[
            self.client._client._transport.bidi_read_object
        ]
        self.first_bidi_read_req = _storage_v2.BidiReadObjectRequest(
            read_object_spec=_storage_v2.BidiReadObjectSpec(
                bucket=self._full_bucket_name, object=object_name
            ),
        )
        self.metadata = (("x-goog-request-params", f"bucket={self._full_bucket_name}"),)
        self.socket_like_rpc: Optional[AsyncBidiRpc] = None
        self._is_stream_open: bool = False

    async def open(self) -> None:
        """Opens the bidi-gRPC connection to read from the object.

        This method sends an initial request to start the stream and receives
        the first response containing metadata and a read handle.
        """
        if self._is_stream_open:
            raise ValueError("Stream is already open")
        self.socket_like_rpc = AsyncBidiRpc(
            self.rpc, initial_request=self.first_bidi_read_req, metadata=self.metadata
        )
        await self.socket_like_rpc.open()  # this is actually 1 send
        response = await self.socket_like_rpc.recv()
        if self.generation_number is None:
            self.generation_number = response.metadata.generation

        self.read_handle = response.read_handle

        self._is_stream_open = True

    async def close(self) -> None:
        """Closes the bidi-gRPC connection."""
        if not self._is_stream_open:
            raise ValueError("Stream is not open")
        await self.socket_like_rpc.close()
        self._is_stream_open = False

    async def send(
        self, bidi_read_object_request: _storage_v2.BidiReadObjectRequest
    ) -> None:
        """Sends a request message on the stream.

        Args:
            bidi_read_object_request (:class:`~google.cloud._storage_v2.types.BidiReadObjectRequest`):
                The request message to send. This is typically used to specify
                the read offset and limit.
        """
        if not self._is_stream_open:
            raise ValueError("Stream is not open")
        await self.socket_like_rpc.send(bidi_read_object_request)

    async def recv(self) -> _storage_v2.BidiReadObjectResponse:
        """Receives a response from the stream.

        This method waits for the next message from the server, which could
        contain object data or metadata.

        Returns:
            :class:`~google.cloud._storage_v2.types.BidiReadObjectResponse`:
                The response message from the server.
        """
        if not self._is_stream_open:
            raise ValueError("Stream is not open")
        response = await self.socket_like_rpc.recv()
        # Update read_handle if present in response
        if response and response.read_handle:
            self.read_handle = response.read_handle
        return response

    @property
    def is_stream_open(self) -> bool:
        return self._is_stream_open

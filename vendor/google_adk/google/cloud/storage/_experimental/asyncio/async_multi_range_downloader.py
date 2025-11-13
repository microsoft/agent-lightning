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

from __future__ import annotations
import asyncio
import google_crc32c
from google.api_core import exceptions
from google_crc32c import Checksum

from typing import List, Optional, Tuple

from google.cloud.storage._experimental.asyncio.async_read_object_stream import (
    _AsyncReadObjectStream,
)
from google.cloud.storage._experimental.asyncio.async_grpc_client import (
    AsyncGrpcClient,
)

from io import BytesIO
from google.cloud import _storage_v2
from google.cloud.storage.exceptions import DataCorruption
from google.cloud.storage._helpers import generate_random_56_bit_integer


_MAX_READ_RANGES_PER_BIDI_READ_REQUEST = 100


class Result:
    """An instance of this class will be populated and retured for each
    `read_range` provided to ``download_ranges`` method.

    """

    def __init__(self, bytes_requested: int):
        # only while instantiation, should not be edited later.
        # hence there's no setter, only getter is provided.
        self._bytes_requested: int = bytes_requested
        self._bytes_written: int = 0

    @property
    def bytes_requested(self) -> int:
        return self._bytes_requested

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @bytes_written.setter
    def bytes_written(self, value: int):
        self._bytes_written = value

    def __repr__(self):
        return f"bytes_requested: {self._bytes_requested}, bytes_written: {self._bytes_written}"


class AsyncMultiRangeDownloader:
    """Provides an interface for downloading multiple ranges of a GCS ``Object``
    concurrently.

    Example usage:

    .. code-block:: python

        client = AsyncGrpcClient().grpc_client
        mrd = await AsyncMultiRangeDownloader.create_mrd(
            client, bucket_name="chandrasiri-rs", object_name="test_open9"
        )
        my_buff1 = open('my_fav_file.txt', 'wb')
        my_buff2 = BytesIO()
        my_buff3 = BytesIO()
        my_buff4 = any_object_which_provides_BytesIO_like_interface()
        await mrd.download_ranges(
            [
                # (start_byte, bytes_to_read, writeable_buffer)
                (0, 100, my_buff1),
                (100, 20, my_buff2),
                (200, 123, my_buff3),
                (300, 789, my_buff4),
            ]
        )

        # verify data in buffers...
        assert my_buff2.getbuffer().nbytes == 20


    """

    @classmethod
    async def create_mrd(
        cls,
        client: AsyncGrpcClient.grpc_client,
        bucket_name: str,
        object_name: str,
        generation_number: Optional[int] = None,
        read_handle: Optional[bytes] = None,
    ) -> AsyncMultiRangeDownloader:
        """Initializes a MultiRangeDownloader and opens the underlying bidi-gRPC
        object for reading.

        :type client: :class:`~google.cloud.storage._experimental.asyncio.async_grpc_client.AsyncGrpcClient.grpc_client`
        :param client: The asynchronous client to use for making API requests.

        :type bucket_name: str
        :param bucket_name: The name of the bucket containing the object.

        :type object_name: str
        :param object_name: The name of the object to be read.

        :type generation_number: int
        :param generation_number: (Optional) If present, selects a specific
                                  revision of this object.

        :type read_handle: bytes
        :param read_handle: (Optional) An existing handle for reading the object.
                            If provided, opening the bidi-gRPC connection will be faster.

        :rtype: :class:`~google.cloud.storage._experimental.asyncio.async_multi_range_downloader.AsyncMultiRangeDownloader`
        :returns: An initialized AsyncMultiRangeDownloader instance for reading.
        """
        mrd = cls(client, bucket_name, object_name, generation_number, read_handle)
        await mrd.open()
        return mrd

    def __init__(
        self,
        client: AsyncGrpcClient.grpc_client,
        bucket_name: str,
        object_name: str,
        generation_number: Optional[int] = None,
        read_handle: Optional[bytes] = None,
    ) -> None:
        """Constructor for AsyncMultiRangeDownloader, clients are not adviced to
         use it directly. Instead it's adviced to use the classmethod `create_mrd`.

        :type client: :class:`~google.cloud.storage._experimental.asyncio.async_grpc_client.AsyncGrpcClient.grpc_client`
        :param client: The asynchronous client to use for making API requests.

        :type bucket_name: str
        :param bucket_name: The name of the bucket containing the object.

        :type object_name: str
        :param object_name: The name of the object to be read.

        :type generation_number: int
        :param generation_number: (Optional) If present, selects a specific revision of
                                  this object.

        :type read_handle: bytes
        :param read_handle: (Optional) An existing read handle.
        """

        # Verify that the fast, C-accelerated version of crc32c is available.
        # If not, raise an error to prevent silent performance degradation.
        if google_crc32c.implementation != "c":
            raise exceptions.NotFound(
                "The google-crc32c package is not installed with C support. "
                "Bidi reads require the C extension for data integrity checks."
                "For more information, see https://github.com/googleapis/python-crc32c."
            )

        self.client = client
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.generation_number = generation_number
        self.read_handle = read_handle
        self.read_obj_str: Optional[_AsyncReadObjectStream] = None
        self._is_stream_open: bool = False

        self._read_id_to_writable_buffer_dict = {}
        self._read_id_to_download_ranges_id = {}
        self._download_ranges_id_to_pending_read_ids = {}

    async def open(self) -> None:
        """Opens the bidi-gRPC connection to read from the object.

        This method initializes and opens an `_AsyncReadObjectStream` (bidi-gRPC stream) to
        for downloading ranges of data from GCS ``Object``.

        "Opening" constitutes fetching object metadata such as generation number
        and read handle and sets them as attributes if not already set.
        """
        if self._is_stream_open:
            raise ValueError("Underlying bidi-gRPC stream is already open")

        if self.read_obj_str is None:
            self.read_obj_str = _AsyncReadObjectStream(
                client=self.client,
                bucket_name=self.bucket_name,
                object_name=self.object_name,
                generation_number=self.generation_number,
                read_handle=self.read_handle,
            )
        await self.read_obj_str.open()
        self._is_stream_open = True
        if self.generation_number is None:
            self.generation_number = self.read_obj_str.generation_number
        self.read_handle = self.read_obj_str.read_handle
        return

    async def download_ranges(
        self, read_ranges: List[Tuple[int, int, BytesIO]], lock: asyncio.Lock = None
    ) -> None:
        """Downloads multiple byte ranges from the object into the buffers
        provided by user.

        :type read_ranges: List[Tuple[int, int, "BytesIO"]]
        :param read_ranges: A list of tuples, where each tuple represents a
            byte range (start_byte, bytes_to_read, writeable_buffer). Buffer has
            to be provided by the user, and user has to make sure appropriate
            memory is available in the application to avoid out-of-memory crash.

        :type lock: asyncio.Lock
        :param lock: (Optional) An asyncio lock to synchronize sends and recvs
            on the underlying bidi-GRPC stream. This is required when multiple
            coroutines are calling this method concurrently.

            i.e. Example usage with multiple coroutines:

            ```
            lock = asyncio.Lock()
            task1 = asyncio.create_task(mrd.download_ranges(ranges1, lock))
            task2 = asyncio.create_task(mrd.download_ranges(ranges2, lock))
            await asyncio.gather(task1, task2)

            ```

            If user want to call this method serially from multiple coroutines,
            then providing a lock is not necessary.

            ```
            await mrd.download_ranges(ranges1)
            await mrd.download_ranges(ranges2)

            # ... some other code code...

            ```


        :raises ValueError: if the underlying bidi-GRPC stream is not open.
        :raises ValueError: if the length of read_ranges is more than 1000.
        :raises DataCorruption: if a checksum mismatch is detected while reading data.

        """

        if len(read_ranges) > 1000:
            raise ValueError(
                "Invalid input - length of read_ranges cannot be more than 1000"
            )

        if not self._is_stream_open:
            raise ValueError("Underlying bidi-gRPC stream is not open")

        if lock is None:
            lock = asyncio.Lock()

        _func_id = generate_random_56_bit_integer()
        read_ids_in_current_func = set()
        for i in range(0, len(read_ranges), _MAX_READ_RANGES_PER_BIDI_READ_REQUEST):
            read_ranges_segment = read_ranges[
                i : i + _MAX_READ_RANGES_PER_BIDI_READ_REQUEST
            ]

            read_ranges_for_bidi_req = []
            for j, read_range in enumerate(read_ranges_segment):
                read_id = generate_random_56_bit_integer()
                read_ids_in_current_func.add(read_id)
                self._read_id_to_download_ranges_id[read_id] = _func_id
                self._read_id_to_writable_buffer_dict[read_id] = read_range[2]
                bytes_requested = read_range[1]
                read_ranges_for_bidi_req.append(
                    _storage_v2.ReadRange(
                        read_offset=read_range[0],
                        read_length=bytes_requested,
                        read_id=read_id,
                    )
                )
            async with lock:
                await self.read_obj_str.send(
                    _storage_v2.BidiReadObjectRequest(
                        read_ranges=read_ranges_for_bidi_req
                    )
                )
        self._download_ranges_id_to_pending_read_ids[
            _func_id
        ] = read_ids_in_current_func

        while len(self._download_ranges_id_to_pending_read_ids[_func_id]) > 0:
            async with lock:
                response = await self.read_obj_str.recv()

            if response is None:
                raise Exception("None response received, something went wrong.")

            for object_data_range in response.object_data_ranges:
                if object_data_range.read_range is None:
                    raise Exception("Invalid response, read_range is None")

                checksummed_data = object_data_range.checksummed_data
                data = checksummed_data.content
                server_checksum = checksummed_data.crc32c

                client_crc32c = Checksum(data).digest()
                client_checksum = int.from_bytes(client_crc32c, "big")

                if server_checksum != client_checksum:
                    raise DataCorruption(
                        response,
                        f"Checksum mismatch for read_id {object_data_range.read_range.read_id}. "
                        f"Server sent {server_checksum}, client calculated {client_checksum}.",
                    )

                read_id = object_data_range.read_range.read_id
                buffer = self._read_id_to_writable_buffer_dict[read_id]
                buffer.write(data)

                if object_data_range.range_end:
                    tmp_dn_ranges_id = self._read_id_to_download_ranges_id[read_id]
                    self._download_ranges_id_to_pending_read_ids[
                        tmp_dn_ranges_id
                    ].remove(read_id)
                    del self._read_id_to_download_ranges_id[read_id]

    async def close(self):
        """
        Closes the underlying bidi-gRPC connection.
        """
        if not self._is_stream_open:
            raise ValueError("Underlying bidi-gRPC stream is not open")
        await self.read_obj_str.close()
        self._is_stream_open = False

    @property
    def is_stream_open(self) -> bool:
        return self._is_stream_open

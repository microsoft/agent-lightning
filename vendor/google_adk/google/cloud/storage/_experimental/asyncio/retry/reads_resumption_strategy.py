from typing import Any, List, IO

from google.cloud import _storage_v2 as storage_v2
from google.cloud.storage.exceptions import DataCorruption
from google.cloud.storage._experimental.asyncio.retry.base_strategy import (
    _BaseResumptionStrategy,
)
from google.cloud._storage_v2.types.storage import BidiReadObjectRedirectedError


class _DownloadState:
    """A helper class to track the state of a single range download."""

    def __init__(
        self, initial_offset: int, initial_length: int, user_buffer: IO[bytes]
    ):
        self.initial_offset = initial_offset
        self.initial_length = initial_length
        self.user_buffer = user_buffer
        self.bytes_written = 0
        self.next_expected_offset = initial_offset
        self.is_complete = False


class _ReadResumptionStrategy(_BaseResumptionStrategy):
    """The concrete resumption strategy for bidi reads."""

    def generate_requests(self, state: dict) -> List[storage_v2.ReadRange]:
        """Generates new ReadRange requests for all incomplete downloads.

        :type state: dict
        :param state: A dictionary mapping a read_id to its corresponding
                  _DownloadState object.
        """
        pending_requests = []
        for read_id, read_state in state.items():
            if not read_state.is_complete:
                new_offset = read_state.initial_offset + read_state.bytes_written
                new_length = read_state.initial_length - read_state.bytes_written

                new_request = storage_v2.ReadRange(
                    read_offset=new_offset,
                    read_length=new_length,
                    read_id=read_id,
                )
                pending_requests.append(new_request)
        return pending_requests

    def update_state_from_response(
        self, response: storage_v2.BidiReadObjectResponse, state: dict
    ) -> None:
        """Processes a server response, performs integrity checks, and updates state."""
        for object_data_range in response.object_data_ranges:
            read_id = object_data_range.read_range.read_id
            read_state = state[read_id]

            # Offset Verification
            chunk_offset = object_data_range.read_range.read_offset
            if chunk_offset != read_state.next_expected_offset:
                raise DataCorruption(response, f"Offset mismatch for read_id {read_id}")

            data = object_data_range.checksummed_data.content
            chunk_size = len(data)
            read_state.bytes_written += chunk_size
            read_state.next_expected_offset += chunk_size
            read_state.user_buffer.write(data)

            # Final Byte Count Verification
            if object_data_range.range_end:
                read_state.is_complete = True
                if (
                    read_state.initial_length != 0
                    and read_state.bytes_written != read_state.initial_length
                ):
                    raise DataCorruption(
                        response, f"Byte count mismatch for read_id {read_id}"
                    )

    async def recover_state_on_failure(self, error: Exception, state: Any) -> None:
        """Handles BidiReadObjectRedirectedError for reads."""
        # This would parse the gRPC error details, extract the routing_token,
        # and store it on the shared state object.
        cause = getattr(error, "cause", error)
        if isinstance(cause, BidiReadObjectRedirectedError):
            state["routing_token"] = cause.routing_token

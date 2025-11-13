# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from google.api_core import exceptions
from google.api_core import operation_async
from google.protobuf import empty_pb2

from google.cloud.bigtable_admin_v2.types import OptimizeRestoredTableMetadata


class AsyncRestoreTableOperation(operation_async.AsyncOperation):
    """A Future for interacting with Bigtable Admin's RestoreTable Long-Running Operation.

    This is needed to expose a potential long-running operation that might run after this operation
    finishes, OptimizeRestoreTable. This is exposed via the the :meth:`optimize_restore_table_operation`
    method.

    **This class should not be instantiated by users** and should only be instantiated by the admin
    client's :meth:`restore_table
    <google.cloud.bigtable_admin_v2.overlay.services.bigtable_table_admin.BigtableTableAdminAsyncClient.restore_table>`
    method.

    Args:
        operations_client (google.api_core.operations_v1.AbstractOperationsClient): The operations
            client from the admin client class's transport.
        restore_table_operation (google.api_core.operation_async.AsyncOperation): A
            :class:`google.api_core.operation_async.AsyncOperation`
            instance resembling a RestoreTable long-running operation
    """

    def __init__(
        self, operations_client, restore_table_operation: operation_async.AsyncOperation
    ):
        self._operations_client = operations_client
        self._optimize_restored_table_operation = None
        super().__init__(
            restore_table_operation._operation,
            restore_table_operation._refresh,
            restore_table_operation._cancel,
            restore_table_operation._result_type,
            restore_table_operation._metadata_type,
            retry=restore_table_operation._retry,
        )

    async def optimize_restored_table_operation(
        self,
    ) -> Optional[operation_async.AsyncOperation]:
        """Gets the OptimizeRestoredTable long-running operation that runs after this operation finishes.
        The current operation might not trigger a follow-up OptimizeRestoredTable operation, in which case, this
        method will return `None`.
        This method must not be called before the parent restore_table operation is complete.
        Returns:
            An object representing a long-running operation, or None if there is no OptimizeRestoredTable operation
                after this one.
        Raises:
            RuntimeError: raised when accessed before the restore_table operation is complete

        Raises:
            google.api_core.GoogleAPIError: raised when accessed before the restore_table operation is complete
        """
        if not await self.done():
            raise exceptions.GoogleAPIError(
                "optimize_restored_table operation can't be accessed until the restore_table operation is complete"
            )

        if self._optimize_restored_table_operation is not None:
            return self._optimize_restored_table_operation

        operation_name = self.metadata.optimize_table_operation_name

        # When the RestoreTable operation finishes, it might not necessarily trigger
        # an optimize operation.
        if operation_name:
            gapic_operation = await self._operations_client.get_operation(
                name=operation_name
            )
            self._optimize_restored_table_operation = operation_async.from_gapic(
                gapic_operation,
                self._operations_client,
                empty_pb2.Empty,
                metadata_type=OptimizeRestoredTableMetadata,
            )
            return self._optimize_restored_table_operation
        else:
            # no optimize operation found
            return None

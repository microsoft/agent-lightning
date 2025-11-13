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

from typing import Union, Callable

from google.api_core.future import polling
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.cloud.bigtable_admin_v2.types import bigtable_table_admin

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore


# The consistency check could take a very long time, so we wait indefinitely.
DEFAULT_RETRY = polling.DEFAULT_POLLING.with_timeout(None)


class _CheckConsistencyPollingFuture(polling.PollingFuture):
    """A Future that polls an underlying `check_consistency` operation until it returns True.

    **This class should not be instantiated by users** and should only be instantiated by the admin
    client's
    :meth:`google.cloud.bigtable_admin_v2.overlay.services.bigtable_table_admin.BigtableTableAdminClient.wait_for_consistency`
    or
    :meth:`google.cloud.bigtable_admin_v2.overlay.services.bigtable_table_admin.BigtableTableAdminClient.wait_for_replication`
    methods.

    Args:
        check_consistency_call(Callable[
            [Optional[google.api_core.retry.Retry],
            google.cloud.bigtable_admin_v2.types.CheckConsistencyResponse]):
            A :meth:`check_consistency
            <google.cloud.bigtable_admin_v2.overlay.services.bigtable_table_admin.BigtableTableAdminClient.check_consistency>`
            call from the admin client. The call should fix every user parameter,
            which will be done via :meth:`functools.partial`.
        polling (google.api_core.retry.Retry): The configuration used for polling.
            This parameter controls how often :meth:`done` is polled. If the
            ``timeout`` argument is specified in the :meth:`result
            <google.api_core.future.polling.PollingFuture.result>` method it will
            override the ``polling.timeout`` property.
    """

    def __init__(
        self,
        check_consistency_call: Callable[
            [OptionalRetry], bigtable_table_admin.CheckConsistencyResponse
        ],
        polling: retries.Retry = DEFAULT_RETRY,
        **kwargs
    ):
        super(_CheckConsistencyPollingFuture, self).__init__(polling=polling, **kwargs)

        # Done is called with two different scenarios, retry is specified or not specified.
        # API_call will be a functools partial with everything except retry specified because of
        # that.
        self._check_consistency_call = check_consistency_call

    def done(self, retry: OptionalRetry = None):
        """Polls the underlying `check_consistency` call to see if the future is complete.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the
                polling RPC (to not be confused with polling configuration. See
                the documentation for :meth:`result <google.api_core.future.polling.PollingFuture.result>`
                for details).

        Returns:
            bool: True if the future is complete, False otherwise.
        """

        if self._result_set:
            return True

        try:
            check_consistency_response = self._check_consistency_call()
            if check_consistency_response.consistent:
                self.set_result(True)

            return check_consistency_response.consistent
        except Exception as e:
            self.set_exception(e)

    def cancel(self):
        raise NotImplementedError("Cannot cancel consistency token operation")

    def cancelled(self):
        raise NotImplementedError("Cannot cancel consistency token operation")

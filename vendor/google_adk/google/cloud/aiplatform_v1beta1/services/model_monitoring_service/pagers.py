# -*- coding: utf-8 -*-
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
#
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import retry_async as retries_async
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Sequence,
    Tuple,
    Optional,
    Iterator,
    Union,
)

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
    OptionalAsyncRetry = Union[
        retries_async.AsyncRetry, gapic_v1.method._MethodDefault, None
    ]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore
    OptionalAsyncRetry = Union[retries_async.AsyncRetry, object, None]  # type: ignore

from google.cloud.aiplatform_v1beta1.types import model_monitor
from google.cloud.aiplatform_v1beta1.types import model_monitoring_alert
from google.cloud.aiplatform_v1beta1.types import model_monitoring_job
from google.cloud.aiplatform_v1beta1.types import model_monitoring_service
from google.cloud.aiplatform_v1beta1.types import model_monitoring_stats


class ListModelMonitorsPager:
    """A pager for iterating through ``list_model_monitors`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``model_monitors`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListModelMonitors`` requests and continue to iterate
    through the ``model_monitors`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., model_monitoring_service.ListModelMonitorsResponse],
        request: model_monitoring_service.ListModelMonitorsRequest,
        response: model_monitoring_service.ListModelMonitorsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListModelMonitorsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.ListModelMonitorsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[model_monitoring_service.ListModelMonitorsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __iter__(self) -> Iterator[model_monitor.ModelMonitor]:
        for page in self.pages:
            yield from page.model_monitors

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListModelMonitorsAsyncPager:
    """A pager for iterating through ``list_model_monitors`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``model_monitors`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListModelMonitors`` requests and continue to iterate
    through the ``model_monitors`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[model_monitoring_service.ListModelMonitorsResponse]
        ],
        request: model_monitoring_service.ListModelMonitorsRequest,
        response: model_monitoring_service.ListModelMonitorsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListModelMonitorsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListModelMonitorsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.ListModelMonitorsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[model_monitoring_service.ListModelMonitorsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[model_monitor.ModelMonitor]:
        async def async_generator():
            async for page in self.pages:
                for response in page.model_monitors:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListModelMonitoringJobsPager:
    """A pager for iterating through ``list_model_monitoring_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``model_monitoring_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListModelMonitoringJobs`` requests and continue to iterate
    through the ``model_monitoring_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., model_monitoring_service.ListModelMonitoringJobsResponse],
        request: model_monitoring_service.ListModelMonitoringJobsRequest,
        response: model_monitoring_service.ListModelMonitoringJobsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.ListModelMonitoringJobsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[model_monitoring_service.ListModelMonitoringJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __iter__(self) -> Iterator[model_monitoring_job.ModelMonitoringJob]:
        for page in self.pages:
            yield from page.model_monitoring_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListModelMonitoringJobsAsyncPager:
    """A pager for iterating through ``list_model_monitoring_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``model_monitoring_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListModelMonitoringJobs`` requests and continue to iterate
    through the ``model_monitoring_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[model_monitoring_service.ListModelMonitoringJobsResponse]
        ],
        request: model_monitoring_service.ListModelMonitoringJobsRequest,
        response: model_monitoring_service.ListModelMonitoringJobsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListModelMonitoringJobsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.ListModelMonitoringJobsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[model_monitoring_service.ListModelMonitoringJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[model_monitoring_job.ModelMonitoringJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.model_monitoring_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelMonitoringStatsPager:
    """A pager for iterating through ``search_model_monitoring_stats`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``monitoring_stats`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``SearchModelMonitoringStats`` requests and continue to iterate
    through the ``monitoring_stats`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., model_monitoring_service.SearchModelMonitoringStatsResponse
        ],
        request: model_monitoring_service.SearchModelMonitoringStatsRequest,
        response: model_monitoring_service.SearchModelMonitoringStatsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.SearchModelMonitoringStatsRequest(
            request
        )
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[model_monitoring_service.SearchModelMonitoringStatsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __iter__(self) -> Iterator[model_monitoring_stats.ModelMonitoringStats]:
        for page in self.pages:
            yield from page.monitoring_stats

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelMonitoringStatsAsyncPager:
    """A pager for iterating through ``search_model_monitoring_stats`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``monitoring_stats`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``SearchModelMonitoringStats`` requests and continue to iterate
    through the ``monitoring_stats`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[model_monitoring_service.SearchModelMonitoringStatsResponse]
        ],
        request: model_monitoring_service.SearchModelMonitoringStatsRequest,
        response: model_monitoring_service.SearchModelMonitoringStatsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.SearchModelMonitoringStatsRequest(
            request
        )
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[model_monitoring_service.SearchModelMonitoringStatsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[model_monitoring_stats.ModelMonitoringStats]:
        async def async_generator():
            async for page in self.pages:
                for response in page.monitoring_stats:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelMonitoringAlertsPager:
    """A pager for iterating through ``search_model_monitoring_alerts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``model_monitoring_alerts`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``SearchModelMonitoringAlerts`` requests and continue to iterate
    through the ``model_monitoring_alerts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., model_monitoring_service.SearchModelMonitoringAlertsResponse
        ],
        request: model_monitoring_service.SearchModelMonitoringAlertsRequest,
        response: model_monitoring_service.SearchModelMonitoringAlertsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.SearchModelMonitoringAlertsRequest(
            request
        )
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[model_monitoring_service.SearchModelMonitoringAlertsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __iter__(self) -> Iterator[model_monitoring_alert.ModelMonitoringAlert]:
        for page in self.pages:
            yield from page.model_monitoring_alerts

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelMonitoringAlertsAsyncPager:
    """A pager for iterating through ``search_model_monitoring_alerts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``model_monitoring_alerts`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``SearchModelMonitoringAlerts`` requests and continue to iterate
    through the ``model_monitoring_alerts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[model_monitoring_service.SearchModelMonitoringAlertsResponse]
        ],
        request: model_monitoring_service.SearchModelMonitoringAlertsRequest,
        response: model_monitoring_service.SearchModelMonitoringAlertsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringAlertsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = model_monitoring_service.SearchModelMonitoringAlertsRequest(
            request
        )
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[model_monitoring_service.SearchModelMonitoringAlertsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request,
                retry=self._retry,
                timeout=self._timeout,
                metadata=self._metadata,
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[model_monitoring_alert.ModelMonitoringAlert]:
        async def async_generator():
            async for page in self.pages:
                for response in page.model_monitoring_alerts:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)

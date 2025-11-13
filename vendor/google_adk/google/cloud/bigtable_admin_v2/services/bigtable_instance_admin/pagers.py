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

from google.cloud.bigtable_admin_v2.types import bigtable_instance_admin
from google.cloud.bigtable_admin_v2.types import instance


class ListAppProfilesPager:
    """A pager for iterating through ``list_app_profiles`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``app_profiles`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListAppProfiles`` requests and continue to iterate
    through the ``app_profiles`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., bigtable_instance_admin.ListAppProfilesResponse],
        request: bigtable_instance_admin.ListAppProfilesRequest,
        response: bigtable_instance_admin.ListAppProfilesResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListAppProfilesRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse):
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
        self._request = bigtable_instance_admin.ListAppProfilesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[bigtable_instance_admin.ListAppProfilesResponse]:
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

    def __iter__(self) -> Iterator[instance.AppProfile]:
        for page in self.pages:
            yield from page.app_profiles

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListAppProfilesAsyncPager:
    """A pager for iterating through ``list_app_profiles`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``app_profiles`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListAppProfiles`` requests and continue to iterate
    through the ``app_profiles`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[bigtable_instance_admin.ListAppProfilesResponse]
        ],
        request: bigtable_instance_admin.ListAppProfilesRequest,
        response: bigtable_instance_admin.ListAppProfilesResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListAppProfilesRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListAppProfilesResponse):
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
        self._request = bigtable_instance_admin.ListAppProfilesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[bigtable_instance_admin.ListAppProfilesResponse]:
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

    def __aiter__(self) -> AsyncIterator[instance.AppProfile]:
        async def async_generator():
            async for page in self.pages:
                for response in page.app_profiles:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListHotTabletsPager:
    """A pager for iterating through ``list_hot_tablets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``hot_tablets`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListHotTablets`` requests and continue to iterate
    through the ``hot_tablets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., bigtable_instance_admin.ListHotTabletsResponse],
        request: bigtable_instance_admin.ListHotTabletsRequest,
        response: bigtable_instance_admin.ListHotTabletsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListHotTabletsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse):
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
        self._request = bigtable_instance_admin.ListHotTabletsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[bigtable_instance_admin.ListHotTabletsResponse]:
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

    def __iter__(self) -> Iterator[instance.HotTablet]:
        for page in self.pages:
            yield from page.hot_tablets

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListHotTabletsAsyncPager:
    """A pager for iterating through ``list_hot_tablets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``hot_tablets`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListHotTablets`` requests and continue to iterate
    through the ``hot_tablets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[bigtable_instance_admin.ListHotTabletsResponse]
        ],
        request: bigtable_instance_admin.ListHotTabletsRequest,
        response: bigtable_instance_admin.ListHotTabletsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListHotTabletsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListHotTabletsResponse):
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
        self._request = bigtable_instance_admin.ListHotTabletsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[bigtable_instance_admin.ListHotTabletsResponse]:
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

    def __aiter__(self) -> AsyncIterator[instance.HotTablet]:
        async def async_generator():
            async for page in self.pages:
                for response in page.hot_tablets:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListLogicalViewsPager:
    """A pager for iterating through ``list_logical_views`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``logical_views`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListLogicalViews`` requests and continue to iterate
    through the ``logical_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., bigtable_instance_admin.ListLogicalViewsResponse],
        request: bigtable_instance_admin.ListLogicalViewsRequest,
        response: bigtable_instance_admin.ListLogicalViewsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListLogicalViewsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse):
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
        self._request = bigtable_instance_admin.ListLogicalViewsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[bigtable_instance_admin.ListLogicalViewsResponse]:
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

    def __iter__(self) -> Iterator[instance.LogicalView]:
        for page in self.pages:
            yield from page.logical_views

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListLogicalViewsAsyncPager:
    """A pager for iterating through ``list_logical_views`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``logical_views`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListLogicalViews`` requests and continue to iterate
    through the ``logical_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[bigtable_instance_admin.ListLogicalViewsResponse]
        ],
        request: bigtable_instance_admin.ListLogicalViewsRequest,
        response: bigtable_instance_admin.ListLogicalViewsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListLogicalViewsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListLogicalViewsResponse):
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
        self._request = bigtable_instance_admin.ListLogicalViewsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[bigtable_instance_admin.ListLogicalViewsResponse]:
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

    def __aiter__(self) -> AsyncIterator[instance.LogicalView]:
        async def async_generator():
            async for page in self.pages:
                for response in page.logical_views:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListMaterializedViewsPager:
    """A pager for iterating through ``list_materialized_views`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``materialized_views`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListMaterializedViews`` requests and continue to iterate
    through the ``materialized_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., bigtable_instance_admin.ListMaterializedViewsResponse],
        request: bigtable_instance_admin.ListMaterializedViewsRequest,
        response: bigtable_instance_admin.ListMaterializedViewsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListMaterializedViewsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse):
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
        self._request = bigtable_instance_admin.ListMaterializedViewsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[bigtable_instance_admin.ListMaterializedViewsResponse]:
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

    def __iter__(self) -> Iterator[instance.MaterializedView]:
        for page in self.pages:
            yield from page.materialized_views

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListMaterializedViewsAsyncPager:
    """A pager for iterating through ``list_materialized_views`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``materialized_views`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListMaterializedViews`` requests and continue to iterate
    through the ``materialized_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[bigtable_instance_admin.ListMaterializedViewsResponse]
        ],
        request: bigtable_instance_admin.ListMaterializedViewsRequest,
        response: bigtable_instance_admin.ListMaterializedViewsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.bigtable_admin_v2.types.ListMaterializedViewsRequest):
                The initial request object.
            response (google.cloud.bigtable_admin_v2.types.ListMaterializedViewsResponse):
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
        self._request = bigtable_instance_admin.ListMaterializedViewsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[bigtable_instance_admin.ListMaterializedViewsResponse]:
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

    def __aiter__(self) -> AsyncIterator[instance.MaterializedView]:
        async def async_generator():
            async for page in self.pages:
                for response in page.materialized_views:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)

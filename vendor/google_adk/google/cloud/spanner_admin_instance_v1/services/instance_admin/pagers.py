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

from google.cloud.spanner_admin_instance_v1.types import spanner_instance_admin
from google.longrunning import operations_pb2  # type: ignore


class ListInstanceConfigsPager:
    """A pager for iterating through ``list_instance_configs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``instance_configs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListInstanceConfigs`` requests and continue to iterate
    through the ``instance_configs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_instance_admin.ListInstanceConfigsResponse],
        request: spanner_instance_admin.ListInstanceConfigsRequest,
        response: spanner_instance_admin.ListInstanceConfigsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse):
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
        self._request = spanner_instance_admin.ListInstanceConfigsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_instance_admin.ListInstanceConfigsResponse]:
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

    def __iter__(self) -> Iterator[spanner_instance_admin.InstanceConfig]:
        for page in self.pages:
            yield from page.instance_configs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstanceConfigsAsyncPager:
    """A pager for iterating through ``list_instance_configs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``instance_configs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListInstanceConfigs`` requests and continue to iterate
    through the ``instance_configs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[spanner_instance_admin.ListInstanceConfigsResponse]
        ],
        request: spanner_instance_admin.ListInstanceConfigsRequest,
        response: spanner_instance_admin.ListInstanceConfigsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsResponse):
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
        self._request = spanner_instance_admin.ListInstanceConfigsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_instance_admin.ListInstanceConfigsResponse]:
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

    def __aiter__(self) -> AsyncIterator[spanner_instance_admin.InstanceConfig]:
        async def async_generator():
            async for page in self.pages:
                for response in page.instance_configs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstanceConfigOperationsPager:
    """A pager for iterating through ``list_instance_config_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListInstanceConfigOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., spanner_instance_admin.ListInstanceConfigOperationsResponse
        ],
        request: spanner_instance_admin.ListInstanceConfigOperationsRequest,
        response: spanner_instance_admin.ListInstanceConfigOperationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse):
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
        self._request = spanner_instance_admin.ListInstanceConfigOperationsRequest(
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
    ) -> Iterator[spanner_instance_admin.ListInstanceConfigOperationsResponse]:
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

    def __iter__(self) -> Iterator[operations_pb2.Operation]:
        for page in self.pages:
            yield from page.operations

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstanceConfigOperationsAsyncPager:
    """A pager for iterating through ``list_instance_config_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListInstanceConfigOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[spanner_instance_admin.ListInstanceConfigOperationsResponse]
        ],
        request: spanner_instance_admin.ListInstanceConfigOperationsRequest,
        response: spanner_instance_admin.ListInstanceConfigOperationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsResponse):
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
        self._request = spanner_instance_admin.ListInstanceConfigOperationsRequest(
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
    ) -> AsyncIterator[spanner_instance_admin.ListInstanceConfigOperationsResponse]:
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

    def __aiter__(self) -> AsyncIterator[operations_pb2.Operation]:
        async def async_generator():
            async for page in self.pages:
                for response in page.operations:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancesPager:
    """A pager for iterating through ``list_instances`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``instances`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListInstances`` requests and continue to iterate
    through the ``instances`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_instance_admin.ListInstancesResponse],
        request: spanner_instance_admin.ListInstancesRequest,
        response: spanner_instance_admin.ListInstancesResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse):
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
        self._request = spanner_instance_admin.ListInstancesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_instance_admin.ListInstancesResponse]:
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

    def __iter__(self) -> Iterator[spanner_instance_admin.Instance]:
        for page in self.pages:
            yield from page.instances

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancesAsyncPager:
    """A pager for iterating through ``list_instances`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``instances`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListInstances`` requests and continue to iterate
    through the ``instances`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[spanner_instance_admin.ListInstancesResponse]],
        request: spanner_instance_admin.ListInstancesRequest,
        response: spanner_instance_admin.ListInstancesResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancesResponse):
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
        self._request = spanner_instance_admin.ListInstancesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_instance_admin.ListInstancesResponse]:
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

    def __aiter__(self) -> AsyncIterator[spanner_instance_admin.Instance]:
        async def async_generator():
            async for page in self.pages:
                for response in page.instances:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancePartitionsPager:
    """A pager for iterating through ``list_instance_partitions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``instance_partitions`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListInstancePartitions`` requests and continue to iterate
    through the ``instance_partitions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_instance_admin.ListInstancePartitionsResponse],
        request: spanner_instance_admin.ListInstancePartitionsRequest,
        response: spanner_instance_admin.ListInstancePartitionsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse):
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
        self._request = spanner_instance_admin.ListInstancePartitionsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_instance_admin.ListInstancePartitionsResponse]:
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

    def __iter__(self) -> Iterator[spanner_instance_admin.InstancePartition]:
        for page in self.pages:
            yield from page.instance_partitions

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancePartitionsAsyncPager:
    """A pager for iterating through ``list_instance_partitions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``instance_partitions`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListInstancePartitions`` requests and continue to iterate
    through the ``instance_partitions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[spanner_instance_admin.ListInstancePartitionsResponse]
        ],
        request: spanner_instance_admin.ListInstancePartitionsRequest,
        response: spanner_instance_admin.ListInstancePartitionsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsResponse):
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
        self._request = spanner_instance_admin.ListInstancePartitionsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_instance_admin.ListInstancePartitionsResponse]:
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

    def __aiter__(self) -> AsyncIterator[spanner_instance_admin.InstancePartition]:
        async def async_generator():
            async for page in self.pages:
                for response in page.instance_partitions:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancePartitionOperationsPager:
    """A pager for iterating through ``list_instance_partition_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListInstancePartitionOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., spanner_instance_admin.ListInstancePartitionOperationsResponse
        ],
        request: spanner_instance_admin.ListInstancePartitionOperationsRequest,
        response: spanner_instance_admin.ListInstancePartitionOperationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse):
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
        self._request = spanner_instance_admin.ListInstancePartitionOperationsRequest(
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
    ) -> Iterator[spanner_instance_admin.ListInstancePartitionOperationsResponse]:
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

    def __iter__(self) -> Iterator[operations_pb2.Operation]:
        for page in self.pages:
            yield from page.operations

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListInstancePartitionOperationsAsyncPager:
    """A pager for iterating through ``list_instance_partition_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListInstancePartitionOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ...,
            Awaitable[spanner_instance_admin.ListInstancePartitionOperationsResponse],
        ],
        request: spanner_instance_admin.ListInstancePartitionOperationsRequest,
        response: spanner_instance_admin.ListInstancePartitionOperationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsResponse):
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
        self._request = spanner_instance_admin.ListInstancePartitionOperationsRequest(
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
    ) -> AsyncIterator[spanner_instance_admin.ListInstancePartitionOperationsResponse]:
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

    def __aiter__(self) -> AsyncIterator[operations_pb2.Operation]:
        async def async_generator():
            async for page in self.pages:
                for response in page.operations:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)

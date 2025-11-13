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

from google.cloud.spanner_admin_database_v1.types import backup
from google.cloud.spanner_admin_database_v1.types import backup_schedule
from google.cloud.spanner_admin_database_v1.types import spanner_database_admin
from google.longrunning import operations_pb2  # type: ignore


class ListDatabasesPager:
    """A pager for iterating through ``list_databases`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``databases`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDatabases`` requests and continue to iterate
    through the ``databases`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_database_admin.ListDatabasesResponse],
        request: spanner_database_admin.ListDatabasesRequest,
        response: spanner_database_admin.ListDatabasesResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabasesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse):
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
        self._request = spanner_database_admin.ListDatabasesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_database_admin.ListDatabasesResponse]:
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

    def __iter__(self) -> Iterator[spanner_database_admin.Database]:
        for page in self.pages:
            yield from page.databases

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatabasesAsyncPager:
    """A pager for iterating through ``list_databases`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``databases`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDatabases`` requests and continue to iterate
    through the ``databases`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[spanner_database_admin.ListDatabasesResponse]],
        request: spanner_database_admin.ListDatabasesRequest,
        response: spanner_database_admin.ListDatabasesResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabasesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabasesResponse):
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
        self._request = spanner_database_admin.ListDatabasesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_database_admin.ListDatabasesResponse]:
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

    def __aiter__(self) -> AsyncIterator[spanner_database_admin.Database]:
        async def async_generator():
            async for page in self.pages:
                for response in page.databases:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBackupsPager:
    """A pager for iterating through ``list_backups`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``backups`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListBackups`` requests and continue to iterate
    through the ``backups`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., backup.ListBackupsResponse],
        request: backup.ListBackupsRequest,
        response: backup.ListBackupsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupsResponse):
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
        self._request = backup.ListBackupsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[backup.ListBackupsResponse]:
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

    def __iter__(self) -> Iterator[backup.Backup]:
        for page in self.pages:
            yield from page.backups

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBackupsAsyncPager:
    """A pager for iterating through ``list_backups`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``backups`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListBackups`` requests and continue to iterate
    through the ``backups`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[backup.ListBackupsResponse]],
        request: backup.ListBackupsRequest,
        response: backup.ListBackupsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupsResponse):
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
        self._request = backup.ListBackupsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[backup.ListBackupsResponse]:
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

    def __aiter__(self) -> AsyncIterator[backup.Backup]:
        async def async_generator():
            async for page in self.pages:
                for response in page.backups:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatabaseOperationsPager:
    """A pager for iterating through ``list_database_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDatabaseOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_database_admin.ListDatabaseOperationsResponse],
        request: spanner_database_admin.ListDatabaseOperationsRequest,
        response: spanner_database_admin.ListDatabaseOperationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse):
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
        self._request = spanner_database_admin.ListDatabaseOperationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_database_admin.ListDatabaseOperationsResponse]:
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


class ListDatabaseOperationsAsyncPager:
    """A pager for iterating through ``list_database_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDatabaseOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[spanner_database_admin.ListDatabaseOperationsResponse]
        ],
        request: spanner_database_admin.ListDatabaseOperationsRequest,
        response: spanner_database_admin.ListDatabaseOperationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabaseOperationsResponse):
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
        self._request = spanner_database_admin.ListDatabaseOperationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_database_admin.ListDatabaseOperationsResponse]:
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


class ListBackupOperationsPager:
    """A pager for iterating through ``list_backup_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListBackupOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., backup.ListBackupOperationsResponse],
        request: backup.ListBackupOperationsRequest,
        response: backup.ListBackupOperationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse):
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
        self._request = backup.ListBackupOperationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[backup.ListBackupOperationsResponse]:
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


class ListBackupOperationsAsyncPager:
    """A pager for iterating through ``list_backup_operations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``operations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListBackupOperations`` requests and continue to iterate
    through the ``operations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[backup.ListBackupOperationsResponse]],
        request: backup.ListBackupOperationsRequest,
        response: backup.ListBackupOperationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupOperationsRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupOperationsResponse):
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
        self._request = backup.ListBackupOperationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[backup.ListBackupOperationsResponse]:
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


class ListDatabaseRolesPager:
    """A pager for iterating through ``list_database_roles`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``database_roles`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDatabaseRoles`` requests and continue to iterate
    through the ``database_roles`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., spanner_database_admin.ListDatabaseRolesResponse],
        request: spanner_database_admin.ListDatabaseRolesRequest,
        response: spanner_database_admin.ListDatabaseRolesResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse):
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
        self._request = spanner_database_admin.ListDatabaseRolesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[spanner_database_admin.ListDatabaseRolesResponse]:
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

    def __iter__(self) -> Iterator[spanner_database_admin.DatabaseRole]:
        for page in self.pages:
            yield from page.database_roles

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatabaseRolesAsyncPager:
    """A pager for iterating through ``list_database_roles`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``database_roles`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDatabaseRoles`` requests and continue to iterate
    through the ``database_roles`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[spanner_database_admin.ListDatabaseRolesResponse]
        ],
        request: spanner_database_admin.ListDatabaseRolesRequest,
        response: spanner_database_admin.ListDatabaseRolesResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListDatabaseRolesResponse):
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
        self._request = spanner_database_admin.ListDatabaseRolesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[spanner_database_admin.ListDatabaseRolesResponse]:
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

    def __aiter__(self) -> AsyncIterator[spanner_database_admin.DatabaseRole]:
        async def async_generator():
            async for page in self.pages:
                for response in page.database_roles:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBackupSchedulesPager:
    """A pager for iterating through ``list_backup_schedules`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``backup_schedules`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListBackupSchedules`` requests and continue to iterate
    through the ``backup_schedules`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., backup_schedule.ListBackupSchedulesResponse],
        request: backup_schedule.ListBackupSchedulesRequest,
        response: backup_schedule.ListBackupSchedulesResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse):
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
        self._request = backup_schedule.ListBackupSchedulesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[backup_schedule.ListBackupSchedulesResponse]:
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

    def __iter__(self) -> Iterator[backup_schedule.BackupSchedule]:
        for page in self.pages:
            yield from page.backup_schedules

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBackupSchedulesAsyncPager:
    """A pager for iterating through ``list_backup_schedules`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``backup_schedules`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListBackupSchedules`` requests and continue to iterate
    through the ``backup_schedules`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[backup_schedule.ListBackupSchedulesResponse]],
        request: backup_schedule.ListBackupSchedulesRequest,
        response: backup_schedule.ListBackupSchedulesResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesRequest):
                The initial request object.
            response (google.cloud.spanner_admin_database_v1.types.ListBackupSchedulesResponse):
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
        self._request = backup_schedule.ListBackupSchedulesRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[backup_schedule.ListBackupSchedulesResponse]:
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

    def __aiter__(self) -> AsyncIterator[backup_schedule.BackupSchedule]:
        async def async_generator():
            async for page in self.pages:
                for response in page.backup_schedules:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)

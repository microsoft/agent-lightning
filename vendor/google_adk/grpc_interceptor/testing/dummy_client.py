"""Defines a service and client for testing interceptors."""

import asyncio
from concurrent import futures
from contextlib import contextmanager
from inspect import iscoroutine
from threading import Event, Thread
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union
)

import grpc

from grpc_interceptor.client import ClientInterceptor
from grpc_interceptor.server import AsyncServerInterceptor, grpc_aio, ServerInterceptor
from grpc_interceptor.testing.protos import dummy_pb2_grpc
from grpc_interceptor.testing.protos.dummy_pb2 import DummyRequest, DummyResponse

SpecialCaseFunction = Callable[
    [str, Union[grpc.ServicerContext, grpc_aio.ServicerContext]], str
]


class _SpecialCaseMixin:
    _special_cases: Dict[str, SpecialCaseFunction]

    def _get_output(self, request: DummyRequest, context: grpc.ServicerContext) -> str:
        input = request.input

        output = input
        if input in self._special_cases:
            output = self._special_cases[input](input, context)

        return output

    async def _get_output_async(
        self,
        request: DummyRequest,
        context: grpc_aio.ServicerContext
    ) -> str:
        input = request.input

        output = input
        if input in self._special_cases:
            output = self._special_cases[input](input, context)
            if iscoroutine(output):
                output = await output

        return output


class DummyService(dummy_pb2_grpc.DummyServiceServicer, _SpecialCaseMixin):
    """A gRPC service used for testing.

    Args:
        special_cases: A dictionary where the keys are strings, and the values are
            functions that take and return strings. The functions can also raise
            exceptions. When the Execute method is given a string in the dict, it
            will call the function with that string instead, and return the result.
            This allows testing special cases, like raising exceptions.
    """

    def __init__(
        self,
        special_cases: Dict[str, SpecialCaseFunction],
    ):
        self._special_cases = special_cases

    def Execute(
        self, request: DummyRequest, context: grpc.ServicerContext
    ) -> DummyResponse:
        """Echo the input, or take on of the special cases actions."""
        return DummyResponse(output=self._get_output(request, context))

    def ExecuteClientStream(
        self, request_iter: Iterable[DummyRequest], context: grpc.ServicerContext
    ) -> DummyResponse:
        """Iterate over the input and concatenates the strings into the output."""
        output = "".join(self._get_output(request, context) for request in request_iter)
        return DummyResponse(output=output)

    def ExecuteServerStream(
        self, request: DummyRequest, context: grpc.ServicerContext
    ) -> Iterable[DummyResponse]:
        """Stream one character at a time from the input."""
        for c in self._get_output(request, context):
            yield DummyResponse(output=c)

    def ExecuteClientServerStream(
        self, request_iter: Iterable[DummyRequest], context: grpc.ServicerContext
    ) -> Iterable[DummyResponse]:
        """Stream input to output."""
        for request in request_iter:
            yield DummyResponse(output=self._get_output(request, context))


class AsyncDummyService(dummy_pb2_grpc.DummyServiceServicer, _SpecialCaseMixin):
    """A gRPC service used for testing, similar to DummyService except async.

    See DummyService for more info.
    """

    def __init__(
        self,
        special_cases: Dict[str, SpecialCaseFunction],
    ):
        self._special_cases = special_cases

    async def Execute(
        self, request: DummyRequest, context: grpc_aio.ServicerContext
    ) -> DummyResponse:
        """Echo the input, or take on of the special cases actions."""
        return DummyResponse(output=await self._get_output_async(request, context))

    async def ExecuteClientStream(
        self,
        request_iter: AsyncIterable[DummyRequest],
        context: grpc_aio.ServicerContext,
    ) -> DummyResponse:
        """Iterate over the input and concatenates the strings into the output."""
        output = "".join([
            await self._get_output_async(request, context)
            async for request in request_iter
        ])  # noqa: E501
        return DummyResponse(output=output)

    async def ExecuteServerStream(
        self, request: DummyRequest, context: grpc_aio.ServicerContext
    ) -> AsyncGenerator[DummyResponse, None]:
        """Stream one character at a time from the input."""
        for c in await self._get_output_async(request, context):
            yield DummyResponse(output=c)

    async def ExecuteClientServerStream(
        self,
        request_iter: AsyncIterable[DummyRequest],
        context: grpc_aio.ServicerContext,
    ) -> AsyncGenerator[DummyResponse, None]:
        """Stream input to output."""
        async for request in request_iter:
            yield DummyResponse(output=await self._get_output_async(request, context))


class AsyncReadWriteDummyService(
    dummy_pb2_grpc.DummyServiceServicer, _SpecialCaseMixin
):
    """Similar to AsyncDummyService except uses the read / write API.

    See DummyService for more info.
    """

    def __init__(
        self,
        special_cases: Dict[str, SpecialCaseFunction],
    ):
        self._special_cases = special_cases

    async def Execute(
        self, request: DummyRequest, context: grpc_aio.ServicerContext
    ) -> DummyResponse:
        """Echo the input, or take on of the special cases actions."""
        return DummyResponse(output=await self._get_output_async(request, context))

    async def ExecuteClientStream(
        self,
        unused_request: Any,
        context: grpc_aio.ServicerContext,
    ) -> DummyResponse:
        """Iterate over the input and concatenates the strings into the output."""
        output = []
        while True:
            request = await context.read()
            if request == grpc_aio.EOF:
                break
            output.append(await self._get_output_async(request, context))

        return DummyResponse(output="".join(output))

    async def ExecuteServerStream(
        self, request: DummyRequest, context: grpc_aio.ServicerContext
    ) -> None:
        """Stream one character at a time from the input."""
        for c in await self._get_output_async(request, context):
            await context.write(DummyResponse(output=c))

    async def ExecuteClientServerStream(
        self,
        request_iter: AsyncIterable[DummyRequest],
        context: grpc_aio.ServicerContext,
    ) -> None:
        """Stream input to output."""
        while True:
            request = await context.read()
            if request == grpc_aio.EOF:
                break
            await context.write(
                DummyResponse(output=await self._get_output_async(request, context))
            )


@contextmanager
def dummy_client(
    special_cases: Dict[str, SpecialCaseFunction],
    interceptors: Optional[List[ServerInterceptor]] = None,
    client_interceptors: Optional[List[ClientInterceptor]] = None,
    aio_server: bool = False,
    aio_client: bool = False,
    aio_read_write: bool = False,
):
    """A context manager that returns a gRPC client connected to a DummyService."""
    # Sanity check that the interceptors are async if using an async server,
    # otherwise the tests will just hang.
    for intr in interceptors or []:
        if aio_server != isinstance(intr, AsyncServerInterceptor):
            raise TypeError("Set aio_server correctly")
    with dummy_channel(
        special_cases,
        interceptors,
        client_interceptors,
        aio_server=aio_server,
        aio_client=aio_client,
        aio_read_write=aio_read_write,
    ) as channel:
        client = dummy_pb2_grpc.DummyServiceStub(channel)
        yield client


@contextmanager
def dummy_channel(
    special_cases: Dict[str, SpecialCaseFunction],
    interceptors: Optional[List[ServerInterceptor]] = None,
    client_interceptors: Optional[List[ClientInterceptor]] = None,
    aio_server: bool = False,
    aio_client: bool = False,
    aio_read_write: bool = False,
):
    """A context manager that returns a gRPC channel connected to a DummyService."""
    if not interceptors:
        interceptors = []

    if aio_server:
        service = (
            AsyncReadWriteDummyService(special_cases)
            if aio_read_write
            else AsyncDummyService(special_cases)
        )
        aio_loop = asyncio.new_event_loop()
        aio_thread = _AsyncServerThread(
            aio_loop,
            service,
            interceptors,
        )
        aio_thread.start()
        aio_thread.wait_for_server()
        port = aio_thread.port
    else:
        dummy_service = DummyService(special_cases)
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1), interceptors=interceptors
        )
        dummy_pb2_grpc.add_DummyServiceServicer_to_server(dummy_service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()

    channel_descriptor = f"localhost:{port}"

    if aio_client:
        channel = grpc_aio.insecure_channel(channel_descriptor)
        # Client interceptors might work, but I haven't tested them yet.
        if client_interceptors:
            raise TypeError("Client interceptors not supported with async channel")
        # We close the channel in _AsyncServerThread because we need to await
        # it, and doing that in this thread is problematic because dummy_client
        # isn't always used in an async context. We could get around that by
        # creating a new loop or something, but will be lazy and use the server
        # thread / loop for now.
        if not aio_server:
            raise ValueError("aio_server must be True if aio_client is True")
        aio_thread.async_channel = channel
    else:
        channel = grpc.insecure_channel(channel_descriptor)
        if client_interceptors:
            channel = grpc.intercept_channel(channel, *client_interceptors)

    try:
        yield channel
    finally:
        if not aio_client:
            # async channel is closed by _AsyncServerThread
            channel.close()

        if aio_server:
            aio_thread.stop()
            aio_thread.join()
        else:
            server.stop(None)


class _AsyncServerThread(Thread):
    port: int = 0
    async_channel = None

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        service,
        interceptors: List[AsyncServerInterceptor],
    ):
        super().__init__()
        self.__loop = loop
        self.__service = service
        self.__interceptors = interceptors
        self.__started = Event()

    def run(self):
        asyncio.set_event_loop(self.__loop)
        self.__loop.run_until_complete(self.__run_server())

    async def __run_server(self):
        self.__server = grpc_aio.server(interceptors=tuple(self.__interceptors))
        dummy_pb2_grpc.add_DummyServiceServicer_to_server(self.__service, self.__server)
        self.port = self.__server.add_insecure_port("localhost:0")
        await self.__server.start()
        self.__started.set()
        await self.__server.wait_for_termination()
        if self.async_channel:
            await self.async_channel.close()

    def wait_for_server(self):
        self.__started.wait()

    def stop(self):
        self.__loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self.__shutdown())
        )

    async def __shutdown(self) -> None:
        await self.__server.stop(None)
        self.__loop.stop()

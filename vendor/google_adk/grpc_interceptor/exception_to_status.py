"""ExceptionToStatusInterceptor catches GrpcException and sets the gRPC context."""

# TODO: use asynccontextmanager
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    NoReturn,
    Optional,
)

import grpc
from grpc import aio as grpc_aio

from grpc_interceptor.exceptions import GrpcException
from grpc_interceptor.server import AsyncServerInterceptor, ServerInterceptor


class ExceptionToStatusInterceptor(ServerInterceptor):
    """An interceptor that catches exceptions and sets the RPC status and details.

    ExceptionToStatusInterceptor will catch any subclass of GrpcException and set the
    status code and details on the gRPC context. You can also extend this and override
    the handle_exception method to catch other types of exceptions, and handle them in
    different ways. E.g., you can catch and handle exceptions that don't derive from
    GrpcException. Or you can set rich error statuses with context.abort_with_status().

    Args:
        status_on_unknown_exception: Specify what to do if an exception which is
            not a subclass of GrpcException is raised. If None, do nothing (by
            default, grpc will set the status to UNKNOWN). If not None, then the
            status code will be set to this value if `context.abort` hasn't been called
            earlier. It must not be OK. The details will be set to the value of repr(e),
            where e is the exception. In any case, the exception will be propagated.

    Raises:
        ValueError: If status_code is OK.
    """

    def __init__(self, status_on_unknown_exception: Optional[grpc.StatusCode] = None):
        if status_on_unknown_exception == grpc.StatusCode.OK:
            raise ValueError("The status code for unknown exceptions cannot be OK")

        self._status_on_unknown_exception = status_on_unknown_exception

    def _generate_responses(
        self,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
        response_iterator: Iterable,
    ) -> Generator[Any, None, None]:
        """Yield all the responses, but check for errors along the way."""
        with self._handle_exception(request_or_iterator, context, method_name):
            yield from response_iterator

    @contextmanager
    def _handle_exception(
        self, request_or_iterator: Any, context: grpc.ServicerContext, method_name: str
    ) -> Iterator[None]:
        try:
            yield
        except Exception as ex:
            self.handle_exception(ex, request_or_iterator, context, method_name)

    def handle_exception(
        self,
        ex: Exception,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> NoReturn:
        """Override this if extending ExceptionToStatusInterceptor.

        This will get called when an exception is raised while handling the RPC.

        Args:
            ex: The exception that was raised.
            request_or_iterator: The RPC request, as a protobuf message if it is a
                unary request, or an iterator of protobuf messages if it is a streaming
                request.
            context: The servicer context. You probably want to call context.abort(...)
            method_name: The name of the RPC being called.

        Raises:
            This method must raise and cannot return, as in general there's no
            meaningful RPC response to return if an exception has occurred. You can
            raise the original exception, ex, or something else.
        """
        if isinstance(ex, GrpcException):
            context.abort(ex.status_code, ex.details)
        elif not context.code():
            if self._status_on_unknown_exception is not None:
                context.abort(self._status_on_unknown_exception, repr(ex))
        raise ex

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:
        """Do not call this directly; use the interceptor kwarg on grpc.server()."""
        with self._handle_exception(request_or_iterator, context, method_name):
            response_or_iterator = method(request_or_iterator, context)

        if isinstance(response_or_iterator, Iterable):
            # multiple responses; return a generator
            return self._generate_responses(
                request_or_iterator, context, method_name, response_or_iterator
            )
        else:
            # return a single response
            return response_or_iterator


class AsyncExceptionToStatusInterceptor(AsyncServerInterceptor):
    """An interceptor that catches exceptions and sets the RPC status and details.

    This is the async analogy to ExceptionToStatusInterceptor. Please see that class'
    documentation for more information.
    """

    def __init__(self, status_on_unknown_exception: Optional[grpc.StatusCode] = None):
        if status_on_unknown_exception == grpc.StatusCode.OK:
            raise ValueError("The status code for unknown exceptions cannot be OK")

        self._status_on_unknown_exception = status_on_unknown_exception

    async def _generate_responses(
        self,
        request_or_iterator: Any,
        context: grpc_aio.ServicerContext,
        method_name: str,
        response_iterator: AsyncIterable,
    ) -> AsyncGenerator[Any, None]:
        """Yield all the responses, but check for errors along the way."""
        try:
            async for r in response_iterator:
                yield r
        except Exception as ex:
            await self.handle_exception(ex, request_or_iterator, context, method_name)

    async def handle_exception(
        self,
        ex: Exception,
        request_or_iterator: Any,
        context: grpc_aio.ServicerContext,
        method_name: str,
    ) -> NoReturn:
        """Override this if extending ExceptionToStatusInterceptor.

        This will get called when an exception is raised while handling the RPC.

        Args:
            ex: The exception that was raised.
            request_or_iterator: The RPC request, as a protobuf message if it is a
                unary request, or an iterator of protobuf messages if it is a streaming
                request.
            context: The servicer context. You probably want to call context.abort(...)
            method_name: The name of the RPC being called.

        Raises:
            This method must raise and cannot return, as in general there's no
            meaningful RPC response to return if an exception has occurred. You can
            raise the original exception, ex, or something else.
        """
        if isinstance(ex, GrpcException):
            await context.abort(ex.status_code, ex.details)
        elif not context.code():
            if self._status_on_unknown_exception is not None:
                await context.abort(self._status_on_unknown_exception, repr(ex))
        raise ex

    async def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc_aio.ServicerContext,
        method_name: str,
    ) -> Any:
        """Do not call this directly; use the interceptor kwarg on grpc.server()."""
        try:
            response_or_iterator = method(request_or_iterator, context)
            if not hasattr(response_or_iterator, "__aiter__"):
                return await response_or_iterator
        except Exception as ex:
            await self.handle_exception(ex, request_or_iterator, context, method_name)

        return self._generate_responses(
            request_or_iterator, context, method_name, response_or_iterator
        )

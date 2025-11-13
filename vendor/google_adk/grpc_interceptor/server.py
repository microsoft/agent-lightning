"""Base class for server-side interceptors."""

import abc
from asyncio import iscoroutine
from typing import Any, Callable, Tuple

import grpc
from grpc import aio as grpc_aio  # Needed for grpcio pre-1.33.2


class ServerInterceptor(grpc.ServerInterceptor, metaclass=abc.ABCMeta):
    """Base class for server-side interceptors.

    To implement an interceptor, subclass this class and override the intercept method.
    """

    @abc.abstractmethod
    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:  # pragma: no cover
        """Override this method to implement a custom interceptor.

        You should call method(request_or_iterator, context) to invoke the next handler
        (either the RPC method implementation, or the next interceptor in the list).

        Args:
            method: Either the RPC method implementation, or the next interceptor in
                the chain.
            request_or_iterator: The RPC request, as a protobuf message if it is a
                unary request, or an iterator of protobuf messages if it is a streaming
                request.
            context: The ServicerContext pass by gRPC to the service.
            method_name: A string of the form "/protobuf.package.Service/Method"

        Returns:
            This should return the result of method(request, context), which
            is typically the RPC method response, as a protobuf message, or an
            iterator of protobuf messages for streaming responses. The interceptor is
            free to modify this in some way, however.
        """
        return method(request_or_iterator, context)

    # Implementation of grpc.ServerInterceptor, do not override.
    def intercept_service(self, continuation, handler_call_details):
        """Implementation of grpc.ServerInterceptor.

        This is not part of the grpc_interceptor.ServerInterceptor API, but must have
        a public name. Do not override it, unless you know what you're doing.
        """
        next_handler = continuation(handler_call_details)
        # Returns None if the method isn't implemented.
        if next_handler is None:
            return

        handler_factory, next_handler_method = _get_factory_and_method(next_handler)

        def invoke_intercept_method(request_or_iterator, context):
            method_name = handler_call_details.method
            return self.intercept(
                next_handler_method,
                request_or_iterator,
                context,
                method_name,
            )

        return handler_factory(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer,
        )


class AsyncServerInterceptor(grpc_aio.ServerInterceptor, metaclass=abc.ABCMeta):
    """Base class for asyncio server-side interceptors.

    To implement an interceptor, subclass this class and override the intercept method.
    """

    @abc.abstractmethod
    async def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc_aio.ServicerContext,
        method_name: str,
    ) -> Any:  # pragma: no cover
        """Override this method to implement a custom interceptor.

        You should await method(request_or_iterator, context) to invoke the next handler
        (either the RPC method implementation, or the next interceptor in the list).

        Args:
            method: Either the RPC method implementation, or the next interceptor in
                the chain.
            request_or_iterator: The RPC request, as a protobuf message if it is a
                unary request, or an iterator of protobuf messages if it is a streaming
                request.
            context: The ServicerContext pass by gRPC to the service.
            method_name: A string of the form "/protobuf.package.Service/Method"

        Returns:
            This should return the result of method(request_or_iterator, context),
            which is typically the RPC method response, as a protobuf message. The
            interceptor is free to modify this in some way, however.
        """
        response_or_iterator = method(request_or_iterator, context)
        if hasattr(response_or_iterator, "__aiter__"):
            return response_or_iterator
        else:
            return await response_or_iterator

    # Implementation of grpc.ServerInterceptor, do not override.
    async def intercept_service(self, continuation, handler_call_details):
        """Implementation of grpc.aio.ServerInterceptor.

        This is not part of the grpc_interceptor.AsyncServerInterceptor API, but must
        have a public name. Do not override it, unless you know what you're doing.
        """
        next_handler = await continuation(handler_call_details)
        # Returns None if the method isn't implemented.
        if not next_handler:
            return

        handler_factory, next_handler_method = _get_factory_and_method(next_handler)

        if next_handler.response_streaming:

            async def invoke_intercept_method(request, context):
                method_name = handler_call_details.method
                coroutine_or_asyncgen = self.intercept(
                    next_handler_method,
                    request,
                    context,
                    method_name,
                )

                # Async server streaming handlers return async_generator, because they
                # use the async def + yield syntax. However, this is NOT a coroutine
                # and hence is not awaitable. This can be a problem if the interceptor
                # ignores the individual streaming response items and simply returns the
                # result of method(request, context). In that case the interceptor IS a
                # coroutine, and hence should be awaited. In both cases, we need
                # something we can iterate over so that THIS function is an
                # async_generator like the actual RPC method.
                if iscoroutine(coroutine_or_asyncgen):
                    asyncgen_or_none = await coroutine_or_asyncgen
                    # If a handler is using the read/write API, it will return None.
                    if not asyncgen_or_none:
                        return
                    asyncgen = asyncgen_or_none
                else:
                    asyncgen = coroutine_or_asyncgen

                async for r in asyncgen:
                    yield r

        else:

            async def invoke_intercept_method(request, context):
                method_name = handler_call_details.method
                return await self.intercept(
                    next_handler_method,
                    request,
                    context,
                    method_name,
                )

        return handler_factory(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer,
        )


def _get_factory_and_method(
    rpc_handler: grpc.RpcMethodHandler,
) -> Tuple[Callable, Callable]:
    if rpc_handler.unary_unary:
        return grpc.unary_unary_rpc_method_handler, rpc_handler.unary_unary
    elif rpc_handler.unary_stream:
        return grpc.unary_stream_rpc_method_handler, rpc_handler.unary_stream
    elif rpc_handler.stream_unary:
        return grpc.stream_unary_rpc_method_handler, rpc_handler.stream_unary
    elif rpc_handler.stream_stream:
        return grpc.stream_stream_rpc_method_handler, rpc_handler.stream_stream
    else:  # pragma: no cover
        raise RuntimeError("RPC handler implementation does not exist")


class MethodName:
    """Represents a gRPC method name.

    gRPC methods are defined by three parts, represented by the three attributes.

    Attributes:
        package: This is defined by the `package foo.bar;` designation in the protocol
            buffer definition, or it could be defined by the protocol buffer directory
            structure, depending on the language
            (see https://developers.google.com/protocol-buffers/docs/proto3#packages).
        service: This is the service name in the protocol buffer definition (e.g.,
            `service SearchService { ... }`.
        method: This is the method name. (e.g., `rpc Search(...) returns (...);`).
    """

    def __init__(self, package: str, service: str, method: str):
        self.package = package
        self.service = service
        self.method = method

    def __repr__(self) -> str:
        """Object-like representation."""
        return (
            f"MethodName(package='{self.package}', service='{self.service}',"
            f" method='{self.method}')"
        )

    @property
    def fully_qualified_service(self):
        """Return the service name prefixed with the package.

        Example:
            >>> MethodName("foo.bar", "SearchService", "Search").fully_qualified_service
            'foo.bar.SearchService'
        """
        return f"{self.package}.{self.service}" if self.package else self.service


def parse_method_name(method_name: str) -> MethodName:
    """Parse a method name into package, service and endpoint components.

    Arguments:
        method_name: A string of the form "/foo.bar.SearchService/Search", as passed to
            ServerInterceptor.intercept().

    Returns:
        A MethodName object.

    Example:
        >>> parse_method_name("/foo.bar.SearchService/Search")
        MethodName(package='foo.bar', service='SearchService', method='Search')
    """
    _, package_and_service, method = method_name.split("/")
    *maybe_package, service = package_and_service.rsplit(".", maxsplit=1)
    package = maybe_package[0] if maybe_package else ""
    return MethodName(package, service, method)

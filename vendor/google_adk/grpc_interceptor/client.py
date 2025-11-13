"""Base class for client-side interceptors."""

import abc
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import grpc


class _ClientCallDetailsFields(NamedTuple):
    method: str
    timeout: Optional[float]
    metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]]
    credentials: Optional[grpc.CallCredentials]
    wait_for_ready: Optional[bool]
    compression: Any  # Type added in grpcio 1.23.0


class ClientCallDetails(_ClientCallDetailsFields, grpc.ClientCallDetails):
    """Describes an RPC to be invoked.

    See https://grpc.github.io/grpc/python/grpc.html#grpc.ClientCallDetails
    """

    pass


class ClientInterceptorReturnType(grpc.Call, grpc.Future):
    """Return type for the ClientInterceptor.intercept method."""

    pass


class ClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
    metaclass=abc.ABCMeta,
):
    """Base class for client-side interceptors.

    To implement an interceptor, subclass this class and override the intercept method.
    """

    @abc.abstractmethod
    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        call_details: grpc.ClientCallDetails,
    ) -> ClientInterceptorReturnType:
        """Override this method to implement a custom interceptor.

        This method is called for all unary and streaming RPCs. The interceptor
        implementation should call `method` using a `grpc.ClientCallDetails` and the
        `request_or_iterator` object as parameters. The `request_or_iterator`
        parameter may be type checked to determine if this is a singluar request
        for unary RPCs or an iterator for client-streaming or client-server streaming
        RPCs.

        Args:
            method: A function that proceeds with the invocation by executing the next
                interceptor in the chain or invoking the actual RPC on the underlying
                channel.
            request_or_iterator: RPC request message or iterator of request messages
                for streaming requests.
            call_details: Describes an RPC to be invoked.

        Returns:
            The type of the return should match the type of the return value received
            by calling `method`. This is an object that is both a
            `Call <https://grpc.github.io/grpc/python/grpc.html#grpc.Call>`_ for the
            RPC and a
            `Future <https://grpc.github.io/grpc/python/grpc.html#grpc.Future>`_.

            The actual result from the RPC can be got by calling `.result()` on the
            value returned from `method`.
        """
        return method(request_or_iterator, call_details)  # pragma: no cover

    def intercept_unary_unary(
        self,
        continuation: Callable,
        call_details: grpc.ClientCallDetails,
        request: Any,
    ):
        """Implementation of grpc.UnaryUnaryClientInterceptor.

        This is not part of the grpc_interceptor.ClientInterceptor API, but must have
        a public name. Do not override it, unless you know what you're doing.
        """
        return self.intercept(_swap_args(continuation), request, call_details)

    def intercept_unary_stream(
        self,
        continuation: Callable,
        call_details: grpc.ClientCallDetails,
        request: Any,
    ):
        """Implementation of grpc.UnaryStreamClientInterceptor.

        This is not part of the grpc_interceptor.ClientInterceptor API, but must have
        a public name. Do not override it, unless you know what you're doing.
        """
        return self.intercept(_swap_args(continuation), request, call_details)

    def intercept_stream_unary(
        self,
        continuation: Callable,
        call_details: grpc.ClientCallDetails,
        request_iterator: Iterator[Any],
    ):
        """Implementation of grpc.StreamUnaryClientInterceptor.

        This is not part of the grpc_interceptor.ClientInterceptor API, but must have
        a public name. Do not override it, unless you know what you're doing.
        """
        return self.intercept(_swap_args(continuation), request_iterator, call_details)

    def intercept_stream_stream(
        self,
        continuation: Callable,
        call_details: grpc.ClientCallDetails,
        request_iterator: Iterator[Any],
    ):
        """Implementation of grpc.StreamStreamClientInterceptor.

        This is not part of the grpc_interceptor.ClientInterceptor API, but must have
        a public name. Do not override it, unless you know what you're doing.
        """
        return self.intercept(_swap_args(continuation), request_iterator, call_details)


def _swap_args(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def new_fn(x, y):
        return fn(y, x)

    return new_fn

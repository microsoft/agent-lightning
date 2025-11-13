"""A framework for testing interceptors."""

from typing import Callable

from grpc_interceptor.testing.dummy_client import (
    dummy_client,
    DummyService,
)
from grpc_interceptor.testing.protos.dummy_pb2 import DummyRequest, DummyResponse


__all__ = [
    "dummy_client",
    "DummyRequest",
    "DummyResponse",
    "DummyService",
    "raises",
]


def raises(e: Exception) -> Callable:
    """Return a function that raises the given exception when called.

    Args:
        e: The exception to be raised.

    Returns:
        A function that can take any arguments, and raises the given exception.
    """

    def f(*args, **kwargs):
        raise (e)

    return f

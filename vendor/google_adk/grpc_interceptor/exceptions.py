"""Exceptions for ExceptionToStatusInterceptor.

See https://grpc.github.io/grpc/core/md_doc_statuscodes.html for the source of truth
on status code meanings.
"""

from typing import Optional

from grpc import StatusCode


class GrpcException(Exception):
    """Base class for gRPC exceptions.

    Generally you would not use this class directly, but rather use a subclass
    representing one of the standard gRPC status codes (see:
    https://grpc.github.io/grpc/core/md_doc_statuscodes.html for the official list).

    Attributes:
        status_code: A grpc.StatusCode other than OK. The only use case for this
            is if gRPC adds a new status code that isn't represented by one of the
            subclasses of GrpcException. Must not be OK, because gRPC will not
            raise an RpcError to the client if the status code is OK.
        details: A string with additional informantion about the error.
    Args:
        details: If not None, specifies a custom error message.
        status_code: If not None, sets the status code.

    Raises:
        ValueError: If status_code is OK.
    """

    status_code: StatusCode = StatusCode.UNKNOWN
    details: str = "Unknown exception occurred"

    def __init__(
        self, details: Optional[str] = None, status_code: Optional[StatusCode] = None
    ):
        if status_code is not None:
            if status_code == StatusCode.OK:
                raise ValueError("The status code for an exception cannot be OK")
            self.status_code = status_code
        if details is not None:
            self.details = details

    def __repr__(self) -> str:
        """Show the status code and details.

        Returns:
            A string displaying the class name, status code, and details.
        """
        clsname = self.__class__.__name__
        sc = self.status_code.name
        return f"{clsname}(status_code={sc}, details={self.details!r})"

    @property
    def status_string(self):
        """Return status_code as a string.

        Returns:
            The status code as a string.

        Example:
            >>> GrpcException(status_code=StatusCode.NOT_FOUND).status_string
            'NOT_FOUND'
        """
        return self.status_code.name


class Aborted(GrpcException):
    """The operation was aborted.

    Typically this is due to a concurrency issue such as a sequencer check failure or
    transaction abort. See the guidelines on other exceptions for deciding between
    FAILED_PRECONDITION, ABORTED, and UNAVAILABLE.
    """

    status_code = StatusCode.ABORTED
    details = "The operation was aborted"


class AlreadyExists(GrpcException):
    """The entity that a client attempted to create already exists.

    E.g., a file or directory that a client is trying to create already exists.
    """

    status_code = StatusCode.ALREADY_EXISTS
    details = "The entity attempted to be created already exists"


class Cancelled(GrpcException):
    """The operation was cancelled, typically by the caller."""

    status_code = StatusCode.CANCELLED
    details = "The operation was cancelled"


class DataLoss(GrpcException):
    """Unrecoverable data loss or corruption."""

    status_code = StatusCode.DATA_LOSS
    details = "There was unrecoverable data loss or corruption"


class DeadlineExceeded(GrpcException):
    """The deadline expired before the operation could complete.

    For operations that change the state of the system, this error may be returned even
    if the operation has completed successfully. For example, a successful response
    from a server could have been delayed long.
    """

    status_code = StatusCode.DEADLINE_EXCEEDED
    details = "Deadline expired before operation could complete"


class FailedPrecondition(GrpcException):
    """The operation failed because the system is in an invalid state for execution.

    For example, the directory to be deleted is non-empty, an rmdir operation is
    applied to a non-directory, etc. Service implementors can use the following
    guidelines to decide between FAILED_PRECONDITION, ABORTED, and UNAVAILABLE:
    (a) Use UNAVAILABLE if the client can retry just the failing call. (b) Use ABORTED
    if the client should retry at a higher level (e.g., when a client-specified
    test-and-set fails, indicating the client should restart a read-modify-write
    sequence). (c) Use FAILED_PRECONDITION if the client should not retry until the
    system state has been explicitly fixed. E.g., if an "rmdir" fails because the
    directory is non-empty, FAILED_PRECONDITION should be returned since the client
    should not retry unless the files are deleted from the directory.
    """

    status_code = StatusCode.FAILED_PRECONDITION
    details = (
        "The operation was rejected because the system is not"
        " in a state required for execution"
    )


class InvalidArgument(GrpcException):
    """The client specified an invalid argument.

    Note that this differs from FAILED_PRECONDITION. INVALID_ARGUMENT indicates
    arguments that are problematic regardless of the state of the system (e.g., a
    malformed file name).
    """

    status_code = StatusCode.INVALID_ARGUMENT
    details = "The client specified an invalid argument"


class Internal(GrpcException):
    """Internal errors.

    This means that some invariants expected by the underlying system have been broken.
    This error code is reserved for serious errors.
    """

    status_code = StatusCode.INTERNAL
    details = "Internal error"


class OutOfRange(GrpcException):
    """The operation was attempted past the valid range.

    E.g., seeking or reading past end-of-file. Unlike INVALID_ARGUMENT, this error
    indicates a problem that may be fixed if the system state changes. For example, a
    32-bit file system will generate INVALID_ARGUMENT if asked to read at an offset
    that is not in the range [0,2^32-1], but it will generate OUT_OF_RANGE if asked to
    read from an offset past the current file size. There is a fair bit of overlap
    between FAILED_PRECONDITION and OUT_OF_RANGE. We recommend using OUT_OF_RANGE (the
    more specific error) when it applies so that callers who are iterating through a
    space can easily look for an OUT_OF_RANGE error to detect when they are done.
    """

    status_code = StatusCode.OUT_OF_RANGE
    details = "The operation was attempted past the valid range"


class NotFound(GrpcException):
    """Some requested entity (e.g., file or directory) was not found.

    Note to server developers: if a request is denied for an entire class of users,
    such as gradual feature rollout or undocumented whitelist, NOT_FOUND may be used.
    If a request is denied for some users within a class of users, such as user-based
    access control, PERMISSION_DENIED must be used.
    """

    status_code = StatusCode.NOT_FOUND
    details = "The requested entity was not found"


class PermissionDenied(GrpcException):
    """The caller does not have permission to execute the specified operation.

    PERMISSION_DENIED must not be used for rejections caused by exhausting some
    resource (use RESOURCE_EXHAUSTED instead for those errors). PERMISSION_DENIED
    must not be used if the caller can not be identified (use UNAUTHENTICATED instead
    for those errors). This error code does not imply the request is valid or the
    requested entity exists or satisfies other pre-conditions.
    """

    status_code = StatusCode.PERMISSION_DENIED
    details = "The caller does not have permission to execute the specified operation"


class ResourceExhausted(GrpcException):
    """Some resource has been exhausted.

    Perhaps a per-user quota, or perhaps the entire file system is out of space.
    """

    status_code = StatusCode.RESOURCE_EXHAUSTED
    details = "A resource has been exhausted"


class Unauthenticated(GrpcException):
    """The request does not have valid authentication credentials for the operation."""

    status_code = StatusCode.UNAUTHENTICATED
    details = (
        "The request does not have valid authentication credentials for the operation"
    )


class Unavailable(GrpcException):
    """The service is currently unavailable.

    This is most likely a transient condition, which can be corrected by retrying with
    a backoff. Note that it is not always safe to retry non-idempotent operations.
    """

    status_code = StatusCode.UNAVAILABLE
    details = "The service is currently unavailable"


class Unimplemented(GrpcException):
    """The operation is not implemented or is not supported/enabled in this service."""

    status_code = StatusCode.UNIMPLEMENTED
    details = (
        "The operation is not implemented or not supported/enabled in this service"
    )


class Unknown(GrpcException):
    """Unknown error.

    For example, this error may be returned when a Status value received from another
    address space belongs to an error space that is not known in this address space.
    Also errors raised by APIs that do not return enough error information may be
    converted to this error.
    """

    pass

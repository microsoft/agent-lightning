from ..base import OAuth2Error

__all__ = [
    "InvalidRequestUriError",
    "InvalidRequestObjectError",
    "RequestNotSupportedError",
    "RequestUriNotSupportedError",
]


class InvalidRequestUriError(OAuth2Error):
    error = "invalid_request_uri"
    description = "The request_uri in the authorization request returns an error or contains invalid data."
    status_code = 400


class InvalidRequestObjectError(OAuth2Error):
    error = "invalid_request_object"
    description = "The request parameter contains an invalid Request Object."
    status_code = 400


class RequestNotSupportedError(OAuth2Error):
    error = "request_not_supported"
    description = (
        "The authorization server does not support the use of the request parameter."
    )
    status_code = 400


class RequestUriNotSupportedError(OAuth2Error):
    error = "request_uri_not_supported"
    description = "The authorization server does not support the use of the request_uri parameter."
    status_code = 400

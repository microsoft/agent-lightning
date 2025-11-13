"""authlib.rfc6750.errors.
~~~~~~~~~~~~~~~~~~~~~~

OAuth Extensions Error Registration. When a request fails,
the resource server responds using the appropriate HTTP
status code and includes one of the following error codes
in the response.

https://tools.ietf.org/html/rfc6750#section-6.2

:copyright: (c) 2017 by Hsiaoming Yang.
"""

from ..base import OAuth2Error

__all__ = ["InvalidTokenError", "InsufficientScopeError"]


class InvalidTokenError(OAuth2Error):
    """The access token provided is expired, revoked, malformed, or
    invalid for other reasons. The resource SHOULD respond with
    the HTTP 401 (Unauthorized) status code.  The client MAY
    request a new access token and retry the protected resource
    request.

    https://tools.ietf.org/html/rfc6750#section-3.1
    """

    error = "invalid_token"
    description = (
        "The access token provided is expired, revoked, malformed, "
        "or invalid for other reasons."
    )
    status_code = 401

    def __init__(
        self,
        description=None,
        uri=None,
        status_code=None,
        state=None,
        realm=None,
        **extra_attributes,
    ):
        super().__init__(description, uri, status_code, state)
        self.realm = realm
        self.extra_attributes = extra_attributes

    def get_headers(self):
        """If the protected resource request does not include authentication
        credentials or does not contain an access token that enables access
        to the protected resource, the resource server MUST include the HTTP
        "WWW-Authenticate" response header field; it MAY include it in
        response to other conditions as well.

        https://tools.ietf.org/html/rfc6750#section-3
        """
        headers = super().get_headers()

        extras = []
        if self.realm:
            extras.append(f'realm="{self.realm}"')
        if self.extra_attributes:
            extras.extend(
                [f'{k}="{self.extra_attributes[k]}"' for k in self.extra_attributes]
            )
        extras.append(f'error="{self.error}"')
        error_description = self.get_error_description()
        extras.append(f'error_description="{error_description}"')
        headers.append(("WWW-Authenticate", "Bearer " + ", ".join(extras)))
        return headers


class InsufficientScopeError(OAuth2Error):
    """The request requires higher privileges than provided by the
    access token. The resource server SHOULD respond with the HTTP
    403 (Forbidden) status code and MAY include the "scope"
    attribute with the scope necessary to access the protected
    resource.

    https://tools.ietf.org/html/rfc6750#section-3.1
    """

    error = "insufficient_scope"
    description = (
        "The request requires higher privileges than provided by the access token."
    )
    status_code = 403

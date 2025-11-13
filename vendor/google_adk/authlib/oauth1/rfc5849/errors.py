"""authlib.oauth1.rfc5849.errors.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RFC5849 has no definition on errors. This module is designed by
Authlib based on OAuth 1.0a `Section 10`_ with some changes.

.. _`Section 10`: https://oauth.net/core/1.0a/#rfc.section.10
"""

from authlib.common.errors import AuthlibHTTPError
from authlib.common.security import is_secure_transport


class OAuth1Error(AuthlibHTTPError):
    def __init__(self, description=None, uri=None, status_code=None):
        super().__init__(None, description, uri, status_code)

    def get_headers(self):
        """Get a list of headers."""
        return [
            ("Content-Type", "application/x-www-form-urlencoded"),
            ("Cache-Control", "no-store"),
            ("Pragma", "no-cache"),
        ]


class InsecureTransportError(OAuth1Error):
    error = "insecure_transport"
    description = "OAuth 2 MUST utilize https."

    @classmethod
    def check(cls, uri):
        if not is_secure_transport(uri):
            raise cls()


class InvalidRequestError(OAuth1Error):
    error = "invalid_request"


class UnsupportedParameterError(OAuth1Error):
    error = "unsupported_parameter"


class UnsupportedSignatureMethodError(OAuth1Error):
    error = "unsupported_signature_method"


class MissingRequiredParameterError(OAuth1Error):
    error = "missing_required_parameter"

    def __init__(self, key):
        description = f'missing "{key}" in parameters'
        super().__init__(description=description)


class DuplicatedOAuthProtocolParameterError(OAuth1Error):
    error = "duplicated_oauth_protocol_parameter"


class InvalidClientError(OAuth1Error):
    error = "invalid_client"
    status_code = 401


class InvalidTokenError(OAuth1Error):
    error = "invalid_token"
    description = 'Invalid or expired "oauth_token" in parameters'
    status_code = 401


class InvalidSignatureError(OAuth1Error):
    error = "invalid_signature"
    status_code = 401


class InvalidNonceError(OAuth1Error):
    error = "invalid_nonce"
    status_code = 401


class AccessDeniedError(OAuth1Error):
    error = "access_denied"
    description = "The resource owner or authorization server denied the request"


class MethodNotAllowedError(OAuth1Error):
    error = "method_not_allowed"
    status_code = 405

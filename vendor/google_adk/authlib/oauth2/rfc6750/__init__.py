"""authlib.oauth2.rfc6750.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
The OAuth 2.0 Authorization Framework: Bearer Token Usage.

https://tools.ietf.org/html/rfc6750
"""

from .errors import InsufficientScopeError
from .errors import InvalidTokenError
from .parameters import add_bearer_token
from .token import BearerTokenGenerator
from .validator import BearerTokenValidator

# TODO: add deprecation
BearerToken = BearerTokenGenerator


__all__ = [
    "InvalidTokenError",
    "InsufficientScopeError",
    "add_bearer_token",
    "BearerToken",
    "BearerTokenGenerator",
    "BearerTokenValidator",
]

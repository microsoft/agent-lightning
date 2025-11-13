"""authlib.oauth2.rfc7662.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
OAuth 2.0 Token Introspection.

https://tools.ietf.org/html/rfc7662
"""

from .introspection import IntrospectionEndpoint
from .models import IntrospectionToken
from .token_validator import IntrospectTokenValidator

__all__ = ["IntrospectionEndpoint", "IntrospectionToken", "IntrospectTokenValidator"]

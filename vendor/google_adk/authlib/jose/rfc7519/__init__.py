"""authlib.jose.rfc7519.
~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
JSON Web Token (JWT).

https://tools.ietf.org/html/rfc7519
"""

from .claims import BaseClaims
from .claims import JWTClaims
from .jwt import JsonWebToken

__all__ = ["JsonWebToken", "BaseClaims", "JWTClaims"]

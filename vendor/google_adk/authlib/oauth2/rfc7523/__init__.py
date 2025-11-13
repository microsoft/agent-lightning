"""authlib.oauth2.rfc7523.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
JSON Web Token (JWT) Profile for OAuth 2.0 Client
Authentication and Authorization Grants.

https://tools.ietf.org/html/rfc7523
"""

from .assertion import client_secret_jwt_sign
from .assertion import private_key_jwt_sign
from .auth import ClientSecretJWT
from .auth import PrivateKeyJWT
from .client import JWTBearerClientAssertion
from .jwt_bearer import JWTBearerGrant
from .token import JWTBearerTokenGenerator
from .validator import JWTBearerToken
from .validator import JWTBearerTokenValidator

__all__ = [
    "JWTBearerGrant",
    "JWTBearerClientAssertion",
    "client_secret_jwt_sign",
    "private_key_jwt_sign",
    "ClientSecretJWT",
    "PrivateKeyJWT",
    "JWTBearerToken",
    "JWTBearerTokenGenerator",
    "JWTBearerTokenValidator",
]

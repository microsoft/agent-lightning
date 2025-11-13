"""authlib.oauth2.rfc6749.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
The OAuth 2.0 Authorization Framework.

https://tools.ietf.org/html/rfc6749
"""

from .authenticate_client import ClientAuthentication
from .authorization_server import AuthorizationServer
from .errors import AccessDeniedError
from .errors import InsecureTransportError
from .errors import InvalidClientError
from .errors import InvalidGrantError
from .errors import InvalidRequestError
from .errors import InvalidScopeError
from .errors import MismatchingStateException
from .errors import MissingAuthorizationError
from .errors import MissingCodeException  # exceptions for clients
from .errors import MissingTokenException
from .errors import MissingTokenTypeException
from .errors import OAuth2Error
from .errors import UnauthorizedClientError
from .errors import UnsupportedGrantTypeError
from .errors import UnsupportedResponseTypeError
from .errors import UnsupportedTokenTypeError
from .grants import AuthorizationCodeGrant
from .grants import AuthorizationEndpointMixin
from .grants import BaseGrant
from .grants import ClientCredentialsGrant
from .grants import ImplicitGrant
from .grants import RefreshTokenGrant
from .grants import ResourceOwnerPasswordCredentialsGrant
from .grants import TokenEndpointMixin
from .models import AuthorizationCodeMixin
from .models import ClientMixin
from .models import TokenMixin
from .requests import JsonPayload
from .requests import JsonRequest
from .requests import OAuth2Payload
from .requests import OAuth2Request
from .resource_protector import ResourceProtector
from .resource_protector import TokenValidator
from .token_endpoint import TokenEndpoint
from .util import list_to_scope
from .util import scope_to_list
from .wrappers import OAuth2Token

__all__ = [
    "OAuth2Payload",
    "OAuth2Token",
    "OAuth2Request",
    "JsonPayload",
    "JsonRequest",
    "OAuth2Error",
    "AccessDeniedError",
    "MissingAuthorizationError",
    "InvalidGrantError",
    "InvalidClientError",
    "InvalidRequestError",
    "InvalidScopeError",
    "InsecureTransportError",
    "UnauthorizedClientError",
    "UnsupportedResponseTypeError",
    "UnsupportedGrantTypeError",
    "UnsupportedTokenTypeError",
    "MissingCodeException",
    "MissingTokenException",
    "MissingTokenTypeException",
    "MismatchingStateException",
    "ClientMixin",
    "AuthorizationCodeMixin",
    "TokenMixin",
    "ClientAuthentication",
    "AuthorizationServer",
    "ResourceProtector",
    "TokenValidator",
    "TokenEndpoint",
    "BaseGrant",
    "AuthorizationEndpointMixin",
    "TokenEndpointMixin",
    "AuthorizationCodeGrant",
    "ImplicitGrant",
    "ResourceOwnerPasswordCredentialsGrant",
    "ClientCredentialsGrant",
    "RefreshTokenGrant",
    "scope_to_list",
    "list_to_scope",
]

"""
authlib.oauth2.rfc6749.grants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for `Section 4`_ of "Obtaining Authorization".

To request an access token, the client obtains authorization from the
resource owner. The authorization is expressed in the form of an
authorization grant, which the client uses to request the access
token. OAuth defines four grant types:

1. authorization code
2. implicit
3. resource owner password credentials
4. client credentials.

It also provides an extension mechanism for defining additional grant
types. Authlib defines refresh_token as a grant type too.

.. _`Section 4`: https://tools.ietf.org/html/rfc6749#section-4
"""

from .authorization_code import AuthorizationCodeGrant
from .base import AuthorizationEndpointMixin
from .base import BaseGrant
from .base import TokenEndpointMixin
from .client_credentials import ClientCredentialsGrant
from .implicit import ImplicitGrant
from .refresh_token import RefreshTokenGrant
from .resource_owner_password_credentials import ResourceOwnerPasswordCredentialsGrant

__all__ = [
    "BaseGrant",
    "AuthorizationEndpointMixin",
    "TokenEndpointMixin",
    "AuthorizationCodeGrant",
    "ImplicitGrant",
    "ResourceOwnerPasswordCredentialsGrant",
    "ClientCredentialsGrant",
    "RefreshTokenGrant",
]

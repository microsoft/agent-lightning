"""authlib.oauth2.rfc7009.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
OAuth 2.0 Token Revocation.

https://tools.ietf.org/html/rfc7009
"""

from .parameters import prepare_revoke_token_request
from .revocation import RevocationEndpoint

__all__ = ["prepare_revoke_token_request", "RevocationEndpoint"]

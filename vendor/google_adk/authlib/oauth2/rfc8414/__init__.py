"""authlib.oauth2.rfc8414.
~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
OAuth 2.0 Authorization Server Metadata.

https://tools.ietf.org/html/rfc8414
"""

from .models import AuthorizationServerMetadata
from .well_known import get_well_known_url

__all__ = ["AuthorizationServerMetadata", "get_well_known_url"]

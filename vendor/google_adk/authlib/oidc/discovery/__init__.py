"""authlib.oidc.discover.
~~~~~~~~~~~~~~~~~~~~~

OpenID Connect Discovery 1.0 Implementation.

https://openid.net/specs/openid-connect-discovery-1_0.html
"""

from .models import OpenIDProviderMetadata
from .well_known import get_well_known_url

__all__ = ["OpenIDProviderMetadata", "get_well_known_url"]

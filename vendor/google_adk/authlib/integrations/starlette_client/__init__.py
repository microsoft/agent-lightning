from ..base_client import BaseOAuth
from ..base_client import OAuthError
from .apps import StarletteOAuth1App
from .apps import StarletteOAuth2App
from .integration import StarletteIntegration


class OAuth(BaseOAuth):
    oauth1_client_cls = StarletteOAuth1App
    oauth2_client_cls = StarletteOAuth2App
    framework_integration_cls = StarletteIntegration

    def __init__(self, config=None, cache=None, fetch_token=None, update_token=None):
        super().__init__(
            cache=cache, fetch_token=fetch_token, update_token=update_token
        )
        self.config = config


__all__ = [
    "OAuth",
    "OAuthError",
    "StarletteIntegration",
    "StarletteOAuth1App",
    "StarletteOAuth2App",
]

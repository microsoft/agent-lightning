from ..base_client import BaseOAuth
from ..base_client import OAuthError
from .apps import DjangoOAuth1App
from .apps import DjangoOAuth2App
from .integration import DjangoIntegration
from .integration import token_update


class OAuth(BaseOAuth):
    oauth1_client_cls = DjangoOAuth1App
    oauth2_client_cls = DjangoOAuth2App
    framework_integration_cls = DjangoIntegration


__all__ = [
    "OAuth",
    "DjangoOAuth1App",
    "DjangoOAuth2App",
    "DjangoIntegration",
    "token_update",
    "OAuthError",
]

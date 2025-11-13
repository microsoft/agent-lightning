from django.conf import settings
from django.dispatch import Signal

from ..base_client import FrameworkIntegration

token_update = Signal()


class DjangoIntegration(FrameworkIntegration):
    def update_token(self, token, refresh_token=None, access_token=None):
        token_update.send(
            sender=self.__class__,
            name=self.name,
            token=token,
            refresh_token=refresh_token,
            access_token=access_token,
        )

    @staticmethod
    def load_config(oauth, name, params):
        config = getattr(settings, "AUTHLIB_OAUTH_CLIENTS", None)
        if config:
            return config.get(name)

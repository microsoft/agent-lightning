from werkzeug.local import LocalProxy

from ..base_client import BaseOAuth
from ..base_client import OAuthError
from .apps import FlaskOAuth1App
from .apps import FlaskOAuth2App
from .integration import FlaskIntegration
from .integration import token_update


class OAuth(BaseOAuth):
    oauth1_client_cls = FlaskOAuth1App
    oauth2_client_cls = FlaskOAuth2App
    framework_integration_cls = FlaskIntegration

    def __init__(self, app=None, cache=None, fetch_token=None, update_token=None):
        super().__init__(
            cache=cache, fetch_token=fetch_token, update_token=update_token
        )
        self.app = app
        if app:
            self.init_app(app)

    def init_app(self, app, cache=None, fetch_token=None, update_token=None):
        """Initialize lazy for Flask app. This is usually used for Flask application
        factory pattern.
        """
        self.app = app
        if cache is not None:
            self.cache = cache

        if fetch_token:
            self.fetch_token = fetch_token
        if update_token:
            self.update_token = update_token

        app.extensions = getattr(app, "extensions", {})
        app.extensions["authlib.integrations.flask_client"] = self

    def create_client(self, name):
        if not self.app:
            raise RuntimeError("OAuth is not init with Flask app.")
        return super().create_client(name)

    def register(self, name, overwrite=False, **kwargs):
        self._registry[name] = (overwrite, kwargs)
        if self.app:
            return self.create_client(name)
        return LocalProxy(lambda: self.create_client(name))


__all__ = [
    "OAuth",
    "FlaskIntegration",
    "FlaskOAuth1App",
    "FlaskOAuth2App",
    "token_update",
    "OAuthError",
]

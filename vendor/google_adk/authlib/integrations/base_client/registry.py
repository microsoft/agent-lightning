import functools

from .framework_integration import FrameworkIntegration

__all__ = ["BaseOAuth"]


OAUTH_CLIENT_PARAMS = (
    "client_id",
    "client_secret",
    "request_token_url",
    "request_token_params",
    "access_token_url",
    "access_token_params",
    "refresh_token_url",
    "refresh_token_params",
    "authorize_url",
    "authorize_params",
    "api_base_url",
    "client_kwargs",
    "server_metadata_url",
)


class BaseOAuth:
    """Registry for oauth clients.

    Create an instance for registry::

        oauth = OAuth()
    """

    oauth1_client_cls = None
    oauth2_client_cls = None
    framework_integration_cls = FrameworkIntegration

    def __init__(self, cache=None, fetch_token=None, update_token=None):
        self._registry = {}
        self._clients = {}
        self.cache = cache
        self.fetch_token = fetch_token
        self.update_token = update_token

    def create_client(self, name):
        """Create or get the given named OAuth client. For instance, the
        OAuth registry has ``.register`` a twitter client, developers may
        access the client with::

            client = oauth.create_client("twitter")

        :param: name: Name of the remote application
        :return: OAuth remote app
        """
        if name in self._clients:
            return self._clients[name]

        if name not in self._registry:
            return None

        overwrite, config = self._registry[name]
        client_cls = config.pop("client_cls", None)

        if client_cls and client_cls.OAUTH_APP_CONFIG:
            kwargs = client_cls.OAUTH_APP_CONFIG
            kwargs.update(config)
        else:
            kwargs = config

        kwargs = self.generate_client_kwargs(name, overwrite, **kwargs)
        framework = self.framework_integration_cls(name, self.cache)
        if client_cls:
            client = client_cls(framework, name, **kwargs)
        elif kwargs.get("request_token_url"):
            client = self.oauth1_client_cls(framework, name, **kwargs)
        else:
            client = self.oauth2_client_cls(framework, name, **kwargs)

        self._clients[name] = client
        return client

    def register(self, name, overwrite=False, **kwargs):
        """Registers a new remote application.

        :param name: Name of the remote application.
        :param overwrite: Overwrite existing config with framework settings.
        :param kwargs: Parameters for :class:`RemoteApp`.

        Find parameters for the given remote app class. When a remote app is
        registered, it can be accessed with *named* attribute::

            oauth.register('twitter', client_id='', ...)
            oauth.twitter.get('timeline')
        """
        self._registry[name] = (overwrite, kwargs)
        return self.create_client(name)

    def generate_client_kwargs(self, name, overwrite, **kwargs):
        fetch_token = kwargs.pop("fetch_token", None)
        update_token = kwargs.pop("update_token", None)

        config = self.load_config(name, OAUTH_CLIENT_PARAMS)
        if config:
            kwargs = _config_client(config, kwargs, overwrite)

        if not fetch_token and self.fetch_token:
            fetch_token = functools.partial(self.fetch_token, name)

        kwargs["fetch_token"] = fetch_token

        if not kwargs.get("request_token_url"):
            if not update_token and self.update_token:
                update_token = functools.partial(self.update_token, name)

            kwargs["update_token"] = update_token
        return kwargs

    def load_config(self, name, params):
        return self.framework_integration_cls.load_config(self, name, params)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as exc:
            if key in self._registry:
                return self.create_client(key)
            raise AttributeError(f"No such client: {key}") from exc


def _config_client(config, kwargs, overwrite):
    for k in OAUTH_CLIENT_PARAMS:
        v = config.get(k, None)
        if k not in kwargs:
            kwargs[k] = v
        elif overwrite and v:
            if isinstance(kwargs[k], dict):
                kwargs[k].update(v)
            else:
                kwargs[k] = v
    return kwargs

import logging
import time

from authlib.common.security import generate_token
from authlib.common.urls import urlparse
from authlib.consts import default_user_agent

from .errors import MismatchingStateError
from .errors import MissingRequestTokenError
from .errors import MissingTokenError

log = logging.getLogger(__name__)


class BaseApp:
    client_cls = None
    OAUTH_APP_CONFIG = None

    def request(self, method, url, token=None, **kwargs):
        raise NotImplementedError()

    def get(self, url, **kwargs):
        """Invoke GET http request.

        If ``api_base_url`` configured, shortcut is available::

            client.get("users/lepture")
        """
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        """Invoke POST http request.

        If ``api_base_url`` configured, shortcut is available::

            client.post("timeline", json={"text": "Hi"})
        """
        return self.request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        """Invoke PATCH http request.

        If ``api_base_url`` configured, shortcut is available::

            client.patch("profile", json={"name": "Hsiaoming Yang"})
        """
        return self.request("PATCH", url, **kwargs)

    def put(self, url, **kwargs):
        """Invoke PUT http request.

        If ``api_base_url`` configured, shortcut is available::

            client.put("profile", json={"name": "Hsiaoming Yang"})
        """
        return self.request("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        """Invoke DELETE http request.

        If ``api_base_url`` configured, shortcut is available::

            client.delete("posts/123")
        """
        return self.request("DELETE", url, **kwargs)


class _RequestMixin:
    def _get_requested_token(self, request):
        if self._fetch_token and request:
            return self._fetch_token(request)

    def _send_token_request(self, session, method, url, token, kwargs):
        request = kwargs.pop("request", None)
        withhold_token = kwargs.get("withhold_token")
        if self.api_base_url and not url.startswith(("https://", "http://")):
            url = urlparse.urljoin(self.api_base_url, url)

        if withhold_token:
            return session.request(method, url, **kwargs)

        if token is None:
            token = self._get_requested_token(request)

        if token is None:
            raise MissingTokenError()

        session.token = token
        return session.request(method, url, **kwargs)


class OAuth1Base:
    client_cls = None

    def __init__(
        self,
        framework,
        name=None,
        fetch_token=None,
        client_id=None,
        client_secret=None,
        request_token_url=None,
        request_token_params=None,
        access_token_url=None,
        access_token_params=None,
        authorize_url=None,
        authorize_params=None,
        api_base_url=None,
        client_kwargs=None,
        user_agent=None,
        **kwargs,
    ):
        self.framework = framework
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.request_token_url = request_token_url
        self.request_token_params = request_token_params
        self.access_token_url = access_token_url
        self.access_token_params = access_token_params
        self.authorize_url = authorize_url
        self.authorize_params = authorize_params
        self.api_base_url = api_base_url
        self.client_kwargs = client_kwargs or {}

        self._fetch_token = fetch_token
        self._user_agent = user_agent or default_user_agent
        self._kwargs = kwargs

    def _get_oauth_client(self):
        session = self.client_cls(
            self.client_id, self.client_secret, **self.client_kwargs
        )
        session.headers["User-Agent"] = self._user_agent
        return session


class OAuth1Mixin(_RequestMixin, OAuth1Base):
    def request(self, method, url, token=None, **kwargs):
        with self._get_oauth_client() as session:
            return self._send_token_request(session, method, url, token, kwargs)

    def create_authorization_url(self, redirect_uri=None, **kwargs):
        """Generate the authorization url and state for HTTP redirect.

        :param redirect_uri: Callback or redirect URI for authorization.
        :param kwargs: Extra parameters to include.
        :return: dict
        """
        if not self.authorize_url:
            raise RuntimeError('Missing "authorize_url" value')

        if self.authorize_params:
            kwargs.update(self.authorize_params)

        with self._get_oauth_client() as client:
            client.redirect_uri = redirect_uri
            params = self.request_token_params or {}
            request_token = client.fetch_request_token(self.request_token_url, **params)
            log.debug(f"Fetch request token: {request_token!r}")
            url = client.create_authorization_url(self.authorize_url, **kwargs)
            state = request_token["oauth_token"]
        return {"url": url, "request_token": request_token, "state": state}

    def fetch_access_token(self, request_token=None, **kwargs):
        """Fetch access token in one step.

        :param request_token: A previous request token for OAuth 1.
        :param kwargs: Extra parameters to fetch access token.
        :return: A token dict.
        """
        with self._get_oauth_client() as client:
            if request_token is None:
                raise MissingRequestTokenError()
            # merge request token with verifier
            token = {}
            token.update(request_token)
            token.update(kwargs)
            client.token = token
            params = self.access_token_params or {}
            token = client.fetch_access_token(self.access_token_url, **params)
        return token


class OAuth2Base:
    client_cls = None

    def __init__(
        self,
        framework,
        name=None,
        fetch_token=None,
        update_token=None,
        client_id=None,
        client_secret=None,
        access_token_url=None,
        access_token_params=None,
        authorize_url=None,
        authorize_params=None,
        api_base_url=None,
        client_kwargs=None,
        server_metadata_url=None,
        compliance_fix=None,
        client_auth_methods=None,
        user_agent=None,
        **kwargs,
    ):
        self.framework = framework
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token_url = access_token_url
        self.access_token_params = access_token_params
        self.authorize_url = authorize_url
        self.authorize_params = authorize_params
        self.api_base_url = api_base_url
        self.client_kwargs = client_kwargs or {}

        self.compliance_fix = compliance_fix
        self.client_auth_methods = client_auth_methods
        self._fetch_token = fetch_token
        self._update_token = update_token
        self._user_agent = user_agent or default_user_agent

        self._server_metadata_url = server_metadata_url
        self.server_metadata = kwargs

    def _on_update_token(self, token, refresh_token=None, access_token=None):
        raise NotImplementedError()

    def _get_oauth_client(self, **metadata):
        client_kwargs = {}
        client_kwargs.update(self.client_kwargs)
        client_kwargs.update(metadata)

        if self.authorize_url:
            client_kwargs["authorization_endpoint"] = self.authorize_url
        if self.access_token_url:
            client_kwargs["token_endpoint"] = self.access_token_url

        session = self.client_cls(
            client_id=self.client_id,
            client_secret=self.client_secret,
            update_token=self._on_update_token,
            **client_kwargs,
        )
        if self.client_auth_methods:
            for f in self.client_auth_methods:
                session.register_client_auth_method(f)

        if self.compliance_fix:
            self.compliance_fix(session)

        session.headers["User-Agent"] = self._user_agent
        return session

    @staticmethod
    def _format_state_params(state_data, params):
        if state_data is None:
            raise MismatchingStateError()

        code_verifier = state_data.get("code_verifier")
        if code_verifier:
            params["code_verifier"] = code_verifier

        redirect_uri = state_data.get("redirect_uri")
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        return params

    @staticmethod
    def _create_oauth2_authorization_url(client, authorization_endpoint, **kwargs):
        rv = {}
        if client.code_challenge_method:
            code_verifier = kwargs.get("code_verifier")
            if not code_verifier:
                code_verifier = generate_token(48)
                kwargs["code_verifier"] = code_verifier
            rv["code_verifier"] = code_verifier
            log.debug(f"Using code_verifier: {code_verifier!r}")

        scope = kwargs.get("scope", client.scope)
        scope = (
            (scope if isinstance(scope, (list, tuple)) else scope.split())
            if scope
            else None
        )
        if scope and "openid" in scope:
            # this is an OpenID Connect service
            nonce = kwargs.get("nonce")
            if not nonce:
                nonce = generate_token(20)
                kwargs["nonce"] = nonce
            rv["nonce"] = nonce

        url, state = client.create_authorization_url(authorization_endpoint, **kwargs)
        rv["url"] = url
        rv["state"] = state
        return rv


class OAuth2Mixin(_RequestMixin, OAuth2Base):
    def _on_update_token(self, token, refresh_token=None, access_token=None):
        if callable(self._update_token):
            self._update_token(
                token,
                refresh_token=refresh_token,
                access_token=access_token,
            )
        self.framework.update_token(
            token,
            refresh_token=refresh_token,
            access_token=access_token,
        )

    def request(self, method, url, token=None, **kwargs):
        metadata = self.load_server_metadata()
        with self._get_oauth_client(**metadata) as session:
            return self._send_token_request(session, method, url, token, kwargs)

    def load_server_metadata(self):
        if self._server_metadata_url and "_loaded_at" not in self.server_metadata:
            with self.client_cls(**self.client_kwargs) as session:
                resp = session.request(
                    "GET", self._server_metadata_url, withhold_token=True
                )
                resp.raise_for_status()
                metadata = resp.json()

            metadata["_loaded_at"] = time.time()
            self.server_metadata.update(metadata)
        return self.server_metadata

    def create_authorization_url(self, redirect_uri=None, **kwargs):
        """Generate the authorization url and state for HTTP redirect.

        :param redirect_uri: Callback or redirect URI for authorization.
        :param kwargs: Extra parameters to include.
        :return: dict
        """
        metadata = self.load_server_metadata()
        authorization_endpoint = self.authorize_url or metadata.get(
            "authorization_endpoint"
        )

        if not authorization_endpoint:
            raise RuntimeError('Missing "authorize_url" value')

        if self.authorize_params:
            kwargs.update(self.authorize_params)

        with self._get_oauth_client(**metadata) as client:
            if redirect_uri is not None:
                client.redirect_uri = redirect_uri
            return self._create_oauth2_authorization_url(
                client, authorization_endpoint, **kwargs
            )

    def fetch_access_token(self, redirect_uri=None, **kwargs):
        """Fetch access token in the final step.

        :param redirect_uri: Callback or Redirect URI that is used in
                             previous :meth:`authorize_redirect`.
        :param kwargs: Extra parameters to fetch access token.
        :return: A token dict.
        """
        metadata = self.load_server_metadata()
        token_endpoint = self.access_token_url or metadata.get("token_endpoint")
        with self._get_oauth_client(**metadata) as client:
            if redirect_uri is not None:
                client.redirect_uri = redirect_uri
            params = {}
            if self.access_token_params:
                params.update(self.access_token_params)
            params.update(kwargs)
            token = client.fetch_token(token_endpoint, **params)
            return token

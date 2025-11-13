from authlib.common.security import generate_token
from authlib.common.urls import url_decode

from .auth import ClientAuth
from .auth import TokenAuth
from .base import OAuth2Error
from .rfc6749.parameters import parse_authorization_code_response
from .rfc6749.parameters import parse_implicit_response
from .rfc6749.parameters import prepare_grant_uri
from .rfc6749.parameters import prepare_token_request
from .rfc7009 import prepare_revoke_token_request
from .rfc7636 import create_s256_code_challenge

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
}


class OAuth2Client:
    """Construct a new OAuth 2 protocol client.

    :param session: Requests session object to communicate with
                    authorization server.
    :param client_id: Client ID, which you get from client registration.
    :param client_secret: Client Secret, which you get from registration.
    :param token_endpoint_auth_method: client authentication method for
        token endpoint.
    :param revocation_endpoint_auth_method: client authentication method for
        revocation endpoint.
    :param scope: Scope that you needed to access user resources.
    :param state: Shared secret to prevent CSRF attack.
    :param redirect_uri: Redirect URI you registered as callback.
    :param code_challenge_method: PKCE method name, only S256 is supported.
    :param token: A dict of token attributes such as ``access_token``,
        ``token_type`` and ``expires_at``.
    :param token_placement: The place to put token in HTTP request. Available
        values: "header", "body", "uri".
    :param update_token: A function for you to update token. It accept a
        :class:`OAuth2Token` as parameter.
    :param leeway: Time window in seconds before the actual expiration of the
        authentication token, that the token is considered expired and will
        be refreshed.
    """

    client_auth_class = ClientAuth
    token_auth_class = TokenAuth
    oauth_error_class = OAuth2Error

    EXTRA_AUTHORIZE_PARAMS = ("response_mode", "nonce", "prompt", "login_hint")
    SESSION_REQUEST_PARAMS = []

    def __init__(
        self,
        session,
        client_id=None,
        client_secret=None,
        token_endpoint_auth_method=None,
        revocation_endpoint_auth_method=None,
        scope=None,
        state=None,
        redirect_uri=None,
        code_challenge_method=None,
        token=None,
        token_placement="header",
        update_token=None,
        leeway=60,
        **metadata,
    ):
        self.session = session
        self.client_id = client_id
        self.client_secret = client_secret
        self.state = state

        if token_endpoint_auth_method is None:
            if client_secret:
                token_endpoint_auth_method = "client_secret_basic"
            else:
                token_endpoint_auth_method = "none"

        self.token_endpoint_auth_method = token_endpoint_auth_method

        if revocation_endpoint_auth_method is None:
            if client_secret:
                revocation_endpoint_auth_method = "client_secret_basic"
            else:
                revocation_endpoint_auth_method = "none"

        self.revocation_endpoint_auth_method = revocation_endpoint_auth_method

        self.scope = scope
        self.redirect_uri = redirect_uri
        self.code_challenge_method = code_challenge_method

        self.token_auth = self.token_auth_class(token, token_placement, self)
        self.update_token = update_token

        token_updater = metadata.pop("token_updater", None)
        if token_updater:
            raise ValueError(
                "update token has been redesigned, checkout the documentation"
            )

        self.metadata = metadata

        self.compliance_hook = {
            "access_token_response": set(),
            "refresh_token_request": set(),
            "refresh_token_response": set(),
            "revoke_token_request": set(),
            "introspect_token_request": set(),
        }
        self._auth_methods = {}

        self.leeway = leeway

    def register_client_auth_method(self, auth):
        """Extend client authenticate for token endpoint.

        :param auth: an instance to sign the request
        """
        if isinstance(auth, tuple):
            self._auth_methods[auth[0]] = auth[1]
        else:
            self._auth_methods[auth.name] = auth

    def client_auth(self, auth_method):
        if isinstance(auth_method, str) and auth_method in self._auth_methods:
            auth_method = self._auth_methods[auth_method]
        return self.client_auth_class(
            client_id=self.client_id,
            client_secret=self.client_secret,
            auth_method=auth_method,
        )

    @property
    def token(self):
        return self.token_auth.token

    @token.setter
    def token(self, token):
        self.token_auth.set_token(token)

    def create_authorization_url(self, url, state=None, code_verifier=None, **kwargs):
        """Generate an authorization URL and state.

        :param url: Authorization endpoint url, must be HTTPS.
        :param state: An optional state string for CSRF protection. If not
                      given it will be generated for you.
        :param code_verifier: An optional code_verifier for code challenge.
        :param kwargs: Extra parameters to include.
        :return: authorization_url, state
        """
        if state is None:
            state = generate_token()

        response_type = self.metadata.get("response_type", "code")
        response_type = kwargs.pop("response_type", response_type)
        if "redirect_uri" not in kwargs:
            kwargs["redirect_uri"] = self.redirect_uri
        if "scope" not in kwargs:
            kwargs["scope"] = self.scope

        if (
            code_verifier
            and response_type == "code"
            and self.code_challenge_method == "S256"
        ):
            kwargs["code_challenge"] = create_s256_code_challenge(code_verifier)
            kwargs["code_challenge_method"] = self.code_challenge_method

        for k in self.EXTRA_AUTHORIZE_PARAMS:
            if k not in kwargs and k in self.metadata:
                kwargs[k] = self.metadata[k]

        uri = prepare_grant_uri(
            url,
            client_id=self.client_id,
            response_type=response_type,
            state=state,
            **kwargs,
        )
        return uri, state

    def fetch_token(
        self,
        url=None,
        body="",
        method="POST",
        headers=None,
        auth=None,
        grant_type=None,
        state=None,
        **kwargs,
    ):
        """Generic method for fetching an access token from the token endpoint.

        :param url: Access Token endpoint URL, if not configured,
                    ``authorization_response`` is used to extract token from
                    its fragment (implicit way).
        :param body: Optional application/x-www-form-urlencoded body to add the
                     include in the token request. Prefer kwargs over body.
        :param method: The HTTP method used to make the request. Defaults
                       to POST, but may also be GET. Other methods should
                       be added as needed.
        :param headers: Dict to default request headers with.
        :param auth: An auth tuple or method as accepted by requests.
        :param grant_type: Use specified grant_type to fetch token.
        :param state: Optional "state" value to fetch token.
        :return: A :class:`OAuth2Token` object (a dict too).
        """
        state = state or self.state
        # implicit  grant_type
        authorization_response = kwargs.pop("authorization_response", None)
        if authorization_response and "#" in authorization_response:
            return self.token_from_fragment(authorization_response, state)

        session_kwargs = self._extract_session_request_params(kwargs)

        if authorization_response and "code=" in authorization_response:
            grant_type = "authorization_code"
            params = parse_authorization_code_response(
                authorization_response,
                state=state,
            )
            kwargs["code"] = params["code"]

        if grant_type is None:
            grant_type = self.metadata.get("grant_type")

        if grant_type is None:
            grant_type = _guess_grant_type(kwargs)
            self.metadata["grant_type"] = grant_type

        body = self._prepare_token_endpoint_body(body, grant_type, **kwargs)

        if auth is None:
            auth = self.client_auth(self.token_endpoint_auth_method)

        if headers is None:
            headers = DEFAULT_HEADERS

        if url is None:
            url = self.metadata.get("token_endpoint")

        return self._fetch_token(
            url, body=body, auth=auth, method=method, headers=headers, **session_kwargs
        )

    def token_from_fragment(self, authorization_response, state=None):
        token = parse_implicit_response(authorization_response, state)
        if "error" in token:
            raise self.oauth_error_class(
                error=token["error"], description=token.get("error_description")
            )
        self.token = token
        return token

    def refresh_token(
        self, url=None, refresh_token=None, body="", auth=None, headers=None, **kwargs
    ):
        """Fetch a new access token using a refresh token.

        :param url: Refresh Token endpoint, must be HTTPS.
        :param refresh_token: The refresh_token to use.
        :param body: Optional application/x-www-form-urlencoded body to add the
                     include in the token request. Prefer kwargs over body.
        :param auth: An auth tuple or method as accepted by requests.
        :param headers: Dict to default request headers with.
        :return: A :class:`OAuth2Token` object (a dict too).
        """
        session_kwargs = self._extract_session_request_params(kwargs)
        refresh_token = refresh_token or self.token.get("refresh_token")
        if "scope" not in kwargs and self.scope:
            kwargs["scope"] = self.scope
        body = prepare_token_request(
            "refresh_token", body, refresh_token=refresh_token, **kwargs
        )

        if headers is None:
            headers = DEFAULT_HEADERS.copy()

        if url is None:
            url = self.metadata.get("token_endpoint")

        for hook in self.compliance_hook["refresh_token_request"]:
            url, headers, body = hook(url, headers, body)

        if auth is None:
            auth = self.client_auth(self.token_endpoint_auth_method)

        return self._refresh_token(
            url,
            refresh_token=refresh_token,
            body=body,
            headers=headers,
            auth=auth,
            **session_kwargs,
        )

    def ensure_active_token(self, token=None):
        if token is None:
            token = self.token
        if not token.is_expired(leeway=self.leeway):
            return True
        refresh_token = token.get("refresh_token")
        url = self.metadata.get("token_endpoint")
        if refresh_token and url:
            self.refresh_token(url, refresh_token=refresh_token)
            return True
        elif self.metadata.get("grant_type") == "client_credentials":
            access_token = token["access_token"]
            new_token = self.fetch_token(url, grant_type="client_credentials")
            if self.update_token:
                self.update_token(new_token, access_token=access_token)
            return True

    def revoke_token(
        self,
        url,
        token=None,
        token_type_hint=None,
        body=None,
        auth=None,
        headers=None,
        **kwargs,
    ):
        """Revoke token method defined via `RFC7009`_.

        :param url: Revoke Token endpoint, must be HTTPS.
        :param token: The token to be revoked.
        :param token_type_hint: The type of the token that to be revoked.
                                It can be "access_token" or "refresh_token".
        :param body: Optional application/x-www-form-urlencoded body to add the
                     include in the token request. Prefer kwargs over body.
        :param auth: An auth tuple or method as accepted by requests.
        :param headers: Dict to default request headers with.
        :return: Revocation Response

        .. _`RFC7009`: https://tools.ietf.org/html/rfc7009
        """
        if auth is None:
            auth = self.client_auth(self.revocation_endpoint_auth_method)
        return self._handle_token_hint(
            "revoke_token_request",
            url,
            token=token,
            token_type_hint=token_type_hint,
            body=body,
            auth=auth,
            headers=headers,
            **kwargs,
        )

    def introspect_token(
        self,
        url,
        token=None,
        token_type_hint=None,
        body=None,
        auth=None,
        headers=None,
        **kwargs,
    ):
        """Implementation of OAuth 2.0 Token Introspection defined via `RFC7662`_.

        :param url: Introspection Endpoint, must be HTTPS.
        :param token: The token to be introspected.
        :param token_type_hint: The type of the token that to be revoked.
                                It can be "access_token" or "refresh_token".
        :param body: Optional application/x-www-form-urlencoded body to add the
                     include in the token request. Prefer kwargs over body.
        :param auth: An auth tuple or method as accepted by requests.
        :param headers: Dict to default request headers with.
        :return: Introspection Response

        .. _`RFC7662`: https://tools.ietf.org/html/rfc7662
        """
        if auth is None:
            auth = self.client_auth(self.token_endpoint_auth_method)
        return self._handle_token_hint(
            "introspect_token_request",
            url,
            token=token,
            token_type_hint=token_type_hint,
            body=body,
            auth=auth,
            headers=headers,
            **kwargs,
        )

    def register_compliance_hook(self, hook_type, hook):
        """Register a hook for request/response tweaking.

        Available hooks are:

        * access_token_response: invoked before token parsing.
        * refresh_token_request: invoked before refreshing token.
        * refresh_token_response: invoked before refresh token parsing.
        * protected_request: invoked before making a request.
        * revoke_token_request: invoked before revoking a token.
        * introspect_token_request: invoked before introspecting a token.
        """
        if hook_type == "protected_request":
            self.token_auth.hooks.add(hook)
            return

        if hook_type not in self.compliance_hook:
            raise ValueError(
                "Hook type %s is not in %s.", hook_type, self.compliance_hook
            )
        self.compliance_hook[hook_type].add(hook)

    def parse_response_token(self, resp):
        if resp.status_code >= 500:
            resp.raise_for_status()

        token = resp.json()
        if "error" in token:
            raise self.oauth_error_class(
                error=token["error"], description=token.get("error_description")
            )
        self.token = token
        return self.token

    def _fetch_token(
        self, url, body="", headers=None, auth=None, method="POST", **kwargs
    ):
        if method.upper() == "POST":
            resp = self.session.post(
                url, data=dict(url_decode(body)), headers=headers, auth=auth, **kwargs
            )
        else:
            if "?" in url:
                url = "&".join([url, body])
            else:
                url = "?".join([url, body])
            resp = self.session.request(
                method, url, headers=headers, auth=auth, **kwargs
            )

        for hook in self.compliance_hook["access_token_response"]:
            resp = hook(resp)

        return self.parse_response_token(resp)

    def _refresh_token(
        self, url, refresh_token=None, body="", headers=None, auth=None, **kwargs
    ):
        resp = self._http_post(url, body=body, auth=auth, headers=headers, **kwargs)

        for hook in self.compliance_hook["refresh_token_response"]:
            resp = hook(resp)

        token = self.parse_response_token(resp)
        if "refresh_token" not in token:
            self.token["refresh_token"] = refresh_token

        if callable(self.update_token):
            self.update_token(self.token, refresh_token=refresh_token)

        return self.token

    def _handle_token_hint(
        self,
        hook,
        url,
        token=None,
        token_type_hint=None,
        body=None,
        auth=None,
        headers=None,
        **kwargs,
    ):
        if token is None and self.token:
            token = self.token.get("refresh_token") or self.token.get("access_token")

        if body is None:
            body = ""

        body, headers = prepare_revoke_token_request(
            token, token_type_hint, body, headers
        )

        for compliance_hook in self.compliance_hook[hook]:
            url, headers, body = compliance_hook(url, headers, body)

        if auth is None:
            auth = self.client_auth(self.revocation_endpoint_auth_method)

        session_kwargs = self._extract_session_request_params(kwargs)
        return self._http_post(url, body, auth=auth, headers=headers, **session_kwargs)

    def _prepare_token_endpoint_body(self, body, grant_type, **kwargs):
        if grant_type == "authorization_code":
            if "redirect_uri" not in kwargs:
                kwargs["redirect_uri"] = self.redirect_uri
            return prepare_token_request(grant_type, body, **kwargs)

        if "scope" not in kwargs and self.scope:
            kwargs["scope"] = self.scope
        return prepare_token_request(grant_type, body, **kwargs)

    def _extract_session_request_params(self, kwargs):
        """Extract parameters for session object from the passing ``**kwargs``."""
        rv = {}
        for k in self.SESSION_REQUEST_PARAMS:
            if k in kwargs:
                rv[k] = kwargs.pop(k)
        return rv

    def _http_post(self, url, body=None, auth=None, headers=None, **kwargs):
        return self.session.post(
            url, data=dict(url_decode(body)), headers=headers, auth=auth, **kwargs
        )

    def __del__(self):
        del self.session


def _guess_grant_type(kwargs):
    if "code" in kwargs:
        grant_type = "authorization_code"
    elif "username" in kwargs and "password" in kwargs:
        grant_type = "password"
    else:
        grant_type = "client_credentials"
    return grant_type

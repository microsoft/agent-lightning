import httpx
from httpx import USE_CLIENT_DEFAULT
from httpx import Response

from authlib.oauth2.rfc7521 import AssertionClient as _AssertionClient
from authlib.oauth2.rfc7523 import JWTBearerGrant

from ..base_client import OAuthError
from .oauth2_client import OAuth2Auth
from .utils import extract_client_kwargs

__all__ = ["AsyncAssertionClient"]


class AsyncAssertionClient(_AssertionClient, httpx.AsyncClient):
    token_auth_class = OAuth2Auth
    oauth_error_class = OAuthError
    JWT_BEARER_GRANT_TYPE = JWTBearerGrant.GRANT_TYPE
    ASSERTION_METHODS = {
        JWT_BEARER_GRANT_TYPE: JWTBearerGrant.sign,
    }
    DEFAULT_GRANT_TYPE = JWT_BEARER_GRANT_TYPE

    def __init__(
        self,
        token_endpoint,
        issuer,
        subject,
        audience=None,
        grant_type=None,
        claims=None,
        token_placement="header",
        scope=None,
        **kwargs,
    ):
        client_kwargs = extract_client_kwargs(kwargs)
        httpx.AsyncClient.__init__(self, **client_kwargs)

        _AssertionClient.__init__(
            self,
            session=None,
            token_endpoint=token_endpoint,
            issuer=issuer,
            subject=subject,
            audience=audience,
            grant_type=grant_type,
            claims=claims,
            token_placement=token_placement,
            scope=scope,
            **kwargs,
        )

    async def request(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ) -> Response:
        """Send request with auto refresh token feature."""
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token or self.token.is_expired():
                await self.refresh_token()

            auth = self.token_auth
        return await super().request(method, url, auth=auth, **kwargs)

    async def _refresh_token(self, data):
        resp = await self.request(
            "POST", self.token_endpoint, data=data, withhold_token=True
        )

        return self.parse_response_token(resp)


class AssertionClient(_AssertionClient, httpx.Client):
    token_auth_class = OAuth2Auth
    oauth_error_class = OAuthError
    JWT_BEARER_GRANT_TYPE = JWTBearerGrant.GRANT_TYPE
    ASSERTION_METHODS = {
        JWT_BEARER_GRANT_TYPE: JWTBearerGrant.sign,
    }
    DEFAULT_GRANT_TYPE = JWT_BEARER_GRANT_TYPE

    def __init__(
        self,
        token_endpoint,
        issuer,
        subject,
        audience=None,
        grant_type=None,
        claims=None,
        token_placement="header",
        scope=None,
        **kwargs,
    ):
        client_kwargs = extract_client_kwargs(kwargs)
        # app keyword was dropped!
        app_value = client_kwargs.pop("app", None)
        if app_value is not None:
            client_kwargs["transport"] = httpx.WSGITransport(app=app_value)

        httpx.Client.__init__(self, **client_kwargs)

        _AssertionClient.__init__(
            self,
            session=self,
            token_endpoint=token_endpoint,
            issuer=issuer,
            subject=subject,
            audience=audience,
            grant_type=grant_type,
            claims=claims,
            token_placement=token_placement,
            scope=scope,
            **kwargs,
        )

    def request(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ):
        """Send request with auto refresh token feature."""
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token or self.token.is_expired():
                self.refresh_token()

            auth = self.token_auth
        return super().request(method, url, auth=auth, **kwargs)

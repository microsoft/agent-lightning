import typing
from contextlib import asynccontextmanager

import httpx
from anyio import Lock  # Import after httpx so import errors refer to httpx
from httpx import USE_CLIENT_DEFAULT
from httpx import Auth
from httpx import Request
from httpx import Response

from authlib.common.urls import url_decode
from authlib.oauth2.auth import ClientAuth
from authlib.oauth2.auth import TokenAuth
from authlib.oauth2.client import OAuth2Client as _OAuth2Client

from ..base_client import InvalidTokenError
from ..base_client import MissingTokenError
from ..base_client import OAuthError
from ..base_client import UnsupportedTokenTypeError
from .utils import HTTPX_CLIENT_KWARGS
from .utils import build_request

__all__ = [
    "OAuth2Auth",
    "OAuth2ClientAuth",
    "AsyncOAuth2Client",
    "OAuth2Client",
]


class OAuth2Auth(Auth, TokenAuth):
    """Sign requests for OAuth 2.0, currently only bearer token is supported."""

    requires_request_body = True

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        try:
            url, headers, body = self.prepare(
                str(request.url), request.headers, request.content
            )
            headers["Content-Length"] = str(len(body))
            yield build_request(
                url=url, headers=headers, body=body, initial_request=request
            )
        except KeyError as error:
            description = f"Unsupported token_type: {str(error)}"
            raise UnsupportedTokenTypeError(description=description) from error


class OAuth2ClientAuth(Auth, ClientAuth):
    requires_request_body = True

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        url, headers, body = self.prepare(
            request.method, str(request.url), request.headers, request.content
        )
        headers["Content-Length"] = str(len(body))
        yield build_request(
            url=url, headers=headers, body=body, initial_request=request
        )


class AsyncOAuth2Client(_OAuth2Client, httpx.AsyncClient):
    SESSION_REQUEST_PARAMS = HTTPX_CLIENT_KWARGS

    client_auth_class = OAuth2ClientAuth
    token_auth_class = OAuth2Auth
    oauth_error_class = OAuthError

    def __init__(
        self,
        client_id=None,
        client_secret=None,
        token_endpoint_auth_method=None,
        revocation_endpoint_auth_method=None,
        scope=None,
        redirect_uri=None,
        token=None,
        token_placement="header",
        update_token=None,
        leeway=60,
        **kwargs,
    ):
        # extract httpx.Client kwargs
        client_kwargs = self._extract_session_request_params(kwargs)
        httpx.AsyncClient.__init__(self, **client_kwargs)

        # We use a Lock to synchronize coroutines to prevent
        # multiple concurrent attempts to refresh the same token
        self._token_refresh_lock = Lock()

        _OAuth2Client.__init__(
            self,
            session=None,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=token_endpoint_auth_method,
            revocation_endpoint_auth_method=revocation_endpoint_auth_method,
            scope=scope,
            redirect_uri=redirect_uri,
            token=token,
            token_placement=token_placement,
            update_token=update_token,
            leeway=leeway,
            **kwargs,
        )

    async def request(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ):
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token:
                raise MissingTokenError()

            await self.ensure_active_token(self.token)

            auth = self.token_auth

        return await super().request(method, url, auth=auth, **kwargs)

    @asynccontextmanager
    async def stream(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ):
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token:
                raise MissingTokenError()

            await self.ensure_active_token(self.token)

            auth = self.token_auth

        async with super().stream(method, url, auth=auth, **kwargs) as resp:
            yield resp

    async def ensure_active_token(self, token):
        async with self._token_refresh_lock:
            if self.token.is_expired(leeway=self.leeway):
                refresh_token = token.get("refresh_token")
                url = self.metadata.get("token_endpoint")
                if refresh_token and url:
                    await self.refresh_token(url, refresh_token=refresh_token)
                elif self.metadata.get("grant_type") == "client_credentials":
                    access_token = token["access_token"]
                    new_token = await self.fetch_token(
                        url, grant_type="client_credentials"
                    )
                    if self.update_token:
                        await self.update_token(new_token, access_token=access_token)
                else:
                    raise InvalidTokenError()

    async def _fetch_token(
        self,
        url,
        body="",
        headers=None,
        auth=USE_CLIENT_DEFAULT,
        method="POST",
        **kwargs,
    ):
        if method.upper() == "POST":
            resp = await self.post(
                url, data=dict(url_decode(body)), headers=headers, auth=auth, **kwargs
            )
        else:
            if "?" in url:
                url = "&".join([url, body])
            else:
                url = "?".join([url, body])
            resp = await self.get(url, headers=headers, auth=auth, **kwargs)

        for hook in self.compliance_hook["access_token_response"]:
            resp = hook(resp)

        return self.parse_response_token(resp)

    async def _refresh_token(
        self,
        url,
        refresh_token=None,
        body="",
        headers=None,
        auth=USE_CLIENT_DEFAULT,
        **kwargs,
    ):
        resp = await self.post(
            url, data=dict(url_decode(body)), headers=headers, auth=auth, **kwargs
        )

        for hook in self.compliance_hook["refresh_token_response"]:
            resp = hook(resp)

        token = self.parse_response_token(resp)
        if "refresh_token" not in token:
            self.token["refresh_token"] = refresh_token

        if self.update_token:
            await self.update_token(self.token, refresh_token=refresh_token)

        return self.token

    def _http_post(
        self, url, body=None, auth=USE_CLIENT_DEFAULT, headers=None, **kwargs
    ):
        return self.post(
            url, data=dict(url_decode(body)), headers=headers, auth=auth, **kwargs
        )


class OAuth2Client(_OAuth2Client, httpx.Client):
    SESSION_REQUEST_PARAMS = HTTPX_CLIENT_KWARGS

    client_auth_class = OAuth2ClientAuth
    token_auth_class = OAuth2Auth
    oauth_error_class = OAuthError

    def __init__(
        self,
        client_id=None,
        client_secret=None,
        token_endpoint_auth_method=None,
        revocation_endpoint_auth_method=None,
        scope=None,
        redirect_uri=None,
        token=None,
        token_placement="header",
        update_token=None,
        **kwargs,
    ):
        # extract httpx.Client kwargs
        client_kwargs = self._extract_session_request_params(kwargs)
        # app keyword was dropped!
        app_value = client_kwargs.pop("app", None)
        if app_value is not None:
            client_kwargs["transport"] = httpx.WSGITransport(app=app_value)

        httpx.Client.__init__(self, **client_kwargs)

        _OAuth2Client.__init__(
            self,
            session=self,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=token_endpoint_auth_method,
            revocation_endpoint_auth_method=revocation_endpoint_auth_method,
            scope=scope,
            redirect_uri=redirect_uri,
            token=token,
            token_placement=token_placement,
            update_token=update_token,
            **kwargs,
        )

    @staticmethod
    def handle_error(error_type, error_description):
        raise OAuthError(error_type, error_description)

    def request(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ):
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token:
                raise MissingTokenError()

            if not self.ensure_active_token(self.token):
                raise InvalidTokenError()

            auth = self.token_auth

        return super().request(method, url, auth=auth, **kwargs)

    def stream(
        self, method, url, withhold_token=False, auth=USE_CLIENT_DEFAULT, **kwargs
    ):
        if not withhold_token and auth is USE_CLIENT_DEFAULT:
            if not self.token:
                raise MissingTokenError()

            if not self.ensure_active_token(self.token):
                raise InvalidTokenError()

            auth = self.token_auth

        return super().stream(method, url, auth=auth, **kwargs)

import logging
import time

from authlib.common.urls import urlparse

from .errors import MissingRequestTokenError
from .errors import MissingTokenError
from .sync_app import OAuth1Base
from .sync_app import OAuth2Base

log = logging.getLogger(__name__)

__all__ = ["AsyncOAuth1Mixin", "AsyncOAuth2Mixin"]


class AsyncOAuth1Mixin(OAuth1Base):
    async def request(self, method, url, token=None, **kwargs):
        async with self._get_oauth_client() as session:
            return await _http_request(self, session, method, url, token, kwargs)

    async def create_authorization_url(self, redirect_uri=None, **kwargs):
        """Generate the authorization url and state for HTTP redirect.

        :param redirect_uri: Callback or redirect URI for authorization.
        :param kwargs: Extra parameters to include.
        :return: dict
        """
        if not self.authorize_url:
            raise RuntimeError('Missing "authorize_url" value')

        if self.authorize_params:
            kwargs.update(self.authorize_params)

        async with self._get_oauth_client() as client:
            client.redirect_uri = redirect_uri
            params = {}
            if self.request_token_params:
                params.update(self.request_token_params)
            request_token = await client.fetch_request_token(
                self.request_token_url, **params
            )
            log.debug(f"Fetch request token: {request_token!r}")
            url = client.create_authorization_url(self.authorize_url, **kwargs)
            state = request_token["oauth_token"]
        return {"url": url, "request_token": request_token, "state": state}

    async def fetch_access_token(self, request_token=None, **kwargs):
        """Fetch access token in one step.

        :param request_token: A previous request token for OAuth 1.
        :param kwargs: Extra parameters to fetch access token.
        :return: A token dict.
        """
        async with self._get_oauth_client() as client:
            if request_token is None:
                raise MissingRequestTokenError()
            # merge request token with verifier
            token = {}
            token.update(request_token)
            token.update(kwargs)
            client.token = token
            params = self.access_token_params or {}
            token = await client.fetch_access_token(self.access_token_url, **params)
        return token


class AsyncOAuth2Mixin(OAuth2Base):
    async def _on_update_token(self, token, refresh_token=None, access_token=None):
        if self._update_token:
            await self._update_token(
                token,
                refresh_token=refresh_token,
                access_token=access_token,
            )

    async def load_server_metadata(self):
        if self._server_metadata_url and "_loaded_at" not in self.server_metadata:
            async with self.client_cls(**self.client_kwargs) as client:
                resp = await client.request(
                    "GET", self._server_metadata_url, withhold_token=True
                )
                resp.raise_for_status()
                metadata = resp.json()
                metadata["_loaded_at"] = time.time()
            self.server_metadata.update(metadata)
        return self.server_metadata

    async def request(self, method, url, token=None, **kwargs):
        metadata = await self.load_server_metadata()
        async with self._get_oauth_client(**metadata) as session:
            return await _http_request(self, session, method, url, token, kwargs)

    async def create_authorization_url(self, redirect_uri=None, **kwargs):
        """Generate the authorization url and state for HTTP redirect.

        :param redirect_uri: Callback or redirect URI for authorization.
        :param kwargs: Extra parameters to include.
        :return: dict
        """
        metadata = await self.load_server_metadata()
        authorization_endpoint = self.authorize_url or metadata.get(
            "authorization_endpoint"
        )
        if not authorization_endpoint:
            raise RuntimeError('Missing "authorize_url" value')

        if self.authorize_params:
            kwargs.update(self.authorize_params)

        async with self._get_oauth_client(**metadata) as client:
            client.redirect_uri = redirect_uri
            return self._create_oauth2_authorization_url(
                client, authorization_endpoint, **kwargs
            )

    async def fetch_access_token(self, redirect_uri=None, **kwargs):
        """Fetch access token in the final step.

        :param redirect_uri: Callback or Redirect URI that is used in
                             previous :meth:`authorize_redirect`.
        :param kwargs: Extra parameters to fetch access token.
        :return: A token dict.
        """
        metadata = await self.load_server_metadata()
        token_endpoint = self.access_token_url or metadata.get("token_endpoint")
        async with self._get_oauth_client(**metadata) as client:
            if redirect_uri is not None:
                client.redirect_uri = redirect_uri
            params = {}
            if self.access_token_params:
                params.update(self.access_token_params)
            params.update(kwargs)
            token = await client.fetch_token(token_endpoint, **params)
        return token


async def _http_request(ctx, session, method, url, token, kwargs):
    request = kwargs.pop("request", None)
    withhold_token = kwargs.get("withhold_token")
    if ctx.api_base_url and not url.startswith(("https://", "http://")):
        url = urlparse.urljoin(ctx.api_base_url, url)

    if withhold_token:
        return await session.request(method, url, **kwargs)

    if token is None and ctx._fetch_token and request:
        token = await ctx._fetch_token(request)
    if token is None:
        raise MissingTokenError()

    session.token = token
    return await session.request(method, url, **kwargs)

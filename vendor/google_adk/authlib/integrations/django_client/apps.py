from django.http import HttpResponseRedirect

from ..base_client import BaseApp
from ..base_client import OAuth1Mixin
from ..base_client import OAuth2Mixin
from ..base_client import OAuthError
from ..base_client import OpenIDMixin
from ..requests_client import OAuth1Session
from ..requests_client import OAuth2Session


class DjangoAppMixin:
    def save_authorize_data(self, request, **kwargs):
        state = kwargs.pop("state", None)
        if state:
            self.framework.set_state_data(request.session, state, kwargs)
        else:
            raise RuntimeError("Missing state value")

    def authorize_redirect(self, request, redirect_uri=None, **kwargs):
        """Create a HTTP Redirect for Authorization Endpoint.

        :param request: HTTP request instance from Django view.
        :param redirect_uri: Callback or redirect URI for authorization.
        :param kwargs: Extra parameters to include.
        :return: A HTTP redirect response.
        """
        rv = self.create_authorization_url(redirect_uri, **kwargs)
        self.save_authorize_data(request, redirect_uri=redirect_uri, **rv)
        return HttpResponseRedirect(rv["url"])


class DjangoOAuth1App(DjangoAppMixin, OAuth1Mixin, BaseApp):
    client_cls = OAuth1Session

    def authorize_access_token(self, request, **kwargs):
        """Fetch access token in one step.

        :param request: HTTP request instance from Django view.
        :return: A token dict.
        """
        params = request.GET.dict()
        state = params.get("oauth_token")
        if not state:
            raise OAuthError(description='Missing "oauth_token" parameter')

        data = self.framework.get_state_data(request.session, state)
        if not data:
            raise OAuthError(description='Missing "request_token" in temporary data')

        params["request_token"] = data["request_token"]
        params.update(kwargs)
        self.framework.clear_state_data(request.session, state)
        return self.fetch_access_token(**params)


class DjangoOAuth2App(DjangoAppMixin, OAuth2Mixin, OpenIDMixin, BaseApp):
    client_cls = OAuth2Session

    def authorize_access_token(self, request, **kwargs):
        """Fetch access token in one step.

        :param request: HTTP request instance from Django view.
        :return: A token dict.
        """
        if request.method == "GET":
            error = request.GET.get("error")
            if error:
                description = request.GET.get("error_description")
                raise OAuthError(error=error, description=description)
            params = {
                "code": request.GET.get("code"),
                "state": request.GET.get("state"),
            }
        else:
            params = {
                "code": request.POST.get("code"),
                "state": request.POST.get("state"),
            }

        state_data = self.framework.get_state_data(request.session, params.get("state"))
        self.framework.clear_state_data(request.session, params.get("state"))
        params = self._format_state_params(state_data, params)

        claims_options = kwargs.pop("claims_options", None)
        claims_cls = kwargs.pop("claims_cls", None)
        leeway = kwargs.pop("leeway", 120)
        token = self.fetch_access_token(**params, **kwargs)

        if "id_token" in token and "nonce" in state_data:
            userinfo = self.parse_id_token(
                token,
                nonce=state_data["nonce"],
                claims_options=claims_options,
                claims_cls=claims_cls,
                leeway=leeway,
            )
            token["userinfo"] = userinfo
        return token

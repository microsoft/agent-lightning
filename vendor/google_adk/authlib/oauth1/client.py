from authlib.common.encoding import json_loads
from authlib.common.urls import add_params_to_uri
from authlib.common.urls import url_decode
from authlib.common.urls import urlparse

from .rfc5849 import SIGNATURE_HMAC_SHA1
from .rfc5849 import SIGNATURE_TYPE_HEADER
from .rfc5849 import ClientAuth


class OAuth1Client:
    auth_class = ClientAuth

    def __init__(
        self,
        session,
        client_id,
        client_secret=None,
        token=None,
        token_secret=None,
        redirect_uri=None,
        rsa_key=None,
        verifier=None,
        signature_method=SIGNATURE_HMAC_SHA1,
        signature_type=SIGNATURE_TYPE_HEADER,
        force_include_body=False,
        realm=None,
        **kwargs,
    ):
        if not client_id:
            raise ValueError('Missing "client_id"')

        self.session = session
        self.auth = self.auth_class(
            client_id,
            client_secret=client_secret,
            token=token,
            token_secret=token_secret,
            redirect_uri=redirect_uri,
            signature_method=signature_method,
            signature_type=signature_type,
            rsa_key=rsa_key,
            verifier=verifier,
            realm=realm,
            force_include_body=force_include_body,
        )
        self._kwargs = kwargs

    @property
    def redirect_uri(self):
        return self.auth.redirect_uri

    @redirect_uri.setter
    def redirect_uri(self, uri):
        self.auth.redirect_uri = uri

    @property
    def token(self):
        return dict(
            oauth_token=self.auth.token,
            oauth_token_secret=self.auth.token_secret,
            oauth_verifier=self.auth.verifier,
        )

    @token.setter
    def token(self, token):
        """This token setter is designed for an easy integration for
        OAuthClient. Make sure both OAuth1Session and OAuth2Session
        have token setters.
        """
        if token is None:
            self.auth.token = None
            self.auth.token_secret = None
            self.auth.verifier = None
        elif "oauth_token" in token:
            self.auth.token = token["oauth_token"]
            if "oauth_token_secret" in token:
                self.auth.token_secret = token["oauth_token_secret"]
            if "oauth_verifier" in token:
                self.auth.verifier = token["oauth_verifier"]
        else:
            message = f"oauth_token is missing: {token!r}"
            self.handle_error("missing_token", message)

    def create_authorization_url(self, url, request_token=None, **kwargs):
        """Create an authorization URL by appending request_token and optional
        kwargs to url.

        This is the second step in the OAuth 1 workflow. The user should be
        redirected to this authorization URL, grant access to you, and then
        be redirected back to you. The redirection back can either be specified
        during client registration or by supplying a callback URI per request.

        :param url: The authorization endpoint URL.
        :param request_token: The previously obtained request token.
        :param kwargs: Optional parameters to append to the URL.
        :returns: The authorization URL with new parameters embedded.
        """
        kwargs["oauth_token"] = request_token or self.auth.token
        if self.auth.redirect_uri:
            kwargs["oauth_callback"] = self.auth.redirect_uri
        return add_params_to_uri(url, kwargs.items())

    def fetch_request_token(self, url, **kwargs):
        """Method for fetching an access token from the token endpoint.

        This is the first step in the OAuth 1 workflow. A request token is
        obtained by making a signed post request to url. The token is then
        parsed from the application/x-www-form-urlencoded response and ready
        to be used to construct an authorization url.

        :param url: Request Token endpoint.
        :param kwargs: Extra parameters to include for fetching token.
        :return: A Request Token dict.
        """
        return self._fetch_token(url, **kwargs)

    def fetch_access_token(self, url, verifier=None, **kwargs):
        """Method for fetching an access token from the token endpoint.

        This is the final step in the OAuth 1 workflow. An access token is
        obtained using all previously obtained credentials, including the
        verifier from the authorization step.

        :param url: Access Token endpoint.
        :param verifier: A verifier string to prove authorization was granted.
        :param kwargs: Extra parameters to include for fetching access token.
        :return: A token dict.
        """
        if verifier:
            self.auth.verifier = verifier
        if not self.auth.verifier:
            self.handle_error("missing_verifier", 'Missing "verifier" value')
        return self._fetch_token(url, **kwargs)

    def parse_authorization_response(self, url):
        """Extract parameters from the post authorization redirect
        response URL.

        :param url: The full URL that resulted from the user being redirected
                    back from the OAuth provider to you, the client.
        :returns: A dict of parameters extracted from the URL.
        """
        token = dict(url_decode(urlparse.urlparse(url).query))
        self.token = token
        return token

    def _fetch_token(self, url, **kwargs):
        resp = self.session.post(url, auth=self.auth, **kwargs)
        token = self.parse_response_token(resp.status_code, resp.text)
        self.token = token
        self.auth.verifier = None
        return token

    def parse_response_token(self, status_code, text):
        if status_code >= 400:
            message = (
                f"Token request failed with code {status_code}, response was '{text}'."
            )
            self.handle_error("fetch_token_denied", message)

        try:
            text = text.strip()
            if text.startswith("{"):
                token = json_loads(text)
            else:
                token = dict(url_decode(text))
        except (TypeError, ValueError) as e:
            error = (
                "Unable to decode token from token response. "
                "This is commonly caused by an unsuccessful request where"
                " a non urlencoded error message is returned. "
                f"The decoding error was {e}"
            )
            raise ValueError(error) from e
        return token

    @staticmethod
    def handle_error(error_type, error_description):
        raise ValueError(f"{error_type}: {error_description}")

    def __del__(self):
        try:
            del self.session
        except AttributeError:
            pass

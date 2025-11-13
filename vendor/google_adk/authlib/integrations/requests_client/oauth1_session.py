from requests import Session
from requests.auth import AuthBase

from authlib.common.encoding import to_native
from authlib.oauth1 import SIGNATURE_HMAC_SHA1
from authlib.oauth1 import SIGNATURE_TYPE_HEADER
from authlib.oauth1 import ClientAuth
from authlib.oauth1.client import OAuth1Client

from ..base_client import OAuthError
from .utils import update_session_configure


class OAuth1Auth(AuthBase, ClientAuth):
    """Signs the request using OAuth 1 (RFC5849)."""

    def __call__(self, req):
        url, headers, body = self.prepare(req.method, req.url, req.headers, req.body)

        req.url = to_native(url)
        req.prepare_headers(headers)
        if body:
            req.body = body
        return req


class OAuth1Session(OAuth1Client, Session):
    auth_class = OAuth1Auth

    def __init__(
        self,
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
        **kwargs,
    ):
        Session.__init__(self)
        update_session_configure(self, kwargs)
        OAuth1Client.__init__(
            self,
            session=self,
            client_id=client_id,
            client_secret=client_secret,
            token=token,
            token_secret=token_secret,
            redirect_uri=redirect_uri,
            rsa_key=rsa_key,
            verifier=verifier,
            signature_method=signature_method,
            signature_type=signature_type,
            force_include_body=force_include_body,
            **kwargs,
        )

    def rebuild_auth(self, prepared_request, response):
        """When being redirected we should always strip Authorization
        header, since nonce may not be reused as per OAuth spec.
        """
        if "Authorization" in prepared_request.headers:
            # If we get redirected to a new host, we should strip out
            # any authentication headers.
            prepared_request.headers.pop("Authorization", True)
            prepared_request.prepare_auth(self.auth)

    @staticmethod
    def handle_error(error_type, error_description):
        raise OAuthError(error_type, error_description)

from authlib.common.urls import add_params_to_qs

from .assertion import client_secret_jwt_sign
from .assertion import private_key_jwt_sign
from .client import ASSERTION_TYPE


class ClientSecretJWT:
    """Authentication method for OAuth 2.0 Client. This authentication
    method is called ``client_secret_jwt``, which is using ``client_id``
    and ``client_secret`` constructed with JWT to identify a client.

    Here is an example of use ``client_secret_jwt`` with Requests Session::

        from authlib.integrations.requests_client import OAuth2Session

        token_endpoint = "https://example.com/oauth/token"
        session = OAuth2Session(
            "your-client-id",
            "your-client-secret",
            token_endpoint_auth_method="client_secret_jwt",
        )
        session.register_client_auth_method(ClientSecretJWT(token_endpoint))
        session.fetch_token(token_endpoint)

    :param token_endpoint: A string URL of the token endpoint
    :param claims: Extra JWT claims
    :param headers: Extra JWT headers
    :param alg: ``alg`` value, default is HS256
    """

    name = "client_secret_jwt"
    alg = "HS256"

    def __init__(self, token_endpoint=None, claims=None, headers=None, alg=None):
        self.token_endpoint = token_endpoint
        self.claims = claims
        self.headers = headers
        if alg is not None:
            self.alg = alg

    def sign(self, auth, token_endpoint):
        return client_secret_jwt_sign(
            auth.client_secret,
            client_id=auth.client_id,
            token_endpoint=token_endpoint,
            claims=self.claims,
            header=self.headers,
            alg=self.alg,
        )

    def __call__(self, auth, method, uri, headers, body):
        token_endpoint = self.token_endpoint
        if not token_endpoint:
            token_endpoint = uri

        client_assertion = self.sign(auth, token_endpoint)
        body = add_params_to_qs(
            body or "",
            [
                ("client_assertion_type", ASSERTION_TYPE),
                ("client_assertion", client_assertion),
            ],
        )
        return uri, headers, body


class PrivateKeyJWT(ClientSecretJWT):
    """Authentication method for OAuth 2.0 Client. This authentication
    method is called ``private_key_jwt``, which is using ``client_id``
    and ``private_key`` constructed with JWT to identify a client.

    Here is an example of use ``private_key_jwt`` with Requests Session::

        from authlib.integrations.requests_client import OAuth2Session

        token_endpoint = "https://example.com/oauth/token"
        session = OAuth2Session(
            "your-client-id",
            "your-client-private-key",
            token_endpoint_auth_method="private_key_jwt",
        )
        session.register_client_auth_method(PrivateKeyJWT(token_endpoint))
        session.fetch_token(token_endpoint)

    :param token_endpoint: A string URL of the token endpoint
    :param claims: Extra JWT claims
    :param headers: Extra JWT headers
    :param alg: ``alg`` value, default is RS256
    """

    name = "private_key_jwt"
    alg = "RS256"

    def sign(self, auth, token_endpoint):
        return private_key_jwt_sign(
            auth.client_secret,
            client_id=auth.client_id,
            token_endpoint=token_endpoint,
            claims=self.claims,
            header=self.headers,
            alg=self.alg,
        )

from authlib.common.errors import ContinueIteration
from authlib.consts import default_json_headers
from authlib.jose.errors import ExpiredTokenError
from authlib.jose.errors import InvalidClaimError
from authlib.oauth2.rfc6750.errors import InvalidTokenError
from authlib.oauth2.rfc9068.token_validator import JWTBearerTokenValidator

from ..rfc7662 import IntrospectionEndpoint


class JWTIntrospectionEndpoint(IntrospectionEndpoint):
    r"""JWTIntrospectionEndpoint inherits from :ref:`specs/rfc7662`
    :class:`~authlib.oauth2.rfc7662.IntrospectionEndpoint` and implements the machinery
    to automatically process the JWT access tokens.

    :param issuer: The issuer identifier for which tokens will be introspected.

    :param \\*\\*kwargs: Other parameters are inherited from
        :class:`~authlib.oauth2.rfc7662.introspection.IntrospectionEndpoint`.

    ::

        class MyJWTAccessTokenIntrospectionEndpoint(JWTIntrospectionEndpoint):
            def get_jwks(self): ...

            def get_username(self, user_id): ...


        # endpoint dedicated to JWT access token introspection
        authorization_server.register_endpoint(
            MyJWTAccessTokenIntrospectionEndpoint(
                issuer="https://authorization-server.example.org",
            )
        )

        # another endpoint dedicated to refresh token introspection
        authorization_server.register_endpoint(MyRefreshTokenIntrospectionEndpoint)

    """

    #: Endpoint name to be registered
    ENDPOINT_NAME = "introspection"

    def __init__(self, issuer, server=None, *args, **kwargs):
        super().__init__(*args, server=server, **kwargs)
        self.issuer = issuer

    def create_endpoint_response(self, request):
        """"""
        # The authorization server first validates the client credentials
        client = self.authenticate_endpoint_client(request)

        # then verifies whether the token was issued to the client making
        # the revocation request
        token = self.authenticate_token(request, client)

        # the authorization server invalidates the token
        body = self.create_introspection_payload(token)
        return 200, body, default_json_headers

    def authenticate_token(self, request, client):
        """"""
        self.check_params(request, client)

        # do not attempt to decode refresh_tokens
        if request.form.get("token_type_hint") not in ("access_token", None):
            raise ContinueIteration()

        validator = JWTBearerTokenValidator(issuer=self.issuer, resource_server=None)
        validator.get_jwks = self.get_jwks
        try:
            token = validator.authenticate_token(request.form["token"])

        # if the token is not a JWT, fall back to the regular flow
        except InvalidTokenError as exc:
            raise ContinueIteration() from exc

        if token and self.check_permission(token, client, request):
            return token

    def create_introspection_payload(self, token):
        if not token:
            return {"active": False}

        try:
            token.validate()
        except ExpiredTokenError:
            return {"active": False}
        except InvalidClaimError as exc:
            if exc.claim_name == "iss":
                raise ContinueIteration() from exc
            raise InvalidTokenError() from exc

        payload = {
            "active": True,
            "token_type": "Bearer",
            "client_id": token["client_id"],
            "scope": token["scope"],
            "sub": token["sub"],
            "aud": token["aud"],
            "iss": token["iss"],
            "exp": token["exp"],
            "iat": token["iat"],
        }

        if username := self.get_username(token["sub"]):
            payload["username"] = username

        return payload

    def get_jwks(self):
        """Return the JWKs that will be used to check the JWT access token signature.
        Developers MUST re-implement this method::

            def get_jwks(self):
                return load_jwks("jwks.json")
        """
        raise NotImplementedError()

    def get_username(self, user_id: str) -> str:
        """Returns an username from a user ID.
        Developers MAY re-implement this method::

            def get_username(self, user_id):
                return User.get(id=user_id).username
        """
        return None

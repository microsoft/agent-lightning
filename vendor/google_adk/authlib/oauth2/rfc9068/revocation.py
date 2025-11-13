from authlib.common.errors import ContinueIteration
from authlib.oauth2.rfc6750.errors import InvalidTokenError
from authlib.oauth2.rfc9068.token_validator import JWTBearerTokenValidator

from ..rfc6749 import UnsupportedTokenTypeError
from ..rfc7009 import RevocationEndpoint


class JWTRevocationEndpoint(RevocationEndpoint):
    r"""JWTRevocationEndpoint inherits from `RFC7009`_
    :class:`~authlib.oauth2.rfc7009.RevocationEndpoint`.

    The JWT access tokens cannot be revoked.
    If the submitted token is a JWT access token, then revocation returns
    a `invalid_token_error`.

    :param issuer: The issuer identifier.

    :param \\*\\*kwargs: Other parameters are inherited from
        :class:`~authlib.oauth2.rfc7009.RevocationEndpoint`.

    Plain text access tokens and other kind of tokens such as refresh_tokens
    will be ignored by this endpoint and passed to the next revocation endpoint::

        class MyJWTAccessTokenRevocationEndpoint(JWTRevocationEndpoint):
            def get_jwks(self): ...


        # endpoint dedicated to JWT access token revokation
        authorization_server.register_endpoint(
            MyJWTAccessTokenRevocationEndpoint(
                issuer="https://authorization-server.example.org",
            )
        )

        # another endpoint dedicated to refresh token revokation
        authorization_server.register_endpoint(MyRefreshTokenRevocationEndpoint)

    .. _RFC7009: https://tools.ietf.org/html/rfc7009
    """

    def __init__(self, issuer, server=None, *args, **kwargs):
        super().__init__(*args, server=server, **kwargs)
        self.issuer = issuer

    def authenticate_token(self, request, client):
        """"""
        self.check_params(request, client)

        # do not attempt to revoke refresh_tokens
        if request.form.get("token_type_hint") not in ("access_token", None):
            raise ContinueIteration()

        validator = JWTBearerTokenValidator(issuer=self.issuer, resource_server=None)
        validator.get_jwks = self.get_jwks

        try:
            validator.authenticate_token(request.form["token"])

        # if the token is not a JWT, fall back to the regular flow
        except InvalidTokenError as exc:
            raise ContinueIteration() from exc

        # JWT access token cannot be revoked
        raise UnsupportedTokenTypeError()

    def get_jwks(self):
        """Return the JWKs that will be used to check the JWT access token signature.
        Developers MUST re-implement this method::

            def get_jwks(self):
                return load_jwks("jwks.json")
        """
        raise NotImplementedError()

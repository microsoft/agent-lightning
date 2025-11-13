import logging
import time

from authlib.jose import JoseError
from authlib.jose import JWTClaims
from authlib.jose import jwt

from ..rfc6749 import TokenMixin
from ..rfc6750 import BearerTokenValidator

logger = logging.getLogger(__name__)


class JWTBearerToken(TokenMixin, JWTClaims):
    def check_client(self, client):
        return self["client_id"] == client.get_client_id()

    def get_scope(self):
        return self.get("scope")

    def get_expires_in(self):
        return self["exp"] - self["iat"]

    def is_expired(self):
        return self["exp"] < time.time()

    def is_revoked(self):
        return False


class JWTBearerTokenValidator(BearerTokenValidator):
    TOKEN_TYPE = "bearer"
    token_cls = JWTBearerToken

    def __init__(self, public_key, issuer=None, realm=None, **extra_attributes):
        super().__init__(realm, **extra_attributes)
        self.public_key = public_key
        claims_options = {
            "exp": {"essential": True},
            "client_id": {"essential": True},
            "grant_type": {"essential": True},
        }
        if issuer:
            claims_options["iss"] = {"essential": True, "value": issuer}
        self.claims_options = claims_options

    def authenticate_token(self, token_string):
        try:
            claims = jwt.decode(
                token_string,
                self.public_key,
                claims_options=self.claims_options,
                claims_cls=self.token_cls,
            )
            claims.validate()
            return claims
        except JoseError as error:
            logger.debug("Authenticate token failed. %r", error)
            return None

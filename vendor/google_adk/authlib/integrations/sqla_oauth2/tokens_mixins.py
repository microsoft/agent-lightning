import time

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text

from authlib.oauth2.rfc6749 import AuthorizationCodeMixin
from authlib.oauth2.rfc6749 import TokenMixin


class OAuth2AuthorizationCodeMixin(AuthorizationCodeMixin):
    code = Column(String(120), unique=True, nullable=False)
    client_id = Column(String(48))
    redirect_uri = Column(Text, default="")
    response_type = Column(Text, default="")
    scope = Column(Text, default="")
    nonce = Column(Text)
    auth_time = Column(Integer, nullable=False, default=lambda: int(time.time()))
    acr = Column(Text, nullable=True)
    amr = Column(Text, nullable=True)

    code_challenge = Column(Text)
    code_challenge_method = Column(String(48))

    def is_expired(self):
        return self.auth_time + 300 < time.time()

    def get_redirect_uri(self):
        return self.redirect_uri

    def get_scope(self):
        return self.scope

    def get_auth_time(self):
        return self.auth_time

    def get_acr(self):
        return self.acr

    def get_amr(self):
        return self.amr.split() if self.amr else []

    def get_nonce(self):
        return self.nonce


class OAuth2TokenMixin(TokenMixin):
    client_id = Column(String(48))
    token_type = Column(String(40))
    access_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), index=True)
    scope = Column(Text, default="")
    issued_at = Column(Integer, nullable=False, default=lambda: int(time.time()))
    access_token_revoked_at = Column(Integer, nullable=False, default=0)
    refresh_token_revoked_at = Column(Integer, nullable=False, default=0)
    expires_in = Column(Integer, nullable=False, default=0)

    def check_client(self, client):
        return self.client_id == client.get_client_id()

    def get_scope(self):
        return self.scope

    def get_expires_in(self):
        return self.expires_in

    def is_revoked(self):
        return self.access_token_revoked_at or self.refresh_token_revoked_at

    def is_expired(self):
        if not self.expires_in:
            return False

        expires_at = self.issued_at + self.expires_in
        return expires_at < time.time()

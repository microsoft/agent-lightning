from requests import Session

from authlib.oauth2.rfc7521 import AssertionClient
from authlib.oauth2.rfc7523 import JWTBearerGrant

from .oauth2_session import OAuth2Auth
from .utils import update_session_configure


class AssertionAuth(OAuth2Auth):
    def ensure_active_token(self):
        if self.client and (
            not self.token or self.token.is_expired(self.client.leeway)
        ):
            return self.client.refresh_token()


class AssertionSession(AssertionClient, Session):
    """Constructs a new Assertion Framework for OAuth 2.0 Authorization Grants
    per RFC7521_.

    .. _RFC7521: https://tools.ietf.org/html/rfc7521
    """

    token_auth_class = AssertionAuth
    JWT_BEARER_GRANT_TYPE = JWTBearerGrant.GRANT_TYPE
    ASSERTION_METHODS = {
        JWT_BEARER_GRANT_TYPE: JWTBearerGrant.sign,
    }
    DEFAULT_GRANT_TYPE = JWT_BEARER_GRANT_TYPE

    def __init__(
        self,
        token_endpoint,
        issuer,
        subject,
        audience=None,
        grant_type=None,
        claims=None,
        token_placement="header",
        scope=None,
        default_timeout=None,
        leeway=60,
        **kwargs,
    ):
        Session.__init__(self)
        self.default_timeout = default_timeout
        update_session_configure(self, kwargs)
        AssertionClient.__init__(
            self,
            session=self,
            token_endpoint=token_endpoint,
            issuer=issuer,
            subject=subject,
            audience=audience,
            grant_type=grant_type,
            claims=claims,
            token_placement=token_placement,
            scope=scope,
            leeway=leeway,
            **kwargs,
        )

    def request(self, method, url, withhold_token=False, auth=None, **kwargs):
        """Send request with auto refresh token feature."""
        if self.default_timeout:
            kwargs.setdefault("timeout", self.default_timeout)
        if not withhold_token and auth is None:
            auth = self.token_auth
        return super().request(method, url, auth=auth, **kwargs)

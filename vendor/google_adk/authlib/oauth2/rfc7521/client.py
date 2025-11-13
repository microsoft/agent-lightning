from authlib.common.encoding import to_native
from authlib.oauth2.base import OAuth2Error


class AssertionClient:
    """Constructs a new Assertion Framework for OAuth 2.0 Authorization Grants
    per RFC7521_.

    .. _RFC7521: https://tools.ietf.org/html/rfc7521
    """

    DEFAULT_GRANT_TYPE = None
    ASSERTION_METHODS = {}
    token_auth_class = None
    oauth_error_class = OAuth2Error

    def __init__(
        self,
        session,
        token_endpoint,
        issuer,
        subject,
        audience=None,
        grant_type=None,
        claims=None,
        token_placement="header",
        scope=None,
        leeway=60,
        **kwargs,
    ):
        self.session = session

        if audience is None:
            audience = token_endpoint

        self.token_endpoint = token_endpoint

        if grant_type is None:
            grant_type = self.DEFAULT_GRANT_TYPE

        self.grant_type = grant_type

        # https://tools.ietf.org/html/rfc7521#section-5.1
        self.issuer = issuer
        self.subject = subject
        self.audience = audience
        self.claims = claims
        self.scope = scope
        if self.token_auth_class is not None:
            self.token_auth = self.token_auth_class(None, token_placement, self)
        self._kwargs = kwargs
        self.leeway = leeway

    @property
    def token(self):
        return self.token_auth.token

    @token.setter
    def token(self, token):
        self.token_auth.set_token(token)

    def refresh_token(self):
        """Using Assertions as Authorization Grants to refresh token as
        described in `Section 4.1`_.

        .. _`Section 4.1`: https://tools.ietf.org/html/rfc7521#section-4.1
        """
        generate_assertion = self.ASSERTION_METHODS[self.grant_type]
        assertion = generate_assertion(
            issuer=self.issuer,
            subject=self.subject,
            audience=self.audience,
            claims=self.claims,
            **self._kwargs,
        )
        data = {
            "assertion": to_native(assertion),
            "grant_type": self.grant_type,
        }
        if self.scope:
            data["scope"] = self.scope

        return self._refresh_token(data)

    def parse_response_token(self, resp):
        if resp.status_code >= 500:
            resp.raise_for_status()

        token = resp.json()
        if "error" in token:
            raise self.oauth_error_class(
                error=token["error"], description=token.get("error_description")
            )

        self.token = token
        return self.token

    def _refresh_token(self, data):
        resp = self.session.request(
            "POST", self.token_endpoint, data=data, withhold_token=True
        )

        return self.parse_response_token(resp)

    def __del__(self):
        if self.session:
            del self.session

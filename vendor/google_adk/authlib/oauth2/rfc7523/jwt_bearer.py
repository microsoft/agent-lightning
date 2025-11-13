import logging

from authlib.jose import JoseError
from authlib.jose import jwt

from ..rfc6749 import BaseGrant
from ..rfc6749 import InvalidClientError
from ..rfc6749 import InvalidGrantError
from ..rfc6749 import InvalidRequestError
from ..rfc6749 import TokenEndpointMixin
from ..rfc6749 import UnauthorizedClientError
from .assertion import sign_jwt_bearer_assertion

log = logging.getLogger(__name__)
JWT_BEARER_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"


class JWTBearerGrant(BaseGrant, TokenEndpointMixin):
    GRANT_TYPE = JWT_BEARER_GRANT_TYPE

    #: Options for verifying JWT payload claims. Developers MAY
    #: overwrite this constant to create a more strict options.
    CLAIMS_OPTIONS = {
        "iss": {"essential": True},
        "aud": {"essential": True},
        "exp": {"essential": True},
    }

    # A small allowance of time, typically no more than a few minutes,
    # to account for clock skew. The default is 60 seconds.
    LEEWAY = 60

    @staticmethod
    def sign(
        key,
        issuer,
        audience,
        subject=None,
        issued_at=None,
        expires_at=None,
        claims=None,
        **kwargs,
    ):
        return sign_jwt_bearer_assertion(
            key, issuer, audience, subject, issued_at, expires_at, claims, **kwargs
        )

    def process_assertion_claims(self, assertion):
        """Extract JWT payload claims from request "assertion", per
        `Section 3.1`_.

        :param assertion: assertion string value in the request
        :return: JWTClaims
        :raise: InvalidGrantError

        .. _`Section 3.1`: https://tools.ietf.org/html/rfc7523#section-3.1
        """
        try:
            claims = jwt.decode(
                assertion, self.resolve_public_key, claims_options=self.CLAIMS_OPTIONS
            )
            claims.validate(leeway=self.LEEWAY)
        except JoseError as e:
            log.debug("Assertion Error: %r", e)
            raise InvalidGrantError(description=e.description) from e
        return claims

    def resolve_public_key(self, headers, payload):
        client = self.resolve_issuer_client(payload["iss"])
        return self.resolve_client_key(client, headers, payload)

    def validate_token_request(self):
        """The client makes a request to the token endpoint by sending the
        following parameters using the "application/x-www-form-urlencoded"
        format per `Section 2.1`_:

        grant_type
             REQUIRED.  Value MUST be set to
             "urn:ietf:params:oauth:grant-type:jwt-bearer".

        assertion
             REQUIRED.  Value MUST contain a single JWT.

        scope
            OPTIONAL.

        The following example demonstrates an access token request with a JWT
        as an authorization grant:

        .. code-block:: http

            POST /token.oauth2 HTTP/1.1
            Host: as.example.com
            Content-Type: application/x-www-form-urlencoded

            grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer
            &assertion=eyJhbGciOiJFUzI1NiIsImtpZCI6IjE2In0.
            eyJpc3Mi[...omitted for brevity...].
            J9l-ZhwP[...omitted for brevity...]

        .. _`Section 2.1`: https://tools.ietf.org/html/rfc7523#section-2.1
        """
        assertion = self.request.form.get("assertion")
        if not assertion:
            raise InvalidRequestError("Missing 'assertion' in request")

        claims = self.process_assertion_claims(assertion)
        client = self.resolve_issuer_client(claims["iss"])
        log.debug("Validate token request of %s", client)

        if not client.check_grant_type(self.GRANT_TYPE):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'grant_type={self.GRANT_TYPE}'"
            )

        self.request.client = client
        self.validate_requested_scope()

        subject = claims.get("sub")
        if subject:
            user = self.authenticate_user(subject)
            if not user:
                raise InvalidGrantError(description="Invalid 'sub' value in assertion")

            log.debug("Check client(%s) permission to User(%s)", client, user)
            if not self.has_granted_permission(client, user):
                raise InvalidClientError(
                    description="Client has no permission to access user data"
                )
            self.request.user = user

    def create_token_response(self):
        """If valid and authorized, the authorization server issues an access
        token.
        """
        token = self.generate_token(
            scope=self.request.payload.scope,
            user=self.request.user,
            include_refresh_token=False,
        )
        log.debug("Issue token %r to %r", token, self.request.client)
        self.save_token(token)
        return 200, token, self.TOKEN_RESPONSE_HEADER

    def resolve_issuer_client(self, issuer):
        """Fetch client via "iss" in assertion claims. Developers MUST
        implement this method in subclass, e.g.::

            def resolve_issuer_client(self, issuer):
                return Client.query_by_iss(issuer)

        :param issuer: "iss" value in assertion
        :return: Client instance
        """
        raise NotImplementedError()

    def resolve_client_key(self, client, headers, payload):
        """Resolve client key to decode assertion data. Developers MUST
        implement this method in subclass. For instance, there is a
        "jwks" column on client table, e.g.::

            def resolve_client_key(self, client, headers, payload):
                # from authlib.jose import JsonWebKey

                key_set = JsonWebKey.import_key_set(client.jwks)
                return key_set.find_by_kid(headers["kid"])

        :param client: instance of OAuth client model
        :param headers: headers part of the JWT
        :param payload: payload part of the JWT
        :return: ``authlib.jose.Key`` instance
        """
        raise NotImplementedError()

    def authenticate_user(self, subject):
        """Authenticate user with the given assertion claims. Developers MUST
        implement it in subclass, e.g.::

            def authenticate_user(self, subject):
                return User.get_by_sub(subject)

        :param subject: "sub" value in claims
        :return: User instance
        """
        raise NotImplementedError()

    def has_granted_permission(self, client, user):
        """Check if the client has permission to access the given user's resource.
        Developers MUST implement it in subclass, e.g.::

            def has_granted_permission(self, client, user):
                permission = ClientUserGrant.query(client=client, user=user)
                return permission.granted

        :param client: instance of OAuth client model
        :param user: instance of User model
        :return: bool
        """
        raise NotImplementedError()

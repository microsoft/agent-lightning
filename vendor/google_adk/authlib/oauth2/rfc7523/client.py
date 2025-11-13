import logging

from authlib.jose import jwt
from authlib.jose.errors import JoseError

from ..rfc6749 import InvalidClientError

ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
log = logging.getLogger(__name__)


class JWTBearerClientAssertion:
    """Implementation of Using JWTs for Client Authentication, which is
    defined by RFC7523.
    """

    #: Value of ``client_assertion_type`` of JWTs
    CLIENT_ASSERTION_TYPE = ASSERTION_TYPE
    #: Name of the client authentication method
    CLIENT_AUTH_METHOD = "client_assertion_jwt"

    def __init__(self, token_url, validate_jti=True, leeway=60):
        self.token_url = token_url
        self._validate_jti = validate_jti
        # A small allowance of time, typically no more than a few minutes,
        # to account for clock skew. The default is 60 seconds.
        self.leeway = leeway

    def __call__(self, query_client, request):
        data = request.form
        assertion_type = data.get("client_assertion_type")
        assertion = data.get("client_assertion")
        if assertion_type == ASSERTION_TYPE and assertion:
            resolve_key = self.create_resolve_key_func(query_client, request)
            self.process_assertion_claims(assertion, resolve_key)
            return self.authenticate_client(request.client)
        log.debug("Authenticate via %r failed", self.CLIENT_AUTH_METHOD)

    def create_claims_options(self):
        """Create a claims_options for verify JWT payload claims. Developers
        MAY overwrite this method to create a more strict options.
        """
        # https://tools.ietf.org/html/rfc7523#section-3
        # The Audience SHOULD be the URL of the Authorization Server's Token Endpoint
        options = {
            "iss": {"essential": True, "validate": _validate_iss},
            "sub": {"essential": True},
            "aud": {"essential": True, "value": self.token_url},
            "exp": {"essential": True},
        }
        if self._validate_jti:
            options["jti"] = {"essential": True, "validate": self.validate_jti}
        return options

    def process_assertion_claims(self, assertion, resolve_key):
        """Extract JWT payload claims from request "assertion", per
        `Section 3.1`_.

        :param assertion: assertion string value in the request
        :param resolve_key: function to resolve the sign key
        :return: JWTClaims
        :raise: InvalidClientError

        .. _`Section 3.1`: https://tools.ietf.org/html/rfc7523#section-3.1
        """
        try:
            claims = jwt.decode(
                assertion, resolve_key, claims_options=self.create_claims_options()
            )
            claims.validate(leeway=self.leeway)
        except JoseError as e:
            log.debug("Assertion Error: %r", e)
            raise InvalidClientError(description=e.description) from e
        return claims

    def authenticate_client(self, client):
        if client.check_endpoint_auth_method(self.CLIENT_AUTH_METHOD, "token"):
            return client
        raise InvalidClientError(
            description=f"The client cannot authenticate with method: {self.CLIENT_AUTH_METHOD}"
        )

    def create_resolve_key_func(self, query_client, request):
        def resolve_key(headers, payload):
            # https://tools.ietf.org/html/rfc7523#section-3
            # For client authentication, the subject MUST be the
            # "client_id" of the OAuth client
            client_id = payload["sub"]
            client = query_client(client_id)
            if not client:
                raise InvalidClientError(
                    description="The client does not exist on this server."
                )
            request.client = client
            return self.resolve_client_public_key(client, headers)

        return resolve_key

    def validate_jti(self, claims, jti):
        """Validate if the given ``jti`` value is used before. Developers
        MUST implement this method::

            def validate_jti(self, claims, jti):
                key = "jti:{}-{}".format(claims["sub"], jti)
                if redis.get(key):
                    return False
                redis.set(key, 1, ex=3600)
                return True
        """
        raise NotImplementedError()

    def resolve_client_public_key(self, client, headers):
        """Resolve the client public key for verifying the JWT signature.
        A client may have many public keys, in this case, we can retrieve it
        via ``kid`` value in headers. Developers MUST implement this method::

            def resolve_client_public_key(self, client, headers):
                return client.public_key
        """
        raise NotImplementedError()


def _validate_iss(claims, iss):
    return claims["sub"] == iss

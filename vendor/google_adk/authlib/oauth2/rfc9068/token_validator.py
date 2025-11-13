"""authlib.oauth2.rfc9068.token_validator.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of Validating JWT Access Tokens per `Section 4`_.

.. _`Section 7`: https://www.rfc-editor.org/rfc/rfc9068.html#name-validating-jwt-access-token
"""

from authlib.jose import jwt
from authlib.jose.errors import DecodeError
from authlib.jose.errors import JoseError
from authlib.oauth2.rfc6750.errors import InsufficientScopeError
from authlib.oauth2.rfc6750.errors import InvalidTokenError
from authlib.oauth2.rfc6750.validator import BearerTokenValidator

from .claims import JWTAccessTokenClaims


class JWTBearerTokenValidator(BearerTokenValidator):
    """JWTBearerTokenValidator can protect your resource server endpoints.

    :param issuer: The issuer from which tokens will be accepted.
    :param resource_server: An identifier for the current resource server,
        which must appear in the JWT ``aud`` claim.

    Developers needs to implement the missing methods::

        class MyJWTBearerTokenValidator(JWTBearerTokenValidator):
            def get_jwks(self): ...


        require_oauth = ResourceProtector()
        require_oauth.register_token_validator(
            MyJWTBearerTokenValidator(
                issuer="https://authorization-server.example.org",
                resource_server="https://resource-server.example.org",
            )
        )

    You can then protect resources depending on the JWT `scope`, `groups`,
    `roles` or `entitlements` claims::

        @require_oauth(
            scope="profile",
            groups="admins",
            roles="student",
            entitlements="captain",
        )
        def resource_endpoint(): ...
    """

    def __init__(self, issuer, resource_server, *args, **kwargs):
        self.issuer = issuer
        self.resource_server = resource_server
        super().__init__(*args, **kwargs)

    def get_jwks(self):
        """Return the JWKs that will be used to check the JWT access token signature.
        Developers MUST re-implement this method. Typically the JWKs are statically
        stored in the resource server configuration, or dynamically downloaded and
        cached using :ref:`specs/rfc8414`::

            def get_jwks(self):
                if "jwks" in cache:
                    return cache.get("jwks")

                server_metadata = get_server_metadata(self.issuer)
                jwks_uri = server_metadata.get("jwks_uri")
                cache["jwks"] = requests.get(jwks_uri).json()
                return cache["jwks"]
        """
        raise NotImplementedError()

    def validate_iss(self, claims, iss: "str") -> bool:
        # The issuer identifier for the authorization server (which is typically
        # obtained during discovery) MUST exactly match the value of the 'iss'
        # claim.
        return iss == self.issuer

    def authenticate_token(self, token_string):
        """"""
        # empty docstring avoids to display the irrelevant parent docstring

        claims_options = {
            "iss": {"essential": True, "validate": self.validate_iss},
            "exp": {"essential": True},
            "aud": {"essential": True, "value": self.resource_server},
            "sub": {"essential": True},
            "client_id": {"essential": True},
            "iat": {"essential": True},
            "jti": {"essential": True},
            "auth_time": {"essential": False},
            "acr": {"essential": False},
            "amr": {"essential": False},
            "scope": {"essential": False},
            "groups": {"essential": False},
            "roles": {"essential": False},
            "entitlements": {"essential": False},
        }
        jwks = self.get_jwks()

        # If the JWT access token is encrypted, decrypt it using the keys and algorithms
        # that the resource server specified during registration. If encryption was
        # negotiated with the authorization server at registration time and the incoming
        # JWT access token is not encrypted, the resource server SHOULD reject it.

        # The resource server MUST validate the signature of all incoming JWT access
        # tokens according to [RFC7515] using the algorithm specified in the JWT 'alg'
        # Header Parameter. The resource server MUST reject any JWT in which the value
        # of 'alg' is 'none'. The resource server MUST use the keys provided by the
        # authorization server.
        try:
            return jwt.decode(
                token_string,
                key=jwks,
                claims_cls=JWTAccessTokenClaims,
                claims_options=claims_options,
            )
        except DecodeError as exc:
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            ) from exc

    def validate_token(
        self, token, scopes, request, groups=None, roles=None, entitlements=None
    ):
        """"""
        # empty docstring avoids to display the irrelevant parent docstring
        try:
            token.validate()
        except JoseError as exc:
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            ) from exc

        # If an authorization request includes a scope parameter, the corresponding
        # issued JWT access token SHOULD include a 'scope' claim as defined in Section
        # 4.2 of [RFC8693]. All the individual scope strings in the 'scope' claim MUST
        # have meaning for the resources indicated in the 'aud' claim. See Section 5 for
        # more considerations about the relationship between scope strings and resources
        # indicated by the 'aud' claim.

        if self.scope_insufficient(token.get("scope", []), scopes):
            raise InsufficientScopeError()

        # Many authorization servers embed authorization attributes that go beyond the
        # delegated scenarios described by [RFC7519] in the access tokens they issue.
        # Typical examples include resource owner memberships in roles and groups that
        # are relevant to the resource being accessed, entitlements assigned to the
        # resource owner for the targeted resource that the authorization server knows
        # about, and so on. An authorization server wanting to include such attributes
        # in a JWT access token SHOULD use the 'groups', 'roles', and 'entitlements'
        # attributes of the 'User' resource schema defined by Section 4.1.2 of
        # [RFC7643]) as claim types.

        if self.scope_insufficient(token.get("groups"), groups):
            raise InvalidTokenError()

        if self.scope_insufficient(token.get("roles"), roles):
            raise InvalidTokenError()

        if self.scope_insufficient(token.get("entitlements"), entitlements):
            raise InvalidTokenError()

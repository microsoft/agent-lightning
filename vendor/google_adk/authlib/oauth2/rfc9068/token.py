import time
from typing import Optional
from typing import Union

from authlib.common.security import generate_token
from authlib.jose import jwt
from authlib.oauth2.rfc6750.token import BearerTokenGenerator


class JWTBearerTokenGenerator(BearerTokenGenerator):
    r"""A JWT formatted access token generator.

    :param issuer: The issuer identifier. Will appear in the JWT ``iss`` claim.

    :param \\*\\*kwargs: Other parameters are inherited from
        :class:`~authlib.oauth2.rfc6750.token.BearerTokenGenerator`.

    This token generator can be registered into the authorization server::

        class MyJWTBearerTokenGenerator(JWTBearerTokenGenerator):
            def get_jwks(self): ...

            def get_extra_claims(self, client, grant_type, user, scope): ...


        authorization_server.register_token_generator(
            "default",
            MyJWTBearerTokenGenerator(
                issuer="https://authorization-server.example.org"
            ),
        )
    """

    def __init__(
        self,
        issuer,
        alg="RS256",
        refresh_token_generator=None,
        expires_generator=None,
    ):
        super().__init__(
            self.access_token_generator, refresh_token_generator, expires_generator
        )
        self.issuer = issuer
        self.alg = alg

    def get_jwks(self):
        """Return the JWKs that will be used to sign the JWT access token.
        Developers MUST re-implement this method::

            def get_jwks(self):
                return load_jwks("jwks.json")
        """
        raise NotImplementedError()

    def get_extra_claims(self, client, grant_type, user, scope):
        """Return extra claims to add in the JWT access token. Developers MAY
        re-implement this method to add identity claims like the ones in
        :ref:`specs/oidc` ID Token, or any other arbitrary claims::

            def get_extra_claims(self, client, grant_type, user, scope):
                return generate_user_info(user, scope)
        """
        return {}

    def get_audiences(self, client, user, scope) -> Union[str, list[str]]:
        """Return the audience for the token. By default this simply returns
        the client ID. Developers MAY re-implement this method to add extra
        audiences::

            def get_audiences(self, client, user, scope):
                return [
                    client.get_client_id(),
                    resource_server.get_id(),
                ]
        """
        return client.get_client_id()

    def get_acr(self, user) -> Optional[str]:
        """Authentication Context Class Reference.
        Returns a user-defined case sensitive string indicating the class of
        authentication the used performed. Token audience may refuse to give access to
        some resources if some ACR criteria are not met.
        :ref:`specs/oidc` defines one special value: ``0`` means that the user
        authentication did not respect `ISO29115`_ level 1, and will be refused monetary
        operations. Developers MAY re-implement this method::

            def get_acr(self, user):
                if user.insecure_session():
                    return "0"
                return "urn:mace:incommon:iap:silver"

        .. _ISO29115: https://www.iso.org/standard/45138.html
        """
        return None

    def get_auth_time(self, user) -> Optional[int]:
        """User authentication time.
        Time when the End-User authentication occurred. Its value is a JSON number
        representing the number of seconds from 1970-01-01T0:0:0Z as measured in UTC
        until the date/time. Developers MAY re-implement this method::

            def get_auth_time(self, user):
                return datetime.timestamp(user.get_auth_time())
        """
        return None

    def get_amr(self, user) -> Optional[list[str]]:
        """Authentication Methods References.
        Defined by :ref:`specs/oidc` as an option list of user-defined case-sensitive
        strings indication which authentication methods have been used to authenticate
        the user. Developers MAY re-implement this method::

            def get_amr(self, user):
                return ["2FA"] if user.has_2fa_enabled() else []
        """
        return None

    def get_jti(self, client, grant_type, user, scope) -> str:
        """JWT ID.
        Create an unique identifier for the token. Developers MAY re-implement
        this method::

            def get_jti(self, client, grant_type, user scope):
                return generate_random_string(16)
        """
        return generate_token(16)

    def access_token_generator(self, client, grant_type, user, scope):
        now = int(time.time())
        expires_in = now + self._get_expires_in(client, grant_type)

        token_data = {
            "iss": self.issuer,
            "exp": expires_in,
            "client_id": client.get_client_id(),
            "iat": now,
            "jti": self.get_jti(client, grant_type, user, scope),
            "scope": scope,
        }

        # In cases of access tokens obtained through grants where a resource owner is
        # involved, such as the authorization code grant, the value of 'sub' SHOULD
        # correspond to the subject identifier of the resource owner.

        if user:
            token_data["sub"] = user.get_user_id()

        # In cases of access tokens obtained through grants where no resource owner is
        # involved, such as the client credentials grant, the value of 'sub' SHOULD
        # correspond to an identifier the authorization server uses to indicate the
        # client application.

        else:
            token_data["sub"] = client.get_client_id()

        # If the request includes a 'resource' parameter (as defined in [RFC8707]), the
        # resulting JWT access token 'aud' claim SHOULD have the same value as the
        # 'resource' parameter in the request.

        # TODO: Implement this with RFC8707
        if False:  # pragma: no cover
            ...

        # If the request does not include a 'resource' parameter, the authorization
        # server MUST use a default resource indicator in the 'aud' claim. If a 'scope'
        # parameter is present in the request, the authorization server SHOULD use it to
        # infer the value of the default resource indicator to be used in the 'aud'
        # claim. The mechanism through which scopes are associated with default resource
        # indicator values is outside the scope of this specification.

        else:
            token_data["aud"] = self.get_audiences(client, user, scope)

        # If the values in the 'scope' parameter refer to different default resource
        # indicator values, the authorization server SHOULD reject the request with
        # 'invalid_scope' as described in Section 4.1.2.1 of [RFC6749].
        # TODO: Implement this with RFC8707

        if auth_time := self.get_auth_time(user):
            token_data["auth_time"] = auth_time

        # The meaning and processing of acr Claim Values is out of scope for this
        # specification.

        if acr := self.get_acr(user):
            token_data["acr"] = acr

        # The definition of particular values to be used in the amr Claim is beyond the
        # scope of this specification.

        if amr := self.get_amr(user):
            token_data["amr"] = amr

        # Authorization servers MAY return arbitrary attributes not defined in any
        # existing specification, as long as the corresponding claim names are collision
        # resistant or the access tokens are meant to be used only within a private
        # subsystem. Please refer to Sections 4.2 and 4.3 of [RFC7519] for details.

        token_data.update(self.get_extra_claims(client, grant_type, user, scope))

        # This specification registers the 'application/at+jwt' media type, which can
        # be used to indicate that the content is a JWT access token. JWT access tokens
        # MUST include this media type in the 'typ' header parameter to explicitly
        # declare that the JWT represents an access token complying with this profile.
        # Per the definition of 'typ' in Section 4.1.9 of [RFC7515], it is RECOMMENDED
        # that the 'application/' prefix be omitted. Therefore, the 'typ' value used
        # SHOULD be 'at+jwt'.

        header = {"alg": self.alg, "typ": "at+jwt"}

        access_token = jwt.encode(
            header,
            token_data,
            key=self.get_jwks(),
            check=False,
        )
        return access_token.decode()

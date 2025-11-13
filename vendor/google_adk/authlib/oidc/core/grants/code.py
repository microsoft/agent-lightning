"""authlib.oidc.core.grants.code.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of Authentication using the Authorization Code Flow
per `Section 3.1`_.

.. _`Section 3.1`: http://openid.net/specs/openid-connect-core-1_0.html#CodeFlowAuth
"""

import logging

from authlib.oauth2.rfc6749 import OAuth2Request

from .util import generate_id_token
from .util import is_openid_scope
from .util import validate_nonce
from .util import validate_request_prompt

log = logging.getLogger(__name__)


class OpenIDToken:
    def get_jwt_config(self, grant):  # pragma: no cover
        """Get the JWT configuration for OpenIDCode extension. The JWT
        configuration will be used to generate ``id_token``.
        If ``alg`` is undefined, the ``id_token_signed_response_alg`` client
        metadata will be used. By default ``RS256`` will be used.
        If ``key`` is undefined, the ``jwks_uri`` or ``jwks`` client metadata
        will be used.
        Developers MUST implement this method in subclass, e.g.::

            def get_jwt_config(self, grant):
                return {
                    "key": read_private_key_file(key_path),
                    "alg": "RS256",
                    "iss": "issuer-identity",
                    "exp": 3600,
                }

        :param grant: AuthorizationCodeGrant instance
        :return: dict
        """
        raise NotImplementedError()

    def generate_user_info(self, user, scope):
        """Provide user information for the given scope. Developers
        MUST implement this method in subclass, e.g.::

            from authlib.oidc.core import UserInfo


            def generate_user_info(self, user, scope):
                user_info = UserInfo(sub=user.id, name=user.name)
                if "email" in scope:
                    user_info["email"] = user.email
                return user_info

        :param user: user instance
        :param scope: scope of the token
        :return: ``authlib.oidc.core.UserInfo`` instance
        """
        raise NotImplementedError()

    def get_audiences(self, request):
        """Parse `aud` value for id_token, default value is client id. Developers
        MAY rewrite this method to provide a customized audience value.
        """
        client = request.client
        return [client.get_client_id()]

    def process_token(self, grant, response):
        _, token, _ = response
        scope = token.get("scope")
        if not scope or not is_openid_scope(scope):
            # standard authorization code flow
            return token

        request: OAuth2Request = grant.request
        authorization_code = request.authorization_code

        config = self.get_jwt_config(grant)
        config["aud"] = self.get_audiences(request)

        # Per OpenID Connect Registration 1.0 Section 2:
        # Use client's id_token_signed_response_alg if specified
        if not config.get("alg") and (
            client_alg := request.client.id_token_signed_response_alg
        ):
            config["alg"] = client_alg

        if authorization_code:
            config["nonce"] = authorization_code.get_nonce()
            config["auth_time"] = authorization_code.get_auth_time()

            if acr := authorization_code.get_acr():
                config["acr"] = acr

            if amr := authorization_code.get_amr():
                config["amr"] = amr

        user_info = self.generate_user_info(request.user, token["scope"])
        id_token = generate_id_token(token, user_info, **config)
        token["id_token"] = id_token
        return token

    def __call__(self, grant):
        grant.register_hook("after_create_token_response", self.process_token)


class OpenIDCode(OpenIDToken):
    """An extension from OpenID Connect for "grant_type=code" request. Developers
    MUST implement the missing methods::

        class MyOpenIDCode(OpenIDCode):
            def get_jwt_config(self, grant):
                return {...}

            def exists_nonce(self, nonce, request):
                return check_if_nonce_in_cache(request.payload.client_id, nonce)

            def generate_user_info(self, user, scope):
                return {...}

    The register this extension with AuthorizationCodeGrant::

        authorization_server.register_grant(
            AuthorizationCodeGrant, extensions=[MyOpenIDCode()]
        )
    """

    def __init__(self, require_nonce=False):
        self.require_nonce = require_nonce

    def exists_nonce(self, nonce, request):
        """Check if the given nonce is existing in your database. Developers
        MUST implement this method in subclass, e.g.::

            def exists_nonce(self, nonce, request):
                exists = AuthorizationCode.query.filter_by(
                    client_id=request.payload.client_id, nonce=nonce
                ).first()
                return bool(exists)

        :param nonce: A string of "nonce" parameter in request
        :param request: OAuth2Request instance
        :return: Boolean
        """
        raise NotImplementedError()

    def validate_openid_authorization_request(self, grant, redirect_uri):
        validate_nonce(grant.request, self.exists_nonce, self.require_nonce)

    def __call__(self, grant):
        grant.register_hook("after_create_token_response", self.process_token)
        if is_openid_scope(grant.request.payload.scope):
            grant.register_hook(
                "after_validate_authorization_request_payload",
                self.validate_openid_authorization_request,
            )
            grant.register_hook(
                "after_validate_consent_request", validate_request_prompt
            )

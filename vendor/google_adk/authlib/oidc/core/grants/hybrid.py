import logging

from authlib.common.security import generate_token
from authlib.oauth2.rfc6749 import InvalidScopeError
from authlib.oauth2.rfc6749.grants.authorization_code import (
    validate_code_authorization_request,
)

from .implicit import OpenIDImplicitGrant
from .util import is_openid_scope
from .util import validate_nonce

log = logging.getLogger(__name__)


class OpenIDHybridGrant(OpenIDImplicitGrant):
    #: Generated "code" length
    AUTHORIZATION_CODE_LENGTH = 48

    RESPONSE_TYPES = {"code id_token", "code token", "code id_token token"}
    GRANT_TYPE = "code"
    DEFAULT_RESPONSE_MODE = "fragment"

    def generate_authorization_code(self):
        """ "The method to generate "code" value for authorization code data.
        Developers may rewrite this method, or customize the code length with::

            class MyAuthorizationCodeGrant(AuthorizationCodeGrant):
                AUTHORIZATION_CODE_LENGTH = 32  # default is 48
        """
        return generate_token(self.AUTHORIZATION_CODE_LENGTH)

    def save_authorization_code(self, code, request):
        """Save authorization_code for later use. Developers MUST implement
        it in subclass. Here is an example::

            def save_authorization_code(self, code, request):
                client = request.client
                auth_code = AuthorizationCode(
                    code=code,
                    client_id=client.client_id,
                    redirect_uri=request.payload.redirect_uri,
                    scope=request.payload.scope,
                    nonce=request.payload.data.get("nonce"),
                    user_id=request.user.id,
                )
                auth_code.save()
        """
        raise NotImplementedError()

    def validate_authorization_request(self):
        if not is_openid_scope(self.request.payload.scope):
            raise InvalidScopeError(
                "Missing 'openid' scope",
                redirect_uri=self.request.payload.redirect_uri,
                redirect_fragment=True,
            )
        self.register_hook(
            "after_validate_authorization_request_payload",
            lambda grant, redirect_uri: validate_nonce(
                grant.request, grant.exists_nonce, required=True
            ),
        )
        return validate_code_authorization_request(self)

    def create_granted_params(self, grant_user):
        self.request.user = grant_user
        client = self.request.client
        code = self.generate_authorization_code()
        self.save_authorization_code(code, self.request)
        params = [("code", code)]
        token = self.generate_token(
            grant_type="implicit",
            user=grant_user,
            scope=self.request.payload.scope,
            include_refresh_token=False,
        )

        response_types = self.request.payload.response_type.split()
        if "token" in response_types:
            log.debug("Grant token %r to %r", token, client)
            self.server.save_token(token, self.request)
            if "id_token" in response_types:
                token = self.process_implicit_token(token, code)
        else:
            # response_type is "code id_token"
            token = {"expires_in": token["expires_in"], "scope": token["scope"]}
            token = self.process_implicit_token(token, code)

        params.extend([(k, token[k]) for k in token])
        return params

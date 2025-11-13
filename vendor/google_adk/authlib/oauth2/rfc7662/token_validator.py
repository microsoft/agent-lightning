from ..rfc6749 import TokenValidator
from ..rfc6750 import InsufficientScopeError
from ..rfc6750 import InvalidTokenError


class IntrospectTokenValidator(TokenValidator):
    TOKEN_TYPE = "bearer"

    def introspect_token(self, token_string):
        """Request introspection token endpoint with the given token string,
        authorization server will return token information in JSON format.
        Developers MUST implement this method before using it::

            def introspect_token(self, token_string):
                # for example, introspection token endpoint has limited
                # internal IPs to access, so there is no need to add
                # authentication.
                url = "https://example.com/oauth/introspect"
                resp = requests.post(url, data={"token": token_string})
                resp.raise_for_status()
                return resp.json()
        """
        raise NotImplementedError()

    def authenticate_token(self, token_string):
        return self.introspect_token(token_string)

    def validate_token(self, token, scopes, request):
        if not token or not token["active"]:
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            )
        if self.scope_insufficient(token.get("scope"), scopes):
            raise InsufficientScopeError()

"""authlib.oauth2.rfc6750.validator.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate Bearer Token for in request, scope and token.
"""

from ..rfc6749 import TokenValidator
from .errors import InsufficientScopeError
from .errors import InvalidTokenError


class BearerTokenValidator(TokenValidator):
    TOKEN_TYPE = "bearer"

    def authenticate_token(self, token_string):
        """A method to query token from database with the given token string.
        Developers MUST re-implement this method. For instance::

            def authenticate_token(self, token_string):
                return get_token_from_database(token_string)

        :param token_string: A string to represent the access_token.
        :return: token
        """
        raise NotImplementedError()

    def validate_token(self, token, scopes, request):
        """Check if token is active and matches the requested scopes."""
        if not token:
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            )
        if token.is_expired():
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            )
        if token.is_revoked():
            raise InvalidTokenError(
                realm=self.realm, extra_attributes=self.extra_attributes
            )
        if self.scope_insufficient(token.get_scope(), scopes):
            raise InsufficientScopeError()

"""authlib.oauth2.rfc6749.resource_protector.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of Accessing Protected Resources per `Section 7`_.

.. _`Section 7`: https://tools.ietf.org/html/rfc6749#section-7
"""

from .errors import MissingAuthorizationError
from .errors import UnsupportedTokenTypeError
from .util import scope_to_list


class TokenValidator:
    """Base token validator class. Subclass this validator to register
    into ResourceProtector instance.
    """

    TOKEN_TYPE = "bearer"

    def __init__(self, realm=None, **extra_attributes):
        self.realm = realm
        self.extra_attributes = extra_attributes

    @staticmethod
    def scope_insufficient(token_scopes, required_scopes):
        if not required_scopes:
            return False

        token_scopes = scope_to_list(token_scopes)
        if not token_scopes:
            return True

        token_scopes = set(token_scopes)
        for scope in required_scopes:
            resource_scopes = set(scope_to_list(scope))
            if token_scopes.issuperset(resource_scopes):
                return False

        return True

    def authenticate_token(self, token_string):
        """A method to query token from database with the given token string.
        Developers MUST re-implement this method. For instance::

            def authenticate_token(self, token_string):
                return get_token_from_database(token_string)

        :param token_string: A string to represent the access_token.
        :return: token
        """
        raise NotImplementedError()

    def validate_request(self, request):
        """A method to validate if the HTTP request is valid or not. Developers MUST
        re-implement this method.  For instance, your server requires a
        "X-Device-Version" in the header::

            def validate_request(self, request):
                if "X-Device-Version" not in request.headers:
                    raise InvalidRequestError()

        Usually, you don't have to detect if the request is valid or not. If you have
        to, you MUST re-implement this method.

        :param request: instance of HttpRequest
        :raise: InvalidRequestError
        """

    def validate_token(self, token, scopes, request):
        """A method to validate if the authorized token is valid, if it has the
        permission on the given scopes. Developers MUST re-implement this method.
        e.g, check if token is expired, revoked::

            def validate_token(self, token, scopes, request):
                if not token:
                    raise InvalidTokenError()
                if token.is_expired() or token.is_revoked():
                    raise InvalidTokenError()
                if not match_token_scopes(token, scopes):
                    raise InsufficientScopeError()
        """
        raise NotImplementedError()


class ResourceProtector:
    def __init__(self):
        self._token_validators = {}
        self._default_realm = None
        self._default_auth_type = None

    def register_token_validator(self, validator: TokenValidator):
        """Register a token validator for a given Authorization type.
        Authlib has a built-in BearerTokenValidator per rfc6750.
        """
        if not self._default_auth_type:
            self._default_realm = validator.realm
            self._default_auth_type = validator.TOKEN_TYPE

        if validator.TOKEN_TYPE not in self._token_validators:
            self._token_validators[validator.TOKEN_TYPE] = validator

    def get_token_validator(self, token_type):
        """Get token validator from registry for the given token type."""
        validator = self._token_validators.get(token_type.lower())
        if not validator:
            raise UnsupportedTokenTypeError(
                self._default_auth_type, self._default_realm
            )
        return validator

    def parse_request_authorization(self, request):
        """Parse the token and token validator from request Authorization header.
        Here is an example of Authorization header::

            Authorization: Bearer a-token-string

        This method will parse this header, if it can find the validator for
        ``Bearer``, it will return the validator and ``a-token-string``.

        :return: validator, token_string
        :raise: MissingAuthorizationError
        :raise: UnsupportedTokenTypeError
        """
        auth = request.headers.get("Authorization")
        if not auth:
            raise MissingAuthorizationError(
                self._default_auth_type, self._default_realm
            )

        # https://tools.ietf.org/html/rfc6749#section-7.1
        token_parts = auth.split(None, 1)
        if len(token_parts) != 2:
            raise UnsupportedTokenTypeError(
                self._default_auth_type, self._default_realm
            )

        token_type, token_string = token_parts
        validator = self.get_token_validator(token_type)
        return validator, token_string

    def validate_request(self, scopes, request, **kwargs):
        """Validate the request and return a token."""
        validator, token_string = self.parse_request_authorization(request)
        validator.validate_request(request)
        token = validator.authenticate_token(token_string)
        validator.validate_token(token, scopes, request, **kwargs)
        return token

import hashlib
import re

from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64encode

from ..rfc6749 import InvalidGrantError
from ..rfc6749 import InvalidRequestError
from ..rfc6749 import OAuth2Request

CODE_VERIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9\-._~]{43,128}$")
CODE_CHALLENGE_PATTERN = re.compile(r"^[a-zA-Z0-9\-._~]{43,128}$")


def create_s256_code_challenge(code_verifier):
    """Create S256 code_challenge with the given code_verifier."""
    data = hashlib.sha256(to_bytes(code_verifier, "ascii")).digest()
    return to_unicode(urlsafe_b64encode(data))


def compare_plain_code_challenge(code_verifier, code_challenge):
    # If the "code_challenge_method" from Section 4.3 was "plain",
    # they are compared directly
    return code_verifier == code_challenge


def compare_s256_code_challenge(code_verifier, code_challenge):
    # BASE64URL-ENCODE(SHA256(ASCII(code_verifier))) == code_challenge
    return create_s256_code_challenge(code_verifier) == code_challenge


class CodeChallenge:
    """CodeChallenge extension to Authorization Code Grant. It is used to
    improve the security of Authorization Code flow for public clients by
    sending extra "code_challenge" and "code_verifier" to the authorization
    server.

    The AuthorizationCodeGrant SHOULD save the ``code_challenge`` and
    ``code_challenge_method`` into database when ``save_authorization_code``.
    Then register this extension via::

        server.register_grant(AuthorizationCodeGrant, [CodeChallenge(required=True)])
    """

    #: defaults to "plain" if not present in the request
    DEFAULT_CODE_CHALLENGE_METHOD = "plain"
    #: supported ``code_challenge_method``
    SUPPORTED_CODE_CHALLENGE_METHOD = ["plain", "S256"]

    CODE_CHALLENGE_METHODS = {
        "plain": compare_plain_code_challenge,
        "S256": compare_s256_code_challenge,
    }

    def __init__(self, required=True):
        self.required = required

    def __call__(self, grant):
        grant.register_hook(
            "after_validate_authorization_request_payload",
            self.validate_code_challenge,
        )
        grant.register_hook(
            "after_validate_token_request",
            self.validate_code_verifier,
        )

    def validate_code_challenge(self, grant, redirect_uri):
        request: OAuth2Request = grant.request
        challenge = request.payload.data.get("code_challenge")
        method = request.payload.data.get("code_challenge_method")
        if not challenge and not method:
            return

        if not challenge:
            raise InvalidRequestError("Missing 'code_challenge'")

        if len(request.payload.datalist.get("code_challenge", [])) > 1:
            raise InvalidRequestError("Multiple 'code_challenge' in request.")

        if not CODE_CHALLENGE_PATTERN.match(challenge):
            raise InvalidRequestError("Invalid 'code_challenge'")

        if method and method not in self.SUPPORTED_CODE_CHALLENGE_METHOD:
            raise InvalidRequestError("Unsupported 'code_challenge_method'")

        if len(request.payload.datalist.get("code_challenge_method", [])) > 1:
            raise InvalidRequestError("Multiple 'code_challenge_method' in request.")

    def validate_code_verifier(self, grant, result):
        request: OAuth2Request = grant.request
        verifier = request.form.get("code_verifier")

        # public client MUST verify code challenge
        if self.required and request.auth_method == "none" and not verifier:
            raise InvalidRequestError("Missing 'code_verifier'")

        authorization_code = request.authorization_code
        challenge = self.get_authorization_code_challenge(authorization_code)

        # ignore, it is the normal RFC6749 authorization_code request
        if not challenge and not verifier:
            return

        # challenge exists, code_verifier is required
        if not verifier:
            raise InvalidRequestError("Missing 'code_verifier'")

        if not CODE_VERIFIER_PATTERN.match(verifier):
            raise InvalidRequestError("Invalid 'code_verifier'")

        # 4.6. Server Verifies code_verifier before Returning the Tokens
        method = self.get_authorization_code_challenge_method(authorization_code)
        if method is None:
            method = self.DEFAULT_CODE_CHALLENGE_METHOD

        func = self.CODE_CHALLENGE_METHODS.get(method)
        if not func:
            raise RuntimeError(f"No verify method for '{method}'")

        # If the values are not equal, an error response indicating
        # "invalid_grant" MUST be returned.
        if not func(verifier, challenge):
            raise InvalidGrantError(description="Code challenge failed.")

    def get_authorization_code_challenge(self, authorization_code):
        """Get "code_challenge" associated with this authorization code.
        Developers MAY re-implement it in subclass, the default logic::

            def get_authorization_code_challenge(self, authorization_code):
                return authorization_code.code_challenge

        :param authorization_code: the instance of authorization_code
        """
        return authorization_code.code_challenge

    def get_authorization_code_challenge_method(self, authorization_code):
        """Get "code_challenge_method" associated with this authorization code.
        Developers MAY re-implement it in subclass, the default logic::

            def get_authorization_code_challenge_method(self, authorization_code):
                return authorization_code.code_challenge_method

        :param authorization_code: the instance of authorization_code
        """
        return authorization_code.code_challenge_method

import time

from .errors import InvalidNonceError
from .errors import InvalidRequestError
from .errors import InvalidSignatureError
from .errors import MissingRequiredParameterError
from .errors import UnsupportedSignatureMethodError
from .signature import SIGNATURE_HMAC_SHA1
from .signature import SIGNATURE_PLAINTEXT
from .signature import SIGNATURE_RSA_SHA1
from .signature import verify_hmac_sha1
from .signature import verify_plaintext
from .signature import verify_rsa_sha1


class BaseServer:
    SIGNATURE_METHODS = {
        SIGNATURE_HMAC_SHA1: verify_hmac_sha1,
        SIGNATURE_RSA_SHA1: verify_rsa_sha1,
        SIGNATURE_PLAINTEXT: verify_plaintext,
    }
    SUPPORTED_SIGNATURE_METHODS = [SIGNATURE_HMAC_SHA1]
    EXPIRY_TIME = 300

    @classmethod
    def register_signature_method(cls, name, verify):
        """Extend signature method verification.

        :param name: A string to represent signature method.
        :param verify: A function to verify signature.

        The ``verify`` method accept ``OAuth1Request`` as parameter::

            def verify_custom_method(request):
                # verify this request, return True or False
                return True


            Server.register_signature_method("custom-name", verify_custom_method)
        """
        cls.SIGNATURE_METHODS[name] = verify

    def validate_timestamp_and_nonce(self, request):
        """Validate ``oauth_timestamp`` and ``oauth_nonce`` in HTTP request.

        :param request: OAuth1Request instance
        """
        timestamp = request.oauth_params.get("oauth_timestamp")
        nonce = request.oauth_params.get("oauth_nonce")

        if request.signature_method == SIGNATURE_PLAINTEXT:
            # The parameters MAY be omitted when using the "PLAINTEXT"
            # signature method
            if not timestamp and not nonce:
                return

        if not timestamp:
            raise MissingRequiredParameterError("oauth_timestamp")

        try:
            # The timestamp value MUST be a positive integer
            timestamp = int(timestamp)
            if timestamp < 0:
                raise InvalidRequestError('Invalid "oauth_timestamp" value')

            if self.EXPIRY_TIME and time.time() - timestamp > self.EXPIRY_TIME:
                raise InvalidRequestError('Invalid "oauth_timestamp" value')
        except (ValueError, TypeError) as exc:
            raise InvalidRequestError('Invalid "oauth_timestamp" value') from exc

        if not nonce:
            raise MissingRequiredParameterError("oauth_nonce")

        if self.exists_nonce(nonce, request):
            raise InvalidNonceError()

    def validate_oauth_signature(self, request):
        """Validate ``oauth_signature`` from HTTP request.

        :param request: OAuth1Request instance
        """
        method = request.signature_method
        if not method:
            raise MissingRequiredParameterError("oauth_signature_method")

        if method not in self.SUPPORTED_SIGNATURE_METHODS:
            raise UnsupportedSignatureMethodError()

        if not request.signature:
            raise MissingRequiredParameterError("oauth_signature")

        verify = self.SIGNATURE_METHODS.get(method)
        if not verify:
            raise UnsupportedSignatureMethodError()

        if not verify(request):
            raise InvalidSignatureError()

    def get_client_by_id(self, client_id):
        """Get client instance with the given ``client_id``.

        :param client_id: A string of client_id
        :return: Client instance
        """
        raise NotImplementedError()

    def exists_nonce(self, nonce, request):
        """The nonce value MUST be unique across all requests with the same
        timestamp, client credentials, and token combinations.

        :param nonce: A string value of ``oauth_nonce``
        :param request: OAuth1Request instance
        :return: Boolean
        """
        raise NotImplementedError()

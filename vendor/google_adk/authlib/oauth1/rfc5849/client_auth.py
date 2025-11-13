import base64
import hashlib
import time

from authlib.common.encoding import to_native
from authlib.common.security import generate_token
from authlib.common.urls import extract_params

from .parameters import prepare_form_encoded_body
from .parameters import prepare_headers
from .parameters import prepare_request_uri_query
from .signature import SIGNATURE_HMAC_SHA1
from .signature import SIGNATURE_PLAINTEXT
from .signature import SIGNATURE_RSA_SHA1
from .signature import SIGNATURE_TYPE_BODY
from .signature import SIGNATURE_TYPE_HEADER
from .signature import SIGNATURE_TYPE_QUERY
from .signature import sign_hmac_sha1
from .signature import sign_plaintext
from .signature import sign_rsa_sha1
from .wrapper import OAuth1Request

CONTENT_TYPE_FORM_URLENCODED = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART = "multipart/form-data"


class ClientAuth:
    SIGNATURE_METHODS = {
        SIGNATURE_HMAC_SHA1: sign_hmac_sha1,
        SIGNATURE_RSA_SHA1: sign_rsa_sha1,
        SIGNATURE_PLAINTEXT: sign_plaintext,
    }

    @classmethod
    def register_signature_method(cls, name, sign):
        """Extend client signature methods.

        :param name: A string to represent signature method.
        :param sign: A function to generate signature.

        The ``sign`` method accept 2 parameters::

            def custom_sign_method(client, request):
                # client is the instance of Client.
                return "your-signed-string"


            Client.register_signature_method("custom-name", custom_sign_method)
        """
        cls.SIGNATURE_METHODS[name] = sign

    def __init__(
        self,
        client_id,
        client_secret=None,
        token=None,
        token_secret=None,
        redirect_uri=None,
        rsa_key=None,
        verifier=None,
        signature_method=SIGNATURE_HMAC_SHA1,
        signature_type=SIGNATURE_TYPE_HEADER,
        realm=None,
        force_include_body=False,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = token
        self.token_secret = token_secret
        self.redirect_uri = redirect_uri
        self.signature_method = signature_method
        self.signature_type = signature_type
        self.rsa_key = rsa_key
        self.verifier = verifier
        self.realm = realm
        self.force_include_body = force_include_body

    def get_oauth_signature(self, method, uri, headers, body):
        """Get an OAuth signature to be used in signing a request.

        To satisfy `section 3.4.1.2`_ item 2, if the request argument's
        headers dict attribute contains a Host item, its value will
        replace any netloc part of the request argument's uri attribute
        value.

        .. _`section 3.4.1.2`: https://tools.ietf.org/html/rfc5849#section-3.4.1.2
        """
        sign = self.SIGNATURE_METHODS.get(self.signature_method)
        if not sign:
            raise ValueError("Invalid signature method.")

        request = OAuth1Request(method, uri, body=body, headers=headers)
        return sign(self, request)

    def get_oauth_params(self, nonce, timestamp):
        oauth_params = [
            ("oauth_nonce", nonce),
            ("oauth_timestamp", timestamp),
            ("oauth_version", "1.0"),
            ("oauth_signature_method", self.signature_method),
            ("oauth_consumer_key", self.client_id),
        ]
        if self.token:
            oauth_params.append(("oauth_token", self.token))
        if self.redirect_uri:
            oauth_params.append(("oauth_callback", self.redirect_uri))
        if self.verifier:
            oauth_params.append(("oauth_verifier", self.verifier))
        return oauth_params

    def _render(self, uri, headers, body, oauth_params):
        if self.signature_type == SIGNATURE_TYPE_HEADER:
            headers = prepare_headers(oauth_params, headers, realm=self.realm)
        elif self.signature_type == SIGNATURE_TYPE_BODY:
            if CONTENT_TYPE_FORM_URLENCODED in headers.get("Content-Type", ""):
                decoded_body = extract_params(body) or []
                body = prepare_form_encoded_body(oauth_params, decoded_body)
                headers["Content-Type"] = CONTENT_TYPE_FORM_URLENCODED
        elif self.signature_type == SIGNATURE_TYPE_QUERY:
            uri = prepare_request_uri_query(oauth_params, uri)
        else:
            raise ValueError("Unknown signature type specified.")
        return uri, headers, body

    def sign(self, method, uri, headers, body):
        """Sign the HTTP request, add OAuth parameters and signature.

        :param method: HTTP method of the request.
        :param uri:  URI of the HTTP request.
        :param body: Body payload of the HTTP request.
        :param headers: Headers of the HTTP request.
        :return: uri, headers, body
        """
        nonce = generate_nonce()
        timestamp = generate_timestamp()
        if body is None:
            body = b""

        # transform int to str
        timestamp = str(timestamp)

        if headers is None:
            headers = {}

        oauth_params = self.get_oauth_params(nonce, timestamp)

        # https://datatracker.ietf.org/doc/html/draft-eaton-oauth-bodyhash-00.html
        # include oauth_body_hash
        if body and headers.get("Content-Type") != CONTENT_TYPE_FORM_URLENCODED:
            oauth_body_hash = base64.b64encode(hashlib.sha1(body).digest())
            oauth_params.append(("oauth_body_hash", oauth_body_hash.decode("utf-8")))

        uri, headers, body = self._render(uri, headers, body, oauth_params)

        sig = self.get_oauth_signature(method, uri, headers, body)
        oauth_params.append(("oauth_signature", sig))

        uri, headers, body = self._render(uri, headers, body, oauth_params)
        return uri, headers, body

    def prepare(self, method, uri, headers, body):
        """Add OAuth parameters to the request.

        Parameters may be included from the body if the content-type is
        urlencoded, if no content type is set, a guess is made.
        """
        content_type = to_native(headers.get("Content-Type", ""))
        if self.signature_type == SIGNATURE_TYPE_BODY:
            content_type = CONTENT_TYPE_FORM_URLENCODED
        elif not content_type and extract_params(body):
            content_type = CONTENT_TYPE_FORM_URLENCODED

        if CONTENT_TYPE_FORM_URLENCODED in content_type:
            headers["Content-Type"] = CONTENT_TYPE_FORM_URLENCODED
            uri, headers, body = self.sign(method, uri, headers, body)
        elif self.force_include_body:
            # To allow custom clients to work on non form encoded bodies.
            uri, headers, body = self.sign(method, uri, headers, body)
        else:
            # Omit body data in the signing of non form-encoded requests
            uri, headers, _ = self.sign(method, uri, headers, b"")
            body = b""
        return uri, headers, body


def generate_nonce():
    return generate_token()


def generate_timestamp():
    return str(int(time.time()))

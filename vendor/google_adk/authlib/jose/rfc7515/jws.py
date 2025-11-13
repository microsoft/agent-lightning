from authlib.common.encoding import json_b64encode
from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64encode
from authlib.jose.errors import BadSignatureError
from authlib.jose.errors import DecodeError
from authlib.jose.errors import InvalidCritHeaderParameterNameError
from authlib.jose.errors import InvalidHeaderParameterNameError
from authlib.jose.errors import MissingAlgorithmError
from authlib.jose.errors import UnsupportedAlgorithmError
from authlib.jose.util import ensure_dict
from authlib.jose.util import extract_header
from authlib.jose.util import extract_segment

from .models import JWSHeader
from .models import JWSObject


class JsonWebSignature:
    #: Registered Header Parameter Names defined by Section 4.1
    REGISTERED_HEADER_PARAMETER_NAMES = frozenset(
        [
            "alg",
            "jku",
            "jwk",
            "kid",
            "x5u",
            "x5c",
            "x5t",
            "x5t#S256",
            "typ",
            "cty",
            "crit",
        ]
    )

    MAX_CONTENT_LENGTH: int = 256000

    #: Defined available JWS algorithms in the registry
    ALGORITHMS_REGISTRY = {}

    def __init__(self, algorithms=None, private_headers=None):
        self._private_headers = private_headers
        self._algorithms = algorithms

    @classmethod
    def register_algorithm(cls, algorithm):
        if not algorithm or algorithm.algorithm_type != "JWS":
            raise ValueError(f"Invalid algorithm for JWS, {algorithm!r}")
        cls.ALGORITHMS_REGISTRY[algorithm.name] = algorithm

    def serialize_compact(self, protected, payload, key):
        """Generate a JWS Compact Serialization. The JWS Compact Serialization
        represents digitally signed or MACed content as a compact, URL-safe
        string, per `Section 7.1`_.

        .. code-block:: text

            BASE64URL(UTF8(JWS Protected Header)) || '.' ||
            BASE64URL(JWS Payload) || '.' ||
            BASE64URL(JWS Signature)

        :param protected: A dict of protected header
        :param payload: A bytes/string of payload
        :param key: Private key used to generate signature
        :return: byte
        """
        jws_header = JWSHeader(protected, None)
        self._validate_private_headers(protected)
        self._validate_crit_headers(protected)
        algorithm, key = self._prepare_algorithm_key(protected, payload, key)

        protected_segment = json_b64encode(jws_header.protected)
        payload_segment = urlsafe_b64encode(to_bytes(payload))

        # calculate signature
        signing_input = b".".join([protected_segment, payload_segment])
        signature = urlsafe_b64encode(algorithm.sign(signing_input, key))
        return b".".join([protected_segment, payload_segment, signature])

    def deserialize_compact(self, s, key, decode=None):
        """Exact JWS Compact Serialization, and validate with the given key.
        If key is not provided, the returned dict will contain the signature,
        and signing input values. Via `Section 7.1`_.

        :param s: text of JWS Compact Serialization
        :param key: key used to verify the signature
        :param decode: a function to decode payload data
        :return: JWSObject
        :raise: BadSignatureError

        .. _`Section 7.1`: https://tools.ietf.org/html/rfc7515#section-7.1
        """
        if len(s) > self.MAX_CONTENT_LENGTH:
            raise ValueError("Serialization is too long.")

        try:
            s = to_bytes(s)
            signing_input, signature_segment = s.rsplit(b".", 1)
            protected_segment, payload_segment = signing_input.split(b".", 1)
        except ValueError as exc:
            raise DecodeError("Not enough segments") from exc

        protected = _extract_header(protected_segment)
        self._validate_crit_headers(protected)
        jws_header = JWSHeader(protected, None)

        payload = _extract_payload(payload_segment)
        if decode:
            payload = decode(payload)

        signature = _extract_signature(signature_segment)
        rv = JWSObject(jws_header, payload, "compact")
        algorithm, key = self._prepare_algorithm_key(jws_header, payload, key)
        if algorithm.verify(signing_input, signature, key):
            return rv
        raise BadSignatureError(rv)

    def serialize_json(self, header_obj, payload, key):
        """Generate a JWS JSON Serialization. The JWS JSON Serialization
        represents digitally signed or MACed content as a JSON object,
        per `Section 7.2`_.

        :param header_obj: A dict/list of header
        :param payload: A string/dict of payload
        :param key: Private key used to generate signature
        :return: JWSObject

        Example ``header_obj`` of JWS JSON Serialization::

            {
                "protected: {"alg": "HS256"},
                "header": {"kid": "jose"}
            }

        Pass a dict to generate flattened JSON Serialization, pass a list of
        header dict to generate standard JSON Serialization.
        """
        payload_segment = json_b64encode(payload)

        def _sign(jws_header):
            self._validate_private_headers(jws_header)
            # RFC 7515 §4.1.11: 'crit' MUST be integrity-protected.
            # Reject if present in unprotected header, and validate only
            # against the protected header parameters.
            self._reject_unprotected_crit(jws_header.header)
            self._validate_crit_headers(jws_header.protected)
            _alg, _key = self._prepare_algorithm_key(jws_header, payload, key)

            protected_segment = json_b64encode(jws_header.protected)
            signing_input = b".".join([protected_segment, payload_segment])
            signature = urlsafe_b64encode(_alg.sign(signing_input, _key))

            rv = {
                "protected": to_unicode(protected_segment),
                "signature": to_unicode(signature),
            }
            if jws_header.header is not None:
                rv["header"] = jws_header.header
            return rv

        if isinstance(header_obj, dict):
            data = _sign(JWSHeader.from_dict(header_obj))
            data["payload"] = to_unicode(payload_segment)
            return data

        signatures = [_sign(JWSHeader.from_dict(h)) for h in header_obj]
        return {"payload": to_unicode(payload_segment), "signatures": signatures}

    def deserialize_json(self, obj, key, decode=None):
        """Exact JWS JSON Serialization, and validate with the given key.
        If key is not provided, it will return a dict without signature
        verification. Header will still be validated. Via `Section 7.2`_.

        :param obj: text of JWS JSON Serialization
        :param key: key used to verify the signature
        :param decode: a function to decode payload data
        :return: JWSObject
        :raise: BadSignatureError

        .. _`Section 7.2`: https://tools.ietf.org/html/rfc7515#section-7.2
        """
        obj = ensure_dict(obj, "JWS")

        payload_segment = obj.get("payload")
        if payload_segment is None:
            raise DecodeError('Missing "payload" value')

        payload_segment = to_bytes(payload_segment)
        payload = _extract_payload(payload_segment)
        if decode:
            payload = decode(payload)

        if "signatures" not in obj:
            # flattened JSON JWS
            jws_header, valid = self._validate_json_jws(
                payload_segment, payload, obj, key
            )

            rv = JWSObject(jws_header, payload, "flat")
            if valid:
                return rv
            raise BadSignatureError(rv)

        headers = []
        is_valid = True
        for header_obj in obj["signatures"]:
            jws_header, valid = self._validate_json_jws(
                payload_segment, payload, header_obj, key
            )
            headers.append(jws_header)
            if not valid:
                is_valid = False

        rv = JWSObject(headers, payload, "json")
        if is_valid:
            return rv
        raise BadSignatureError(rv)

    def serialize(self, header, payload, key):
        """Generate a JWS Serialization. It will automatically generate a
        Compact or JSON Serialization depending on the given header. If a
        header is in a JSON header format, it will call
        :meth:`serialize_json`, otherwise it will call
        :meth:`serialize_compact`.

        :param header: A dict/list of header
        :param payload: A string/dict of payload
        :param key: Private key used to generate signature
        :return: byte/dict
        """
        if isinstance(header, (list, tuple)):
            return self.serialize_json(header, payload, key)
        if "protected" in header:
            return self.serialize_json(header, payload, key)
        return self.serialize_compact(header, payload, key)

    def deserialize(self, s, key, decode=None):
        """Deserialize JWS Serialization, both compact and JSON format.
        It will automatically deserialize depending on the given JWS.

        :param s: text of JWS Compact/JSON Serialization
        :param key: key used to verify the signature
        :param decode: a function to decode payload data
        :return: dict
        :raise: BadSignatureError

        If key is not provided, it will still deserialize the serialization
        without verification.
        """
        if isinstance(s, dict):
            return self.deserialize_json(s, key, decode)

        s = to_bytes(s)
        if s.startswith(b"{") and s.endswith(b"}"):
            return self.deserialize_json(s, key, decode)
        return self.deserialize_compact(s, key, decode)

    def _prepare_algorithm_key(self, header, payload, key):
        if "alg" not in header:
            raise MissingAlgorithmError()

        alg = header["alg"]
        if self._algorithms is not None and alg not in self._algorithms:
            raise UnsupportedAlgorithmError()
        if alg not in self.ALGORITHMS_REGISTRY:
            raise UnsupportedAlgorithmError()

        algorithm = self.ALGORITHMS_REGISTRY[alg]
        if callable(key):
            key = key(header, payload)
        elif key is None and "jwk" in header:
            key = header["jwk"]
        key = algorithm.prepare_key(key)
        return algorithm, key

    def _validate_private_headers(self, header):
        # only validate private headers when developers set
        # private headers explicitly
        if self._private_headers is not None:
            names = self.REGISTERED_HEADER_PARAMETER_NAMES.copy()
            names = names.union(self._private_headers)

            for k in header:
                if k not in names:
                    raise InvalidHeaderParameterNameError(k)

    def _reject_unprotected_crit(self, unprotected_header):
        """Reject 'crit' when found in the unprotected header (RFC 7515 §4.1.11)."""
        if unprotected_header and "crit" in unprotected_header:
            raise InvalidHeaderParameterNameError("crit")

    def _validate_crit_headers(self, header):
        if "crit" in header:
            crit_headers = header["crit"]
            # Type enforcement for robustness and predictable errors
            if not isinstance(crit_headers, list) or not all(
                isinstance(x, str) for x in crit_headers
            ):
                raise InvalidHeaderParameterNameError("crit")
            names = self.REGISTERED_HEADER_PARAMETER_NAMES.copy()
            if self._private_headers:
                names = names.union(self._private_headers)
            for k in crit_headers:
                if k not in names:
                    raise InvalidCritHeaderParameterNameError(k)
                elif k not in header:
                    raise InvalidCritHeaderParameterNameError(k)

    def _validate_json_jws(self, payload_segment, payload, header_obj, key):
        protected_segment = header_obj.get("protected")
        if not protected_segment:
            raise DecodeError('Missing "protected" value')

        signature_segment = header_obj.get("signature")
        if not signature_segment:
            raise DecodeError('Missing "signature" value')

        protected_segment = to_bytes(protected_segment)
        protected = _extract_header(protected_segment)
        header = header_obj.get("header")
        if header and not isinstance(header, dict):
            raise DecodeError('Invalid "header" value')
        # RFC 7515 §4.1.11: 'crit' MUST be integrity-protected. If present in
        # the unprotected header object, reject the JWS.
        self._reject_unprotected_crit(header)

        # Enforce must-understand semantics for names listed in protected
        # 'crit'. This will also ensure each listed name is present in the
        # protected header.
        self._validate_crit_headers(protected)
        jws_header = JWSHeader(protected, header)
        algorithm, key = self._prepare_algorithm_key(jws_header, payload, key)
        signing_input = b".".join([protected_segment, payload_segment])
        signature = _extract_signature(to_bytes(signature_segment))
        if algorithm.verify(signing_input, signature, key):
            return jws_header, True
        return jws_header, False


def _extract_header(header_segment):
    return extract_header(header_segment, DecodeError)


def _extract_signature(signature_segment):
    return extract_segment(signature_segment, DecodeError, "signature")


def _extract_payload(payload_segment):
    return extract_segment(payload_segment, DecodeError, "payload")

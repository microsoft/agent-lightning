from collections import OrderedDict
from copy import deepcopy

from authlib.common.encoding import json_b64encode
from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64encode
from authlib.jose.errors import DecodeError
from authlib.jose.errors import InvalidAlgorithmForMultipleRecipientsMode
from authlib.jose.errors import InvalidHeaderParameterNameError
from authlib.jose.errors import KeyMismatchError
from authlib.jose.errors import MissingAlgorithmError
from authlib.jose.errors import MissingEncryptionAlgorithmError
from authlib.jose.errors import UnsupportedAlgorithmError
from authlib.jose.errors import UnsupportedCompressionAlgorithmError
from authlib.jose.errors import UnsupportedEncryptionAlgorithmError
from authlib.jose.rfc7516.models import JWEAlgorithmWithTagAwareKeyAgreement
from authlib.jose.rfc7516.models import JWEHeader
from authlib.jose.rfc7516.models import JWESharedHeader
from authlib.jose.util import ensure_dict
from authlib.jose.util import extract_header
from authlib.jose.util import extract_segment


class JsonWebEncryption:
    #: Registered Header Parameter Names defined by Section 4.1
    REGISTERED_HEADER_PARAMETER_NAMES = frozenset(
        [
            "alg",
            "enc",
            "zip",
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

    ALG_REGISTRY = {}
    ENC_REGISTRY = {}
    ZIP_REGISTRY = {}

    def __init__(self, algorithms=None, private_headers=None):
        self._algorithms = algorithms
        self._private_headers = private_headers

    @classmethod
    def register_algorithm(cls, algorithm):
        """Register an algorithm for ``alg`` or ``enc`` or ``zip`` of JWE."""
        if not algorithm or algorithm.algorithm_type != "JWE":
            raise ValueError(f"Invalid algorithm for JWE, {algorithm!r}")

        if algorithm.algorithm_location == "alg":
            cls.ALG_REGISTRY[algorithm.name] = algorithm
        elif algorithm.algorithm_location == "enc":
            cls.ENC_REGISTRY[algorithm.name] = algorithm
        elif algorithm.algorithm_location == "zip":
            cls.ZIP_REGISTRY[algorithm.name] = algorithm

    def serialize_compact(self, protected, payload, key, sender_key=None):
        """Generate a JWE Compact Serialization.

        The JWE Compact Serialization represents encrypted content as a compact,
        URL-safe string. This string is::

            BASE64URL(UTF8(JWE Protected Header)) || '.' ||
            BASE64URL(JWE Encrypted Key) || '.' ||
            BASE64URL(JWE Initialization Vector) || '.' ||
            BASE64URL(JWE Ciphertext) || '.' ||
            BASE64URL(JWE Authentication Tag)

        Only one recipient is supported by the JWE Compact Serialization and
        it provides no syntax to represent JWE Shared Unprotected Header, JWE
        Per-Recipient Unprotected Header, or JWE AAD values.

        :param protected: A dict of protected header
        :param payload: Payload (bytes or a value convertible to bytes)
        :param key: Public key used to encrypt payload
        :param sender_key: Sender's private key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: JWE compact serialization as bytes
        """
        # step 1: Prepare algorithms & key
        alg = self.get_header_alg(protected)
        enc = self.get_header_enc(protected)
        zip_alg = self.get_header_zip(protected)

        self._validate_sender_key(sender_key, alg)
        self._validate_private_headers(protected, alg)

        key = prepare_key(alg, protected, key)
        if sender_key is not None:
            sender_key = alg.prepare_key(sender_key)

        # self._post_validate_header(protected, algorithm)

        # step 2: Generate a random Content Encryption Key (CEK)
        # use enc_alg.generate_cek() in scope of upcoming .wrap
        # or .generate_keys_and_prepare_headers call

        # step 3: Encrypt the CEK with the recipient's public key
        if (
            isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement)
            and alg.key_size is not None
        ):
            # For a JWE algorithm with tag-aware key agreement in case key agreement
            # with key wrapping mode is used:
            # Defer key agreement with key wrapping until
            # authentication tag is computed
            prep = alg.generate_keys_and_prepare_headers(enc, key, sender_key)
            epk = prep["epk"]
            cek = prep["cek"]
            protected.update(prep["header"])
        else:
            # In any other case:
            # Keep the normal steps order defined by RFC 7516
            if isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement):
                wrapped = alg.wrap(enc, protected, key, sender_key)
            else:
                wrapped = alg.wrap(enc, protected, key)
            cek = wrapped["cek"]
            ek = wrapped["ek"]
            if "header" in wrapped:
                protected.update(wrapped["header"])

        # step 4: Generate a random JWE Initialization Vector
        iv = enc.generate_iv()

        # step 5: Let the Additional Authenticated Data encryption parameter
        # be ASCII(BASE64URL(UTF8(JWE Protected Header)))
        protected_segment = json_b64encode(protected)
        aad = to_bytes(protected_segment, "ascii")

        # step 6: compress message if required
        if zip_alg:
            msg = zip_alg.compress(to_bytes(payload))
        else:
            msg = to_bytes(payload)

        # step 7: perform encryption
        ciphertext, tag = enc.encrypt(msg, aad, iv, cek)

        if (
            isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement)
            and alg.key_size is not None
        ):
            # For a JWE algorithm with tag-aware key agreement in case key agreement
            # with key wrapping mode is used:
            # Perform key agreement with key wrapping deferred at step 3
            wrapped = alg.agree_upon_key_and_wrap_cek(
                enc, protected, key, sender_key, epk, cek, tag
            )
            ek = wrapped["ek"]

        # step 8: build resulting message
        return b".".join(
            [
                protected_segment,
                urlsafe_b64encode(ek),
                urlsafe_b64encode(iv),
                urlsafe_b64encode(ciphertext),
                urlsafe_b64encode(tag),
            ]
        )

    def serialize_json(self, header_obj, payload, keys, sender_key=None):  # noqa: C901
        """Generate a JWE JSON Serialization (in fully general syntax).

        The JWE JSON Serialization represents encrypted content as a JSON
        object.  This representation is neither optimized for compactness nor
        URL safe.

        The following members are defined for use in top-level JSON objects
        used for the fully general JWE JSON Serialization syntax:

        protected
            The "protected" member MUST be present and contain the value
            BASE64URL(UTF8(JWE Protected Header)) when the JWE Protected
            Header value is non-empty; otherwise, it MUST be absent.  These
            Header Parameter values are integrity protected.

        unprotected
            The "unprotected" member MUST be present and contain the value JWE
            Shared Unprotected Header when the JWE Shared Unprotected Header
            value is non-empty; otherwise, it MUST be absent.  This value is
            represented as an unencoded JSON object, rather than as a string.
            These Header Parameter values are not integrity protected.

        iv
            The "iv" member MUST be present and contain the value
            BASE64URL(JWE Initialization Vector) when the JWE Initialization
            Vector value is non-empty; otherwise, it MUST be absent.

        aad
            The "aad" member MUST be present and contain the value
            BASE64URL(JWE AAD)) when the JWE AAD value is non-empty;
            otherwise, it MUST be absent.  A JWE AAD value can be included to
            supply a base64url-encoded value to be integrity protected but not
            encrypted.

        ciphertext
            The "ciphertext" member MUST be present and contain the value
            BASE64URL(JWE Ciphertext).

        tag
            The "tag" member MUST be present and contain the value
            BASE64URL(JWE Authentication Tag) when the JWE Authentication Tag
            value is non-empty; otherwise, it MUST be absent.

        recipients
            The "recipients" member value MUST be an array of JSON objects.
            Each object contains information specific to a single recipient.
            This member MUST be present with exactly one array element per
            recipient, even if some or all of the array element values are the
            empty JSON object "{}" (which can happen when all Header Parameter
            values are shared between all recipients and when no encrypted key
            is used, such as when doing Direct Encryption).

        The following members are defined for use in the JSON objects that
        are elements of the "recipients" array:

        header
            The "header" member MUST be present and contain the value JWE Per-
            Recipient Unprotected Header when the JWE Per-Recipient
            Unprotected Header value is non-empty; otherwise, it MUST be
            absent.  This value is represented as an unencoded JSON object,
            rather than as a string.  These Header Parameter values are not
            integrity protected.

        encrypted_key
            The "encrypted_key" member MUST be present and contain the value
            BASE64URL(JWE Encrypted Key) when the JWE Encrypted Key value is
            non-empty; otherwise, it MUST be absent.

        This implementation assumes that "alg" and "enc" header fields are
        contained in the protected or shared unprotected header.

        :param header_obj: A dict of headers (in addition optionally contains JWE AAD)
        :param payload: Payload (bytes or a value convertible to bytes)
        :param keys: Public keys (or a single public key) used to encrypt payload
        :param sender_key: Sender's private key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: JWE JSON serialization (in fully general syntax) as dict

        Example of `header_obj`::

            {
                "protected": {
                    "alg": "ECDH-1PU+A128KW",
                    "enc": "A256CBC-HS512",
                    "apu": "QWxpY2U",
                    "apv": "Qm9iIGFuZCBDaGFybGll",
                },
                "unprotected": {"jku": "https://alice.example.com/keys.jwks"},
                "recipients": [
                    {"header": {"kid": "bob-key-2"}},
                    {"header": {"kid": "2021-05-06"}},
                ],
                "aad": b"Authenticate me too.",
            }
        """
        if not isinstance(keys, list):  # single key
            keys = [keys]

        if not keys:
            raise ValueError("No keys have been provided")

        header_obj = deepcopy(header_obj)

        shared_header = JWESharedHeader.from_dict(header_obj)

        recipients = header_obj.get("recipients")
        if recipients is None:
            recipients = [{} for _ in keys]
        for i in range(len(recipients)):
            if recipients[i] is None:
                recipients[i] = {}
            if "header" not in recipients[i]:
                recipients[i]["header"] = {}

        jwe_aad = header_obj.get("aad")

        if len(keys) != len(recipients):
            raise ValueError(
                f"Count of recipient keys {len(keys)} does not equal to count of recipients {len(recipients)}"
            )

        # step 1: Prepare algorithms & key
        alg = self.get_header_alg(shared_header)
        enc = self.get_header_enc(shared_header)
        zip_alg = self.get_header_zip(shared_header)

        self._validate_sender_key(sender_key, alg)
        self._validate_private_headers(shared_header, alg)
        for recipient in recipients:
            self._validate_private_headers(recipient["header"], alg)

        for i in range(len(keys)):
            keys[i] = prepare_key(alg, recipients[i]["header"], keys[i])
        if sender_key is not None:
            sender_key = alg.prepare_key(sender_key)

        # self._post_validate_header(protected, algorithm)

        # step 2: Generate a random Content Encryption Key (CEK)
        # use enc_alg.generate_cek() in scope of upcoming .wrap
        # or .generate_keys_and_prepare_headers call

        # step 3: Encrypt the CEK with the recipient's public key
        preset = alg.generate_preset(enc, keys[0])
        if "cek" in preset:
            cek = preset["cek"]
        else:
            cek = None
        if len(keys) > 1 and cek is None:
            raise InvalidAlgorithmForMultipleRecipientsMode(alg.name)
        if "header" in preset:
            shared_header.update_protected(preset["header"])

        if (
            isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement)
            and alg.key_size is not None
        ):
            # For a JWE algorithm with tag-aware key agreement in case key agreement
            # with key wrapping mode is used:
            # Defer key agreement with key wrapping until authentication tag is computed
            epks = []
            for i in range(len(keys)):
                prep = alg.generate_keys_and_prepare_headers(
                    enc, keys[i], sender_key, preset
                )
                if cek is None:
                    cek = prep["cek"]
                epks.append(prep["epk"])
                recipients[i]["header"].update(prep["header"])
        else:
            # In any other case:
            # Keep the normal steps order defined by RFC 7516
            for i in range(len(keys)):
                if isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement):
                    wrapped = alg.wrap(enc, shared_header, keys[i], sender_key, preset)
                else:
                    wrapped = alg.wrap(enc, shared_header, keys[i], preset)
                if cek is None:
                    cek = wrapped["cek"]
                recipients[i]["encrypted_key"] = wrapped["ek"]
                if "header" in wrapped:
                    recipients[i]["header"].update(wrapped["header"])

        # step 4: Generate a random JWE Initialization Vector
        iv = enc.generate_iv()

        # step 5: Compute the Encoded Protected Header value
        # BASE64URL(UTF8(JWE Protected Header)). If the JWE Protected Header
        # is not present, let this value be the empty string.
        # Let the Additional Authenticated Data encryption parameter be
        # ASCII(Encoded Protected Header). However, if a JWE AAD value is
        # present, instead let the Additional Authenticated Data encryption
        # parameter be ASCII(Encoded Protected Header || '.' || BASE64URL(JWE AAD)).
        aad = (
            json_b64encode(shared_header.protected) if shared_header.protected else b""
        )
        if jwe_aad is not None:
            aad += b"." + urlsafe_b64encode(jwe_aad)
        aad = to_bytes(aad, "ascii")

        # step 6: compress message if required
        if zip_alg:
            msg = zip_alg.compress(to_bytes(payload))
        else:
            msg = to_bytes(payload)

        # step 7: perform encryption
        ciphertext, tag = enc.encrypt(msg, aad, iv, cek)

        if (
            isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement)
            and alg.key_size is not None
        ):
            # For a JWE algorithm with tag-aware key agreement in case key agreement
            # with key wrapping mode is used:
            # Perform key agreement with key wrapping deferred at step 3
            for i in range(len(keys)):
                wrapped = alg.agree_upon_key_and_wrap_cek(
                    enc, shared_header, keys[i], sender_key, epks[i], cek, tag
                )
                recipients[i]["encrypted_key"] = wrapped["ek"]

        # step 8: build resulting message
        obj = OrderedDict()

        if shared_header.protected:
            obj["protected"] = to_unicode(json_b64encode(shared_header.protected))

        if shared_header.unprotected:
            obj["unprotected"] = shared_header.unprotected

        for recipient in recipients:
            if not recipient["header"]:
                del recipient["header"]
            recipient["encrypted_key"] = to_unicode(
                urlsafe_b64encode(recipient["encrypted_key"])
            )
            for member in set(recipient.keys()):
                if member not in {"header", "encrypted_key"}:
                    del recipient[member]
        obj["recipients"] = recipients

        if jwe_aad is not None:
            obj["aad"] = to_unicode(urlsafe_b64encode(jwe_aad))

        obj["iv"] = to_unicode(urlsafe_b64encode(iv))

        obj["ciphertext"] = to_unicode(urlsafe_b64encode(ciphertext))

        obj["tag"] = to_unicode(urlsafe_b64encode(tag))

        return obj

    def serialize(self, header, payload, key, sender_key=None):
        """Generate a JWE Serialization.

        It will automatically generate a compact or JSON serialization depending
        on `header` argument. If `header` is a dict with "protected",
        "unprotected" and/or "recipients" keys, it will call `serialize_json`,
        otherwise it will call `serialize_compact`.

        :param header: A dict of header(s)
        :param payload: Payload (bytes or a value convertible to bytes)
        :param key: Public key(s) used to encrypt payload
        :param sender_key: Sender's private key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: JWE compact serialization as bytes or
            JWE JSON serialization as dict
        """
        if "protected" in header or "unprotected" in header or "recipients" in header:
            return self.serialize_json(header, payload, key, sender_key)

        return self.serialize_compact(header, payload, key, sender_key)

    def deserialize_compact(self, s, key, decode=None, sender_key=None):
        """Extract JWE Compact Serialization.

        :param s: JWE Compact Serialization as bytes
        :param key: Private key used to decrypt payload
            (optionally can be a tuple of kid and essentially key)
        :param decode: Function to decode payload data
        :param sender_key: Sender's public key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: dict with `header` and `payload` keys where `header` value is
            a dict containing protected header fields
        """
        try:
            s = to_bytes(s)
            protected_s, ek_s, iv_s, ciphertext_s, tag_s = s.rsplit(b".")
        except ValueError as exc:
            raise DecodeError("Not enough segments") from exc

        protected = extract_header(protected_s, DecodeError)
        ek = extract_segment(ek_s, DecodeError, "encryption key")
        iv = extract_segment(iv_s, DecodeError, "initialization vector")
        ciphertext = extract_segment(ciphertext_s, DecodeError, "ciphertext")
        tag = extract_segment(tag_s, DecodeError, "authentication tag")

        alg = self.get_header_alg(protected)
        enc = self.get_header_enc(protected)
        zip_alg = self.get_header_zip(protected)

        self._validate_sender_key(sender_key, alg)
        self._validate_private_headers(protected, alg)

        if isinstance(key, tuple) and len(key) == 2:
            # Ignore separately provided kid, extract essentially key only
            key = key[1]

        key = prepare_key(alg, protected, key)

        if sender_key is not None:
            sender_key = alg.prepare_key(sender_key)

        if isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement):
            # For a JWE algorithm with tag-aware key agreement:
            if alg.key_size is not None:
                # In case key agreement with key wrapping mode is used:
                # Provide authentication tag to .unwrap method
                cek = alg.unwrap(enc, ek, protected, key, sender_key, tag)
            else:
                # Otherwise, don't provide authentication tag to .unwrap method
                cek = alg.unwrap(enc, ek, protected, key, sender_key)
        else:
            # For any other JWE algorithm:
            # Don't provide authentication tag to .unwrap method
            cek = alg.unwrap(enc, ek, protected, key)

        aad = to_bytes(protected_s, "ascii")
        msg = enc.decrypt(ciphertext, aad, iv, tag, cek)

        if zip_alg:
            payload = zip_alg.decompress(to_bytes(msg))
        else:
            payload = msg

        if decode:
            payload = decode(payload)
        return {"header": protected, "payload": payload}

    def deserialize_json(self, obj, key, decode=None, sender_key=None):  # noqa: C901
        """Extract JWE JSON Serialization.

        :param obj: JWE JSON Serialization as dict or str
        :param key: Private key used to decrypt payload
            (optionally can be a tuple of kid and essentially key)
        :param decode: Function to decode payload data
        :param sender_key: Sender's public key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: dict with `header` and `payload` keys where `header` value is
            a dict containing `protected`, `unprotected`, `recipients` and/or
            `aad` keys
        """
        obj = ensure_dict(obj, "JWE")
        obj = deepcopy(obj)

        if "protected" in obj:
            protected = extract_header(to_bytes(obj["protected"]), DecodeError)
        else:
            protected = None

        unprotected = obj.get("unprotected")

        recipients = obj["recipients"]
        for recipient in recipients:
            if "header" not in recipient:
                recipient["header"] = {}
            recipient["encrypted_key"] = extract_segment(
                to_bytes(recipient["encrypted_key"]), DecodeError, "encrypted key"
            )

        if "aad" in obj:
            jwe_aad = extract_segment(to_bytes(obj["aad"]), DecodeError, "JWE AAD")
        else:
            jwe_aad = None

        iv = extract_segment(to_bytes(obj["iv"]), DecodeError, "initialization vector")

        ciphertext = extract_segment(
            to_bytes(obj["ciphertext"]), DecodeError, "ciphertext"
        )

        tag = extract_segment(to_bytes(obj["tag"]), DecodeError, "authentication tag")

        shared_header = JWESharedHeader(protected, unprotected)

        alg = self.get_header_alg(shared_header)
        enc = self.get_header_enc(shared_header)
        zip_alg = self.get_header_zip(shared_header)

        self._validate_sender_key(sender_key, alg)
        self._validate_private_headers(shared_header, alg)
        for recipient in recipients:
            self._validate_private_headers(recipient["header"], alg)

        kid = None
        if isinstance(key, tuple) and len(key) == 2:
            # Extract separately provided kid and essentially key
            kid = key[0]
            key = key[1]

        key = alg.prepare_key(key)

        if kid is None:
            # If kid has not been provided separately, try to get it from key itself
            kid = key.kid

        if sender_key is not None:
            sender_key = alg.prepare_key(sender_key)

        def _unwrap_with_sender_key_and_tag(ek, header):
            return alg.unwrap(enc, ek, header, key, sender_key, tag)

        def _unwrap_with_sender_key_and_without_tag(ek, header):
            return alg.unwrap(enc, ek, header, key, sender_key)

        def _unwrap_without_sender_key_and_tag(ek, header):
            return alg.unwrap(enc, ek, header, key)

        def _unwrap_for_matching_recipient(unwrap_func):
            if kid is not None:
                for recipient in recipients:
                    if recipient["header"].get("kid") == kid:
                        header = JWEHeader(protected, unprotected, recipient["header"])
                        return unwrap_func(recipient["encrypted_key"], header)

            # Since no explicit match has been found, iterate over all the recipients
            error = None
            for recipient in recipients:
                header = JWEHeader(protected, unprotected, recipient["header"])
                try:
                    return unwrap_func(recipient["encrypted_key"], header)
                except Exception as e:
                    error = e
            else:
                if error is None:
                    raise KeyMismatchError()
                else:
                    raise error

        if isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement):
            # For a JWE algorithm with tag-aware key agreement:
            if alg.key_size is not None:
                # In case key agreement with key wrapping mode is used:
                # Provide authentication tag to .unwrap method
                cek = _unwrap_for_matching_recipient(_unwrap_with_sender_key_and_tag)
            else:
                # Otherwise, don't provide authentication tag to .unwrap method
                cek = _unwrap_for_matching_recipient(
                    _unwrap_with_sender_key_and_without_tag
                )
        else:
            # For any other JWE algorithm:
            # Don't provide authentication tag to .unwrap method
            cek = _unwrap_for_matching_recipient(_unwrap_without_sender_key_and_tag)

        aad = to_bytes(obj.get("protected", ""))
        if "aad" in obj:
            aad += b"." + to_bytes(obj["aad"])
        aad = to_bytes(aad, "ascii")

        msg = enc.decrypt(ciphertext, aad, iv, tag, cek)

        if zip_alg:
            payload = zip_alg.decompress(to_bytes(msg))
        else:
            payload = msg

        if decode:
            payload = decode(payload)

        for recipient in recipients:
            if not recipient["header"]:
                del recipient["header"]
            for member in set(recipient.keys()):
                if member != "header":
                    del recipient[member]

        header = {}
        if protected:
            header["protected"] = protected
        if unprotected:
            header["unprotected"] = unprotected
        header["recipients"] = recipients
        if jwe_aad is not None:
            header["aad"] = jwe_aad

        return {"header": header, "payload": payload}

    def deserialize(self, obj, key, decode=None, sender_key=None):
        """Extract a JWE Serialization.

        It supports both compact and JSON serialization.

        :param obj: JWE compact serialization as bytes or
            JWE JSON serialization as dict or str
        :param key: Private key used to decrypt payload
            (optionally can be a tuple of kid and essentially key)
        :param decode: Function to decode payload data
        :param sender_key: Sender's public key in case
            JWEAlgorithmWithTagAwareKeyAgreement is used
        :return: dict with `header` and `payload` keys
        """
        if isinstance(obj, dict):
            return self.deserialize_json(obj, key, decode, sender_key)

        obj = to_bytes(obj)
        if obj.startswith(b"{") and obj.endswith(b"}"):
            return self.deserialize_json(obj, key, decode, sender_key)

        return self.deserialize_compact(obj, key, decode, sender_key)

    @staticmethod
    def parse_json(obj):
        """Parse JWE JSON Serialization.

        :param obj: JWE JSON Serialization as str or dict
        :return: Parsed JWE JSON Serialization as dict if `obj` is an str,
            or `obj` as is if `obj` is already a dict
        """
        return ensure_dict(obj, "JWE")

    def get_header_alg(self, header):
        if "alg" not in header:
            raise MissingAlgorithmError()

        alg = header["alg"]
        if self._algorithms is not None and alg not in self._algorithms:
            raise UnsupportedAlgorithmError()
        if alg not in self.ALG_REGISTRY:
            raise UnsupportedAlgorithmError()
        return self.ALG_REGISTRY[alg]

    def get_header_enc(self, header):
        if "enc" not in header:
            raise MissingEncryptionAlgorithmError()
        enc = header["enc"]
        if self._algorithms is not None and enc not in self._algorithms:
            raise UnsupportedEncryptionAlgorithmError()
        if enc not in self.ENC_REGISTRY:
            raise UnsupportedEncryptionAlgorithmError()
        return self.ENC_REGISTRY[enc]

    def get_header_zip(self, header):
        if "zip" in header:
            z = header["zip"]
            if self._algorithms is not None and z not in self._algorithms:
                raise UnsupportedCompressionAlgorithmError()
            if z not in self.ZIP_REGISTRY:
                raise UnsupportedCompressionAlgorithmError()
            return self.ZIP_REGISTRY[z]

    def _validate_sender_key(self, sender_key, alg):
        if isinstance(alg, JWEAlgorithmWithTagAwareKeyAgreement):
            if sender_key is None:
                raise ValueError(
                    f"{alg.name} algorithm requires sender_key but passed sender_key value is None"
                )
        else:
            if sender_key is not None:
                raise ValueError(
                    f"{alg.name} algorithm does not use sender_key but passed sender_key value is not None"
                )

    def _validate_private_headers(self, header, alg):
        # only validate private headers when developers set
        # private headers explicitly
        if self._private_headers is None:
            return

        names = self.REGISTERED_HEADER_PARAMETER_NAMES.copy()
        names = names.union(self._private_headers)

        if alg.EXTRA_HEADERS:
            names = names.union(alg.EXTRA_HEADERS)

        for k in header:
            if k not in names:
                raise InvalidHeaderParameterNameError(k)


def prepare_key(alg, header, key):
    if callable(key):
        key = key(header, None)
    elif key is None and "jwk" in header:
        key = header["jwk"]
    return alg.prepare_key(key)

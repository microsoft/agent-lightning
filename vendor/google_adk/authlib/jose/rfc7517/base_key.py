import hashlib
from collections import OrderedDict

from authlib.common.encoding import json_dumps
from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64encode

from ..errors import InvalidUseError


class Key:
    """This is the base class for a JSON Web Key."""

    kty = "_"

    ALLOWED_PARAMS = ["use", "key_ops", "alg", "kid", "x5u", "x5c", "x5t", "x5t#S256"]

    PRIVATE_KEY_OPS = [
        "sign",
        "decrypt",
        "unwrapKey",
    ]
    PUBLIC_KEY_OPS = [
        "verify",
        "encrypt",
        "wrapKey",
    ]

    REQUIRED_JSON_FIELDS = []

    def __init__(self, options=None):
        self.options = options or {}
        self._dict_data = {}

    @property
    def tokens(self):
        if not self._dict_data:
            self.load_dict_key()

        rv = dict(self._dict_data)
        rv["kty"] = self.kty
        for k in self.ALLOWED_PARAMS:
            if k not in rv and k in self.options:
                rv[k] = self.options[k]
        return rv

    @property
    def kid(self):
        return self.tokens.get("kid")

    def keys(self):
        return self.tokens.keys()

    def __getitem__(self, item):
        return self.tokens[item]

    @property
    def public_only(self):
        raise NotImplementedError()

    def load_raw_key(self):
        raise NotImplementedError()

    def load_dict_key(self):
        raise NotImplementedError()

    def check_key_op(self, operation):
        """Check if the given key_op is supported by this key.

        :param operation: key operation value, such as "sign", "encrypt".
        :raise: ValueError
        """
        key_ops = self.tokens.get("key_ops")
        if key_ops is not None and operation not in key_ops:
            raise ValueError(f'Unsupported key_op "{operation}"')

        if operation in self.PRIVATE_KEY_OPS and self.public_only:
            raise ValueError(f'Invalid key_op "{operation}" for public key')

        use = self.tokens.get("use")
        if use:
            if operation in ["sign", "verify"]:
                if use != "sig":
                    raise InvalidUseError()
            elif operation in ["decrypt", "encrypt", "wrapKey", "unwrapKey"]:
                if use != "enc":
                    raise InvalidUseError()

    def as_dict(self, is_private=False, **params):
        raise NotImplementedError()

    def as_json(self, is_private=False, **params):
        """Represent this key as a JSON string."""
        obj = self.as_dict(is_private, **params)
        return json_dumps(obj)

    def thumbprint(self):
        """Implementation of RFC7638 JSON Web Key (JWK) Thumbprint."""
        fields = list(self.REQUIRED_JSON_FIELDS)
        fields.append("kty")
        fields.sort()
        data = OrderedDict()

        for k in fields:
            data[k] = self.tokens[k]

        json_data = json_dumps(data)
        digest_data = hashlib.sha256(to_bytes(json_data)).digest()
        return to_unicode(urlsafe_b64encode(digest_data))

    @classmethod
    def check_required_fields(cls, data):
        for k in cls.REQUIRED_JSON_FIELDS:
            if k not in data:
                raise ValueError(f'Missing required field: "{k}"')

    @classmethod
    def validate_raw_key(cls, key):
        raise NotImplementedError()

import secrets

from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64decode
from authlib.common.encoding import urlsafe_b64encode

from ..rfc7517 import Key

POSSIBLE_UNSAFE_KEYS = (
    b"-----BEGIN ",
    b"---- BEGIN ",
    b"ssh-rsa ",
    b"ssh-dss ",
    b"ssh-ed25519 ",
    b"ecdsa-sha2-",
)


class OctKey(Key):
    """Key class of the ``oct`` key type."""

    kty = "oct"
    REQUIRED_JSON_FIELDS = ["k"]

    def __init__(self, raw_key=None, options=None):
        super().__init__(options)
        self.raw_key = raw_key

    @property
    def public_only(self):
        return False

    def get_op_key(self, operation):
        """Get the raw key for the given key_op. This method will also
        check if the given key_op is supported by this key.

        :param operation: key operation value, such as "sign", "encrypt".
        :return: raw key
        """
        self.check_key_op(operation)
        if not self.raw_key:
            self.load_raw_key()
        return self.raw_key

    def load_raw_key(self):
        self.raw_key = urlsafe_b64decode(to_bytes(self.tokens["k"]))

    def load_dict_key(self):
        k = to_unicode(urlsafe_b64encode(self.raw_key))
        self._dict_data = {"kty": self.kty, "k": k}

    def as_dict(self, is_private=False, **params):
        tokens = self.tokens
        if "kid" not in tokens:
            tokens["kid"] = self.thumbprint()

        tokens.update(params)
        return tokens

    @classmethod
    def validate_raw_key(cls, key):
        return isinstance(key, bytes)

    @classmethod
    def import_key(cls, raw, options=None):
        """Import a key from bytes, string, or dict data."""
        if isinstance(raw, cls):
            if options is not None:
                raw.options.update(options)
            return raw

        if isinstance(raw, dict):
            cls.check_required_fields(raw)
            key = cls(options=options)
            key._dict_data = raw
        else:
            raw_key = to_bytes(raw)

            # security check
            if raw_key.startswith(POSSIBLE_UNSAFE_KEYS):
                raise ValueError("This key may not be safe to import")

            key = cls(raw_key=raw_key, options=options)
        return key

    @classmethod
    def generate_key(cls, key_size=256, options=None, is_private=True):
        """Generate a ``OctKey`` with the given bit size."""
        if not is_private:
            raise ValueError("oct key can not be generated as public")

        if key_size % 8 != 0:
            raise ValueError("Invalid bit size for oct key")

        return cls.import_key(secrets.token_bytes(int(key_size / 8)), options)

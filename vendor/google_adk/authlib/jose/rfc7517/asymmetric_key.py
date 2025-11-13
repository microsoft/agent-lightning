from cryptography.hazmat.primitives.serialization import BestAvailableEncryption
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat

from authlib.common.encoding import to_bytes

from ._cryptography_key import load_pem_key
from .base_key import Key


class AsymmetricKey(Key):
    """This is the base class for a JSON Web Key."""

    PUBLIC_KEY_FIELDS = []
    PRIVATE_KEY_FIELDS = []
    PRIVATE_KEY_CLS = bytes
    PUBLIC_KEY_CLS = bytes
    SSH_PUBLIC_PREFIX = b""

    def __init__(self, private_key=None, public_key=None, options=None):
        super().__init__(options)
        self.private_key = private_key
        self.public_key = public_key

    @property
    def public_only(self):
        if self.private_key:
            return False
        if "d" in self.tokens:
            return False
        return True

    def get_op_key(self, operation):
        """Get the raw key for the given key_op. This method will also
        check if the given key_op is supported by this key.

        :param operation: key operation value, such as "sign", "encrypt".
        :return: raw key
        """
        self.check_key_op(operation)
        if operation in self.PUBLIC_KEY_OPS:
            return self.get_public_key()
        return self.get_private_key()

    def get_public_key(self):
        if self.public_key:
            return self.public_key

        private_key = self.get_private_key()
        if private_key:
            return private_key.public_key()

        return self.public_key

    def get_private_key(self):
        if self.private_key:
            return self.private_key

        if self.tokens:
            self.load_raw_key()
        return self.private_key

    def load_raw_key(self):
        if "d" in self.tokens:
            self.private_key = self.load_private_key()
        else:
            self.public_key = self.load_public_key()

    def load_dict_key(self):
        if self.private_key:
            self._dict_data.update(self.dumps_private_key())
        else:
            self._dict_data.update(self.dumps_public_key())

    def dumps_private_key(self):
        raise NotImplementedError()

    def dumps_public_key(self):
        raise NotImplementedError()

    def load_private_key(self):
        raise NotImplementedError()

    def load_public_key(self):
        raise NotImplementedError()

    def as_dict(self, is_private=False, **params):
        """Represent this key as a dict of the JSON Web Key."""
        tokens = self.tokens
        if is_private and "d" not in tokens:
            raise ValueError("This is a public key")

        kid = tokens.get("kid")
        if "d" in tokens and not is_private:
            # filter out private fields
            tokens = {k: tokens[k] for k in tokens if k in self.PUBLIC_KEY_FIELDS}
            tokens["kty"] = self.kty
            if kid:
                tokens["kid"] = kid

        if not kid:
            tokens["kid"] = self.thumbprint()

        tokens.update(params)
        return tokens

    def as_key(self, is_private=False):
        """Represent this key as raw key."""
        if is_private:
            return self.get_private_key()
        return self.get_public_key()

    def as_bytes(self, encoding=None, is_private=False, password=None):
        """Export key into PEM/DER format bytes.

        :param encoding: "PEM" or "DER"
        :param is_private: export private key or public key
        :param password: encrypt private key with password
        :return: bytes
        """
        if encoding is None or encoding == "PEM":
            encoding = Encoding.PEM
        elif encoding == "DER":
            encoding = Encoding.DER
        else:
            raise ValueError(f"Invalid encoding: {encoding!r}")

        raw_key = self.as_key(is_private)
        if is_private:
            if not raw_key:
                raise ValueError("This is a public key")
            if password is None:
                encryption_algorithm = NoEncryption()
            else:
                encryption_algorithm = BestAvailableEncryption(to_bytes(password))
            return raw_key.private_bytes(
                encoding=encoding,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=encryption_algorithm,
            )
        return raw_key.public_bytes(
            encoding=encoding,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

    def as_pem(self, is_private=False, password=None):
        return self.as_bytes(is_private=is_private, password=password)

    def as_der(self, is_private=False, password=None):
        return self.as_bytes(encoding="DER", is_private=is_private, password=password)

    @classmethod
    def import_dict_key(cls, raw, options=None):
        cls.check_required_fields(raw)
        key = cls(options=options)
        key._dict_data = raw
        return key

    @classmethod
    def import_key(cls, raw, options=None):
        if isinstance(raw, cls):
            if options is not None:
                raw.options.update(options)
            return raw

        if isinstance(raw, cls.PUBLIC_KEY_CLS):
            key = cls(public_key=raw, options=options)
        elif isinstance(raw, cls.PRIVATE_KEY_CLS):
            key = cls(private_key=raw, options=options)
        elif isinstance(raw, dict):
            key = cls.import_dict_key(raw, options)
        else:
            if options is not None:
                password = options.pop("password", None)
            else:
                password = None
            raw_key = load_pem_key(raw, cls.SSH_PUBLIC_PREFIX, password=password)
            if isinstance(raw_key, cls.PUBLIC_KEY_CLS):
                key = cls(public_key=raw_key, options=options)
            elif isinstance(raw_key, cls.PRIVATE_KEY_CLS):
                key = cls(private_key=raw_key, options=options)
            else:
                raise ValueError("Invalid data for importing key")
        return key

    @classmethod
    def validate_raw_key(cls, key):
        return isinstance(key, cls.PUBLIC_KEY_CLS) or isinstance(
            key, cls.PRIVATE_KEY_CLS
        )

    @classmethod
    def generate_key(cls, crv_or_size, options=None, is_private=False):
        raise NotImplementedError()

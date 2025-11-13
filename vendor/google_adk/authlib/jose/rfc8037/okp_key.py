from cryptography.hazmat.primitives.asymmetric.ed448 import Ed448PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed448 import Ed448PublicKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x448 import X448PrivateKey
from cryptography.hazmat.primitives.asymmetric.x448 import X448PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat

from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64decode
from authlib.common.encoding import urlsafe_b64encode

from ..rfc7517 import AsymmetricKey

PUBLIC_KEYS_MAP = {
    "Ed25519": Ed25519PublicKey,
    "Ed448": Ed448PublicKey,
    "X25519": X25519PublicKey,
    "X448": X448PublicKey,
}
PRIVATE_KEYS_MAP = {
    "Ed25519": Ed25519PrivateKey,
    "Ed448": Ed448PrivateKey,
    "X25519": X25519PrivateKey,
    "X448": X448PrivateKey,
}


class OKPKey(AsymmetricKey):
    """Key class of the ``OKP`` key type."""

    kty = "OKP"
    REQUIRED_JSON_FIELDS = ["crv", "x"]
    PUBLIC_KEY_FIELDS = REQUIRED_JSON_FIELDS
    PRIVATE_KEY_FIELDS = ["crv", "d"]
    PUBLIC_KEY_CLS = tuple(PUBLIC_KEYS_MAP.values())
    PRIVATE_KEY_CLS = tuple(PRIVATE_KEYS_MAP.values())
    SSH_PUBLIC_PREFIX = b"ssh-ed25519"

    def exchange_shared_key(self, pubkey):
        # used in ECDHESAlgorithm
        private_key = self.get_private_key()
        if private_key and isinstance(private_key, (X25519PrivateKey, X448PrivateKey)):
            return private_key.exchange(pubkey)
        raise ValueError("Invalid key for exchanging shared key")

    @staticmethod
    def get_key_curve(key):
        if isinstance(key, (Ed25519PublicKey, Ed25519PrivateKey)):
            return "Ed25519"
        elif isinstance(key, (Ed448PublicKey, Ed448PrivateKey)):
            return "Ed448"
        elif isinstance(key, (X25519PublicKey, X25519PrivateKey)):
            return "X25519"
        elif isinstance(key, (X448PublicKey, X448PrivateKey)):
            return "X448"

    def load_private_key(self):
        crv_key = PRIVATE_KEYS_MAP[self._dict_data["crv"]]
        d_bytes = urlsafe_b64decode(to_bytes(self._dict_data["d"]))
        return crv_key.from_private_bytes(d_bytes)

    def load_public_key(self):
        crv_key = PUBLIC_KEYS_MAP[self._dict_data["crv"]]
        x_bytes = urlsafe_b64decode(to_bytes(self._dict_data["x"]))
        return crv_key.from_public_bytes(x_bytes)

    def dumps_private_key(self):
        obj = self.dumps_public_key(self.private_key.public_key())
        d_bytes = self.private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        obj["d"] = to_unicode(urlsafe_b64encode(d_bytes))
        return obj

    def dumps_public_key(self, public_key=None):
        if public_key is None:
            public_key = self.public_key
        x_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return {
            "crv": self.get_key_curve(public_key),
            "x": to_unicode(urlsafe_b64encode(x_bytes)),
        }

    @classmethod
    def generate_key(cls, crv="Ed25519", options=None, is_private=False) -> "OKPKey":
        if crv not in PRIVATE_KEYS_MAP:
            raise ValueError(f'Invalid crv value: "{crv}"')
        private_key_cls = PRIVATE_KEYS_MAP[crv]
        raw_key = private_key_cls.generate()
        if not is_private:
            raw_key = raw_key.public_key()
        return cls.import_key(raw_key, options=options)

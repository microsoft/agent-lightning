from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import SECP256K1
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1
from cryptography.hazmat.primitives.asymmetric.ec import SECP384R1
from cryptography.hazmat.primitives.asymmetric.ec import SECP521R1
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKeyWithSerialization,
)
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateNumbers
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicNumbers

from authlib.common.encoding import base64_to_int
from authlib.common.encoding import int_to_base64

from ..rfc7517 import AsymmetricKey


class ECKey(AsymmetricKey):
    """Key class of the ``EC`` key type."""

    kty = "EC"
    DSS_CURVES = {
        "P-256": SECP256R1,
        "P-384": SECP384R1,
        "P-521": SECP521R1,
        # https://tools.ietf.org/html/rfc8812#section-3.1
        "secp256k1": SECP256K1,
    }
    CURVES_DSS = {
        SECP256R1.name: "P-256",
        SECP384R1.name: "P-384",
        SECP521R1.name: "P-521",
        SECP256K1.name: "secp256k1",
    }
    REQUIRED_JSON_FIELDS = ["crv", "x", "y"]

    PUBLIC_KEY_FIELDS = REQUIRED_JSON_FIELDS
    PRIVATE_KEY_FIELDS = ["crv", "d", "x", "y"]

    PUBLIC_KEY_CLS = EllipticCurvePublicKey
    PRIVATE_KEY_CLS = EllipticCurvePrivateKeyWithSerialization
    SSH_PUBLIC_PREFIX = b"ecdsa-sha2-"

    def exchange_shared_key(self, pubkey):
        # # used in ECDHESAlgorithm
        private_key = self.get_private_key()
        if private_key:
            return private_key.exchange(ec.ECDH(), pubkey)
        raise ValueError("Invalid key for exchanging shared key")

    @property
    def curve_key_size(self):
        raw_key = self.get_private_key()
        if not raw_key:
            raw_key = self.public_key
        return raw_key.curve.key_size

    def load_private_key(self):
        curve = self.DSS_CURVES[self._dict_data["crv"]]()
        public_numbers = EllipticCurvePublicNumbers(
            base64_to_int(self._dict_data["x"]),
            base64_to_int(self._dict_data["y"]),
            curve,
        )
        private_numbers = EllipticCurvePrivateNumbers(
            base64_to_int(self.tokens["d"]), public_numbers
        )
        return private_numbers.private_key(default_backend())

    def load_public_key(self):
        curve = self.DSS_CURVES[self._dict_data["crv"]]()
        public_numbers = EllipticCurvePublicNumbers(
            base64_to_int(self._dict_data["x"]),
            base64_to_int(self._dict_data["y"]),
            curve,
        )
        return public_numbers.public_key(default_backend())

    def dumps_private_key(self):
        numbers = self.private_key.private_numbers()
        return {
            "crv": self.CURVES_DSS[self.private_key.curve.name],
            "x": int_to_base64(numbers.public_numbers.x),
            "y": int_to_base64(numbers.public_numbers.y),
            "d": int_to_base64(numbers.private_value),
        }

    def dumps_public_key(self):
        numbers = self.public_key.public_numbers()
        return {
            "crv": self.CURVES_DSS[numbers.curve.name],
            "x": int_to_base64(numbers.x),
            "y": int_to_base64(numbers.y),
        }

    @classmethod
    def generate_key(cls, crv="P-256", options=None, is_private=False) -> "ECKey":
        if crv not in cls.DSS_CURVES:
            raise ValueError(f'Invalid crv value: "{crv}"')
        raw_key = ec.generate_private_key(
            curve=cls.DSS_CURVES[crv](),
            backend=default_backend(),
        )
        if not is_private:
            raw_key = raw_key.public_key()
        return cls.import_key(raw_key, options=options)

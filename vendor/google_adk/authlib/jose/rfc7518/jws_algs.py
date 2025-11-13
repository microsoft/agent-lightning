"""authlib.jose.rfc7518.
~~~~~~~~~~~~~~~~~~~~

"alg" (Algorithm) Header Parameter Values for JWS per `Section 3`_.

.. _`Section 3`: https://tools.ietf.org/html/rfc7518#section-3
"""

import hashlib
import hmac

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from ..rfc7515 import JWSAlgorithm
from .ec_key import ECKey
from .oct_key import OctKey
from .rsa_key import RSAKey
from .util import decode_int
from .util import encode_int


class NoneAlgorithm(JWSAlgorithm):
    name = "none"
    description = "No digital signature or MAC performed"

    def prepare_key(self, raw_data):
        return None

    def sign(self, msg, key):
        return b""

    def verify(self, msg, sig, key):
        return sig == b""


class HMACAlgorithm(JWSAlgorithm):
    """HMAC using SHA algorithms for JWS. Available algorithms:

    - HS256: HMAC using SHA-256
    - HS384: HMAC using SHA-384
    - HS512: HMAC using SHA-512
    """

    SHA256 = hashlib.sha256
    SHA384 = hashlib.sha384
    SHA512 = hashlib.sha512

    def __init__(self, sha_type):
        self.name = f"HS{sha_type}"
        self.description = f"HMAC using SHA-{sha_type}"
        self.hash_alg = getattr(self, f"SHA{sha_type}")

    def prepare_key(self, raw_data):
        return OctKey.import_key(raw_data)

    def sign(self, msg, key):
        # it is faster than the one in cryptography
        op_key = key.get_op_key("sign")
        return hmac.new(op_key, msg, self.hash_alg).digest()

    def verify(self, msg, sig, key):
        op_key = key.get_op_key("verify")
        v_sig = hmac.new(op_key, msg, self.hash_alg).digest()
        return hmac.compare_digest(sig, v_sig)


class RSAAlgorithm(JWSAlgorithm):
    """RSA using SHA algorithms for JWS. Available algorithms:

    - RS256: RSASSA-PKCS1-v1_5 using SHA-256
    - RS384: RSASSA-PKCS1-v1_5 using SHA-384
    - RS512: RSASSA-PKCS1-v1_5 using SHA-512
    """

    SHA256 = hashes.SHA256
    SHA384 = hashes.SHA384
    SHA512 = hashes.SHA512

    def __init__(self, sha_type):
        self.name = f"RS{sha_type}"
        self.description = f"RSASSA-PKCS1-v1_5 using SHA-{sha_type}"
        self.hash_alg = getattr(self, f"SHA{sha_type}")
        self.padding = padding.PKCS1v15()

    def prepare_key(self, raw_data):
        return RSAKey.import_key(raw_data)

    def sign(self, msg, key):
        op_key = key.get_op_key("sign")
        return op_key.sign(msg, self.padding, self.hash_alg())

    def verify(self, msg, sig, key):
        op_key = key.get_op_key("verify")
        try:
            op_key.verify(sig, msg, self.padding, self.hash_alg())
            return True
        except InvalidSignature:
            return False


class ECAlgorithm(JWSAlgorithm):
    """ECDSA using SHA algorithms for JWS. Available algorithms:

    - ES256: ECDSA using P-256 and SHA-256
    - ES384: ECDSA using P-384 and SHA-384
    - ES512: ECDSA using P-521 and SHA-512
    """

    SHA256 = hashes.SHA256
    SHA384 = hashes.SHA384
    SHA512 = hashes.SHA512

    def __init__(self, name, curve, sha_type):
        self.name = name
        self.curve = curve
        self.description = f"ECDSA using {self.curve} and SHA-{sha_type}"
        self.hash_alg = getattr(self, f"SHA{sha_type}")

    def prepare_key(self, raw_data):
        key = ECKey.import_key(raw_data)
        if key["crv"] != self.curve:
            raise ValueError(
                f'Key for "{self.name}" not supported, only "{self.curve}" allowed'
            )
        return key

    def sign(self, msg, key):
        op_key = key.get_op_key("sign")
        der_sig = op_key.sign(msg, ECDSA(self.hash_alg()))
        r, s = decode_dss_signature(der_sig)
        size = key.curve_key_size
        return encode_int(r, size) + encode_int(s, size)

    def verify(self, msg, sig, key):
        key_size = key.curve_key_size
        length = (key_size + 7) // 8

        if len(sig) != 2 * length:
            return False

        r = decode_int(sig[:length])
        s = decode_int(sig[length:])
        der_sig = encode_dss_signature(r, s)

        try:
            op_key = key.get_op_key("verify")
            op_key.verify(der_sig, msg, ECDSA(self.hash_alg()))
            return True
        except InvalidSignature:
            return False


class RSAPSSAlgorithm(JWSAlgorithm):
    """RSASSA-PSS using SHA algorithms for JWS. Available algorithms:

    - PS256: RSASSA-PSS using SHA-256 and MGF1 with SHA-256
    - PS384: RSASSA-PSS using SHA-384 and MGF1 with SHA-384
    - PS512: RSASSA-PSS using SHA-512 and MGF1 with SHA-512
    """

    SHA256 = hashes.SHA256
    SHA384 = hashes.SHA384
    SHA512 = hashes.SHA512

    def __init__(self, sha_type):
        self.name = f"PS{sha_type}"
        tpl = "RSASSA-PSS using SHA-{} and MGF1 with SHA-{}"
        self.description = tpl.format(sha_type, sha_type)
        self.hash_alg = getattr(self, f"SHA{sha_type}")

    def prepare_key(self, raw_data):
        return RSAKey.import_key(raw_data)

    def sign(self, msg, key):
        op_key = key.get_op_key("sign")
        return op_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(self.hash_alg()), salt_length=self.hash_alg.digest_size
            ),
            self.hash_alg(),
        )

    def verify(self, msg, sig, key):
        op_key = key.get_op_key("verify")
        try:
            op_key.verify(
                sig,
                msg,
                padding.PSS(
                    mgf=padding.MGF1(self.hash_alg()),
                    salt_length=self.hash_alg.digest_size,
                ),
                self.hash_alg(),
            )
            return True
        except InvalidSignature:
            return False


JWS_ALGORITHMS = [
    NoneAlgorithm(),  # none
    HMACAlgorithm(256),  # HS256
    HMACAlgorithm(384),  # HS384
    HMACAlgorithm(512),  # HS512
    RSAAlgorithm(256),  # RS256
    RSAAlgorithm(384),  # RS384
    RSAAlgorithm(512),  # RS512
    ECAlgorithm("ES256", "P-256", 256),
    ECAlgorithm("ES384", "P-384", 384),
    ECAlgorithm("ES512", "P-521", 512),
    ECAlgorithm("ES256K", "secp256k1", 256),  # defined in RFC8812
    RSAPSSAlgorithm(256),  # PS256
    RSAPSSAlgorithm(384),  # PS384
    RSAPSSAlgorithm(512),  # PS512
]

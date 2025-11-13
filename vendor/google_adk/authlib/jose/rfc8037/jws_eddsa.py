from cryptography.exceptions import InvalidSignature

from ..rfc7515 import JWSAlgorithm
from .okp_key import OKPKey


class EdDSAAlgorithm(JWSAlgorithm):
    name = "EdDSA"
    description = "Edwards-curve Digital Signature Algorithm for JWS"

    def prepare_key(self, raw_data):
        return OKPKey.import_key(raw_data)

    def sign(self, msg, key):
        op_key = key.get_op_key("sign")
        return op_key.sign(msg)

    def verify(self, msg, sig, key):
        op_key = key.get_op_key("verify")
        try:
            op_key.verify(sig, msg)
            return True
        except InvalidSignature:
            return False


def register_jws_rfc8037(cls):
    cls.register_algorithm(EdDSAAlgorithm())

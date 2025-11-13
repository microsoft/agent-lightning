from .ec_key import ECKey
from .jwe_algs import JWE_ALG_ALGORITHMS
from .jwe_algs import AESAlgorithm
from .jwe_algs import ECDHESAlgorithm
from .jwe_algs import u32be_len_input
from .jwe_encs import JWE_ENC_ALGORITHMS
from .jwe_encs import CBCHS2EncAlgorithm
from .jwe_zips import DeflateZipAlgorithm
from .jws_algs import JWS_ALGORITHMS
from .oct_key import OctKey
from .rsa_key import RSAKey


def register_jws_rfc7518(cls):
    for algorithm in JWS_ALGORITHMS:
        cls.register_algorithm(algorithm)


def register_jwe_rfc7518(cls):
    for algorithm in JWE_ALG_ALGORITHMS:
        cls.register_algorithm(algorithm)

    for algorithm in JWE_ENC_ALGORITHMS:
        cls.register_algorithm(algorithm)

    cls.register_algorithm(DeflateZipAlgorithm())


__all__ = [
    "register_jws_rfc7518",
    "register_jwe_rfc7518",
    "OctKey",
    "RSAKey",
    "ECKey",
    "u32be_len_input",
    "AESAlgorithm",
    "ECDHESAlgorithm",
    "CBCHS2EncAlgorithm",
]

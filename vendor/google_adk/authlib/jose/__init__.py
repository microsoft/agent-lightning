"""authlib.jose
~~~~~~~~~~~~

JOSE implementation in Authlib. Tracking the status of JOSE specs at
https://tools.ietf.org/wg/jose/
"""

from .errors import JoseError
from .rfc7515 import JsonWebSignature
from .rfc7515 import JWSAlgorithm
from .rfc7515 import JWSHeader
from .rfc7515 import JWSObject
from .rfc7516 import JsonWebEncryption
from .rfc7516 import JWEAlgorithm
from .rfc7516 import JWEEncAlgorithm
from .rfc7516 import JWEZipAlgorithm
from .rfc7517 import JsonWebKey
from .rfc7517 import Key
from .rfc7517 import KeySet
from .rfc7518 import ECDHESAlgorithm
from .rfc7518 import ECKey
from .rfc7518 import OctKey
from .rfc7518 import RSAKey
from .rfc7518 import register_jwe_rfc7518
from .rfc7518 import register_jws_rfc7518
from .rfc7519 import BaseClaims
from .rfc7519 import JsonWebToken
from .rfc7519 import JWTClaims
from .rfc8037 import OKPKey
from .rfc8037 import register_jws_rfc8037

# register algorithms
register_jws_rfc7518(JsonWebSignature)
register_jws_rfc8037(JsonWebSignature)

register_jwe_rfc7518(JsonWebEncryption)

# attach algorithms
ECDHESAlgorithm.ALLOWED_KEY_CLS = (ECKey, OKPKey)

# register supported keys
JsonWebKey.JWK_KEY_CLS = {
    OctKey.kty: OctKey,
    RSAKey.kty: RSAKey,
    ECKey.kty: ECKey,
    OKPKey.kty: OKPKey,
}

jwt = JsonWebToken(list(JsonWebSignature.ALGORITHMS_REGISTRY.keys()))


__all__ = [
    "JoseError",
    "JsonWebSignature",
    "JWSAlgorithm",
    "JWSHeader",
    "JWSObject",
    "JsonWebEncryption",
    "JWEAlgorithm",
    "JWEEncAlgorithm",
    "JWEZipAlgorithm",
    "JsonWebKey",
    "Key",
    "KeySet",
    "OctKey",
    "RSAKey",
    "ECKey",
    "OKPKey",
    "JsonWebToken",
    "BaseClaims",
    "JWTClaims",
    "jwt",
]

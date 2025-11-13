"""authlib.jose.rfc7515.
~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of
JSON Web Signature (JWS).

https://tools.ietf.org/html/rfc7515
"""

from .jws import JsonWebSignature
from .models import JWSAlgorithm
from .models import JWSHeader
from .models import JWSObject

__all__ = ["JsonWebSignature", "JWSAlgorithm", "JWSHeader", "JWSObject"]

"""authlib.oauth2.rfc8628.
~~~~~~~~~~~~~~~~~~~~~~

This module represents an implementation of
OAuth 2.0 Device Authorization Grant.

https://tools.ietf.org/html/rfc8628
"""

from .device_code import DEVICE_CODE_GRANT_TYPE
from .device_code import DeviceCodeGrant
from .endpoint import DeviceAuthorizationEndpoint
from .errors import AuthorizationPendingError
from .errors import ExpiredTokenError
from .errors import SlowDownError
from .models import DeviceCredentialDict
from .models import DeviceCredentialMixin

__all__ = [
    "DeviceAuthorizationEndpoint",
    "DeviceCodeGrant",
    "DEVICE_CODE_GRANT_TYPE",
    "DeviceCredentialMixin",
    "DeviceCredentialDict",
    "AuthorizationPendingError",
    "SlowDownError",
    "ExpiredTokenError",
]

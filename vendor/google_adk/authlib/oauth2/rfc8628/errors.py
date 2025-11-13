from ..rfc6749.errors import OAuth2Error

# https://tools.ietf.org/html/rfc8628#section-3.5


class AuthorizationPendingError(OAuth2Error):
    """The authorization request is still pending as the end user hasn't
    yet completed the user-interaction steps (Section 3.3).
    """

    error = "authorization_pending"


class SlowDownError(OAuth2Error):
    """A variant of "authorization_pending", the authorization request is
    still pending and polling should continue, but the interval MUST
    be increased by 5 seconds for this and all subsequent requests.
    """

    error = "slow_down"


class ExpiredTokenError(OAuth2Error):
    """The "device_code" has expired, and the device authorization
    session has concluded.  The client MAY commence a new device
    authorization request but SHOULD wait for user interaction before
    restarting to avoid unnecessary polling.
    """

    error = "expired_token"

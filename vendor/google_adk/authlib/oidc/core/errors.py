from authlib.oauth2 import OAuth2Error


class InteractionRequiredError(OAuth2Error):
    """The Authorization Server requires End-User interaction of some form
    to proceed. This error MAY be returned when the prompt parameter value
    in the Authentication Request is none, but the Authentication Request
    cannot be completed without displaying a user interface for End-User
    interaction.

    http://openid.net/specs/openid-connect-core-1_0.html#AuthError
    """

    error = "interaction_required"


class LoginRequiredError(OAuth2Error):
    """The Authorization Server requires End-User authentication. This error
    MAY be returned when the prompt parameter value in the Authentication
    Request is none, but the Authentication Request cannot be completed
    without displaying a user interface for End-User authentication.

    http://openid.net/specs/openid-connect-core-1_0.html#AuthError
    """

    error = "login_required"


class AccountSelectionRequiredError(OAuth2Error):
    """The End-User is REQUIRED to select a session at the Authorization
    Server. The End-User MAY be authenticated at the Authorization Server
    with different associated accounts, but the End-User did not select a
    session. This error MAY be returned when the prompt parameter value in
    the Authentication Request is none, but the Authentication Request cannot
    be completed without displaying a user interface to prompt for a session
    to use.

    http://openid.net/specs/openid-connect-core-1_0.html#AuthError
    """

    error = "account_selection_required"


class ConsentRequiredError(OAuth2Error):
    """The Authorization Server requires End-User consent. This error MAY be
    returned when the prompt parameter value in the Authentication Request is
    none, but the Authentication Request cannot be completed without
    displaying a user interface for End-User consent.

    http://openid.net/specs/openid-connect-core-1_0.html#AuthError
    """

    error = "consent_required"


class InvalidRequestURIError(OAuth2Error):
    """The request_uri in the Authorization Request returns an error or
    contains invalid data.

    http://openid.net/specs/openid-connect-core-1_0.html#AuthError
    """

    error = "invalid_request_uri"


class InvalidRequestObjectError(OAuth2Error):
    """The request parameter contains an invalid Request Object."""

    error = "invalid_request_object"


class RequestNotSupportedError(OAuth2Error):
    """The OP does not support use of the request parameter."""

    error = "request_not_supported"


class RequestURINotSupportedError(OAuth2Error):
    """The OP does not support use of the request_uri parameter."""

    error = "request_uri_not_supported"


class RegistrationNotSupportedError(OAuth2Error):
    """The OP does not support use of the registration parameter."""

    error = "registration_not_supported"

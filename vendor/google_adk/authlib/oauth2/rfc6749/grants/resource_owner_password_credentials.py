import logging

from ..errors import InvalidRequestError
from ..errors import UnauthorizedClientError
from ..hooks import hooked
from .base import BaseGrant
from .base import TokenEndpointMixin

log = logging.getLogger(__name__)


class ResourceOwnerPasswordCredentialsGrant(BaseGrant, TokenEndpointMixin):
    """The resource owner password credentials grant type is suitable in
    cases where the resource owner has a trust relationship with the
    client, such as the device operating system or a highly privileged.

    application.  The authorization server should take special care when
    enabling this grant type and only allow it when other flows are not
    viable.

    This grant type is suitable for clients capable of obtaining the
    resource owner's credentials (username and password, typically using
    an interactive form).  It is also used to migrate existing clients
    using direct authentication schemes such as HTTP Basic or Digest
    authentication to OAuth by converting the stored credentials to an
    access token::

        +----------+
        | Resource |
        |  Owner   |
        |          |
        +----------+
            v
            |    Resource Owner
           (A) Password Credentials
            |
            v
        +---------+                                  +---------------+
        |         |>--(B)---- Resource Owner ------->|               |
        |         |         Password Credentials     | Authorization |
        | Client  |                                  |     Server    |
        |         |<--(C)---- Access Token ---------<|               |
        |         |    (w/ Optional Refresh Token)   |               |
        +---------+                                  +---------------+
    """

    GRANT_TYPE = "password"

    def validate_token_request(self):
        """The client makes a request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per Appendix B with a character encoding of UTF-8 in the HTTP
        request entity-body:

        grant_type
             REQUIRED.  Value MUST be set to "password".

        username
             REQUIRED.  The resource owner username.

        password
             REQUIRED.  The resource owner password.

        scope
             OPTIONAL.  The scope of the access request as described by
             Section 3.3.

        If the client type is confidential or the client was issued client
        credentials (or assigned other authentication requirements), the
        client MUST authenticate with the authorization server as described
        in Section 3.2.1.

        For example, the client makes the following HTTP request using
        transport-layer security (with extra line breaks for display purposes
        only):

        .. code-block:: http

            POST /token HTTP/1.1
            Host: server.example.com
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
            Content-Type: application/x-www-form-urlencoded

            grant_type=password&username=johndoe&password=A3ddj3w
        """
        # ignore validate for grant_type, since it is validated by
        # check_token_endpoint
        client = self.authenticate_token_endpoint_client()
        log.debug("Validate token request of %r", client)

        if not client.check_grant_type(self.GRANT_TYPE):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'grant_type={self.GRANT_TYPE}'"
            )

        params = self.request.form
        if "username" not in params:
            raise InvalidRequestError("Missing 'username' in request.")
        if "password" not in params:
            raise InvalidRequestError("Missing 'password' in request.")

        log.debug("Authenticate user of %r", params["username"])
        user = self.authenticate_user(params["username"], params["password"])
        if not user:
            raise InvalidRequestError(
                "Invalid 'username' or 'password' in request.",
            )
        self.request.client = client
        self.request.user = user
        self.validate_requested_scope()

    @hooked
    def create_token_response(self):
        """If the access token request is valid and authorized, the
        authorization server issues an access token and optional refresh
        token as described in Section 5.1.  If the request failed client
        authentication or is invalid, the authorization server returns an
        error response as described in Section 5.2.

        An example successful response:

        .. code-block:: http

            HTTP/1.1 200 OK
            Content-Type: application/json
            Cache-Control: no-store
            Pragma: no-cache

            {
                "access_token":"2YotnFZFEjr1zCsicMWpAA",
                "token_type":"example",
                "expires_in":3600,
                "refresh_token":"tGzv3JOkF0XG5Qx2TlKWIA",
                "example_parameter":"example_value"
            }

        :returns: (status_code, body, headers)
        """
        user = self.request.user
        scope = self.request.payload.scope
        token = self.generate_token(user=user, scope=scope)
        log.debug("Issue token %r to %r", token, self.client)
        self.save_token(token)
        return 200, token, self.TOKEN_RESPONSE_HEADER

    def authenticate_user(self, username, password):
        """Validate the resource owner password credentials using its
        existing password validation algorithm::

            def authenticate_user(self, username, password):
                user = get_user_by_username(username)
                if user.check_password(password):
                    return user
        """
        raise NotImplementedError()

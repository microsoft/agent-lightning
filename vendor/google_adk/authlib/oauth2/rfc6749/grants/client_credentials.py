import logging

from ..errors import UnauthorizedClientError
from ..hooks import hooked
from .base import BaseGrant
from .base import TokenEndpointMixin

log = logging.getLogger(__name__)


class ClientCredentialsGrant(BaseGrant, TokenEndpointMixin):
    """The client can request an access token using only its client
    credentials (or other supported means of authentication) when the
    client is requesting access to the protected resources under its
    control, or those of another resource owner that have been previously
    arranged with the authorization server.

    The client credentials grant type MUST only be used by confidential
    clients::

        +---------+                                  +---------------+
        |         |                                  |               |
        |         |>--(A)- Client Authentication --->| Authorization |
        | Client  |                                  |     Server    |
        |         |<--(B)---- Access Token ---------<|               |
        |         |                                  |               |
        +---------+                                  +---------------+

    https://tools.ietf.org/html/rfc6749#section-4.4
    """

    GRANT_TYPE = "client_credentials"

    def validate_token_request(self):
        """The client makes a request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per Appendix B with a character encoding of UTF-8 in the HTTP
        request entity-body:

        grant_type
             REQUIRED.  Value MUST be set to "client_credentials".

        scope
             OPTIONAL.  The scope of the access request as described by
             Section 3.3.

        The client MUST authenticate with the authorization server as
        described in Section 3.2.1.

        For example, the client makes the following HTTP request using
        transport-layer security (with extra line breaks for display purposes
        only):

        .. code-block:: http

            POST /token HTTP/1.1
            Host: server.example.com
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
            Content-Type: application/x-www-form-urlencoded

            grant_type=client_credentials

        The authorization server MUST authenticate the client.
        """
        # ignore validate for grant_type, since it is validated by
        # check_token_endpoint
        client = self.authenticate_token_endpoint_client()
        log.debug("Validate token request of %r", client)

        if not client.check_grant_type(self.GRANT_TYPE):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'grant_type={self.GRANT_TYPE}'"
            )

        self.request.client = client
        self.validate_requested_scope()

    @hooked
    def create_token_response(self):
        """If the access token request is valid and authorized, the
        authorization server issues an access token as described in
        Section 5.1.  A refresh token SHOULD NOT be included.  If the request
        failed client authentication or is invalid, the authorization server
        returns an error response as described in Section 5.2.

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
                "example_parameter":"example_value"
            }

        :returns: (status_code, body, headers)
        """
        token = self.generate_token(
            scope=self.request.payload.scope, include_refresh_token=False
        )
        log.debug("Issue token %r to %r", token, self.client)
        self.save_token(token)
        return 200, token, self.TOKEN_RESPONSE_HEADER

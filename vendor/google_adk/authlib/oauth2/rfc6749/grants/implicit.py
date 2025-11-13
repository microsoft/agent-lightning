import logging

from authlib.common.urls import add_params_to_uri

from ..errors import AccessDeniedError
from ..errors import OAuth2Error
from ..errors import UnauthorizedClientError
from ..hooks import hooked
from .base import AuthorizationEndpointMixin
from .base import BaseGrant

log = logging.getLogger(__name__)


class ImplicitGrant(BaseGrant, AuthorizationEndpointMixin):
    """The implicit grant type is used to obtain access tokens (it does not
    support the issuance of refresh tokens) and is optimized for public
    clients known to operate a particular redirection URI.  These clients
    are typically implemented in a browser using a scripting language
    such as JavaScript.

    Since this is a redirection-based flow, the client must be capable of
    interacting with the resource owner's user-agent (typically a web
    browser) and capable of receiving incoming requests (via redirection)
    from the authorization server.

    Unlike the authorization code grant type, in which the client makes
    separate requests for authorization and for an access token, the
    client receives the access token as the result of the authorization
    request.

    The implicit grant type does not include client authentication, and
    relies on the presence of the resource owner and the registration of
    the redirection URI.  Because the access token is encoded into the
    redirection URI, it may be exposed to the resource owner and other
    applications residing on the same device::

        +----------+
        | Resource |
        |  Owner   |
        |          |
        +----------+
             ^
             |
            (B)
        +----|-----+          Client Identifier     +---------------+
        |         -+----(A)-- & Redirection URI --->|               |
        |  User-   |                                | Authorization |
        |  Agent  -|----(B)-- User authenticates -->|     Server    |
        |          |                                |               |
        |          |<---(C)--- Redirection URI ----<|               |
        |          |          with Access Token     +---------------+
        |          |            in Fragment
        |          |                                +---------------+
        |          |----(D)--- Redirection URI ---->|   Web-Hosted  |
        |          |          without Fragment      |     Client    |
        |          |                                |    Resource   |
        |     (F)  |<---(E)------- Script ---------<|               |
        |          |                                +---------------+
        +-|--------+
          |    |
         (A)  (G) Access Token
          |    |
          ^    v
        +---------+
        |         |
        |  Client |
        |         |
        +---------+
    """

    #: authorization_code grant type has authorization endpoint
    AUTHORIZATION_ENDPOINT = True
    #: Allowed client auth methods for token endpoint
    TOKEN_ENDPOINT_AUTH_METHODS = ["none"]

    RESPONSE_TYPES = {"token"}
    GRANT_TYPE = "implicit"
    ERROR_RESPONSE_FRAGMENT = True

    @hooked
    def validate_authorization_request(self):
        """The client constructs the request URI by adding the following
        parameters to the query component of the authorization endpoint URI
        using the "application/x-www-form-urlencoded" format.
        Per `Section 4.2.1`_.

        response_type
             REQUIRED.  Value MUST be set to "token".

        client_id
             REQUIRED.  The client identifier as described in Section 2.2.

        redirect_uri
             OPTIONAL.  As described in Section 3.1.2.

        scope
             OPTIONAL.  The scope of the access request as described by
             Section 3.3.

        state
             RECOMMENDED.  An opaque value used by the client to maintain
             state between the request and callback.  The authorization
             server includes this value when redirecting the user-agent back
             to the client.  The parameter SHOULD be used for preventing
             cross-site request forgery as described in Section 10.12.

        The client directs the resource owner to the constructed URI using an
        HTTP redirection response, or by other means available to it via the
        user-agent.

        For example, the client directs the user-agent to make the following
        HTTP request using TLS:

        .. code-block:: http

            GET /authorize?response_type=token&client_id=s6BhdRkqt3&state=xyz
            &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1
            Host: server.example.com

        .. _`Section 4.2.1`: https://tools.ietf.org/html/rfc6749#section-4.2.1
        """
        # ignore validate for response_type, since it is validated by
        # check_authorization_endpoint

        # The implicit grant type is optimized for public clients
        client = self.authenticate_token_endpoint_client()
        log.debug("Validate authorization request of %r", client)

        redirect_uri = self.validate_authorization_redirect_uri(self.request, client)

        response_type = self.request.payload.response_type
        if not client.check_response_type(response_type):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'response_type={response_type}'",
                redirect_uri=redirect_uri,
                redirect_fragment=True,
            )

        try:
            self.request.client = client
            self.validate_requested_scope()
        except OAuth2Error as error:
            error.redirect_uri = redirect_uri
            error.redirect_fragment = True
            raise error
        return redirect_uri

    @hooked
    def create_authorization_response(self, redirect_uri, grant_user):
        """If the resource owner grants the access request, the authorization
        server issues an access token and delivers it to the client by adding
        the following parameters to the fragment component of the redirection
        URI using the "application/x-www-form-urlencoded" format.
        Per `Section 4.2.2`_.

        access_token
             REQUIRED.  The access token issued by the authorization server.

        token_type
             REQUIRED.  The type of the token issued as described in
             Section 7.1.  Value is case insensitive.

        expires_in
             RECOMMENDED.  The lifetime in seconds of the access token.  For
             example, the value "3600" denotes that the access token will
             expire in one hour from the time the response was generated.
             If omitted, the authorization server SHOULD provide the
             expiration time via other means or document the default value.

        scope
             OPTIONAL, if identical to the scope requested by the client;
             otherwise, REQUIRED.  The scope of the access token as
             described by Section 3.3.

        state
             REQUIRED if the "state" parameter was present in the client
             authorization request.  The exact value received from the
             client.

        The authorization server MUST NOT issue a refresh token.

        For example, the authorization server redirects the user-agent by
        sending the following HTTP response:

        .. code-block:: http

            HTTP/1.1 302 Found
            Location: http://example.com/cb#access_token=2YotnFZFEjr1zCsicMWpAA
                   &state=xyz&token_type=example&expires_in=3600

        Developers should note that some user-agents do not support the
        inclusion of a fragment component in the HTTP "Location" response
        header field.  Such clients will require using other methods for
        redirecting the client than a 3xx redirection response -- for
        example, returning an HTML page that includes a 'continue' button
        with an action linked to the redirection URI.

        .. _`Section 4.2.2`: https://tools.ietf.org/html/rfc6749#section-4.2.2

        :param redirect_uri: Redirect to the given URI for the authorization
        :param grant_user: if resource owner granted the request, pass this
            resource owner, otherwise pass None.
        :returns: (status_code, body, headers)
        """
        state = self.request.payload.state
        if grant_user:
            self.request.user = grant_user
            token = self.generate_token(
                user=grant_user,
                scope=self.request.payload.scope,
                include_refresh_token=False,
            )
            log.debug("Grant token %r to %r", token, self.request.client)

            self.save_token(token)
            params = [(k, token[k]) for k in token]
            if state:
                params.append(("state", state))

            uri = add_params_to_uri(redirect_uri, params, fragment=True)
            headers = [("Location", uri)]
            return 302, "", headers
        else:
            raise AccessDeniedError(redirect_uri=redirect_uri, redirect_fragment=True)

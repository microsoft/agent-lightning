import logging

from authlib.common.security import generate_token
from authlib.common.urls import add_params_to_uri

from ..errors import AccessDeniedError
from ..errors import InvalidClientError
from ..errors import InvalidGrantError
from ..errors import InvalidRequestError
from ..errors import OAuth2Error
from ..errors import UnauthorizedClientError
from ..hooks import hooked
from .base import AuthorizationEndpointMixin
from .base import BaseGrant
from .base import TokenEndpointMixin

log = logging.getLogger(__name__)


class AuthorizationCodeGrant(BaseGrant, AuthorizationEndpointMixin, TokenEndpointMixin):
    """The authorization code grant type is used to obtain both access
    tokens and refresh tokens and is optimized for confidential clients.
    Since this is a redirection-based flow, the client must be capable of
    interacting with the resource owner's user-agent (typically a web
    browser) and capable of receiving incoming requests (via redirection)
    from the authorization server::

        +----------+
        | Resource |
        |   Owner  |
        |          |
        +----------+
             ^
             |
            (B)
        +----|-----+          Client Identifier      +---------------+
        |         -+----(A)-- & Redirection URI ---->|               |
        |  User-   |                                 | Authorization |
        |  Agent  -+----(B)-- User authenticates --->|     Server    |
        |          |                                 |               |
        |         -+----(C)-- Authorization Code ---<|               |
        +-|----|---+                                 +---------------+
          |    |                                         ^      v
         (A)  (C)                                        |      |
          |    |                                         |      |
          ^    v                                         |      |
        +---------+                                      |      |
        |         |>---(D)-- Authorization Code ---------'      |
        |  Client |          & Redirection URI                  |
        |         |                                             |
        |         |<---(E)----- Access Token -------------------'
        +---------+       (w/ Optional Refresh Token)
    """

    #: Allowed client auth methods for token endpoint
    TOKEN_ENDPOINT_AUTH_METHODS = ["client_secret_basic", "client_secret_post"]

    #: Generated "code" length
    AUTHORIZATION_CODE_LENGTH = 48

    RESPONSE_TYPES = {"code"}
    GRANT_TYPE = "authorization_code"

    def validate_authorization_request(self):
        """The client constructs the request URI by adding the following
        parameters to the query component of the authorization endpoint URI
        using the "application/x-www-form-urlencoded" format.
        Per `Section 4.1.1`_.

        response_type
             REQUIRED.  Value MUST be set to "code".

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
        HTTP request using TLS (with extra line breaks for display purposes
        only):

        .. code-block:: http

            GET /authorize?response_type=code&client_id=s6BhdRkqt3&state=xyz
            &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1
            Host: server.example.com

        The authorization server validates the request to ensure that all
        required parameters are present and valid.  If the request is valid,
        the authorization server authenticates the resource owner and obtains
        an authorization decision (by asking the resource owner or by
        establishing approval via other means).

        .. _`Section 4.1.1`: https://tools.ietf.org/html/rfc6749#section-4.1.1
        """
        return validate_code_authorization_request(self)

    def create_authorization_response(self, redirect_uri: str, grant_user):
        """If the resource owner grants the access request, the authorization
        server issues an authorization code and delivers it to the client by
        adding the following parameters to the query component of the
        redirection URI using the "application/x-www-form-urlencoded" format.
        Per `Section 4.1.2`_.

        code
            REQUIRED.  The authorization code generated by the
            authorization server. The authorization code MUST expire
            shortly after it is issued to mitigate the risk of leaks. A
            maximum authorization code lifetime of 10 minutes is
            RECOMMENDED. The client MUST NOT use the authorization code
            more than once. If an authorization code is used more than
            once, the authorization server MUST deny the request and SHOULD
            revoke (when possible) all tokens previously issued based on
            that authorization code.  The authorization code is bound to
            the client identifier and redirection URI.
        state
            REQUIRED if the "state" parameter was present in the client
            authorization request.  The exact value received from the
            client.

        For example, the authorization server redirects the user-agent by
        sending the following HTTP response.

        .. code-block:: http

            HTTP/1.1 302 Found
            Location: https://client.example.com/cb?code=SplxlOBeZQQYbYS6WxSbIA
                   &state=xyz

        .. _`Section 4.1.2`: https://tools.ietf.org/html/rfc6749#section-4.1.2

        :param redirect_uri: Redirect to the given URI for the authorization
        :param grant_user: if resource owner granted the request, pass this
            resource owner, otherwise pass None.
        :returns: (status_code, body, headers)
        """
        if not grant_user:
            raise AccessDeniedError(redirect_uri=redirect_uri)

        self.request.user = grant_user

        code = self.generate_authorization_code()
        self.save_authorization_code(code, self.request)

        params = [("code", code)]
        if self.request.payload.state:
            params.append(("state", self.request.payload.state))
        uri = add_params_to_uri(redirect_uri, params)
        headers = [("Location", uri)]
        return 302, "", headers

    @hooked
    def validate_token_request(self):
        """The client makes a request to the token endpoint by sending the
        following parameters using the "application/x-www-form-urlencoded"
        format per `Section 4.1.3`_:

        grant_type
             REQUIRED.  Value MUST be set to "authorization_code".

        code
             REQUIRED.  The authorization code received from the
             authorization server.

        redirect_uri
             REQUIRED, if the "redirect_uri" parameter was included in the
             authorization request as described in Section 4.1.1, and their
             values MUST be identical.

        client_id
             REQUIRED, if the client is not authenticating with the
             authorization server as described in Section 3.2.1.

        If the client type is confidential or the client was issued client
        credentials (or assigned other authentication requirements), the
        client MUST authenticate with the authorization server as described
        in Section 3.2.1.

        For example, the client makes the following HTTP request using TLS:

        .. code-block:: http

            POST /token HTTP/1.1
            Host: server.example.com
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
            Content-Type: application/x-www-form-urlencoded

            grant_type=authorization_code&code=SplxlOBeZQQYbYS6WxSbIA
            &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb

        .. _`Section 4.1.3`: https://tools.ietf.org/html/rfc6749#section-4.1.3
        """
        # ignore validate for grant_type, since it is validated by
        # check_token_endpoint

        # authenticate the client if client authentication is included
        client = self.authenticate_token_endpoint_client()

        log.debug("Validate token request of %r", client)
        if not client.check_grant_type(self.GRANT_TYPE):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'grant_type={self.GRANT_TYPE}'"
            )

        code = self.request.form.get("code")
        if code is None:
            raise InvalidRequestError("Missing 'code' in request.")

        # ensure that the authorization code was issued to the authenticated
        # confidential client, or if the client is public, ensure that the
        # code was issued to "client_id" in the request
        authorization_code = self.query_authorization_code(code, client)
        if not authorization_code:
            raise InvalidGrantError("Invalid 'code' in request.")

        # validate redirect_uri parameter
        log.debug("Validate token redirect_uri of %r", client)
        redirect_uri = self.request.payload.redirect_uri
        original_redirect_uri = authorization_code.get_redirect_uri()
        if original_redirect_uri and redirect_uri != original_redirect_uri:
            raise InvalidGrantError("Invalid 'redirect_uri' in request.")

        # save for create_token_response
        self.request.client = client
        self.request.authorization_code = authorization_code

    @hooked
    def create_token_response(self):
        """If the access token request is valid and authorized, the
        authorization server issues an access token and optional refresh
        token as described in Section 5.1.  If the request client
        authentication failed or is invalid, the authorization server returns
        an error response as described in Section 5.2. Per `Section 4.1.4`_.

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

        .. _`Section 4.1.4`: https://tools.ietf.org/html/rfc6749#section-4.1.4
        """
        client = self.request.client
        authorization_code = self.request.authorization_code

        user = self.authenticate_user(authorization_code)
        if not user:
            raise InvalidGrantError("There is no 'user' for this code.")
        self.request.user = user

        scope = authorization_code.get_scope()
        token = self.generate_token(
            user=user,
            scope=scope,
            include_refresh_token=client.check_grant_type("refresh_token"),
        )
        log.debug("Issue token %r to %r", token, client)

        self.save_token(token)
        self.delete_authorization_code(authorization_code)
        return 200, token, self.TOKEN_RESPONSE_HEADER

    def generate_authorization_code(self):
        """ "The method to generate "code" value for authorization code data.
        Developers may rewrite this method, or customize the code length with::

            class MyAuthorizationCodeGrant(AuthorizationCodeGrant):
                AUTHORIZATION_CODE_LENGTH = 32  # default is 48
        """
        return generate_token(self.AUTHORIZATION_CODE_LENGTH)

    def save_authorization_code(self, code, request):
        """Save authorization_code for later use. Developers MUST implement
        it in subclass. Here is an example::

            def save_authorization_code(self, code, request):
                client = request.client
                item = AuthorizationCode(
                    code=code,
                    client_id=client.client_id,
                    redirect_uri=request.payload.redirect_uri,
                    scope=request.payload.scope,
                    user_id=request.user.id,
                )
                item.save()
        """
        raise NotImplementedError()

    def query_authorization_code(self, code, client):  # pragma: no cover
        """Get authorization_code from previously savings. Developers MUST
        implement it in subclass::

            def query_authorization_code(self, code, client):
                return Authorization.get(code=code, client_id=client.client_id)

        :param code: a string represent the code.
        :param client: client related to this code.
        :return: authorization_code object
        """
        raise NotImplementedError()

    def delete_authorization_code(self, authorization_code):
        """Delete authorization code from database or cache. Developers MUST
        implement it in subclass, e.g.::

            def delete_authorization_code(self, authorization_code):
                authorization_code.delete()

        :param authorization_code: the instance of authorization_code
        """
        raise NotImplementedError()

    def authenticate_user(self, authorization_code):
        """Authenticate the user related to this authorization_code. Developers
        MUST implement this method in subclass, e.g.::

            def authenticate_user(self, authorization_code):
                return User.get(authorization_code.user_id)

        :param authorization_code: AuthorizationCode object
        :return: user
        """
        raise NotImplementedError()


def validate_code_authorization_request(grant):
    request = grant.request
    client_id = request.payload.client_id
    log.debug("Validate authorization request of %r", client_id)

    if client_id is None:
        raise InvalidClientError(
            description="Missing 'client_id' parameter.",
        )

    client = grant.server.query_client(client_id)
    if not client:
        raise InvalidClientError(
            description="The client does not exist on this server.",
        )

    redirect_uri = grant.validate_authorization_redirect_uri(request, client)
    response_type = request.payload.response_type
    if not client.check_response_type(response_type):
        raise UnauthorizedClientError(
            f"The client is not authorized to use 'response_type={response_type}'",
            redirect_uri=redirect_uri,
        )

    grant.request.client = client

    @hooked
    def validate_authorization_request_payload(grant, redirect_uri):
        grant.validate_requested_scope()

    try:
        validate_authorization_request_payload(grant, redirect_uri)
    except OAuth2Error as error:
        error.redirect_uri = redirect_uri
        raise error
    return redirect_uri

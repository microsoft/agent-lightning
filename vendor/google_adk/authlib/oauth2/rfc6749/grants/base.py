from authlib.consts import default_json_headers

from ..errors import InvalidRequestError
from ..hooks import Hookable
from ..hooks import hooked
from ..requests import OAuth2Request


class BaseGrant(Hookable):
    #: Allowed client auth methods for token endpoint
    TOKEN_ENDPOINT_AUTH_METHODS = ["client_secret_basic"]

    #: Designed for which "grant_type"
    GRANT_TYPE = None

    # NOTE: there is no charset for application/json, since
    # application/json should always in UTF-8.
    # The example on RFC is incorrect.
    # https://tools.ietf.org/html/rfc4627
    TOKEN_RESPONSE_HEADER = default_json_headers

    def __init__(self, request: OAuth2Request, server):
        super().__init__()
        self.prompt = None
        self.redirect_uri = None
        self.request = request
        self.server = server

    @property
    def client(self):
        return self.request.client

    def generate_token(
        self,
        user=None,
        scope=None,
        grant_type=None,
        expires_in=None,
        include_refresh_token=True,
    ):
        if grant_type is None:
            grant_type = self.GRANT_TYPE
        return self.server.generate_token(
            client=self.request.client,
            grant_type=grant_type,
            user=user,
            scope=scope,
            expires_in=expires_in,
            include_refresh_token=include_refresh_token,
        )

    def authenticate_token_endpoint_client(self):
        """Authenticate client with the given methods for token endpoint.

        For example, the client makes the following HTTP request using TLS:

        .. code-block:: http

            POST /token HTTP/1.1
            Host: server.example.com
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
            Content-Type: application/x-www-form-urlencoded

            grant_type=authorization_code&code=SplxlOBeZQQYbYS6WxSbIA
            &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb

        Default available methods are: "none", "client_secret_basic" and
        "client_secret_post".

        :return: client
        """
        client = self.server.authenticate_client(
            self.request, self.TOKEN_ENDPOINT_AUTH_METHODS
        )
        self.server.send_signal("after_authenticate_client", client=client, grant=self)
        return client

    def save_token(self, token):
        """A method to save token into database."""
        return self.server.save_token(token, self.request)

    def validate_requested_scope(self):
        """Validate if requested scope is supported by Authorization Server."""
        scope = self.request.payload.scope
        return self.server.validate_requested_scope(scope)


class TokenEndpointMixin:
    #: Allowed HTTP methods of this token endpoint
    TOKEN_ENDPOINT_HTTP_METHODS = ["POST"]

    #: Designed for which "grant_type"
    GRANT_TYPE = None

    @classmethod
    def check_token_endpoint(cls, request: OAuth2Request):
        return (
            request.payload.grant_type == cls.GRANT_TYPE
            and request.method in cls.TOKEN_ENDPOINT_HTTP_METHODS
        )

    def validate_token_request(self):
        raise NotImplementedError()

    def create_token_response(self):
        raise NotImplementedError()


class AuthorizationEndpointMixin:
    RESPONSE_TYPES = set()
    ERROR_RESPONSE_FRAGMENT = False

    @classmethod
    def check_authorization_endpoint(cls, request: OAuth2Request):
        return request.payload.response_type in cls.RESPONSE_TYPES

    @staticmethod
    def validate_authorization_redirect_uri(request: OAuth2Request, client):
        if request.payload.redirect_uri:
            if not client.check_redirect_uri(request.payload.redirect_uri):
                raise InvalidRequestError(
                    f"Redirect URI {request.payload.redirect_uri} is not supported by client.",
                )
            return request.payload.redirect_uri
        else:
            redirect_uri = client.get_default_redirect_uri()
            if not redirect_uri:
                raise InvalidRequestError(
                    "Missing 'redirect_uri' in request.", state=request.payload.state
                )
            return redirect_uri

    @staticmethod
    def validate_no_multiple_request_parameter(request: OAuth2Request):
        """For the Authorization Endpoint, request and response parameters MUST NOT be included
        more than once. Per `Section 3.1`_.

        .. _`Section 3.1`: https://tools.ietf.org/html/rfc6749#section-3.1
        """
        datalist = request.payload.datalist
        parameters = ["response_type", "client_id", "redirect_uri", "scope", "state"]
        for param in parameters:
            if len(datalist.get(param, [])) > 1:
                raise InvalidRequestError(
                    f"Multiple '{param}' in request.", state=request.payload.state
                )

    @hooked
    def validate_consent_request(self):
        redirect_uri = self.validate_authorization_request()
        self.redirect_uri = redirect_uri
        return redirect_uri

    def validate_authorization_request(self):
        raise NotImplementedError()

    def create_authorization_response(self, redirect_uri: str, grant_user):
        raise NotImplementedError()
